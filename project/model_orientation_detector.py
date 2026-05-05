"""
Определение ориентации модели с помощью мультимодальной LLM.
Рендерит модель с разных углов и спрашивает у LLM, где "перед".

МУЛЬТИМОДАЛЬНЫЕ МОДЕЛИ:
По умолчанию пытается использовать текущую модель проекта.
Если она не поддерживает изображения, нужна мультимодальная модель.

Популярные мультимодальные модели:
- minicpm-v:8b (легковесная, ~5GB)
- llama3.2-vision:11b (~7GB)
- llava:13b (~8GB)

Установка:
  ollama pull minicpm-v:8b

Использование:
  # По умолчанию использует модель из DEFAULT_MODEL
  offset = compute_yaw_offset(model_path, "chair")

  # Или укажите мультимодальную модель:
  offset = compute_yaw_offset(
      model_path, "chair",
      llm_model="minicpm-v:8b"
  )
"""

import os
import json
import hashlib
import threading
try:
    import numpy as np
except Exception:  # pragma: no cover
    np = None

try:
    import trimesh
except Exception:  # pragma: no cover
    trimesh = None

try:
    from PIL import Image
except Exception:  # pragma: no cover
    Image = None
from project.llm_request import DEFAULT_MODEL
import base64
try:
    import requests
except Exception:  # pragma: no cover
    requests = None


# ---------------------------------------------------------------------------
# Persistent disk cache for orientation results.
# Key: sha1 of the absolute model path. Value: dict with offset, front_yaw, etc.
# Avoids re-querying the VLM for the same GLB across runs.
# ---------------------------------------------------------------------------
_CACHE_LOCK = threading.Lock()


def _cache_path(cache_dir):
    return os.path.join(cache_dir, "cache.json")


def _cache_key(model_path):
    return hashlib.sha1(os.path.abspath(model_path).encode("utf-8")).hexdigest()


def _cache_load(cache_dir):
    path = _cache_path(cache_dir)
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def _cache_save(cache_dir, data):
    os.makedirs(cache_dir, exist_ok=True)
    path = _cache_path(cache_dir)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


def render_model_view_trimesh(mesh, yaw_deg=0, resolution=(256, 256)):
    """
    Рендерит модель с помощью trimesh (fallback метод).

    Args:
        mesh: trimesh объект
        yaw_deg: угол поворота модели вокруг вертикальной оси (0-360)
        resolution: разрешение изображения (width, height)

    Returns:
        PIL.Image: отрендеренное изображение
    """
    if trimesh is None or np is None or Image is None:
        raise RuntimeError(
            "render_model_view_trimesh requires trimesh, numpy, Pillow"
        )
    # Применяем поворот к мешу
    yaw_rad = np.radians(yaw_deg)
    rotation_matrix = trimesh.transformations.rotation_matrix(
        yaw_rad, [0, 1, 0]
    )
    mesh_copy = mesh.copy()
    mesh_copy.apply_transform(rotation_matrix)

    # Создаём сцену
    scene = mesh_copy.scene()

    # Рендерим с помощью trimesh
    try:
        png_data = scene.save_image(resolution=resolution)
        from io import BytesIO
        return Image.open(BytesIO(png_data))
    except Exception as e:
        print(f"[orientation_detector] Ошибка рендеринга: {e}")
        # Создаём пустое изображение как fallback
        return Image.new('RGB', resolution, color=(200, 200, 200))


def render_model_view(mesh, yaw_deg=0, distance=2.0, resolution=(256, 256)):
    """
    Рендерит модель с заданного угла обзора.

    Args:
        mesh: trimesh объект
        yaw_deg: угол поворота модели вокруг вертикальной оси (0-360)
        distance: расстояние камеры от модели
        resolution: разрешение изображения (width, height)

    Returns:
        PIL.Image: отрендеренное изображение
    """
    if trimesh is None or np is None or Image is None:
        raise RuntimeError("render_model_view requires trimesh, numpy, Pillow")
    # Пробуем использовать pyrender с OSMesa
    try:
        os.environ['PYOPENGL_PLATFORM'] = 'osmesa'
        import pyrender

        # Создаём сцену pyrender
        scene = pyrender.Scene(ambient_light=[0.3, 0.3, 0.3])

        # Применяем поворот к мешу
        yaw_rad = np.radians(yaw_deg)
        rotation_matrix = trimesh.transformations.rotation_matrix(
            yaw_rad, [0, 1, 0]
        )
        mesh_copy = mesh.copy()
        mesh_copy.apply_transform(rotation_matrix)

        # Добавляем меш в сцену
        mesh_pyrender = pyrender.Mesh.from_trimesh(mesh_copy, smooth=True)
        scene.add(mesh_pyrender)

        # Настраиваем камеру
        camera = pyrender.PerspectiveCamera(yfov=np.pi / 3.0)
        camera_pose = np.array([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, distance],
            [0.0, 0.0, 0.0, 1.0]
        ])
        scene.add(camera, pose=camera_pose)

        # Добавляем свет
        light = pyrender.DirectionalLight(
            color=[1.0, 1.0, 1.0], intensity=3.0
        )
        scene.add(light, pose=camera_pose)

        # Рендерим
        renderer = pyrender.OffscreenRenderer(*resolution)
        try:
            color, depth = renderer.render(scene)
            return Image.fromarray(color)
        finally:
            renderer.delete()

    except Exception as e:
        print(f"[orientation_detector] pyrender не работает: {e}")
        print("[orientation_detector] Используем trimesh рендеринг...")
        return render_model_view_trimesh(mesh, yaw_deg, resolution)


def detect_model_front_with_llm(
    model_path,
    model_name="object",
    cache_dir=".cache/orientation",
    llm_model=None,
    use_remote=True,
    remote_url="http://94.19.29.186:11434",
    num_views=4
):
    """
    Определяет "перед" модели с помощью мультимодальной LLM.

    Args:
        model_path: путь к GLB/OBJ файлу модели
        model_name: название модели для контекста
        cache_dir: директория для кэширования рендеров
        llm_model: название мультимодальной модели
        use_remote: использовать удаленный сервер (по умолчанию True)
        remote_url: URL удаленного Ollama сервера
        num_views: 2 (только 0°/180°) или 4 (0°/90°/180°/270°). По умолчанию 4.

    Returns:
        dict: {
            "front_yaw": int,  # угол в градусах (0, 90, 180, 270)
            "confidence": str,  # "high", "medium", "low"
            "reasoning": str    # объяснение от LLM
        }
    """
    if trimesh is None or np is None or Image is None:
        raise RuntimeError(
            "VLM detector requires trimesh, numpy, Pillow (and optionally requests)"
        )
    if use_remote and requests is None:
        raise RuntimeError("Remote VLM detector requires 'requests'")
    # Определяем модель для использования
    if llm_model is None:
        if use_remote:
            llm_model = "qwen3.5:9b-q4_K_M"  # Удаленная мультимодальная
        else:
            llm_model = DEFAULT_MODEL
        print(f"[orientation_detector] Используем модель по умолчанию: "
              f"{llm_model}")

    server_info = f"удаленный {remote_url}" if use_remote else "локальный"
    print(f"[orientation_detector] Сервер: {server_info}")
    print(f"[orientation_detector] Модель: {llm_model}")
    os.makedirs(cache_dir, exist_ok=True)

    # Загружаем модель
    print(f"[orientation_detector] Загрузка модели: {model_name}")
    mesh = trimesh.load(model_path, force='mesh')

    # Центрируем и нормализуем размер
    mesh.vertices -= mesh.center_mass
    scale = 1.0 / np.max(mesh.extents)
    mesh.vertices *= scale

    if num_views == 2:
        angles = [0, 180]
    else:
        angles = [0, 90, 180, 270]
    image_paths = []

    print(f"[orientation_detector] Рендеринг с {len(angles)} углов: "
          f"{angles}")
    for angle in angles:
        img = render_model_view(mesh, yaw_deg=angle)
        img_path = os.path.join(
            cache_dir, f"{model_name}_yaw{angle}.png"
        )
        img.save(img_path)
        image_paths.append(img_path)
        print(f"[orientation_detector]   Сохранён вид {angle}°: "
              f"{img_path}")

    # Формируем промпт для мультимодальной LLM
    system_prompt = (
        "You are a 3D model orientation expert. "
        "Analyze rendered views of 3D models to identify their front side.\n\n"
        "The FRONT of an object is:\n"
        "- For furniture (chair, sofa, bed): where a person sits/lies\n"
        "- For appliances (TV, monitor, oven): the side with controls/screen\n"
        "- For vehicles: the direction of movement\n"
        "- For containers (box, crate): the side with opening/label\n"
        "- For decorative objects: the most detailed/decorated side\n\n"
        "Respond ONLY with valid JSON, no markdown, no explanations."
    )

    angles_lines = "\n".join(
        f"- Image {i + 1}: rotated to {a}°" for i, a in enumerate(angles)
    )
    angles_choices = " or ".join(str(a) for a in angles)
    user_prompt = (
        f"Look at these {len(angles)} images of a {model_name}:\n"
        f"{angles_lines}\n\n"
        "TASK: Identify which image shows the FRONT view (the camera "
        "is looking AT the object from the front).\n\n"
        "Analyze visual features:\n"
        "- Functional elements (controls, handles, openings, screen)\n"
        "- Asymmetry and detail level\n"
        "- User interaction side (where a person sits / faces)\n\n"
        "Respond with JSON:\n"
        "{\n"
        f'  "front_yaw": <one of: {angles_choices}>,\n'
        '  "confidence": "<high, medium, or low>"\n'
        "}"
    )

    # Отправляем запрос к мультимодальной LLM с изображениями
    print("[orientation_detector] Отправка запроса к LLM...")

    try:
        if use_remote:
            # Используем удаленный сервер
            url = f"{remote_url}/api/chat"
            
            # Конвертируем изображения в base64
            images_base64 = []
            for img_path in image_paths:
                with open(img_path, "rb") as f:
                    img_base64 = base64.b64encode(f.read()).decode("utf-8")
                    images_base64.append(img_base64)
            
            payload = {
                "model": llm_model,
                "stream": False,
                "messages": [
                    {
                        "role": "system",
                        "content": (
                            "You are a JSON-only API. "
                            "Respond ONLY with valid JSON, no markdown, "
                            "no explanations."
                        )
                    },
                    {
                        "role": "user",
                        "content": user_prompt,
                        "images": images_base64
                    }
                ],
                "options": {
                    "temperature": 0,
                    "seed": 42
                },
                "format": "json"  # Принудительный JSON формат
            }
            
            response = requests.post(url, json=payload, timeout=300)
            
            if response.status_code != 200:
                raise RuntimeError(
                    f"Remote Ollama HTTP {response.status_code}: "
                    f"{response.text}"
                )
            
            result_data = response.json()
            response_text = result_data.get("message", {}).get("content", "")
            
            print(f"[orientation_detector] DEBUG: response_text = "
                  f"{response_text[:200]}")  # Первые 200 символов
        else:
            # Используем локальный сервер через llm_request
            from project.llm_request import request as llm_request
            response_text = llm_request(
                system=system_prompt,
                user=user_prompt,
                model=llm_model,
                images=image_paths,
                timeout_s=300
            )

        import json
        
        # Парсим JSON из ответа (может быть обернут в ```json```)
        if isinstance(response_text, str):
            # Убираем markdown обертку если есть
            text = response_text.strip()
            if "```json" in text:
                start = text.find("```json") + len("```json")
                end = text.find("```", start)
                json_str = text[start:end].strip()
            elif "```" in text:
                # Может быть просто ```
                start = text.find("```") + len("```")
                end = text.find("```", start)
                json_str = text[start:end].strip()
            else:
                json_str = text
            
            print(f"[orientation_detector] DEBUG: json_str = "
                  f"{json_str[:200]}")
            
            result = json.loads(json_str)
        else:
            result = response_text

        print(f"[orientation_detector] Результат: "
              f"front_yaw={result['front_yaw']}°, "
              f"confidence={result['confidence']}")

        return result

    except Exception as e:
        error_msg = str(e)
        print(f"[orientation_detector] Ошибка при запросе к LLM: {e}")
        
        # Проверяем, не связана ли ошибка с отсутствием поддержки изображений
        if "image" in error_msg.lower() or "vision" in error_msg.lower():
            print()
            print("="*60)
            print("ОШИБКА: Модель не поддерживает изображения!")
            print("="*60)
            print(f"Текущая модель: {llm_model}")
            print()
            print("Установите мультимодальную модель:")
            print("  ollama pull minicpm-v:8b")
            print()
            print("Затем запустите с указанием модели:")
            print("  python model_orientation_detector.py minicpm-v:8b")
            print("="*60)
        
        # Fallback: предполагаем, что перед на 0°
        return {
            "front_yaw": 0,
            "confidence": "low"
        }


def _load_single_mesh(model_path: str):
    if trimesh is None:
        raise RuntimeError("trimesh is required for mesh loading")
    loaded = trimesh.load(model_path, force="scene")
    if isinstance(loaded, trimesh.Scene):
        geometries = list(loaded.geometry.values())
        if not geometries:
            raise ValueError("Empty scene: no geometries")
        meshes = []
        for geom in geometries:
            if isinstance(geom, trimesh.Trimesh):
                meshes.append(geom)
        if not meshes:
            raise ValueError("Scene contains no meshes")
        if len(meshes) == 1:
            mesh = meshes[0]
        else:
            mesh = trimesh.util.concatenate(meshes)
    elif isinstance(loaded, trimesh.Trimesh):
        mesh = loaded
    else:
        raise TypeError(f"Unsupported trimesh load result: {type(loaded)}")

    if mesh.vertices is None or len(mesh.vertices) == 0:
        raise ValueError("Mesh has no vertices")
    return mesh


def _extract_yaw_features(
    model_path: str,
    num_views: int = 4,
    sample_vertices: int = 200_000,
):
    if trimesh is None or np is None:
        raise RuntimeError("Feature extraction requires trimesh and numpy")

    if num_views == 2:
        angles = [0, 180]
    else:
        angles = [0, 90, 180, 270]

    mesh = _load_single_mesh(model_path)
    verts = np.asarray(mesh.vertices, dtype=np.float64)
    if verts.shape[0] > sample_vertices:
        rng = np.random.default_rng(42)
        idx = rng.choice(verts.shape[0], size=sample_vertices, replace=False)
        verts = verts[idx]

    # Center + normalize scale for stable features.
    center = verts.mean(axis=0)
    verts = verts - center
    extents = np.ptp(verts, axis=0)
    scale = float(np.max(extents))
    if not np.isfinite(scale) or scale <= 0:
        scale = 1.0
    verts = verts / scale

    # Face normals / areas (for "which side has the main flat face").
    face_normals = None
    face_areas = None
    try:
        if mesh.faces is not None and len(mesh.faces) > 0:
            face_normals = np.asarray(mesh.face_normals, dtype=np.float64)
            face_areas = np.asarray(mesh.area_faces, dtype=np.float64)
    except Exception:
        face_normals = None
        face_areas = None

    features = []
    for yaw_deg in angles:
        yaw_rad = np.radians(float(yaw_deg))
        rot = trimesh.transformations.rotation_matrix(yaw_rad, [0, 1, 0])
        R = rot[:3, :3]
        v = trimesh.transformations.transform_points(verts, rot)
        x = v[:, 0]
        y = v[:, 1]
        z = v[:, 2]

        y_hi = float(np.quantile(y, 0.80))
        y_lo = float(np.quantile(y, 0.20))
        hi_mask = y >= y_hi
        lo_mask = y <= y_lo
        top80_mean_z = float(np.mean(z[hi_mask])) if int(hi_mask.sum()) else 0.0
        bot20_mean_z = float(np.mean(z[lo_mask])) if int(lo_mask.sum()) else 0.0

        bbox_extents = [
            float(np.ptp(x)),
            float(np.ptp(y)),
            float(np.ptp(z)),
        ]

        area_pos_z = 0.0
        area_neg_z = 0.0
        if face_normals is not None and face_areas is not None:
            n = face_normals @ R.T
            nz = n[:, 2]
            area_pos_z = float(np.sum(face_areas * np.clip(nz, 0.0, 1.0)))
            area_neg_z = float(np.sum(face_areas * np.clip(-nz, 0.0, 1.0)))

        features.append(
            {
                "yaw": int(yaw_deg),
                "bbox_extents": bbox_extents,
                "top80_mean_z": top80_mean_z,
                "bot20_mean_z": bot20_mean_z,
                "area_pos_z": area_pos_z,
                "area_neg_z": area_neg_z,
            }
        )

    return features


def detect_model_front_with_text_llm(
    model_path,
    model_name="object",
    llm_model=None,
    base_url=None,
    timeout_s=180,
    num_views=4,
):
    """Определяет "перед" модели через обычный (текстовый) LLM.

    В LLM передаются вычисленные геометрические признаки для yaw-кандидатов.
    Это НЕ VLM: изображения не используются.

    Returns: {"front_yaw": int, "confidence": "high|medium|low"}
    """
    if llm_model is None:
        llm_model = DEFAULT_MODEL

    try:
        feats = _extract_yaw_features(model_path, num_views=num_views)
    except Exception as e:
        print(f"[orientation_detector] feature extraction failed: {e}")
        return {"front_yaw": 0, "confidence": "low"}

    system_prompt = (
        "You are a 3D model orientation assistant. "
        "You must choose the FRONT direction among candidate yaws. "
        "Respond ONLY with valid JSON, no markdown, no extra keys."
    )

    # Give the model a small rulebook + measured features.
    user_prompt = (
        f"We analyze a 3D mesh of '{model_name}'. "
        "For each candidate yaw, the mesh was rotated around +Y by that yaw, "
        "then features were measured assuming the camera looks from +Z.\n\n"
        "Features per yaw:\n"
        "- bbox_extents: [range_x, range_y, range_z] after rotation\n"
        "- top80_mean_z: mean Z of vertices in top 20% by height (Y)\n"
        "- bot20_mean_z: mean Z of vertices in bottom 20% by height (Y)\n"
        "- area_pos_z: area-weighted sum of face normals pointing towards +Z\n"
        "- area_neg_z: area-weighted sum of face normals pointing towards -Z\n\n"
        "Heuristics (use them but don't hardcode):\n"
        "- Chair/sofa/bed: backrest/headboard is tall; FRONT is opposite of it. "
        "So FRONT yaw usually makes top80_mean_z negative (tall parts behind).\n"
        "- TV/monitor/appliance: FRONT often has a large flat face; prefer yaw with larger area_pos_z.\n"
        "- If symmetric/ambiguous: choose best guess, confidence=low.\n\n"
        "Candidates:\n"
        + "\n".join(
            json.dumps(item, ensure_ascii=False) for item in feats
        )
        + "\n\nRespond with JSON ONLY:\n"
        + "{\n  \"front_yaw\": <one of the candidate yaw values>,\n  \"confidence\": \"high\"|\"medium\"|\"low\"\n}"
    )

    try:
        from project.llm_request import request as llm_request

        result = llm_request(
            system=system_prompt,
            user=user_prompt,
            model=llm_model,
            base_url=base_url or os.environ.get("OLLAMA_BASE_URL") or "http://localhost:11434",
            timeout_s=timeout_s,
            images=None,
        )
        front_yaw = int(result.get("front_yaw", 0))
        conf = str(result.get("confidence", "low")).lower()
        if conf not in ("high", "medium", "low"):
            conf = "low"
        return {"front_yaw": front_yaw, "confidence": conf}
    except Exception as e:
        print(f"[orientation_detector] text-LLM failed: {e}")
        return {"front_yaw": 0, "confidence": "low"}


def compute_yaw_offset(
    model_path,
    model_name="object",
    llm_model=None,
    use_remote=True,
    remote_url="http://94.19.29.186:11434",
    cache_dir=".cache/orientation",
    num_views=4,
    force_refresh=False,
    method=None,
    env_method_var="WORLD_CREATOR_ORIENTATION_METHOD",
    env_disable_var="WORLD_CREATOR_DISABLE_ORIENTATION",
):
    """
    Вычисляет offset угла для модели.

    Offset нужно прибавлять к плановому yaw_deg при сборке MuJoCo XML
    (см. ``mujoco_assembler``), чтобы интринсик «перёд» модели смотрел
    в направление, заданное планировщиком (как если бы модель была
    +Z-forward в GLB).

    Результат кешируется на диск в ``{cache_dir}/cache.json`` —
    повторные вызовы для того же файла не идут в LLM.

    Returns:
        int: offset в градусах (0, 90, 180, 270)
    """
    # Decide method: param > env > default.
    if str(os.environ.get(env_disable_var, "")).lower() in ("1", "true", "yes"):
        method_effective = "off"
    else:
        method_effective = (method or os.environ.get(env_method_var) or "llm")
        method_effective = str(method_effective).strip().lower()
    if method_effective in ("none", "disable", "disabled"):
        method_effective = "off"
    if method_effective in ("text", "text_llm", "text-llm"):
        method_effective = "llm"
    if method_effective in ("vlm", "vision"):
        method_effective = "vlm"

    if method_effective == "off":
        return 0

    # Disk cache lookup.
    key = _cache_key(f"{model_path}|{method_effective}")
    with _CACHE_LOCK:
        cache = _cache_load(cache_dir)
        if not force_refresh and key in cache:
            entry = cache[key]
            if isinstance(entry, dict) and "offset" in entry:
                offset = int(entry["offset"]) % 360
                print(f"[orientation_detector] Cache hit '{model_name}': "
                      f"offset={offset}° (front_yaw={entry.get('front_yaw')}°, "
                      f"method={entry.get('method', method_effective)})")
                return offset

    if method_effective == "vlm":
        result = detect_model_front_with_llm(
            model_path,
            model_name,
            cache_dir=cache_dir,
            llm_model=llm_model,
            use_remote=use_remote,
            remote_url=remote_url,
            num_views=num_views,
        )
    elif method_effective == "llm":
        # Text-only LLM: no images.
        # If use_remote is requested, treat remote_url as Ollama base URL.
        base_url = remote_url if use_remote else os.environ.get("OLLAMA_BASE_URL")
        result = detect_model_front_with_text_llm(
            model_path,
            model_name=model_name,
            llm_model=llm_model,
            base_url=base_url,
            timeout_s=180,
            num_views=num_views,
        )
    else:
        raise ValueError(
            f"Unknown orientation method '{method_effective}'. "
            "Use: llm|vlm|off"
        )
    front_yaw = int(result["front_yaw"]) % 360

    # Derivation: trimesh renders mesh rotated by R_y(front_yaw) around
    # the GLB up-axis +Y; the camera at +Z sees the face whose original
    # angle from +Z (CCW about +Y) is α = -front_yaw.
    # MuJoCo applies euler="90 yaw 0" (up_axis=y), so a body-local
    # vector (sin α, 0, cos α) maps to world facing angle
    # β = α + yaw - 90°. Planner assumes a +Z-forward model
    # (α = 0), so to match for an arbitrary α we need
    # yaw_actual = yaw_planned - α = yaw_planned + front_yaw.
    offset = front_yaw

    confidence = (result.get("confidence") or "").lower()
    if confidence == "low":
        # LLM не смогло уверенно определить — не кешируем,
        # чтобы при следующем запуске была попытка снова.
        print(f"[orientation_detector] Модель '{model_name}': "
              f"перед на {front_yaw}°, offset={offset}° "
              f"(confidence=low, not cached)")
    else:
        with _CACHE_LOCK:
            cache = _cache_load(cache_dir)
            cache[key] = {
                "model_path": os.path.abspath(model_path),
                "model_name": model_name,
                "front_yaw": front_yaw,
                "offset": offset,
                "confidence": result.get("confidence"),
                "num_views": num_views,
                "method": method_effective,
            }
            _cache_save(cache_dir, cache)
        print(f"[orientation_detector] Модель '{model_name}': "
              f"перед на {front_yaw}°, offset={offset}° (saved to cache)")

    return offset


if __name__ == "__main__":
    import sys
    
    # Тест на модели стула
    chair_path = (
        "/home/sakin153/.cache/huggingface/hub/"
        "datasets--HorizonRobotics--EmbodiedGenData/snapshots/"
        "0da911ca5b1c26c64533d8579b2fe9d3f042acb4/"
        "dataset/basic_furniture/chair/aeea9a5e33665bb3b31544206cb2a929/"
        "mesh/computer_chair_44.glb"
    )

    # Можно передать модель как аргумент
    llm_model = sys.argv[1] if len(sys.argv) > 1 else None

    print("="*60)
    print("ТЕСТ ОПРЕДЕЛЕНИЯ ОРИЕНТАЦИИ МОДЕЛИ")
    print("="*60)
    print("Используем удаленный сервер: http://94.19.29.186:11434")
    if llm_model:
        print(f"Модель: {llm_model}")
    else:
        print("Модель: qwen3.5:9b-q4_K_M (по умолчанию)")
    print("="*60)
    print()

    offset = compute_yaw_offset(
        chair_path,
        "computer_chair",
        llm_model,
        use_remote=True
    )

    print()
    print("="*60)
    print(f"РЕЗУЛЬТАТ: offset = {offset}°")
    print("="*60)
    print()
    print("Этот offset нужно добавить ко всем yaw углам "
          "при размещении этой модели.")
    print()
    print("Например, если LLM сказал yaw=0° (север),")
    print(f"то реальный yaw = (0 + {offset}) % 360 = {offset}°")
