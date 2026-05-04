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
import trimesh
import numpy as np
from PIL import Image
from project.llm_request import DEFAULT_MODEL
import base64
import requests


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
    remote_url="http://94.19.29.186:11434"
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

    Returns:
        dict: {
            "front_yaw": int,  # угол в градусах (0, 90, 180, 270)
            "confidence": str,  # "high", "medium", "low"
            "reasoning": str    # объяснение от LLM
        }
    """
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

    # Рендерим только с 2 углов (0° и 180°) для скорости
    angles = [0, 180]
    image_paths = []

    print("[orientation_detector] Рендеринг с 2 углов (0° и 180°)...")
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

    user_prompt = (
        f"Look at these 2 images of a {model_name}:\n"
        "- Image 1: rotated to 0°\n"
        "- Image 2: rotated to 180°\n\n"
        "TASK: Identify which image shows the FRONT.\n\n"
        "Analyze visual features:\n"
        "- Functional elements (controls, handles, openings)\n"
        "- Asymmetry and detail level\n"
        "- User interaction side\n\n"
        "Respond with JSON:\n"
        "{\n"
        '  "front_yaw": <0 or 180>,\n'
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


def compute_yaw_offset(
    model_path,
    model_name="object",
    llm_model=None,
    use_remote=True,
    remote_url="http://94.19.29.186:11434"
):
    """
    Вычисляет offset угла для модели.

    Если "перед" модели находится не на 0°, возвращает offset,
    который нужно добавить к yaw углам при размещении.

    Args:
        model_path: путь к модели
        model_name: название модели
        llm_model: название мультимодальной модели (опционально)
        use_remote: использовать удаленный сервер (по умолчанию True)
        remote_url: URL удаленного Ollama сервера

    Returns:
        int: offset в градусах (0, 90, 180, 270)
    """
    result = detect_model_front_with_llm(
        model_path, model_name,
        llm_model=llm_model,
        use_remote=use_remote,
        remote_url=remote_url
    )
    front_yaw = result["front_yaw"]

    # Если перед на 0°, offset не нужен
    # Если перед на 180°, нужно вычесть 180° (offset = 180)

    offset = (360 - front_yaw) % 360

    print(f"[orientation_detector] Модель '{model_name}': "
          f"перед на {front_yaw}°, offset={offset}°")

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
