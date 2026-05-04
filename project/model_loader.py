# Стадия 2 - загружаем размеры моделей и вычисляем размер комнаты.

import re
import trimesh

from project.llm_request import request as _llm_request, DEFAULT_MODEL, OLLAMA_BASE_URL

# Таблица типичных высот (если LLM недоступен — fallback)
_HEIGHT_TABLE = {
    "chair":      0.90,
    "stool":      0.75,
    "sofa":       0.85,
    "couch":      0.85,
    "desk":       0.75,
    "table":      0.75,
    "dining table": 0.75,
    "bed":        0.55,
    "wardrobe":   1.80,
    "cabinet":    1.00,
    "shelf":      1.80,
    "bookshelf":  1.80,
    "monitor":    0.45,
    "lamp":       1.50,
    "door":       2.10,
    "box":        0.40,
    "bottle":     0.30,
    "cup":        0.12,
    "mug":        0.12,
    "plate":      0.03,
    "book":       0.25,
    "plant":      0.60,
    "laptop":     0.02,
    "keyboard":   0.04,
    "phone":      0.15,
    "apple":      0.08,
    "bowl":       0.10,
    "tray":       0.05,
}

# Максимальные горизонтальные размеры (ширина/глубина) для плоских объектов
_MAX_HORIZ_M = [
    (("crate", "container", "box"),  0.45),
    (("plate", "dish"),              0.32),
    (("bowl",),                      0.25),
    (("cup", "mug"),                 0.12),
    (("laptop",),                    0.40),
    (("keyboard",),                  0.45),
    (("remote",),                    0.22),
    (("phone", "smartphone"),        0.08),
    (("apple", "fruit"),             0.10),
    (("candle",),                    0.10),
    (("book",),                      0.35),
    (("tray",),                      0.50),
]


def _lookup_height(model_name):
    name = model_name.lower()
    for key, h in _HEIGHT_TABLE.items():
        if key in name:
            return h
    return None


def _ask_llm_height(model_name):
    system = (
        "You are a 3D scene assistant. "
        "Answer with ONLY a single decimal number in metres, no units, no text."
    )
    user = (
        f"Total height from floor to top (including backrest for chairs, lid for boxes) "
        f"in metres of a '{model_name}' (common household/furniture item). "
        f"Example answers: 0.75, 1.80, 0.12"
    )
    try:
        ans = _llm_request(system, user, DEFAULT_MODEL, OLLAMA_BASE_URL, timeout_s=60)
        text = str(ans) if not isinstance(ans, str) else ans
        nums = re.findall(r"\d+(?:\.\d+)?", text.strip())
        if nums:
            val = float(nums[0])
            if 0.01 < val < 10.0:
                print(f"[model_loader] Высота '{model_name}': {val}м (LLM)")
                return val
    except Exception as e:
        print(f"[model_loader] LLM высота для '{model_name}' не получена: {e}")
    return None


def _target_height(model_name):
    h = _lookup_height(model_name)
    if h is not None:
        print(f"[model_loader] Высота '{model_name}': {h}м (таблица)")
        return h
    
    # Если нет в таблице - вызываем LLM
    h = _ask_llm_height(model_name)
    if h is not None:
        return h
    
    # Только если LLM тоже не помог - используем fallback
    print(f"[model_loader] Высота '{model_name}': 1.0м (fallback)")
    return 1.0


def _infer_unit_scale(max_dim):
    if max_dim <= 0.0:
        return 1.0
    if max_dim > 100.0:
        return 0.001
    if max_dim > 10.0:
        return 0.01
    if max_dim > 3.5:
        return 0.1
    if max_dim < 0.01:
        return 100.0
    if max_dim < 0.05:
        return 10.0
    return 1.0


def _load_sizes(models):
    for m in models:
        path = m.get("model_loc", "")
        if not path:
            continue
        try:
            mesh = trimesh.load(path, force="mesh")
            m["size"] = list(mesh.extents)
        except Exception as e:
            print(f"[model_loader] Не удалось загрузить {m.get('Model')}: {e}")
    return models


def _normalize_scale(models):
    print(f"[model_loader] Нормализация масштаба ({len(models)} моделей):")
    for m in models:
        size = m.get("size")
        if not size or len(size) < 3:
            m["scale"] = float(m.get("scale", 1.0))
            continue

        sx = max(1e-6, float(size[0]))
        sy = max(1e-6, float(size[1]))
        sz = max(1e-6, float(size[2]))
        max_dim = max(sx, sy, sz)
        name = str(m.get("Model") or m.get("name") or "")
        name_lower = name.lower()
        up_axis = str(m.get("_up_axis", "y"))
        unit = _infer_unit_scale(max_dim)

        # Определяем высоту: удлинённые объекты (банан, огурец, ручка) — берём меньший горизонт
        is_elongated = any(kw in name_lower for kw in ["banana", "cucumber", "carrot", "stick", "rod", "pencil", "pen"])
        if is_elongated:
            height_raw = min(sx, sz) if up_axis != "z" else min(sx, sy)
        elif up_axis == "z":
            height_raw = sz if sz >= 0.05 * max_dim else max_dim
        elif sy >= 0.05 * max_dim:
            height_raw = sy
        else:
            height_raw = max_dim

        height_m = height_raw * unit
        target_h = _target_height(name)

        # Если объект уже близок к нужному размеру — минимальное масштабирование
        if 0.8 * target_h <= height_m <= 1.2 * target_h:
            final_scale = unit
            m["scale"] = final_scale
            if up_axis == "z":
                m["size"] = [sx * final_scale, sz * final_scale, sy * final_scale]
                m["_height_m"] = sz * final_scale
            else:
                m["size"] = [sx * final_scale, sy * final_scale, sz * final_scale]
                m["_height_m"] = sy * final_scale
            print(f"  {name:<30} scale={final_scale:.4f}  h={m['_height_m']:.3f}м  (уже норм)")
            continue

        # Основной масштаб
        det_scale = unit * (target_h / max(1e-6, height_m))

        # Зажим высоты в допустимый диапазон [65%, 160%] от target
        min_h = max(0.03, target_h * 0.65)
        max_h = max(0.06, target_h * 1.60)
        actual_h = height_raw * det_scale
        if actual_h < min_h:
            det_scale *= min_h / max(1e-9, actual_h)
        elif actual_h > max_h:
            det_scale *= max_h / max(1e-9, actual_h)

        final_scale = max(1e-4, min(1000.0, det_scale))

        # Зажим горизонтальных размеров для плоских/компактных объектов
        if any(kw in name_lower for kw in ["banana", "cucumber", "carrot"]):
            max_len = 0.20
            actual_len = max(sx, sz) * final_scale
            if actual_len > max_len:
                final_scale *= max_len / actual_len
        elif any(kw in name_lower for kw in ["stick", "rod", "pencil", "pen"]):
            max_len = 0.20  # Ручка/карандаш максимум 20см
            actual_len = max(sx, sz) * final_scale
            if actual_len > max_len:
                final_scale *= max_len / actual_len
        else:
            for keywords, max_horiz in _MAX_HORIZ_M:
                if any(kw in name_lower for kw in keywords):
                    actual_horiz = max(sx, sz) * final_scale
                    if actual_horiz > max_horiz:
                        final_scale *= max_horiz / actual_horiz
                    break

        final_scale = max(1e-4, min(1000.0, final_scale))
        m["scale"] = final_scale

        if up_axis == "z":
            m["size"] = [sx * final_scale, sz * final_scale, sy * final_scale]
            m["_height_m"] = sz * final_scale
        else:
            m["size"] = [sx * final_scale, sy * final_scale, sz * final_scale]
            m["_height_m"] = sy * final_scale

        print(f"  {name:<30} scale={final_scale:.4f}  h={m['_height_m']:.3f}м → цель {target_h:.2f}м")

    return models


def load_and_scale_models(chosen_models):
    """Стадия 2: загружает размеры моделей и нормализует масштаб.

    Принимает список из pick_models(), возвращает тот же список
    с добавленными полями 'size', 'scale', '_height_m'.
    """
    models = _load_sizes(chosen_models)
    models = _normalize_scale(models)
    return models
