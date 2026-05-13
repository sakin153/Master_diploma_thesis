"""Stage 2 - Model Loading and Scaling
Loads model dimensions and normalizes scales using LLM-based batch autoscaling
"""

import re
import statistics
from collections import Counter

import trimesh

from project.llm_request import request as _llm_request, DEFAULT_MODEL, OLLAMA_BASE_URL

# LLM Batch Autoscaling - no fallback table, LLM is required


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


def _model_scale_key(m):
    """Return a stable key for sharing one computed scale across duplicates."""
    return str(
        m.get("uuid")
        or m.get("model_loc")
        or m.get("Model")
        or m.get("name")
        or ""
    ).strip().lower()


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


# LLM Batch Autoscaling
_DIM_BUCKET_M = 0.05  # Bucket size for mode calculation


def _build_unique_entries(models):
    """Deduplicate models by name and return unique entries with averaged geometry."""
    grouped = {}
    for m in models:
        name = str(m.get("Model") or m.get("name") or "").strip()
        if not name:
            continue
        size = m.get("size")
        if not size or len(size) < 3:
            continue
        try:
            sx, sy, sz = float(size[0]), float(size[1]), float(size[2])
        except (TypeError, ValueError):
            continue
        max_dim = max(sx, sy, sz)
        unit = _infer_unit_scale(max_dim)
        # Dimensions in meters after unit detection (raw mesh proportions for LLM)
        dims_m = [sx * unit, sy * unit, sz * unit]
        key = name.lower()
        if key not in grouped:
            grouped[key] = {"name": name, "raw_dims_m": dims_m, "count": 1}
        else:
            grouped[key]["count"] += 1
    return list(grouped.values())


def _format_batch_prompt(entries, scene_desc):
    lines = []
    for e in entries:
        wx, wy, wz = e["raw_dims_m"]
        lines.append(
            f'- "{e["name"]}": raw mesh extents ≈ [{wx:.3f}, {wy:.3f}, {wz:.3f}] m '
            f"(proportions only, may be wrong absolute size)"
        )
    objects_block = "\n".join(lines)

    system = (
        "You are a 3D scene assistant. Estimate realistic real-world dimensions "
        "(in metres) for each object below. The objects appear together in ONE "
        "scene, so their sizes must be RELATIVE TO EACH OTHER and consistent "
        "(a mug must be smaller than the table it sits on). Return ONE entry per "
        "unique object name — never repeat the same object. "
        "Output ONLY a JSON array, no prose, no markdown fences."
    )
    user = (
        f"Scene description:\n{scene_desc.strip() or '(generic indoor scene)'}\n\n"
        f"Objects (each listed exactly once):\n{objects_block}\n\n"
        "For every object return realistic dimensions in metres:\n"
        '[{"name": "<exact name from list>", "height_m": <h>, '
        '"width_m": <w>, "depth_m": <d>}, ...]\n\n'
        "Rules:\n"
        "- height_m = vertical extent floor→top (e.g. chair 0.85, mug 0.10).\n"
        "- width_m, depth_m = horizontal footprint.\n"
        "- Proportions between objects must match the scene "
        "(small items on tables must be smaller than the table).\n"
        "- One entry per unique name; do not duplicate objects.\n"
        "- Numbers only, no units, no comments."
    )
    return system, user


def _parse_batch_response(parsed, valid_names):
    """Превращает ответ LLM в {name_lower: (h, w, d)}; пропускает мусор."""
    rows = []
    if isinstance(parsed, list):
        rows = [r for r in parsed if isinstance(r, dict)]
    elif isinstance(parsed, dict):
        for key in ("answer", "objects", "models", "items"):
            if isinstance(parsed.get(key), list):
                rows = [r for r in parsed[key] if isinstance(r, dict)]
                break
        if not rows and "name" in parsed:
            rows = [parsed]

    out = {}
    valid = {n.lower() for n in valid_names}
    for row in rows:
        name = row.get("name") or row.get("Model") or row.get("model")
        if not isinstance(name, str):
            continue
        key = name.strip().lower()
        if key not in valid or key in out:
            continue
        try:
            h = float(row.get("height_m") or row.get("height") or 0.0)
            w = float(row.get("width_m") or row.get("width") or 0.0)
            d = float(row.get("depth_m") or row.get("depth") or 0.0)
        except (TypeError, ValueError):
            continue
        if not (0.005 < h < 10.0):
            continue
        if not (0.005 < w < 10.0):
            continue
        if not (0.005 < d < 10.0):
            continue
        out[key] = (h, w, d)
    return out


def _aggregate_dim(values):
    """Агрегируем серию ответов: мода (по корзинам _DIM_BUCKET_M),
    при равенстве — медиана."""
    if not values:
        return None
    if len(values) == 1:
        return values[0]
    buckets = [round(v / _DIM_BUCKET_M) for v in values]
    counts = Counter(buckets)
    top_count = max(counts.values())
    winners = [b for b, c in counts.items() if c == top_count]
    if len(winners) == 1 and top_count >= 2:
        # одна явная мода — берём среднее значений из этой корзины
        chosen = [v for v, b in zip(values, buckets) if b == winners[0]]
        return statistics.fmean(chosen)
    # мода неоднозначна → медиана всех ответов
    return statistics.median(values)


def _ask_llm_dimensions_batch(models, scene_desc, n_calls=3, base_seed=17):
    """Batch LLM autoscaling with N repeated calls and aggregation.
    
    Returns:
        Dict mapping name_lower to {"h": float, "w": float, "d": float}
        Only successful aggregations included; failures fall back to _HEIGHT_TABLE
    """
    entries = _build_unique_entries(models)
    if not entries:
        return {}

    valid_names = [e["name"] for e in entries]
    system, user = _format_batch_prompt(entries, scene_desc)

    print(f"[model_loader] LLM autoscale: {len(entries)} уникальных моделей, "
          f"{n_calls} запросов")

    per_call = []  # список {name_lower: (h, w, d)}
    for i in range(n_calls):
        # Vary temperature+seed for independent samples (avoid identical responses)
        temperature = 0.2 + 0.2 * i  # 0.2, 0.4, 0.6
        seed = base_seed + i * 1000
        try:
            parsed = _llm_request(
                system, user, DEFAULT_MODEL, OLLAMA_BASE_URL,
                timeout_s=180, temperature=temperature, seed=seed,
            )
            mapping = _parse_batch_response(parsed, valid_names)
            print(f"[model_loader]   call {i+1}/{n_calls}: получено "
                  f"{len(mapping)}/{len(entries)} записей")
            per_call.append(mapping)
        except Exception as e:
            print(f"[model_loader]   call {i+1}/{n_calls} ошибка: {e}")
            per_call.append({})

    # Aggregate results per object name and axis
    result = {}
    for e in entries:
        key = e["name"].lower()
        hs = [m[key][0] for m in per_call if key in m]
        ws = [m[key][1] for m in per_call if key in m]
        ds = [m[key][2] for m in per_call if key in m]
        if not hs:
            continue
        h = _aggregate_dim(hs)
        w = _aggregate_dim(ws)
        d = _aggregate_dim(ds)
        if h is None or w is None or d is None:
            continue
        result[key] = {"h": h, "w": w, "d": d}
        print(f"[model_loader]   {e['name']:<28} → h={h:.3f} w={w:.3f} d={d:.3f}m "
              f"(из {len(hs)} ответов)")
    return result


# Scale Normalization


def _normalize_scale(models, llm_dims):
    print(f"[model_loader] Нормализация масштаба ({len(models)} моделей):")
    scale_cache = {}
    for m in models:
        cache_key = _model_scale_key(m)
        if cache_key in scale_cache:
            cached = scale_cache[cache_key]
            m["scale"] = cached["scale"]
            m["size"] = list(cached["size"])
            m["_height_m"] = cached["_height_m"]
            continue

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

        # Determine raw height direction for scale calculation
        is_elongated = any(kw in name_lower for kw in
                           ["banana", "cucumber", "carrot", "stick", "rod", "pencil", "pen"])
        if is_elongated:
            height_raw = min(sx, sz) if up_axis != "z" else min(sx, sy)
        elif up_axis == "z":
            height_raw = sz if sz >= 0.05 * max_dim else max_dim
        elif sy >= 0.05 * max_dim:
            height_raw = sy
        else:
            height_raw = max_dim

        height_m = height_raw * unit

        # Get target dimensions from LLM batch results - NO FALLBACK
        llm_entry = llm_dims.get(name_lower)
        if llm_entry is None:
            raise RuntimeError(
                f"[model_loader] LLM не вернул размеры для '{name}'. "
                f"Все 3 попытки не дали результата для этого объекта. "
                f"Fallback-таблица отключена."
            )

        target_h = llm_entry["h"]
        target_w = llm_entry["w"]
        target_d = llm_entry["d"]

        # Scale by longest axis (axis-invariant for elongated objects)
        target_max = max(target_h, target_w, target_d)
        # Calculate scale: raw_max * scale = target_max
        det_scale = target_max / max(1e-6, max_dim)

        # Clamp longest dimension: ±25% of LLM target
        min_max = max(0.005, target_max * 0.75)
        max_max = max(0.010, target_max * 1.25)
        actual_max = max_dim * det_scale
        if actual_max < min_max:
            det_scale *= min_max / max(1e-9, actual_max)
        elif actual_max > max_max:
            det_scale *= max_max / max(1e-9, actual_max)

        final_scale = max(1e-4, min(1000.0, det_scale))

        final_scale = max(1e-4, min(1000.0, final_scale))
        m["scale"] = final_scale

        if up_axis == "z":
            m["size"] = [sx * final_scale, sz * final_scale, sy * final_scale]
            m["_height_m"] = sz * final_scale
        else:
            m["size"] = [sx * final_scale, sy * final_scale, sz * final_scale]
            m["_height_m"] = sy * final_scale

        scale_cache[cache_key] = {
            "scale": final_scale,
            "size": list(m["size"]),
            "_height_m": m["_height_m"],
        }

        print(f"  {name:<30} scale={final_scale:.4f}  h={m['_height_m']:.3f}м "
              f"→ цель {target_h:.2f}м (LLM)")

    return models


def load_and_scale_models(chosen_models, scene_desc=""):
    """Stage 2: Load model dimensions and normalize scales.
    
    Args:
        chosen_models: List from pick_models()
        scene_desc: Expanded scene description for LLM context
        
    Returns:
        Same list with added fields: 'size', 'scale', '_height_m'
    """
    models = _load_sizes(chosen_models)
    llm_dims = _ask_llm_dimensions_batch(models, scene_desc)
    models = _normalize_scale(models, llm_dims)
    return models
