"""Stage 1 - Model Selection
Selects 3D models from catalog for each scene object
"""

import json
import re
from pathlib import Path

from project.llm_request import DEFAULT_MODEL, request as _default_request

# Fallback mappings for common object synonyms
_FALLBACKS = {
    "book": "box",
    "lamp": "bottle",
    "light": "bottle",
    "pillow": "cushion",
    "cushion": "pillow",
    "tv": "television",
    "tv stand": "table",
    "sofa": "lounge chair",
    "couch": "lounge chair",
    "rug": "carpet",
}


def _load_prompt(filename):
    """Load a prompt from the prompts directory."""
    prompt_file = Path(__file__).parent / "prompts" / filename
    if not prompt_file.exists():
        raise FileNotFoundError(f"Prompt file not found: {prompt_file}")
    return prompt_file.read_text(encoding="utf-8")


_DISAMBIGUATION_PROMPT = _load_prompt("model_disambiguation.txt")


def _tokenize(text):
    return re.findall(r"[a-zа-яё0-9]+", (text or "").lower())


def _singularize(word):
    w = (word or "").strip().lower()
    if len(w) > 3 and w.endswith("ves"):
        return w[:-3] + "f"
    if len(w) > 3 and w.endswith("ies"):
        return w[:-3] + "y"
    if len(w) > 2 and w.endswith("s"):
        return w[:-1]
    return w


def _score(obj_name, model):
    obj = obj_name.strip().lower()
    name = str(model.get("name", "")).lower()
    categories = model.get("categories") or []
    meta = " ".join(str(x).lower() for x in (model.get("tags") or []) + categories if x)

    score = 0
    if obj == name:       score += 20
    elif obj in name:     score += 12
    if obj in meta:       score += 6

    tokens = [t for t in _tokenize(obj) if len(t) > 1]
    for t in tokens:
        if t in name:     score += 4
        if t in meta:     score += 2

    head = _singularize(tokens[-1]) if tokens else obj
    category_str = " ".join(str(c).lower() for c in categories)

    if head in category_str:                                              score += 5
    if any(k in category_str for k in ["abstract", "icon", "logo", "ui"]): score -= 15
    if any(k in name for k in ["miniature", "toy", "lego"]) and head != "toy": score -= 10
    if head == "chair" and "wheelchair" in name:                          score -= 12
    if head in {"desk", "table"} and any(k in name for k in ["lamp", "light", "fan"]): score -= 15

    return score


def _rank(obj_name, catalog, limit=10):
    scored = sorted(
        [(m, _score(obj_name, m)) for m in catalog],
        key=lambda x: x[1], reverse=True
    )
    seen = set()
    result = []
    for m, s in scored:
        if s < 3:
            break
        key = (m.get("uuid"), m.get("name"))
        if key not in seen:
            seen.add(key)
            result.append(m)
        if len(result) >= limit:
            break
    return result


def _llm_pick(obj_name, scene_query, candidates, llm):
    payload = [
        {"uuid": str(c.get("uuid", "")), "name": str(c.get("name", "")), "description": str(c.get("description", ""))}
        for c in candidates if c.get("uuid")
    ]
    if not payload:
        return None

    prompt = _DISAMBIGUATION_PROMPT.format(
        scene_query=scene_query,
        object=obj_name,
        candidates=json.dumps(payload, ensure_ascii=False),
    )

    try:
        result = llm(prompt, obj_name)
    except Exception as e:
        print(f"[model_picker] LLM ошибка для '{obj_name}': {e}")
        return None

    chosen_uuid = ""
    if isinstance(result, dict):
        chosen_uuid = str(result.get("uuid", "")).strip()
    elif isinstance(result, str):
        m = re.search(r"[0-9a-f]{16,32}", result)
        chosen_uuid = m.group(0) if m else ("none" if "none" in result.lower() else "")

    if chosen_uuid.lower() == "none":
        return None

    by_uuid = {str(c.get("uuid", "")): c for c in candidates}
    return by_uuid.get(chosen_uuid)


def pick_models(scene_spec, catalog, llm_model=DEFAULT_MODEL, prompt_model_fn=None):
    """Stage 1: Select models from catalog for each object in SceneSpec.
    
    Returns:
        List of dicts: [{"Model": "Desk", "uuid": "abc123", "model_loc": "/path/to/model.glb"}, ...]
    """
    llm = prompt_model_fn if prompt_model_fn is not None else _default_request

    # Expand objects by quantity: desk x3 → ["desk", "desk", "desk"]
    object_names = []
    for hint in scene_spec.estimated_objects:
        object_names.extend([hint.name] * hint.quantity)

    chosen = []
    picked_by_type = {}  # object type → selected model (one per type)

    for obj in object_names:
        obj_key = obj.lower()
        search = _FALLBACKS.get(obj_key, obj_key)

        # Reuse model if this type was already selected
        if obj_key in picked_by_type:
            picked = picked_by_type[obj_key]
        else:
            ranked = _rank(search, catalog)
            if not ranked:
                print(f"[model_picker] '{obj}' — нет совпадений, пропускаем")
                continue

            if len(ranked) == 1:
                picked = ranked[0]
            else:
                picked = _llm_pick(obj, scene_spec.expanded_description, ranked[:10], llm)
                if picked is None:
                    raise RuntimeError(
                        f"[model_picker] LLM не смог выбрать модель для '{obj}' из {len(ranked)} кандидатов. "
                        f"Fallback на первый ранжированный отключён."
                    )

            picked_by_type[obj_key] = picked

        name = str(picked.get("name", "")).strip()
        uuid = str(picked.get("uuid", "")).strip()

        chosen.append({
            "Model": name,
            "uuid": uuid,
            "model_loc": str(picked.get("model_loc", "")),
            "_query_name": obj,
        })

    if not chosen:
        raise RuntimeError("No models found. Try a different query.")

    _print_results(chosen)
    return chosen


def _print_results(chosen):
    rows = "\n".join(
        f"    - {m['_query_name']} → {m['Model']} (uuid={m['uuid'][:8]})"
        for m in chosen
    )
    print(
        f"[model_picker] Подобраны модели:\n"
        f"  найдено     : {len(chosen)}\n"
        f"  модели      :\n{rows}"
    )
