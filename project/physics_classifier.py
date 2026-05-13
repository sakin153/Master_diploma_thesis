"""Stage 2.5 - Physics Classification
Classifies objects as static (fixed to world) or dynamic (free to move).
Uses LLM to make robotics-informed decisions based on object properties.
"""

import json
from pathlib import Path

from project.llm_request import DEFAULT_MODEL, request as _default_request


_PROMPT_FILE = Path(__file__).parent / "prompts" / "physics_classify.txt"


def _load_prompt():
    if not _PROMPT_FILE.exists():
        raise FileNotFoundError(f"Prompt file not found: {_PROMPT_FILE}")
    return _PROMPT_FILE.read_text(encoding="utf-8")


_SYSTEM_PROMPT = _load_prompt()

# Volume guard: override dynamic classification for large objects (prevents "flying furniture")
_VOLUME_GUARD_M3 = 0.5

# Heuristic fallback: classify by volume if LLM fails
_HEURISTIC_VOLUME_M3 = 0.1


def _bbox_volume(size):
    if not size or len(size) < 3:
        return 0.0
    try:
        return float(size[0]) * float(size[1]) * float(size[2])
    except (TypeError, ValueError):
        return 0.0




def _build_user_payload(models, scene_spec, query):
    items = []
    for m in models:
        size = m.get("size") or [0.0, 0.0, 0.0]
        size_rounded = [round(float(size[i]), 3) for i in range(min(3, len(size)))]
        items.append({
            "uuid": str(m.get("uuid", "")),
            "name": str(m.get("Model") or m.get("_query_name") or ""),
            "size_wdh_m": size_rounded,
        })

    payload = {
        "query": str(query or ""),
        "scene_description": str(getattr(scene_spec, "expanded_description", "") or ""),
        "room_w_m": round(float(getattr(scene_spec, "room_half_size", 2.5)) * 2.0, 2),
        "room_l_m": round(float(getattr(scene_spec, "room_half_size", 2.5)) * 2.0, 2),
        "objects": items,
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)


def _normalize_physics(value):
    if not isinstance(value, str):
        return None
    v = value.strip().lower()
    if v in ("static", "статичный", "статика"):
        return True
    if v in ("dynamic", "динамичный", "динамика"):
        return False
    return None


def _parse_response(raw):
    if isinstance(raw, dict):
        items = raw.get("items")
    elif isinstance(raw, list):
        items = raw
    else:
        items = None

    if not isinstance(items, list):
        return {}

    decisions = {}
    for it in items:
        if not isinstance(it, dict):
            continue
        uid = str(it.get("uuid", "")).strip()
        if not uid:
            continue
        decision = _normalize_physics(it.get("physics"))
        if decision is None:
            continue
        decisions[uid] = decision
    return decisions


def classify_physics(models, scene_spec, query, prompt_model_fn=None,
                     llm_model=DEFAULT_MODEL, verbose=True):
    """Stage 2.5: Classify objects as static or dynamic using LLM.
    
    Args:
        models: List of models after load_and_scale_models
        scene_spec: SceneSpec from stage 0
        query: Original user query or expanded_description
        prompt_model_fn: Optional custom LLM function
        llm_model: Ollama model ID
        verbose: Print classification results
        
    Returns:
        Same models list with is_static field added (True=static, False=dynamic)
    """
    if not models:
        return models

    llm = prompt_model_fn if prompt_model_fn is not None else _default_request

    if verbose:
        print(f"[physics_classifier] Classifying {len(models)} objects")

    user_payload = _build_user_payload(models, scene_spec, query)

    decisions = {}
    try:
        raw = llm(_SYSTEM_PROMPT, user_payload, llm_model)
        decisions = _parse_response(raw)
    except Exception as exc:
        print(f"[physics_classifier] LLM failed ({exc}); using volume heuristic")

    for m in models:
        uid = str(m.get("uuid", ""))
        decision = decisions.get(uid)

        if decision is None:
            raise RuntimeError(
                f"[physics_classifier] LLM не вернул решение для '{m.get('Model', '?')}' (uuid={uid}). "
                f"Heuristic fallback отключён."
            )
        source = "llm"

        if decision is False:
            vol = _bbox_volume(m.get("size"))
            if vol >= _VOLUME_GUARD_M3:
                if verbose:
                    print(
                        f"[physics_classifier]   guard: {m.get('Model', '?')} "
                        f"vol={vol:.2f}m³ ≥ {_VOLUME_GUARD_M3}m³ → принудительно static"
                    )
                decision = True
                source = "guard"

        m["is_static"] = bool(decision)
        if verbose:
            label = "static " if decision else "dynamic"
            print(f"[physics_classifier]   {m.get('Model', '?'):<24} → {label} ({source})")

    return models
