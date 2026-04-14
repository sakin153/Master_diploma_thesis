from typing import Any, Dict, List, Sequence


def _safe_str(v: Any) -> str:
    if v is None:
        return ""
    return str(v).strip()


def _normalize_constraint(c: Dict[str, Any]) -> Dict[str, Any]:
    out = {
        "type": _safe_str(c.get("type")).lower(),
        "target": _safe_str(c.get("target")),
        "value": c.get("value"),
        "hard": bool(c.get("hard", False)),
        "weight": float(c.get("weight", 1.0)),
    }
    has_distance = "distance" in c
    is_pair = isinstance(c.get("distance"), (list, tuple)) and len(c["distance"]) == 2
    if has_distance and is_pair:
        out["distance"] = [float(c["distance"][0]), float(c["distance"][1])]
    return out


def _fallback_plan(chosen_models: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    objects: List[Dict[str, Any]] = []
    for i, m in enumerate(chosen_models):
        name = _safe_str(m.get("Model") or m.get("name"))
        if not name:
            continue
        # Keep first object more central and others near it as a simple baseline.
        constraints = [{"type": "region", "value": "middle", "hard": False, "weight": 1.0}]
        if i > 0 and objects:
            constraints.append(
                {
                    "type": "near",
                    "target": objects[0]["Model"],
                    "distance": [0.3, 2.0],
                    "hard": False,
                    "weight": 0.8,
                }
            )
        objects.append({"Model": name, "constraints": constraints})
    return {"objects": objects}


def build_semantic_plan(
    *,
    prompt_model,
    prompt_template: str,
    query: str,
    chosen_model: str,
    chosen_models: Sequence[Dict[str, Any]],
    context_models: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    catalog = []
    for m in context_models:
        name = _safe_str(m.get("name"))
        if not name:
            continue
        catalog.append(
            {
                "name": name,
                "tags": m.get("tags") or [],
                "categories": m.get("categories") or [],
            }
        )

    content = prompt_template.format(
        catalog_str=str(catalog),
        models_str=str(list(chosen_models)),
    )

    try:
        try:
            raw = prompt_model(content, query)
        except TypeError:
            raw = prompt_model(content, query, chosen_model)
    except Exception:
        return _fallback_plan(chosen_models)

    if not isinstance(raw, dict):
        return _fallback_plan(chosen_models)

    objects_raw = raw.get("objects")
    if not isinstance(objects_raw, list):
        return _fallback_plan(chosen_models)

    objects: List[Dict[str, Any]] = []
    for item in objects_raw:
        if not isinstance(item, dict):
            continue
        model_name = _safe_str(item.get("Model") or item.get("name"))
        if not model_name:
            continue
        cons = item.get("constraints")
        norm_constraints: List[Dict[str, Any]] = []
        if isinstance(cons, list):
            for c in cons:
                if isinstance(c, dict):
                    nc = _normalize_constraint(c)
                    if nc["type"]:
                        norm_constraints.append(nc)
        objects.append({"Model": model_name, "constraints": norm_constraints})

    if not objects:
        return _fallback_plan(chosen_models)

    return {"objects": objects}
