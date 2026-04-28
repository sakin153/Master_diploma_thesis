from typing import Any, Dict, List, Sequence, Tuple


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
    side = _safe_str(c.get("side")).lower()
    if side in ("front", "back", "left", "right"):
        out["side"] = side
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


def _ensure_plan_covers_chosen_models(
    objects: List[Dict[str, Any]],
    chosen_models: Sequence[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Ensure the semantic plan has one row per chosen model instance.

    Some LLM outputs collapse duplicates or drop object types. This function
    preserves multiplicity and injects simple region constraints for missing
    entries so downstream placement still receives a complete plan.
    """
    import re
    
    def _normalize_name(name: str) -> str:
        """Remove _N suffixes: box_1 -> box, apple_2 -> apple"""
        return re.sub(r'_\d+$', '', name)
    
    out = [dict(o) for o in objects if isinstance(o, dict) and _safe_str(o.get("Model"))]

    remaining: Dict[str, int] = {}
    for o in out:
        n = _safe_str(o.get("Model"))
        base = _normalize_name(n)
        remaining[base] = remaining.get(base, 0) + 1

    for cm in chosen_models:
        name = _safe_str(cm.get("Model") or cm.get("name"))
        if not name:
            continue
        base = _normalize_name(name)
        cnt = remaining.get(base, 0)
        if cnt > 0:
            remaining[base] = cnt - 1
            continue
        out.append(
            {
                "Model": name,
                "constraints": [
                    {
                        "type": "region",
                        "target": "",
                        "value": "middle",
                        "hard": False,
                        "weight": 1.0,
                    }
                ],
            }
        )

    return out


_FACING_OBJECT_HINTS: frozenset = frozenset({
    "chair", "armchair", "stool", "seat", "bench", "sofa", "couch",
    "loveseat", "settee", "ottoman", "barstool", "bar stool",
    "tv", "television", "monitor", "screen", "display",
    "lamp", "fan",
})


def _name_is_facing_subject(name: str) -> bool:
    """Heuristic: this object meaningfully has a "front" that should face a target."""
    lname = (name or "").lower()
    return any(k in lname for k in _FACING_OBJECT_HINTS)


def _distribute_identical_around_anchor(objects: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Auto-add directional constraints for identical objects around common anchor."""
    groups: Dict[Tuple[str, str], List[Tuple[int, Dict]]] = {}
    for idx, obj in enumerate(objects):
        name = _safe_str(obj.get("Model"))
        anchor = None
        for c in obj.get("constraints", []):
            if isinstance(c, dict) and _safe_str(c.get("type")).lower() == "face_to":
                anchor = _safe_str(c.get("target"))
                break
        if anchor:
            groups.setdefault((name, anchor), []).append((idx, obj))
    
    result = list(objects)
    for (_, anchor), items in groups.items():
        n = len(items)
        if not (2 <= n <= 8):
            continue
        dirs = (["left_of", "right_of", "in_front_of", "behind"] * 2)[:n]
        for i, (idx, obj) in enumerate(items):
            new_obj = dict(obj)
            cons = []
            # Remove conflicting region constraints, boost face_to heavily
            for c in obj.get("constraints", []):
                if not isinstance(c, dict):
                    continue
                ctype = _safe_str(c.get("type")).lower()
                if ctype == "region":
                    continue  # Remove region constraint
                if ctype == "face_to":
                    c = dict(c)
                    c["weight"] = 10.0
                cons.append(c)
            cons.extend([
                {"type": dirs[i], "target": anchor, "value": None, "hard": False, "weight": 0.5},
                {"type": "near", "target": anchor, "distance": [0.4, 1.2], "hard": False, "weight": 0.6}
            ])
            new_obj["constraints"] = cons
            result[idx] = new_obj
    return result


def _infer_face_to_from_near(
    objects: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """For each `near(A,B)` where A is a chair/sofa/lamp/screen-like object,
    add a soft `face_to(A,B)` so the front of A is oriented towards B.

    This eliminates the most visible artefact: a chair turned with its back
    to the desk it's "near", or a TV facing a wall instead of the sofa.
    """
    out: List[Dict[str, Any]] = []
    for obj in objects:
        if not isinstance(obj, dict):
            out.append(obj)
            continue
        name = _safe_str(obj.get("Model"))
        cons = obj.get("constraints") if isinstance(obj.get("constraints"), list) else []
        new_cons: List[Dict[str, Any]] = list(cons)
        if _name_is_facing_subject(name):
            existing_face_targets = {
                _safe_str(c.get("target"))
                for c in cons
                if isinstance(c, dict)
                and _safe_str(c.get("type")).lower() in {"face_to", "face_same_as"}
            }
            for c in cons:
                if not isinstance(c, dict):
                    continue
                if _safe_str(c.get("type")).lower() != "near":
                    continue
                target = _safe_str(c.get("target"))
                if not target or target in existing_face_targets:
                    continue
                new_cons.append({
                    "type": "face_to",
                    "target": target,
                    "value": None,
                    "hard": False,
                    "weight": 0.5,
                })
                existing_face_targets.add(target)
        merged = dict(obj)
        merged["constraints"] = new_cons
        out.append(merged)
    return out


def _stabilize_dense_group_constraints(
    objects: List[Dict[str, Any]],
    *,
    dense_threshold: int = 4,
) -> List[Dict[str, Any]]:
    """Remove noisy relative constraints for dense repeated groups.

    For scenes like classrooms/warehouses, LLM often generates long chains of
    near/left/right constraints among many identical instances. This can
    override deterministic grid/pair arrangements and produce tangled layouts.
    """
    counts: Dict[str, int] = {}
    for o in objects:
        name = _safe_str(o.get("Model"))
        if not name:
            continue
        counts[name] = counts.get(name, 0) + 1

    dense = {name for name, c in counts.items() if c >= dense_threshold}
    if not dense:
        return objects

    relative_types = {
        "near",
        "far",
        "left_of",
        "right_of",
        "in_front_of",
        "behind",
        "center_aligned",
    }

    out: List[Dict[str, Any]] = []
    for o in objects:
        name = _safe_str(o.get("Model"))
        cons = o.get("constraints") if isinstance(o.get("constraints"), list) else []
        keep: List[Dict[str, Any]] = []
        for c in cons:
            if not isinstance(c, dict):
                continue
            ctype = _safe_str(c.get("type")).lower()
            target = _safe_str(c.get("target"))

            # Keep dense groups mostly region-constrained so arrangement tools
            # can produce clean rows/grids without chain-collapse artifacts.
            if name in dense and ctype in relative_types:
                continue
            if name in dense and target in dense and ctype in relative_types:
                continue

            keep.append(c)

        if not keep:
            keep = [{"type": "region", "target": "", "value": "middle", "hard": False, "weight": 1.0}]

        # beside and on_top_of are mutually exclusive
        has_beside = any(_safe_str(c.get("type")).lower() == "beside" for c in keep)
        if has_beside:
            keep = [c for c in keep if _safe_str(c.get("type")).lower() != "on_top_of"]

        out.append({"Model": name, "constraints": keep})

    return out


def build_semantic_plan(
    *,
    prompt_model,
    prompt_template: str,
    query: str,
    chosen_model: str,
    chosen_models: Sequence[Dict[str, Any]],
    context_models: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    # Build prompt with query and chosen models
    content = prompt_template.format(
        query=query,
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
        
        obj_entry: Dict[str, Any] = {
            "Model": model_name,
            "constraints": norm_constraints
        }
        
        # beside and on_top_of are mutually exclusive: an object cannot
        # simultaneously stand beside a target AND rest on top of it.
        has_beside = any(c["type"] == "beside" for c in norm_constraints)
        if has_beside:
            norm_constraints = [c for c in norm_constraints if c["type"] != "on_top_of"]
            obj_entry["constraints"] = norm_constraints
        
        # Parse is_static from LLM response
        if "is_static" in item:
            obj_entry["is_static"] = bool(item["is_static"])
        
        objects.append(obj_entry)

    if not objects:
        return _fallback_plan(chosen_models)

    objects = _ensure_plan_covers_chosen_models(objects, chosen_models)
    objects = _stabilize_dense_group_constraints(objects)
    objects = _distribute_identical_around_anchor(objects)
    objects = _infer_face_to_from_near(objects)
    
    # Auto-set is_static=True for wall-mounted objects
    for obj in objects:
        constraints = obj.get("constraints", [])
        if isinstance(constraints, list):
            for c in constraints:
                if isinstance(c, dict) and c.get("type") in ("wall_mount", "wall_mounted"):
                    obj["is_static"] = True
                    break

    return {"objects": objects}
