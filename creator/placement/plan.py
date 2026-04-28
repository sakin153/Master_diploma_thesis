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
    elif has_distance and isinstance(c.get("distance"), (int, float)):
        out["distance"] = float(c["distance"])
    side = _safe_str(c.get("side")).lower()
    if side in ("front", "back", "left", "right"):
        out["side"] = side
    if "offset" in c:
        out["offset"] = float(c["offset"])
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
    preserves multiplicity and injects constraints for missing entries by
    distributing them evenly across existing anchor targets instead of
    dumping them all at region:middle.
    """
    import re

    def _normalize_name(name: str) -> str:
        return re.sub(r'_\d+$', '', name)

    out = [dict(o) for o in objects if isinstance(o, dict) and _safe_str(o.get("Model"))]

    remaining: Dict[str, int] = {}
    for o in out:
        n = _safe_str(o.get("Model"))
        base = _normalize_name(n)
        remaining[base] = remaining.get(base, 0) + 1

    # Collect anchor targets per object type (for round-robin distribution)
    # e.g. chairs have beside(table_1), beside(table_2) → targets = [table_1, table_2]
    anchor_targets_by_type: Dict[str, List[str]] = {}
    sides_cycle = ["front", "back", "left", "right"]
    for o in out:
        name = _safe_str(o.get("Model"))
        base = _normalize_name(name)
        for c in (o.get("constraints") or []):
            if isinstance(c, dict) and _safe_str(c.get("type")).lower() == "beside":
                t = _safe_str(c.get("target"))
                if t:
                    anchor_targets_by_type.setdefault(base, [])
                    if t not in anchor_targets_by_type[base]:
                        anchor_targets_by_type[base].append(t)

    # Also collect all anchor names (objects with region:middle or region:edge)
    all_anchors: List[str] = []
    for o in out:
        for c in (o.get("constraints") or []):
            if isinstance(c, dict) and _safe_str(c.get("type")).lower() == "region":
                all_anchors.append(_safe_str(o.get("Model")))

    rr: Dict[str, int] = {}  # round-robin cursor per type

    for cm in chosen_models:
        name = _safe_str(cm.get("Model") or cm.get("name"))
        if not name:
            continue
        base = _normalize_name(name)
        cnt = remaining.get(base, 0)
        if cnt > 0:
            remaining[base] = cnt - 1
            continue

        # Distribute missing instance to the least-populated anchor target
        targets = anchor_targets_by_type.get(base) or all_anchors
        if targets:
            # Count how many of this type already point to each target
            target_counts: Dict[str, int] = {t: 0 for t in targets}
            for o in out:
                if _normalize_name(_safe_str(o.get("Model"))) != base:
                    continue
                for c in (o.get("constraints") or []):
                    if isinstance(c, dict) and _safe_str(c.get("type")).lower() == "beside":
                        t = _safe_str(c.get("target"))
                        if t in target_counts:
                            target_counts[t] += 1
            target = min(target_counts, key=target_counts.get)
            # Pick a side not yet used for this target
            used_sides = {
                _safe_str(c.get("side"))
                for o in out
                if _normalize_name(_safe_str(o.get("Model"))) == base
                for c in (o.get("constraints") or [])
                if isinstance(c, dict) and _safe_str(c.get("type")).lower() == "beside"
                and _safe_str(c.get("target")) == target
            }
            side = next((s for s in sides_cycle if s not in used_sides), sides_cycle[0])
            constraints = [
                {"type": "beside", "target": target, "side": side,
                 "distance": [0.8, 1.2], "hard": False, "weight": 3.0},
                {"type": "face_to", "target": target, "hard": False, "weight": 2.0},
            ]
        else:
            constraints = [{"type": "region", "value": "middle", "hard": False, "weight": 1.0}]

        out.append({"Model": name, "constraints": constraints})

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
    """Remove noisy peer-to-peer constraints within dense repeated groups.

    Only removes constraints where BOTH the subject AND target are in the
    dense group (e.g. chair→chair near/left_of chains). Preserves:
    - `beside` / `face_to` pointing at an anchor (chair→table)
    - `on_top_of` pointing at any surface
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

    # Constraints that are only noisy when pointing at another dense peer
    peer_noise_types = {
        "near", "far", "left_of", "right_of", "in_front_of", "behind",
        "center_aligned",
    }
    # These are always kept regardless of target
    always_keep_types = {"beside", "face_to", "on_top_of", "region", "wall_mount"}

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

            # Always keep structural constraints
            if ctype in always_keep_types:
                keep.append(c)
                continue

            # Remove peer-noise only when BOTH subject and target are dense
            if name in dense and target in dense and ctype in peer_noise_types:
                continue

            keep.append(c)

        if not keep:
            keep = [{"type": "region", "target": "", "value": "middle", "hard": False, "weight": 1.0}]

        out.append({"Model": name, "constraints": keep, **{k: v for k, v in o.items() if k not in ("Model", "constraints")}})

    return out


def _build_models_str(chosen_models: Sequence[Dict[str, Any]]) -> str:
    """Build a structured string for the LLM that makes grouping explicit.

    Instead of a flat list, we number each instance and show counts so the
    LLM knows exactly how many of each type exist and can assign constraints
    without guessing.

    Example output:
      Objects (20 total):
        table x4  → table_1, table_2, table_3, table_4
        computer chair x16  → computer chair_1 ... computer chair_16
    """
    from collections import Counter
    counts: Counter = Counter()
    for m in chosen_models:
        name = _safe_str(m.get("Model") or m.get("name"))
        if name:
            counts[name] += 1

    lines = [f"Objects ({len(chosen_models)} total):"]
    instance_idx: Dict[str, int] = {}
    for name, count in counts.items():
        start = 1
        end = count
        if count == 1:
            lines.append(f"  {name} x1  → {name}_1")
        else:
            lines.append(f"  {name} x{count}  → {name}_1 ... {name}_{count}")

    return "\n".join(lines)


def build_semantic_plan(
    *,
    prompt_model,
    prompt_template: str,
    query: str,
    chosen_model: str,
    chosen_models: Sequence[Dict[str, Any]],
    context_models: Sequence[Dict[str, Any]],
    scene_spec: Any = None,
) -> Dict[str, Any]:
    from creator.contexts_prompts.constraints import fmt_seating_plan_tmpl

    # ---------------------------------------------------------------
    # Stage A: Floor plan — anchors + large furniture only
    # ---------------------------------------------------------------
    # Split chosen_models into "anchors/large" and "children" (seating/small)
    _CHILD_HINTS = {"chair", "stool", "seat", "bench", "sofa", "couch"}

    def _is_child(m: Dict[str, Any]) -> bool:
        name = _safe_str(m.get("Model") or m.get("name")).lower()
        return any(h in name for h in _CHILD_HINTS)

    anchor_models = [m for m in chosen_models if not _is_child(m)]
    child_models  = [m for m in chosen_models if _is_child(m)]

    # If nothing to split, fall back to single call
    if not child_models:
        return _build_plan_single(
            prompt_model=prompt_model,
            prompt_template=prompt_template,
            query=query,
            chosen_model=chosen_model,
            chosen_models=chosen_models,
            scene_spec=scene_spec,
        )

    # Call 1: place anchors + large furniture
    anchor_plan = _build_plan_single(
        prompt_model=prompt_model,
        prompt_template=prompt_template,
        query=query,
        chosen_model=chosen_model,
        chosen_models=anchor_models,
        scene_spec=None,  # don't hint children — they come from seating plan
    )

    # Build anchor name list from anchor_models only (not from plan output
    # which may include chairs added by _ensure_plan_covers_chosen_models)
    from collections import Counter
    anchor_counts: Counter = Counter()
    anchors_list: List[str] = []
    anchor_names = {_safe_str(m.get("Model") or m.get("name")).lower() for m in anchor_models}
    for obj in anchor_plan.get("objects", []):
        name = _safe_str(obj.get("Model"))
        if name.lower() not in anchor_names:
            continue
        anchor_counts[name] += 1
        anchors_list.append(f"{name}_{anchor_counts[name]}")

    anchors_str = "\n".join(f"  - {a}" for a in anchors_list)
    children_str = _build_models_str(child_models)

    # Call 2: assign seating around the placed anchors
    seating_content = fmt_seating_plan_tmpl.format(
        anchors_str=anchors_str,
        children_str=children_str,
    )
    try:
        try:
            raw2 = prompt_model(seating_content, query)
        except TypeError:
            raw2 = prompt_model(seating_content, query, chosen_model)
    except Exception:
        raw2 = None

    child_objects: List[Dict[str, Any]] = []
    child_names = {_safe_str(m.get("Model") or m.get("name")).lower() for m in child_models}
    if isinstance(raw2, dict):
        for item in (raw2.get("objects") or []):
            if not isinstance(item, dict):
                continue
            model_name = _safe_str(item.get("Model") or item.get("name"))
            # Only accept objects that are actually children — ignore tables/anchors LLM may have included
            if model_name.lower() not in child_names:
                continue
            cons = [_normalize_constraint(c) for c in (item.get("constraints") or [])
                    if isinstance(c, dict)]
            cons = [c for c in cons if c["type"]]
            # Keep only the FIRST beside constraint — drop duplicates that cause solver conflicts
            seen_beside = False
            deduped = []
            for c in cons:
                if c["type"] == "beside":
                    if seen_beside:
                        continue
                    seen_beside = True
                deduped.append(c)
            cons = deduped
            cons = [c for c in cons if c["type"]]
            entry: Dict[str, Any] = {"Model": model_name, "constraints": cons}
            if "is_static" in item:
                entry["is_static"] = bool(item["is_static"])
            child_objects.append(entry)

    # Merge: anchor plan objects + child objects
    all_objects = list(anchor_plan.get("objects", [])) + child_objects
    all_objects = _ensure_plan_covers_chosen_models(all_objects, chosen_models)
    all_objects = _stabilize_dense_group_constraints(all_objects)

    for obj in all_objects:
        for c in (obj.get("constraints") or []):
            if isinstance(c, dict) and c.get("type") in ("wall_mount", "wall_mounted"):
                obj["is_static"] = True

    return {"objects": all_objects}


def _build_plan_single(
    *,
    prompt_model,
    prompt_template: str,
    query: str,
    chosen_model: str,
    chosen_models: Sequence[Dict[str, Any]],
    scene_spec: Any = None,
) -> Dict[str, Any]:
    """Single LLM call for the full plan (used when no seating split needed)."""
    models_str = _build_models_str(chosen_models)

    group_hint = ""
    if scene_spec is not None:
        hints = getattr(scene_spec, "estimated_objects", []) or []
        if hints:
            group_hint = "\n\nIntended grouping:\n"
            for h in hints:
                name = str(getattr(h, "name", "")).strip()
                qty = int(getattr(h, "quantity", 1))
                notes = str(getattr(h, "notes", "")).strip()
                group_hint += f"  - {name} x{qty}"
                if notes:
                    group_hint += f"  ({notes})"
                group_hint += "\n"

    content = prompt_template.format(query=query, models_str=models_str + group_hint)

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

        obj_entry: Dict[str, Any] = {"Model": model_name, "constraints": norm_constraints}

        has_beside = any(c["type"] == "beside" for c in norm_constraints)
        if has_beside:
            norm_constraints = [c for c in norm_constraints if c["type"] != "on_top_of"]
            obj_entry["constraints"] = norm_constraints

        if "is_static" in item:
            obj_entry["is_static"] = bool(item["is_static"])

        objects.append(obj_entry)

    if not objects:
        return _fallback_plan(chosen_models)

    objects = _ensure_plan_covers_chosen_models(objects, chosen_models)
    objects = _stabilize_dense_group_constraints(objects)

    for obj in objects:
        for c in (obj.get("constraints") or []):
            if isinstance(c, dict) and c.get("type") in ("wall_mount", "wall_mounted"):
                obj["is_static"] = True

    return {"objects": objects}

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
    
    # Auto-set is_static=True for wall-mounted objects
    for obj in objects:
        constraints = obj.get("constraints", [])
        if isinstance(constraints, list):
            for c in constraints:
                if isinstance(c, dict) and c.get("type") in ("wall_mount", "wall_mounted"):
                    obj["is_static"] = True
                    break

    return {"objects": objects}
