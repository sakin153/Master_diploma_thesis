from typing import Any, Dict, List, Optional, Sequence, Tuple

from creator.placement.command_interpreter import CommandInterpreter
from creator.placement.scene_graph import SceneGraph


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
        # CRITICAL FIX: Add unique 'id' field
        clean_name = name.replace(" ", "_").replace("-", "_")
        obj_id = f"{clean_name}_{i}"
        objects.append({"id": obj_id, "Model": name, "constraints": constraints})
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
        computer chair x16  → computer chair_1, computer chair_2, ..., computer chair_15, computer chair_16
    """
    from collections import Counter
    counts: Counter = Counter()
    for m in chosen_models:
        name = _safe_str(m.get("Model") or m.get("name"))
        if name:
            counts[name] += 1

    lines = [f"Objects ({len(chosen_models)} total):"]
    for name, count in counts.items():
        if count == 1:
            lines.append(f"  {name} x1  → {name}_1")
        elif count <= 3:
            # Small count: show all explicitly
            instances = ", ".join(f"{name}_{i}" for i in range(1, count + 1))
            lines.append(f"  {name} x{count}  → {instances}")
        else:
            # Large count: show first 2, last 2, and total with critical reminder
            lines.append(f"  {name} x{count}  → {name}_1, {name}_2, ..., {name}_{count-1}, {name}_{count}")
            lines.append(f"    (CRITICAL: ALL {count} instances must be included)")

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
    use_scene_graph: bool = False,
    command_interpreter: Optional[CommandInterpreter] = None,
) -> Dict[str, Any]:
    from creator.contexts_prompts.constraints import fmt_seating_plan_tmpl

    # Initialize command interpreter if not provided
    if command_interpreter is None:
        command_interpreter = CommandInterpreter()

    # Parse the query for precise spatial constraints
    spatial_command = command_interpreter.parse_spatial_command(query)
    
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
        plan = _build_plan_single(
            prompt_model=prompt_model,
            prompt_template=prompt_template,
            query=query,
            chosen_model=chosen_model,
            chosen_models=chosen_models,
            scene_spec=scene_spec,
        )
        
        # Apply precise constraints from command interpreter
        if spatial_command.precise_constraints:
            plan = command_interpreter.validate_llm_plan(query, plan)
        
        # Build scene graph if requested
        if use_scene_graph:
            plan = _build_scene_graph_from_plan(plan, spatial_command)
        
        return plan

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

    # CRITICAL: Verify all anchors are present before proceeding to seating
    import re
    import logging
    def _normalize_name_local(name: str) -> str:
        return re.sub(r'_\d+$', '', name)
    
    expected_anchor_counts = Counter(_normalize_name_local(_safe_str(m.get("Model") or m.get("name"))) 
                                     for m in anchor_models if _safe_str(m.get("Model") or m.get("name")))
    actual_anchor_counts = Counter(_normalize_name_local(a.split('_')[0] if '_' in a else a) 
                                   for a in anchors_list)
    
    # Log anchor placement results
    logging.info(f"Stage 1 (anchors): Expected {dict(expected_anchor_counts)}, Got {dict(actual_anchor_counts)}")
    logging.info(f"Stage 1 (anchors): Placed anchors: {anchors_list}")
    
    # Log if anchors are missing
    for anchor_type, expected_count in expected_anchor_counts.items():
        actual_count = actual_anchor_counts.get(anchor_type, 0)
        if actual_count < expected_count:
            logging.warning(
                f"Stage 1 (anchors): Expected {expected_count} {anchor_type}(s), "
                f"but only {actual_count} were placed. Fallback will add missing anchors."
            )

    anchors_str = "\n".join(f"  - {a}" for a in anchors_list)
    children_str = _build_models_str(child_models)
    
    # Calculate expected distribution for the seating prompt
    num_anchors = len(anchors_list)
    num_children = len(child_models)
    children_per_anchor = num_children // num_anchors if num_anchors > 0 else 0
    
    distribution_hint = ""
    if num_anchors > 0 and children_per_anchor > 0:
        distribution_hint = f"\n\nDISTRIBUTION: {num_children} children across {num_anchors} anchors = {children_per_anchor} children per anchor"

    # Call 2: assign seating around the placed anchors (with retry logic)
    seating_content = fmt_seating_plan_tmpl.format(
        anchors_str=anchors_str,
        children_str=children_str + distribution_hint,
    )
    
    raw2 = None
    max_retries = 3
    for attempt in range(max_retries):
        try:
            try:
                raw2 = prompt_model(seating_content, query)
            except TypeError:
                raw2 = prompt_model(seating_content, query, chosen_model)
            
            # If we got a valid response, break
            if isinstance(raw2, dict) and "objects" in raw2:
                logging.info(f"Stage 2 (seating): LLM returned valid response on attempt {attempt + 1}")
                break
            else:
                logging.warning(f"Stage 2 (seating): LLM returned invalid response type on attempt {attempt + 1}: {type(raw2)}")
                raw2 = None
                
        except Exception as e:
            logging.error(f"Stage 2 (seating): LLM call failed on attempt {attempt + 1}: {e}")
            raw2 = None
            
        if attempt < max_retries - 1:
            logging.info(f"Stage 2 (seating): Retrying... (attempt {attempt + 2}/{max_retries})")
    
    if raw2 is None:
        logging.error("Stage 2 (seating): All retry attempts failed, proceeding with fallback")

    child_objects: List[Dict[str, Any]] = []
    child_names = {_safe_str(m.get("Model") or m.get("name")).lower() for m in child_models}
    if isinstance(raw2, dict):
        logging.info(f"Stage 2 (seating): LLM returned {len(raw2.get('objects', []))} objects")
        for item in (raw2.get("objects") or []):
            if not isinstance(item, dict):
                continue
            model_name = _safe_str(item.get("Model") or item.get("name"))
            # Only accept objects that are actually children — ignore tables/anchors LLM may have included
            if model_name.lower() not in child_names:
                logging.debug(f"Stage 2 (seating): Skipping non-child object: {model_name}")
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
        logging.info(f"Stage 2 (seating): Accepted {len(child_objects)} child objects")
    else:
        logging.warning(f"Stage 2 (seating): LLM returned invalid response type: {type(raw2)}")

    # Merge: anchor plan objects + child objects
    all_objects = list(anchor_plan.get("objects", [])) + child_objects
    all_objects = _ensure_plan_covers_chosen_models(all_objects, chosen_models)
    all_objects = _stabilize_dense_group_constraints(all_objects)

    for obj in all_objects:
        for c in (obj.get("constraints") or []):
            if isinstance(c, dict) and c.get("type") in ("wall_mount", "wall_mounted"):
                obj["is_static"] = True

    # CRITICAL FIX: Add unique 'id' field to each object
    # Universal Placement System requires 'id' and 'type' fields for each object
    for i, obj in enumerate(all_objects):
        if "id" not in obj:
            model_name = _safe_str(obj.get("Model", f"object_{i}"))
            # Create unique ID from model name + index
            clean_name = model_name.replace(" ", "_").replace("-", "_")
            obj["id"] = f"{clean_name}_{i}"
        
        # Add 'type' field if missing
        if "type" not in obj:
            # Infer type from is_static or model name
            if obj.get("is_static", True):
                obj["type"] = "furniture"
            else:
                obj["type"] = "small_object"

    plan = {"objects": all_objects}
    
    # Apply precise constraints from command interpreter
    if spatial_command.precise_constraints:
        plan = command_interpreter.validate_llm_plan(query, plan)
    
    # Build scene graph if requested
    if use_scene_graph:
        plan = _build_scene_graph_from_plan(plan, spatial_command)
    
    return plan


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

    # Validate that all chosen_models are present (post-validation)
    from collections import Counter
    import re
    def _normalize_name_local(name: str) -> str:
        return re.sub(r'_\d+$', '', name)
    
    expected_counts = Counter(_normalize_name_local(_safe_str(m.get("Model") or m.get("name"))) 
                              for m in chosen_models if _safe_str(m.get("Model") or m.get("name")))
    actual_counts = Counter(_normalize_name_local(_safe_str(obj.get("Model"))) 
                           for obj in objects if _safe_str(obj.get("Model")))
    
    for obj_type, expected_count in expected_counts.items():
        actual_count = actual_counts.get(obj_type, 0)
        if actual_count < expected_count:
            # Log warning - _ensure_plan_covers_chosen_models should have fixed this
            import logging
            logging.warning(
                f"Post-validation: Expected {expected_count} {obj_type}(s), "
                f"but plan contains {actual_count}. This should have been fixed by fallback."
            )

    for obj in objects:
        for c in (obj.get("constraints") or []):
            if isinstance(c, dict) and c.get("type") in ("wall_mount", "wall_mounted"):
                obj["is_static"] = True

    # CRITICAL FIX: Add unique 'id' field to each object
    # Universal Placement System requires 'id' field for each object
    for i, obj in enumerate(objects):
        if "id" not in obj:
            model_name = _safe_str(obj.get("Model", f"object_{i}"))
            # Create unique ID from model name + index
            clean_name = model_name.replace(" ", "_").replace("-", "_")
            obj["id"] = f"{clean_name}_{i}"

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
    
    # Validate that all chosen_models are present (post-validation)
    from collections import Counter
    import re
    def _normalize_name_local(name: str) -> str:
        return re.sub(r'_\d+$', '', name)
    
    expected_counts = Counter(_normalize_name_local(_safe_str(m.get("Model") or m.get("name"))) 
                              for m in chosen_models if _safe_str(m.get("Model") or m.get("name")))
    actual_counts = Counter(_normalize_name_local(_safe_str(obj.get("Model"))) 
                           for obj in objects if _safe_str(obj.get("Model")))
    
    for obj_type, expected_count in expected_counts.items():
        actual_count = actual_counts.get(obj_type, 0)
        if actual_count < expected_count:
            # Log warning - _ensure_plan_covers_chosen_models should have fixed this
            import logging
            logging.warning(
                f"Post-validation: Expected {expected_count} {obj_type}(s), "
                f"but plan contains {actual_count}. This should have been fixed by fallback."
            )
    
    # Auto-set is_static=True for wall-mounted objects
    for obj in objects:
        constraints = obj.get("constraints", [])
        if isinstance(constraints, list):
            for c in constraints:
                if isinstance(c, dict) and c.get("type") in ("wall_mount", "wall_mounted"):
                    obj["is_static"] = True
                    break

    return {"objects": objects}



def _build_scene_graph_from_plan(
    plan: Dict[str, Any],
    spatial_command: Any,
) -> Dict[str, Any]:
    """Build a hierarchical scene graph from a flat semantic plan.

    This function analyzes the constraints in the plan to determine
    parent-child relationships and builds a SceneGraph structure.
    Objects with 'on_top_of' or 'beside' constraints become children
    of their target objects.

    Args:
        plan: Flat semantic plan with objects and constraints
        spatial_command: Parsed spatial command with positioning info

    Returns:
        Plan with added 'scene_graph' key containing SceneGraph structure
    """
    scene_graph = SceneGraph()
    objects = plan.get("objects", [])

    # Track which objects have been added to the graph
    added_objects = {}

    # First pass: identify anchor objects (no parent relationships)
    anchor_objects = []
    child_objects = []

    for obj in objects:
        name = _safe_str(obj.get("Model"))
        if not name:
            continue

        constraints = obj.get("constraints", [])
        has_parent_constraint = False

        for c in constraints:
            if not isinstance(c, dict):
                continue
            ctype = _safe_str(c.get("type")).lower()

            # These constraint types indicate a parent-child relationship
            if ctype in ("on_top_of", "beside", "inside"):
                has_parent_constraint = True
                break

        if has_parent_constraint:
            child_objects.append(obj)
        else:
            anchor_objects.append(obj)

    # Second pass: add anchor objects to scene graph (children of root)
    for obj in anchor_objects:
        name = _safe_str(obj.get("Model"))
        object_type = name.split("_")[0] if "_" in name else name

        # Extract anchor points based on object type
        anchor_points = _generate_anchor_points_for_object(
            name, object_type
        )

        node = scene_graph.add_object(
            object_id=name,
            object_type=object_type,
            parent=None,  # Will be added to root
            constraints=obj.get("constraints", []),
            anchor_points=anchor_points,
            metadata={
                "is_static": obj.get("is_static", False),
                "original_model": obj.get("Model"),
            }
        )
        added_objects[name] = node

    # Third pass: add child objects with parent relationships
    for obj in child_objects:
        name = _safe_str(obj.get("Model"))
        object_type = name.split("_")[0] if "_" in name else name

        # Find parent from constraints
        parent_node = None
        for c in obj.get("constraints", []):
            if not isinstance(c, dict):
                continue
            ctype = _safe_str(c.get("type")).lower()

            if ctype in ("on_top_of", "beside", "inside"):
                target = _safe_str(c.get("target"))
                if target in added_objects:
                    parent_node = added_objects[target]
                    break

        # If no parent found, add to root
        anchor_points = _generate_anchor_points_for_object(
            name, object_type
        )

        node = scene_graph.add_object(
            object_id=name,
            object_type=object_type,
            parent=parent_node,
            constraints=obj.get("constraints", []),
            anchor_points=anchor_points,
            metadata={
                "is_static": obj.get("is_static", False),
                "original_model": obj.get("Model"),
            }
        )
        added_objects[name] = node

    # Add semantic relationships based on spatial command
    semantic_relationships = _extract_semantic_relationships(
        objects, spatial_command
    )

    # Add scene graph to plan
    plan_with_graph = dict(plan)
    plan_with_graph["scene_graph"] = scene_graph
    plan_with_graph["semantic_relationships"] = semantic_relationships

    return plan_with_graph


def _generate_anchor_points_for_object(
    object_id: str,
    object_type: str,
) -> List[Dict[str, Any]]:
    """Generate anchor points for an object based on its type.

    Args:
        object_id: Unique identifier for the object
        object_type: Type/category of the object

    Returns:
        List of anchor point dictionaries
    """
    anchor_points = []

    # Surface objects (tables, shelves, etc.) provide top surface anchors
    surface_types = {"table", "desk", "shelf", "counter", "cabinet"}
    if any(t in object_type.lower() for t in surface_types):
        anchor_points.extend([
            {
                "id": f"{object_id}_top_center",
                "anchor_type": "surface",
                "position": "top_center",
                "available": True,
            },
            {
                "id": f"{object_id}_top_front",
                "anchor_type": "surface",
                "position": "top_front",
                "available": True,
            },
            {
                "id": f"{object_id}_top_back",
                "anchor_type": "surface",
                "position": "top_back",
                "available": True,
            },
        ])

    # All objects provide edge anchors for beside positioning
    for side in ["front", "back", "left", "right"]:
        anchor_points.append({
            "id": f"{object_id}_edge_{side}",
            "anchor_type": "edge",
            "position": f"edge_{side}",
            "available": True,
        })

    return anchor_points


def _extract_semantic_relationships(
    objects: List[Dict[str, Any]],
    spatial_command: Any,
) -> List[Dict[str, Any]]:
    """Extract semantic relationships between objects from constraints.

    Args:
        objects: List of objects with constraints
        spatial_command: Parsed spatial command

    Returns:
        List of semantic relationship dictionaries
    """
    relationships = []

    for obj in objects:
        source = _safe_str(obj.get("Model"))
        if not source:
            continue

        for c in obj.get("constraints", []):
            if not isinstance(c, dict):
                continue

            ctype = _safe_str(c.get("type")).lower()
            target = _safe_str(c.get("target"))

            if not target:
                continue

            # Map constraint types to semantic relationships
            relationship_type = None
            if ctype == "on_top_of":
                relationship_type = "supported_by"
            elif ctype == "beside":
                relationship_type = "adjacent_to"
            elif ctype == "inside":
                relationship_type = "contained_by"
            elif ctype == "near":
                relationship_type = "near"
            elif ctype == "face_to":
                relationship_type = "facing"

            if relationship_type:
                relationships.append({
                    "source": source,
                    "target": target,
                    "type": relationship_type,
                    "constraint": c,
                })

    # Add relationships from spatial command precise constraints
    for precise in getattr(spatial_command, "precise_constraints", []):
        if precise.source_object and precise.target_object:
            rel_type = (
                "precise_distance"
                if precise.constraint_type == "distance"
                else "precise_angle"
            )
            relationships.append({
                "source": precise.source_object,
                "target": precise.target_object,
                "type": rel_type,
                "value": precise.value,
                "unit": precise.unit,
            })

    return relationships
