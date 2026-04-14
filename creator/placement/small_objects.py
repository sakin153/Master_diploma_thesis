from typing import Any, Dict, List, Optional, Sequence


def _volume(size: Sequence[float]) -> float:
    return float(size[0]) * float(size[1]) * float(size[2])


def _get_size(model: Dict[str, Any]) -> Sequence[float]:
    size = model.get("size")
    if size is None:
        return [1.0, 1.0, 1.0]
    return size


def _instance_constraints(
    placed_models: Sequence[Dict[str, Any]],
    semantic_plan: Dict[str, Any],
) -> List[List[Dict[str, Any]]]:
    objects = semantic_plan.get("objects", []) if isinstance(semantic_plan, dict) else []
    by_name: Dict[str, List[List[Dict[str, Any]]]] = {}
    for obj in objects:
        if not isinstance(obj, dict):
            continue
        name = str(obj.get("Model") or obj.get("name") or "")
        if not name:
            continue
        constraints = obj.get("constraints") if isinstance(obj.get("constraints"), list) else []
        by_name.setdefault(name, []).append(
            [c for c in constraints if isinstance(c, dict)]
        )

    rr_idx: Dict[str, int] = {}
    rows: List[List[Dict[str, Any]]] = []
    for model in placed_models:
        name = str(model.get("Model") or model.get("name") or "")
        variants = by_name.get(name, [])
        if not variants:
            rows.append([])
            continue
        i = rr_idx.get(name, 0) % len(variants)
        rr_idx[name] = rr_idx.get(name, 0) + 1
        rows.append(variants[i])
    return rows


def _find_target_by_name(
    *,
    item_index: int,
    target_name: str,
    placed_models: Sequence[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    item_pose = placed_models[item_index].get("Pose") or {"x": 0.0, "y": 0.0}
    ix = float(item_pose.get("x", 0.0))
    iy = float(item_pose.get("y", 0.0))

    best = None
    best_d = float("inf")
    for idx, candidate in enumerate(placed_models):
        if idx == item_index:
            continue
        if str(candidate.get("Model") or candidate.get("name") or "") != target_name:
            continue
        cp = candidate.get("Pose") or {"x": 0.0, "y": 0.0}
        cx = float(cp.get("x", 0.0))
        cy = float(cp.get("y", 0.0))
        d = (ix - cx) ** 2 + (iy - cy) ** 2
        if d < best_d:
            best_d = d
            best = candidate
    return best


def _find_best_receptacle(
    item: Dict[str, Any],
    receptacles: Sequence[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    if not receptacles:
        return None
    # Place near the closest large object in XY.
    ip = item.get("Pose") or {"x": 0.0, "y": 0.0}
    ix, iy = float(ip.get("x", 0.0)), float(ip.get("y", 0.0))

    best = None
    best_d = float("inf")
    for r in receptacles:
        rp = r.get("Pose") or {"x": 0.0, "y": 0.0}
        rx, ry = float(rp.get("x", 0.0)), float(rp.get("y", 0.0))
        d = (ix - rx) ** 2 + (iy - ry) ** 2
        if d < best_d:
            best_d = d
            best = r
    return best


def solve_small_object_placements(
    *,
    placed_models: Sequence[Dict[str, Any]],
    semantic_plan: Optional[Dict[str, Any]] = None,
    small_threshold_volume: float = 0.06,
) -> List[Dict[str, Any]]:
    out = [dict(m) for m in placed_models]
    constraint_rows = _instance_constraints(out, semantic_plan or {"objects": []})

    receptacles = []
    for m in out:
        size = _get_size(m)
        if _volume(size) >= small_threshold_volume:
            receptacles.append(m)

    for i, m in enumerate(out):
        size = _get_size(m)

        constraints = constraint_rows[i] if i < len(constraint_rows) else []
        on_target = ""
        for c in constraints:
            ctype = str(c.get("type", "")).lower()
            if ctype in {"on", "on_top_of", "on-top-of", "on top of", "on_top"}:
                on_target = str(c.get("target", ""))
                if on_target:
                    break

        # If graph explicitly says this object must be on another one, enforce it.
        if on_target:
            receptacle = _find_target_by_name(
                item_index=i,
                target_name=on_target,
                placed_models=out,
            )
            if receptacle is not None:
                rp = receptacle.get("Pose") or {"x": 0.0, "y": 0.0, "z": 0.0}
                rsize = _get_size(receptacle)

                pose = dict(m.get("Pose") or {"x": 0.0, "y": 0.0, "z": 0.0})
                pose["x"] = float(rp.get("x", 0.0))
                pose["y"] = float(rp.get("y", 0.0))
                z_base = float(rp.get("z", 0.0))
                z_receptacle = float(rsize[2]) / 2.0
                z_item = float(size[2]) / 2.0
                pose["z"] = z_base + z_receptacle + z_item + 0.01
                out[i]["Pose"] = pose
                continue

        if _volume(size) >= small_threshold_volume:
            continue

        receptacle = _find_best_receptacle(m, receptacles)
        if receptacle is None:
            continue

        rp = receptacle.get("Pose") or {"x": 0.0, "y": 0.0, "z": 0.0}
        rsize = _get_size(receptacle)

        pose = dict(m.get("Pose") or {"x": 0.0, "y": 0.0, "z": 0.0})
        pose["x"] = float(rp.get("x", 0.0))
        pose["y"] = float(rp.get("y", 0.0))
        z_base = float(rp.get("z", 0.0))
        z_receptacle = float(rsize[2]) / 2.0
        z_item = float(size[2]) / 2.0
        pose["z"] = z_base + z_receptacle + z_item + 0.01
        out[i]["Pose"] = pose

    return out
