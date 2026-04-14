import math
from typing import Any, Dict, List, Sequence, Tuple


def _half_xy(size: Sequence[float], scale: float) -> Tuple[float, float]:
    sx = float(size[0]) * float(scale)
    sy = float(size[1]) * float(scale)
    return max(0.01, sx / 2.0), max(0.01, sy / 2.0)


def _aabb_overlap(a: Dict[str, float], b: Dict[str, float], margin: float = 0.02) -> bool:
    separated_x = (a["max_x"] + margin < b["min_x"]) or (
        b["max_x"] + margin < a["min_x"]
    )
    separated_y = (a["max_y"] + margin < b["min_y"]) or (
        b["max_y"] + margin < a["min_y"]
    )
    return not (separated_x or separated_y)


def _candidate_grid(room_half_size: float, step: float) -> List[Tuple[float, float]]:
    pts: List[Tuple[float, float]] = []
    n = int((room_half_size * 2) / step)
    start = -room_half_size
    for ix in range(n + 1):
        x = start + ix * step
        for iy in range(n + 1):
            y = start + iy * step
            pts.append((x, y))
    return pts


def _point_in_polygon(x: float, y: float, polygon: Sequence[Sequence[float]]) -> bool:
    inside = False
    n = len(polygon)
    if n < 3:
        return False
    j = n - 1
    for i in range(n):
        xi, yi = float(polygon[i][0]), float(polygon[i][1])
        xj, yj = float(polygon[j][0]), float(polygon[j][1])
        intersects = ((yi > y) != (yj > y)) and (
            x < (xj - xi) * (y - yi) / max(1e-9, (yj - yi)) + xi
        )
        if intersects:
            inside = not inside
        j = i
    return inside


def _inside_room(
    x: float,
    y: float,
    hx: float,
    hy: float,
    room_half_size: float,
    room_polygon: Sequence[Sequence[float]],
) -> bool:
    if room_polygon:
        corners = [
            (x - hx, y - hy),
            (x - hx, y + hy),
            (x + hx, y - hy),
            (x + hx, y + hy),
        ]
        return all(_point_in_polygon(cx, cy, room_polygon) for cx, cy in corners)
    cond1 = x - hx < -room_half_size
    cond2 = x + hx > room_half_size
    cond3 = y - hy < -room_half_size
    cond4 = y + hy > room_half_size
    out_of_bounds = cond1 or cond2 or cond3 or cond4
    return not out_of_bounds


def _forbidden_overlap(
    candidate_box: Dict[str, float],
    forbidden_zones: Sequence[Dict[str, Any]],
) -> bool:
    for z in forbidden_zones:
        polygon = z.get("polygon")
        if not isinstance(polygon, list) or len(polygon) < 3:
            continue
        xs = [float(p[0]) for p in polygon]
        ys = [float(p[1]) for p in polygon]
        zbox = {
            "min_x": min(xs),
            "max_x": max(xs),
            "min_y": min(ys),
            "max_y": max(ys),
        }
        if _aabb_overlap(candidate_box, zbox, margin=0.0):
            return True
    return False


def _region_score(x: float, y: float, room_half_size: float, pref: str) -> float:
    d_center = math.sqrt(x * x + y * y)
    max_d = math.sqrt(2.0 * room_half_size * room_half_size)
    center_score = 1.0 - min(1.0, d_center / max_d)
    if pref == "middle":
        return center_score
    if pref == "edge":
        return 1.0 - center_score
    return 0.0


def _relative_score(
    x: float,
    y: float,
    yaw: float,
    constraint: Dict[str, Any],
    placed: Dict[str, Dict[str, Any]],
) -> float:
    ctype = str(constraint.get("type", "")).lower()
    target = constraint.get("target")
    if not target:
        return 0.0

    target_item = placed.get(target)
    if target_item is None:
        for p in placed.values():
            if str(p.get("Model", "")) == str(target):
                target_item = p
                break
    if target_item is None:
        return 0.0

    t = target_item["Pose"]
    tx, ty = float(t["x"]), float(t["y"])
    dx, dy = x - tx, y - ty
    dist = math.sqrt(dx * dx + dy * dy)

    if ctype == "near":
        low, high = constraint.get("distance", [0.2, 2.0])
        low = float(low)
        high = float(high)
        if low <= dist <= high:
            return 1.0
        return max(0.0, 1.0 - min(abs(dist - low), abs(dist - high)) / max(0.1, high))

    if ctype == "far":
        low, high = constraint.get("distance", [2.0, 8.0])
        low = float(low)
        high = float(high)
        if low <= dist <= high:
            return 1.0
        return max(0.0, min(dist / max(0.1, low), 1.0))

    if ctype == "left_of":
        return 1.0 if x < tx else 0.0
    if ctype == "right_of":
        return 1.0 if x > tx else 0.0
    if ctype == "in_front_of":
        return 1.0 if y > ty else 0.0
    if ctype == "behind":
        return 1.0 if y < ty else 0.0

    if ctype == "face_to":
        target_yaw = math.degrees(math.atan2(ty - y, tx - x))
        diff = abs((yaw - target_yaw + 180.0) % 360.0 - 180.0)
        return max(0.0, 1.0 - diff / 180.0)

    if ctype == "face_same_as":
        tyaw = float(target_item.get("yaw_deg", 0.0))
        diff = abs((yaw - tyaw + 180.0) % 360.0 - 180.0)
        return max(0.0, 1.0 - diff / 180.0)

    if ctype == "center_aligned":
        return max(0.0, 1.0 - abs(x - tx) - abs(y - ty))

    return 0.0


def _score_candidate(
    *,
    x: float,
    y: float,
    yaw: float,
    room_half_size: float,
    constraints: Sequence[Dict[str, Any]],
    placed: Dict[str, Dict[str, Any]],
    model_name: str,
    ring_target: str,
    ring_index: int,
    ring_total: int,
) -> float:
    total = 0.0
    for c in constraints:
        ctype = str(c.get("type", "")).lower()
        w = float(c.get("weight", 1.0))
        if ctype == "region":
            pref = str(c.get("value", "")).lower()
            total += w * _region_score(x, y, room_half_size, pref)
        else:
            total += w * _relative_score(x, y, yaw, c, placed)

    # Spread duplicates of the same object class to avoid row-like packing.
    if ring_target and ring_total > 1 and ring_target in placed:
        tpose = placed[ring_target]["Pose"]
        tx, ty = float(tpose["x"]), float(tpose["y"])
        ang = math.atan2(y - ty, x - tx)
        desired_ang = (2.0 * math.pi * float(ring_index)) / float(ring_total)
        diff = abs((ang - desired_ang + math.pi) % (2.0 * math.pi) - math.pi)
        total += 0.8 * max(0.0, 1.0 - diff / math.pi)

    # Mild repulsion from already placed same-class objects.
    for p in placed.values():
        if str(p.get("Model", "")) != model_name:
            continue
        pp = p.get("Pose") or {}
        dx = x - float(pp.get("x", 0.0))
        dy = y - float(pp.get("y", 0.0))
        d = math.sqrt(dx * dx + dy * dy)
        if d < 0.8:
            total -= (0.8 - d) * 1.5

    return total


def solve_floor_placements(
    *,
    full_placed_models: Sequence[Dict[str, Any]],
    semantic_plan: Dict[str, Any],
    room_half_size: float = 5.0,
    grid_step: float = 0.8,
    yaw_candidates_deg: Sequence[float] = (0.0, 90.0, 180.0, 270.0),
    beam_width: int = 12,
) -> List[Dict[str, Any]]:
    model_plan = {}
    objects = semantic_plan.get("objects", []) if isinstance(semantic_plan, dict) else []
    room = semantic_plan.get("room", {}) if isinstance(semantic_plan, dict) else {}
    room_polygon = room.get("polygon") if isinstance(room, dict) else []
    forbidden_zones = room.get("forbidden_zones") if isinstance(room, dict) else []
    if isinstance(objects, list):
        for o in objects:
            if isinstance(o, dict) and o.get("Model"):
                model_plan[str(o["Model"])] = o.get("constraints") or []

    indexed_models = []
    rr_instance_idx: Dict[str, int] = {}
    for i, m in enumerate(full_placed_models):
        mm = dict(m)
        name = str(mm.get("Model") or mm.get("name") or "")
        mm["Model"] = name
        mm["_index"] = i
        constraints_pool = model_plan.get(name, [])
        if isinstance(constraints_pool, list) and constraints_pool and isinstance(
            constraints_pool[0], dict
        ):
            mm["_constraints"] = constraints_pool
        elif isinstance(constraints_pool, list):
            # constraints_pool is list of constraints for a single instance
            mm["_constraints"] = constraints_pool
        else:
            mm["_constraints"] = []

        # If plan has multiple entries of same model name, map them instance-wise.
        if isinstance(objects, list):
            per_model_rows = [
                o for o in objects if isinstance(o, dict) and str(o.get("Model", "")) == name
            ]
            if per_model_rows:
                row_idx = rr_instance_idx.get(name, 0) % len(per_model_rows)
                row_constraints = per_model_rows[row_idx].get("constraints")
                if isinstance(row_constraints, list):
                    mm["_constraints"] = row_constraints
                rr_instance_idx[name] = rr_instance_idx.get(name, 0) + 1

        indexed_models.append(mm)

    # Duplicate-aware ring metadata for near(target) constraints.
    groups: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
    for m in indexed_models:
        constraints = m.get("_constraints", [])
        near_target = ""
        for c in constraints:
            if str(c.get("type", "")).lower() == "near":
                near_target = str(c.get("target", ""))
                if near_target:
                    break
        m["_ring_target"] = near_target
        key = (m["Model"], near_target)
        groups.setdefault(key, []).append(m)

    for (_name, target), group in groups.items():
        if not target:
            continue
        total = len(group)
        for idx, item in enumerate(group):
            item["_ring_index"] = idx
            item["_ring_total"] = total

    # Order larger objects first to reduce dead-ends.
    indexed_models.sort(
        key=lambda x: float(x.get("size", [1.0, 1.0, 1.0])[0]) * float(
            x.get("size", [1.0, 1.0, 1.0])[1]
        ),
        reverse=True,
    )

    grid = _candidate_grid(room_half_size=room_half_size, step=grid_step)

    states = [
        {
            "placed": {},
            "boxes": [],
            "score": 0.0,
        }
    ]

    for m in indexed_models:
        name = m["Model"]
        instance_id = f"{name}__{m.get('_index', 0)}"
        constraints = m.get("_constraints", [])
        ring_target = str(m.get("_ring_target", ""))
        ring_index = int(m.get("_ring_index", 0))
        ring_total = int(m.get("_ring_total", 1))
        hx, hy = _half_xy(m.get("size", [1.0, 1.0, 1.0]), scale=1.0)
        hz = max(0.01, float(m.get("size", [1.0, 1.0, 1.0])[2]) / 2.0)

        next_states = []
        for st in states:
            for x, y in grid:
                # Hard room bound check.
                if not _inside_room(
                    x,
                    y,
                    hx,
                    hy,
                    room_half_size,
                    room_polygon if isinstance(room_polygon, list) else [],
                ):
                    continue

                candidate_box = {
                    "min_x": x - hx,
                    "max_x": x + hx,
                    "min_y": y - hy,
                    "max_y": y + hy,
                }

                if isinstance(forbidden_zones, list) and _forbidden_overlap(
                    candidate_box,
                    forbidden_zones,
                ):
                    continue

                collision = False
                for b in st["boxes"]:
                    if _aabb_overlap(candidate_box, b):
                        collision = True
                        break
                if collision:
                    continue

                for yaw in yaw_candidates_deg:
                    item = dict(m)
                    item["Pose"] = {"x": float(x), "y": float(y), "z": float(hz)}
                    item["yaw_deg"] = float(yaw)

                    rel_score = _score_candidate(
                        x=float(x),
                        y=float(y),
                        yaw=float(yaw),
                        room_half_size=room_half_size,
                        constraints=constraints,
                        placed=st["placed"],
                        model_name=name,
                        ring_target=ring_target,
                        ring_index=ring_index,
                        ring_total=ring_total,
                    )

                    new_state = {
                        "placed": dict(st["placed"]),
                        "boxes": list(st["boxes"]),
                        "score": float(st["score"]) + float(rel_score),
                    }
                    new_state["placed"][instance_id] = item
                    new_state["boxes"].append(candidate_box)
                    next_states.append(new_state)

        if not next_states:
            # Fallback for this object if no feasible candidate found.
            fallback_item = dict(m)
            fallback_item["Pose"] = {"x": 0.0, "y": 0.0, "z": float(hz)}
            fallback_item["yaw_deg"] = 0.0
            for st in states:
                st["placed"][instance_id] = fallback_item
            continue

        next_states.sort(key=lambda s: s["score"], reverse=True)
        states = next_states[: max(1, beam_width)]

    best = max(states, key=lambda s: s["score"])

    out = []
    # Keep original model order for downstream pipeline compatibility.
    by_name: Dict[str, List[Dict[str, Any]]] = {}
    for p in best["placed"].values():
        by_name.setdefault(p["Model"], []).append(p)

    for original in full_placed_models:
        name = str(original.get("Model") or original.get("name"))
        if by_name.get(name):
            out.append(by_name[name].pop(0))
        else:
            fallback = dict(original)
            hz = max(0.01, float(fallback.get("size", [1.0, 1.0, 1.0])[2]) / 2.0)
            fallback["Pose"] = {"x": 0.0, "y": 0.0, "z": float(hz)}
            fallback["yaw_deg"] = 0.0
            out.append(fallback)

    return out
