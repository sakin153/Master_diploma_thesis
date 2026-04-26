"""Wall-mounted object placement solver.

Improvements over original:
- Distribute wall objects along all 4 walls instead of fixed X=room_half_size
- Collision checking against floor objects and other wall objects (3D AABB)
- Height-aware placement based on object type
- Spacing between wall-mounted objects
"""

import math
from typing import Any, Dict, List, Sequence, Tuple


_WALL_KEYWORDS = {
    "wall lamp": 1.8,
    "wall light": 1.8,
    "sconce": 1.7,
    "wall sconce": 1.7,
    "picture frame": 1.5,
    "painting": 1.5,
    "wall art": 1.5,
    "wall clock": 1.6,
    "wall mirror": 1.4,
    "wall shelf": 1.3,
    "floating shelf": 1.3,
    "wall sign": 1.5,
    "wall poster": 1.4,
}

# Exact-word keywords: only match if the word appears as a standalone token
_WALL_KEYWORDS_EXACT = {
    "sconce": 1.7,
    "painting": 1.5,
}


def _is_likely_floor_item(name: str) -> bool:
    """Detect objects that contain wall keywords as substrings but are floor items."""
    floor_prefixes = (
        "tv stand", "book", "floor lamp", "standing lamp", "table lamp",
        "desk lamp", "side table", "coffee table", "end table", "nightstand",
        "wardrobe", "cabinet", "dresser", "chest",
    )
    lname = name.lower()
    return any(lname.startswith(p) or p in lname for p in floor_prefixes)


def _is_wall_item(name: str, constraints: List[Dict[str, Any]]) -> bool:
    # Only mount objects that explicitly request wall mounting (not just edge preference)
    wants_wall_mount = any(
        str(c.get("type", "")).lower() == "wall_mount"
        for c in constraints
    )
    if wants_wall_mount:
        return True

    lname = name.lower()

    # Exclude known floor items that contain wall-related substrings
    if _is_likely_floor_item(lname):
        return False

    return any(k in lname for k in _WALL_KEYWORDS)


def _wall_height(name: str) -> float:
    lname = name.lower()
    for keyword, height in _WALL_KEYWORDS.items():
        if keyword in lname:
            return height
    for keyword, height in _WALL_KEYWORDS_EXACT.items():
        if keyword in lname:
            return height
    return 1.3


def _size_xyz(model: Dict[str, Any]) -> Tuple[float, float, float]:
    size = model.get("size")
    if not isinstance(size, (list, tuple)) or len(size) < 3:
        return 0.3, 0.3, 0.3
    # Canonical size convention in this project:
    # size[0]=scene X width, size[1]=scene Z height, size[2]=scene Y depth.
    return (
        max(0.01, float(size[0])),
        max(0.01, float(size[2])),
        max(0.01, float(size[1])),
    )


def _aabb_3d_overlap(
    a: Dict[str, float], b: Dict[str, float], margin: float = 0.05,
) -> bool:
    """Strict 3D AABB overlap.

    For wall vs floor checks, use _aabb_overlap_with_clearance instead — that
    one allows a wall-mounted painting at z=1.5 m above a couch at z=0.45 m
    even when their XY footprints intersect, as long as there's vertical
    clearance.
    """
    sep_x = a["max_x"] + margin < b["min_x"] or b["max_x"] + margin < a["min_x"]
    sep_y = a["max_y"] + margin < b["min_y"] or b["max_y"] + margin < a["min_y"]
    sep_z = a["max_z"] + margin < b["min_z"] or b["max_z"] + margin < a["min_z"]
    return not (sep_x or sep_y or sep_z)


def _aabb_overlap_with_clearance(
    wall_aabb: Dict[str, float],
    floor_aabb: Dict[str, float],
    *,
    xy_margin: float = 0.05,
    vertical_clearance: float = 0.10,
) -> bool:
    """Wall vs floor collision: only blocks when vertical separation is small.

    A painting at 1.5 m on a wall can hang above a sofa at 0.45 m even if
    their XY rectangles overlap — that's a desired setup, not a collision.
    The check returns True only when the wall object's lower edge is within
    `vertical_clearance` of the floor object's upper edge, AND the XY
    rectangles overlap.
    """
    # No XY overlap → no collision regardless of Z
    sep_x = (
        wall_aabb["max_x"] + xy_margin < floor_aabb["min_x"]
        or floor_aabb["max_x"] + xy_margin < wall_aabb["min_x"]
    )
    sep_y = (
        wall_aabb["max_y"] + xy_margin < floor_aabb["min_y"]
        or floor_aabb["max_y"] + xy_margin < wall_aabb["min_y"]
    )
    if sep_x or sep_y:
        return False

    # XY-overlapping: the wall object must be safely ABOVE the floor object,
    # otherwise the painting/clock would clip through the furniture below.
    if wall_aabb["min_z"] >= floor_aabb["max_z"] + vertical_clearance:
        return False
    if floor_aabb["min_z"] >= wall_aabb["max_z"] + vertical_clearance:
        # Wall object is BELOW floor object — should not happen but tolerate.
        return False
    return True


def _model_aabb_3d(model: Dict[str, Any]) -> Dict[str, float]:
    pose = model.get("Pose") or {"x": 0.0, "y": 0.0, "z": 0.0}
    sx, sy, sz = _size_xyz(model)
    x = float(pose.get("x", 0.0))
    y = float(pose.get("y", 0.0))
    z = float(pose.get("z", 0.0))
    return {
        "min_x": x - sx / 2.0,
        "max_x": x + sx / 2.0,
        "min_y": y - sy / 2.0,
        "max_y": y + sy / 2.0,
        "min_z": z - sz / 2.0,
        "max_z": z + sz / 2.0,
    }


def solve_wall_placements(
    *,
    placed_models: Sequence[Dict[str, Any]],
    semantic_plan: Dict[str, Any],
    room_half_size: float = 5.0,
) -> List[Dict[str, Any]]:
    """Place wall-mounted objects along room walls with collision avoidance."""
    plan_rows_by_model: Dict[str, List[List[Dict[str, Any]]]] = {}
    objects = semantic_plan.get("objects", []) if isinstance(semantic_plan, dict) else []
    if isinstance(objects, list):
        for o in objects:
            if isinstance(o, dict) and o.get("Model"):
                key = str(o["Model"])
                plan_rows_by_model.setdefault(key, []).append(
                    o.get("constraints") or []
                )

    # Walls: (axis, sign)
    # Wall position is computed per-object to account for object half-width
    walls = [
        ("x", +1),   # +X wall
        ("x", -1),   # -X wall
        ("y", +1),   # +Y wall
        ("y", -1),   # -Y wall
    ]

    out: List[Dict[str, Any]] = []
    wall_placed_aabbs: List[Dict[str, float]] = []
    floor_aabbs = [_model_aabb_3d(m) for m in placed_models]
    wall_idx = 0
    rr_idx: Dict[str, int] = {}

    for m in placed_models:
        item = dict(m)
        name = str(item.get("Model") or item.get("name") or "")
        rows = plan_rows_by_model.get(name, [])
        if rows:
            i = rr_idx.get(name, 0) % len(rows)
            constraints = rows[i]
            rr_idx[name] = rr_idx.get(name, 0) + 1
        else:
            constraints = []

        if not _is_wall_item(name, constraints):
            out.append(item)
            continue

        sx, sy, sz = _size_xyz(item)
        target_z = _wall_height(name)
        target_z = max(target_z, sz / 2.0 + 0.1)

        # Half-extents of this object for proper wall placement
        # Use the smaller dimension as depth (into wall), larger as width (along wall)
        half_depth = min(sx, sy) / 2.0
        half_width_along_wall = max(sx, sy) / 2.0

        # Try placing on different walls, rotating through them
        placed = False
        for attempt in range(len(walls)):
            w_axis, w_sign = walls[(wall_idx + attempt) % len(walls)]

            # Wall position: push object half-depth away from wall surface + margin
            w_pos = w_sign * (room_half_size - half_depth - 0.05)

            # Sample positions along the wall
            for offset in _wall_offsets(room_half_size, half_width_along_wall * 2):
                if w_axis == "x":
                    px, py, pz = w_pos, offset, target_z
                    # Face inward: yaw = 180 if +X wall, 0 if -X wall
                    yaw = 180.0 if w_sign > 0 else 0.0
                else:
                    px, py, pz = offset, w_pos, target_z
                    yaw = 270.0 if w_sign > 0 else 90.0

                candidate = dict(item)
                candidate["Pose"] = {"x": px, "y": py, "z": pz}
                candidate["yaw_deg"] = yaw
                c_aabb = _model_aabb_3d(candidate)

                # Check against floor objects and other wall objects.
                # Floor check uses vertical-clearance rule: a high-mounted
                # painting can hang above furniture below it without conflict.
                collision = False
                for fa in floor_aabbs:
                    if _aabb_overlap_with_clearance(c_aabb, fa):
                        collision = True
                        break
                if not collision:
                    for wa in wall_placed_aabbs:
                        if _aabb_3d_overlap(c_aabb, wa):
                            collision = True
                            break

                if not collision:
                    item = candidate
                    wall_placed_aabbs.append(c_aabb)
                    placed = True
                    break

            if placed:
                wall_idx = (wall_idx + attempt + 1) % len(walls)
                break

        if not placed:
            # Fallback: place on +X wall at center, accounting for object half-depth
            half_d = min(sx, sy) / 2.0
            fallback_x = room_half_size - half_d - 0.05
            pose = dict(item.get("Pose") or {"x": 0.0, "y": 0.0, "z": target_z})
            pose["x"] = fallback_x
            pose["y"] = 0.0
            pose["z"] = target_z
            item["Pose"] = pose
            item["yaw_deg"] = 180.0

        out.append(item)

    return out


def _wall_offsets(room_half_size: float, obj_width: float) -> List[float]:
    """Generate candidate positions along a wall, spread from center."""
    offsets = [0.0]
    step = max(0.4, obj_width + 0.2)
    pos = step
    while pos < room_half_size - obj_width / 2.0 - 0.1:
        offsets.append(pos)
        offsets.append(-pos)
        pos += step
    return offsets
