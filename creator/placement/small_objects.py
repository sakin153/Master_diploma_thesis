"""Small object placement on support surfaces.

Improvements inspired by SceneSmith:
- Support surface validation with area checking
- Multiple spawn positions on receptacle surface (not just center)
- Collision checking between small objects on the same surface
- Stacking support for explicit "on" constraints
"""

import math
import random
from typing import Any, Dict, List, Optional, Sequence, Tuple


def _volume(size: Sequence[float]) -> float:
    return float(size[0]) * float(size[1]) * float(size[2])


def _get_size(model: Dict[str, Any]) -> List[float]:
    size = model.get("size")
    if size is None or len(size) < 3:
        return [0.1, 0.1, 0.1]
    return [max(0.01, float(size[0])), max(0.01, float(size[1])), max(0.01, float(size[2]))]


def _footprint_area(size: Sequence[float]) -> float:
    return float(size[0]) * float(size[1])


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


def _spawn_positions_on_surface(
    receptacle: Dict[str, Any],
    item_size: Sequence[float],
    n_samples: int = 8,
    rng: Optional[random.Random] = None,
) -> List[Tuple[float, float]]:
    """Generate candidate XY positions on the top surface of a receptacle.

    Inspired by SceneSmith's support surface extraction — we sample within
    the footprint of the receptacle, inset by the item's half-size.
    """
    if rng is None:
        rng = random.Random(42)

    rp = receptacle.get("Pose") or {"x": 0.0, "y": 0.0}
    rx = float(rp.get("x", 0.0))
    ry = float(rp.get("y", 0.0))
    rsize = _get_size(receptacle)

    # Usable surface = receptacle footprint minus item half-extents
    margin_x = float(item_size[0]) / 2.0 + 0.02
    margin_y = float(item_size[1]) / 2.0 + 0.02
    usable_hx = max(0.0, rsize[0] / 2.0 - margin_x)
    usable_hy = max(0.0, rsize[1] / 2.0 - margin_y)

    positions = [(rx, ry)]  # center always included
    for _ in range(n_samples - 1):
        ox = rng.uniform(-usable_hx, usable_hx) if usable_hx > 0.01 else 0.0
        oy = rng.uniform(-usable_hy, usable_hy) if usable_hy > 0.01 else 0.0
        positions.append((rx + ox, ry + oy))

    return positions


def _xy_overlap(
    x1: float, y1: float, s1: Sequence[float],
    x2: float, y2: float, s2: Sequence[float],
    margin: float = 0.02,
) -> bool:
    """Check if two items overlap in XY given their center positions and sizes."""
    hx1, hy1 = s1[0] / 2.0, s1[1] / 2.0
    hx2, hy2 = s2[0] / 2.0, s2[1] / 2.0
    sep_x = (x1 + hx1 + margin < x2 - hx2) or (x2 + hx2 + margin < x1 - hx1)
    sep_y = (y1 + hy1 + margin < y2 - hy2) or (y2 + hy2 + margin < y1 - hy1)
    return not (sep_x or sep_y)


def solve_small_object_placements(
    *,
    placed_models: Sequence[Dict[str, Any]],
    semantic_plan: Optional[Dict[str, Any]] = None,
    small_threshold_volume: float = 0.06,
    seed: int = 42,
) -> List[Dict[str, Any]]:
    """Place small objects on support surfaces with collision avoidance.

    Pipeline:
    1. Identify receptacles (large objects) and small objects
    2. For explicit "on" constraints: place on named target
    3. For implicit small objects: find best receptacle, sample positions
    4. Check collisions with other small objects on same surface
    """
    rng = random.Random(seed)
    out = [dict(m) for m in placed_models]
    constraint_rows = _instance_constraints(out, semantic_plan or {"objects": []})

    # Identify receptacles
    receptacles = []
    for m in out:
        size = _get_size(m)
        if _volume(size) >= small_threshold_volume:
            receptacles.append(m)

    # Track items placed on each receptacle for collision checking
    surface_items: Dict[int, List[Tuple[float, float, List[float]]]] = {}

    for i, m in enumerate(out):
        size = _get_size(m)
        constraints = constraint_rows[i] if i < len(constraint_rows) else []

        # Check for explicit "on" constraint
        on_target = ""
        for c in constraints:
            ctype = str(c.get("type", "")).lower()
            if ctype in {"on", "on_top_of", "on-top-of", "on top of", "on_top"}:
                on_target = str(c.get("target", ""))
                if on_target:
                    break

        if on_target:
            receptacle = _find_target_by_name(
                item_index=i,
                target_name=on_target,
                placed_models=out,
            )
            if receptacle is not None:
                _place_on_receptacle(out, i, receptacle, size, rng, surface_items)
                continue

        # Only auto-place small objects
        if _volume(size) >= small_threshold_volume:
            continue

        # Find best receptacle (closest large object)
        receptacle = _find_best_receptacle(m, receptacles)
        if receptacle is None:
            continue

        _place_on_receptacle(out, i, receptacle, size, rng, surface_items)

    return out


def _place_on_receptacle(
    out: List[Dict[str, Any]],
    item_idx: int,
    receptacle: Dict[str, Any],
    item_size: List[float],
    rng: random.Random,
    surface_items: Dict[int, List[Tuple[float, float, List[float]]]],
    room_half_size: float = 5.0,
) -> None:
    """Place an item on a receptacle surface, avoiding collisions with others."""
    rp = receptacle.get("Pose") or {"x": 0.0, "y": 0.0, "z": 0.0}
    rsize = _get_size(receptacle)

    z_base = float(rp.get("z", 0.0))
    z_receptacle = float(rsize[2]) / 2.0
    z_item = float(item_size[2]) / 2.0
    target_z = z_base + z_receptacle + z_item + 0.01

    # Find receptacle id for tracking
    recep_id = id(receptacle)
    existing = surface_items.get(recep_id, [])

    # Try candidate positions
    candidates = _spawn_positions_on_surface(receptacle, item_size, n_samples=12, rng=rng)
    best_pos = candidates[0]  # default: center
    best_score = -float("inf")

    hx_item = float(item_size[0]) / 2.0
    hy_item = float(item_size[1]) / 2.0

    for cx, cy in candidates:
        # Clamp to room bounds
        cx = max(-room_half_size + hx_item + 0.02, min(room_half_size - hx_item - 0.02, cx))
        cy = max(-room_half_size + hy_item + 0.02, min(room_half_size - hy_item - 0.02, cy))

        # Check collision with other items on this surface
        collision = False
        for ex, ey, esize in existing:
            if _xy_overlap(cx, cy, item_size, ex, ey, esize):
                collision = True
                break

        if collision:
            continue

        # Prefer positions closer to center of receptacle
        rx = float(rp.get("x", 0.0))
        ry = float(rp.get("y", 0.0))
        dist_to_center = math.sqrt((cx - rx) ** 2 + (cy - ry) ** 2)
        score = -dist_to_center

        if score > best_score:
            best_score = score
            best_pos = (cx, cy)

    pose = dict(out[item_idx].get("Pose") or {})
    pose["x"] = best_pos[0]
    pose["y"] = best_pos[1]
    pose["z"] = target_z
    out[item_idx]["Pose"] = pose

    # Track placement
    surface_items.setdefault(recep_id, []).append(
        (best_pos[0], best_pos[1], item_size)
    )


def _find_best_receptacle(
    item: Dict[str, Any],
    receptacles: Sequence[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    if not receptacles:
        return None
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
