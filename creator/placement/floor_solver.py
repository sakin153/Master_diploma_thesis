"""Floor object placement solver using OBB/SAT collision and multi-loss scoring.

Best practices incorporated:
- OBB + SAT collision detection (ImperativeScene-inspired)
- Multi-loss scoring: inbound + overlap + clearance + constraint (ImperativeScene)
- DFS + beam search with largest-first ordering (MUJOCO_SCENE_PLACEMENT_PRINCIPLE)
- Clearance zones for doors/windows (SceneSmith)
- Ring layout for duplicate objects around targets
- Gradient-based post-pass overlap resolution (ImperativeScene)
- Adaptive candidate sampling: grid + jittered + near-target biased
"""

import math
import random
from typing import Any, Dict, List, Optional, Sequence, Tuple

from creator.placement.arranger import apply_arrangements
from creator.placement.targets import parse_instance_target
from creator.placement.geometry import (
    AABB,
    ClearanceZone,
    OBB,
    PlacementLoss,
    Vec2,
    build_door_clearance,
    compute_clearance_loss,
    compute_inbound_loss,
    compute_overlap_loss,
    gradient_resolve_overlaps,
    model_half_height,
    model_to_obb,
    obb_inside_polygon,
    obb_inside_room,
    obb_overlap,
    obb_overlap_depth,
    point_in_polygon,
)


# ---------------------------------------------------------------------------
# Candidate generation
# ---------------------------------------------------------------------------

def _grid_candidates(room_half_size: float, step: float) -> List[Tuple[float, float]]:
    pts: List[Tuple[float, float]] = []
    n = int((room_half_size * 2) / step)
    start = -room_half_size
    for ix in range(n + 1):
        x = start + ix * step
        for iy in range(n + 1):
            y = start + iy * step
            pts.append((x, y))
    return pts


def _jittered_candidates(
    base: Sequence[Tuple[float, float]],
    jitter: float,
    rng: random.Random,
    count: int = 2,
) -> List[Tuple[float, float]]:
    """Add jittered variants around base grid points for finer placement."""
    pts: List[Tuple[float, float]] = []
    for x, y in base:
        for _ in range(count):
            pts.append((
                x + rng.uniform(-jitter, jitter),
                y + rng.uniform(-jitter, jitter),
            ))
    return pts


def _near_target_candidates(
    target_pos: Tuple[float, float],
    distance_range: Tuple[float, float],
    n_samples: int,
    rng: random.Random,
) -> List[Tuple[float, float]]:
    """Generate candidates in an annular region around a target."""
    pts: List[Tuple[float, float]] = []
    lo, hi = distance_range
    for _ in range(n_samples):
        angle = rng.uniform(0, 2 * math.pi)
        r = rng.uniform(lo, hi)
        pts.append((
            target_pos[0] + r * math.cos(angle),
            target_pos[1] + r * math.sin(angle),
        ))
    return pts


# ---------------------------------------------------------------------------
# Constraint scoring (enhanced from original)
# ---------------------------------------------------------------------------

def _region_score(x: float, y: float, room_half_size: float, pref: str) -> float:
    d_center = math.sqrt(x * x + y * y)
    max_d = math.sqrt(2.0) * room_half_size
    center_ratio = min(1.0, d_center / max_d) if max_d > 0 else 0.0
    if pref == "middle":
        return 1.0 - center_ratio
    if pref == "edge":
        return center_ratio
    return 0.0


def _edge_wall_proximity_score(
    x: float, y: float, room_half_size: float,
) -> float:
    """Bonus for being close to a wall (for edge-region objects)."""
    dx = room_half_size - abs(x)
    dy = room_half_size - abs(y)
    min_dist = min(dx, dy)
    return max(0.0, 1.0 - min_dist / room_half_size)


def _compute_face_center_yaw(x: float, y: float) -> float:
    """Compute yaw angle to face the center of the room (0, 0).
    
    For objects placed at edges/walls, this makes them face inward.
    
    Args:
        x: Object x position
        y: Object y position
    
    Returns:
        Yaw angle in degrees that makes the object face toward (0, 0)
    """
    # Angle from object position toward center
    angle_to_center = math.atan2(-y, -x)  # Negative because we want to face center
    yaw_deg = math.degrees(angle_to_center)
    return yaw_deg % 360.0


def _get_yaw_candidates_for_position(
    x: float,
    y: float,
    room_half_size: float,
    constraints: List[Dict[str, Any]],
    default_yaw_candidates: Sequence[float],
) -> List[float]:
    """Get yaw candidates for a position, with special handling for edge objects.
    
    Objects near walls (within 20% of room size) automatically get face-center yaw
    candidates to make them face toward the room interior, regardless of constraints.
    
    Args:
        x: Position x
        y: Position y
        room_half_size: Room half size
        constraints: Object constraints (not used in current implementation)
        default_yaw_candidates: Default yaw values to try
    
    Returns:
        List of yaw angles to try for this position
    """
    # Check if position is near a wall (within 20% of room size)
    dx = room_half_size - abs(x)
    dy = room_half_size - abs(y)
    min_dist_to_wall = min(dx, dy)
    
    # If close to wall, add face-center yaw to make object face inward
    if min_dist_to_wall < room_half_size * 0.2:
        face_center_yaw = _compute_face_center_yaw(x, y)
        # Add face-center yaw plus small variations as primary candidates
        yaw_candidates = [
            face_center_yaw,
            (face_center_yaw + 15) % 360.0,
            (face_center_yaw - 15) % 360.0,
        ]
        # Also include default candidates as fallback
        yaw_candidates.extend(default_yaw_candidates)
        return yaw_candidates
    
    return list(default_yaw_candidates)


def _relative_score(
    x: float,
    y: float,
    yaw_deg: float,
    constraint: Dict[str, Any],
    placed: Dict[str, Dict[str, Any]],
) -> float:
    ctype = str(constraint.get("type", "")).lower()
    target = constraint.get("target")
    if not target:
        return 0.0

    target_item = placed.get(target)
    if target_item is None:
        base_target, _ = parse_instance_target(str(target))
        for p in placed.values():
            if str(p.get("Model", "")) == base_target:
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
        low, high = float(low), float(high)
        if low <= dist <= high:
            # Peak score at midpoint of range
            mid = (low + high) * 0.5
            return 1.0 - 0.3 * abs(dist - mid) / max(0.1, (high - low) * 0.5)
        return max(0.0, 1.0 - min(abs(dist - low), abs(dist - high)) / max(0.1, high))

    if ctype == "far":
        low, high = constraint.get("distance", [2.0, 8.0])
        low, high = float(low), float(high)
        if low <= dist <= high:
            return 1.0
        return max(0.0, min(dist / max(0.1, low), 1.0))

    if ctype == "left_of":
        return 1.0 if x < tx else max(0.0, 1.0 - (x - tx) * 2.0)
    if ctype == "right_of":
        return 1.0 if x > tx else max(0.0, 1.0 - (tx - x) * 2.0)
    if ctype == "in_front_of":
        return 1.0 if y > ty else max(0.0, 1.0 - (ty - y) * 2.0)
    if ctype == "behind":
        return 1.0 if y < ty else max(0.0, 1.0 - (y - ty) * 2.0)

    if ctype == "beside":
        # Placement relative to target's local orientation
        side = str(constraint.get("side", "")).lower()
        
        # Get target's yaw (default 0 if not set)
        target_yaw_deg = float(target_item.get("yaw_deg", 0.0))
        target_yaw_rad = math.radians(target_yaw_deg)
        
        # Transform to target's local frame
        dx_global = x - tx
        dy_global = y - ty
        
        # Rotate by -target_yaw to get local coordinates
        cos_t = math.cos(-target_yaw_rad)
        sin_t = math.sin(-target_yaw_rad)
        dx_local = dx_global * cos_t - dy_global * sin_t
        dy_local = dx_global * sin_t + dy_global * cos_t
        
        # Check distance constraint (default: close proximity for functional placement)
        dist_range = constraint.get("distance", [0.2, 0.6])
        dist_min, dist_max = float(dist_range[0]), float(dist_range[1])
        actual_dist = math.sqrt(dx_local**2 + dy_local**2)
        
        # If no side specified, just check distance (like "near")
        if not side:
            if dist_min <= actual_dist <= dist_max:
                return 1.0
            elif actual_dist < dist_min:
                return max(0.0, 1.0 - (dist_min - actual_dist) / dist_max)
            else:
                return max(0.0, 1.0 - (actual_dist - dist_max) / dist_max)
        
        if actual_dist < dist_min or actual_dist > dist_max:
            return max(0.0, 1.0 - min(abs(actual_dist - dist_min), abs(actual_dist - dist_max)) / dist_max)
        
        # Check side alignment
        score = 0.0
        if side == "right":
            # Right side: positive dx_local, small dy_local
            if dx_local > 0:
                score = 1.0 - abs(dy_local) / max(0.1, abs(dx_local))
        elif side == "left":
            # Left side: negative dx_local, small dy_local
            if dx_local < 0:
                score = 1.0 - abs(dy_local) / max(0.1, abs(dx_local))
        elif side == "front":
            # Front: positive dy_local, small dx_local
            if dy_local > 0:
                score = 1.0 - abs(dx_local) / max(0.1, abs(dy_local))
        elif side == "back":
            # Back: negative dy_local, small dx_local
            if dy_local < 0:
                score = 1.0 - abs(dx_local) / max(0.1, abs(dy_local))
        
        return max(0.0, score)

    if ctype == "face_to":
        if dist < 1e-6:
            return 1.0
        # Angle from object toward target — object's front should point at target
        target_yaw = math.degrees(math.atan2(ty - y, tx - x))
        diff = abs((yaw_deg - target_yaw) % 360.0 - 180.0)
        diff = min(diff, 360.0 - diff)
        score = max(0.0, 1.0 - diff / 90.0)
        return score

    if ctype == "face_same_as":
        tyaw = float(target_item.get("yaw_deg", 0.0))
        diff = abs((yaw_deg - tyaw + 180.0) % 360.0 - 180.0)
        return max(0.0, 1.0 - diff / 180.0)

    if ctype == "center_aligned":
        # Center alignment: align along the axis perpendicular to the
        # direction from target. If chair is in front of table (dy > dx),
        # align X; if to the side (dx > dy), align Y.
        dx_abs = abs(x - tx)
        dy_abs = abs(y - ty)
        
        # Determine primary approach direction
        if dx_abs > dy_abs:
            # Approaching from side → align Y (vertical centering)
            alignment_error = dy_abs
        else:
            # Approaching from front/back → align X (horizontal centering)
            alignment_error = dx_abs
        
        # Strong penalty for misalignment (×6 to dominate other constraints)
        return max(0.0, 1.0 - alignment_error * 6.0)

    return 0.0


def _score_candidate(
    *,
    x: float,
    y: float,
    yaw_deg: float,
    obb: OBB,
    room_half_size: float,
    constraints: Sequence[Dict[str, Any]],
    placed: Dict[str, Dict[str, Any]],
    placed_obbs: Sequence[OBB],
    clearance_zones: Sequence[ClearanceZone],
    model_name: str,
    ring_target: str,
    ring_index: int,
    ring_total: int,
) -> Tuple[float, PlacementLoss]:
    """Compute composite score and loss for a candidate placement."""
    loss = PlacementLoss()

    # Hard losses
    loss.inbound = compute_inbound_loss(obb, room_half_size)
    loss.overlap = compute_overlap_loss(obb, placed_obbs, margin=0.01)
    loss.clearance = compute_clearance_loss(obb, clearance_zones)

    if not loss.is_feasible():
        return -loss.total() * 10.0, loss

    # Soft constraint scoring
    total = 0.0
    is_anchor = any(
        c.get("anchor") or (str(c.get("type","")).lower() == "region" and str(c.get("value","")).lower() == "middle")
        for c in constraints if isinstance(c, dict)
    )
    
    for c in constraints:
        ctype = str(c.get("type", "")).lower()
        w = float(c.get("weight", 1.0))
        
        # Anchor objects get massive weight boost for region constraint
        if is_anchor and ctype == "region":
            w *= 10.0
        
        if ctype == "region":
            pref = str(c.get("value", "")).lower()
            s = _region_score(x, y, room_half_size, pref)
            if pref == "edge":
                s = (s + _edge_wall_proximity_score(x, y, room_half_size)) * 0.5
            total += w * s
        else:
            total += w * _relative_score(x, y, yaw_deg, c, placed)

    # Ring spread for duplicate objects around their target.
    # `placed` is keyed by composite instance IDs ("table__0", "table__arr_0"),
    # while `ring_target` is the bare model name from the LLM constraints —
    # so we have to look the anchor up by Model field, mirroring the same
    # fallback used by `_relative_score`.
    ring_anchor = None
    if ring_target and ring_total > 1:
        base_target, _ = parse_instance_target(str(ring_target))
        for p in placed.values():
            if str(p.get("Model", "")) == base_target:
                ring_anchor = p
                break

    if ring_anchor is not None and ring_total > 1:
        tpose = ring_anchor["Pose"]
        tx, ty = float(tpose["x"]), float(tpose["y"])
        ang = math.atan2(y - ty, x - tx)
        anchor_yaw_rad = math.radians(float(ring_anchor.get("yaw_deg", 0.0)))
        desired_ang = (2.0 * math.pi * float(ring_index)) / float(ring_total) + anchor_yaw_rad
        diff = abs((ang - desired_ang + math.pi) % (2.0 * math.pi) - math.pi)
        ring_score = max(0.0, 1.0 - diff / math.pi)
        # High weight so angular spread dominates other soft preferences:
        # without this, `face_to(table)` weight=10 (typical from the LLM)
        # pulls every chair toward whichever side gets picked first and
        # they clump on one side.
        total += 5.0 * ring_score
        # Reward yaw aligned to face the anchor (180° from outward radial).
        outward = math.degrees(math.atan2(y - ty, x - tx))
        face_yaw_target = (outward + 180.0) % 360.0
        yaw_diff = abs(((yaw_deg - face_yaw_target + 180.0) % 360.0) - 180.0)
        total += 2.0 * max(0.0, 1.0 - yaw_diff / 180.0)

    # Repulsion from same-class objects (avoid clustering)
    for p in placed.values():
        if str(p.get("Model", "")) != model_name:
            continue
        pp = p.get("Pose") or {}
        pdx = x - float(pp.get("x", 0.0))
        pdy = y - float(pp.get("y", 0.0))
        d = math.sqrt(pdx * pdx + pdy * pdy)
        if d < 0.8:
            total -= (0.8 - d) * 1.5

    loss.constraint = max(0.0, -total)
    return total, loss


# ---------------------------------------------------------------------------
# Main solver
# ---------------------------------------------------------------------------

def solve_floor_placements(
    *,
    full_placed_models: Sequence[Dict[str, Any]],
    semantic_plan: Dict[str, Any],
    room_half_size: float = 5.0,
    grid_step: Optional[float] = None,
    yaw_candidates_deg: Sequence[float] = (0.0, 90.0, 180.0, 270.0),
    beam_width: int = 12,
    collision_inflation: float = 0.05,
    seed: int = 42,
    max_backtracks: int = 4,
) -> List[Dict[str, Any]]:
    """Place floor objects using DFS + beam search with OBB/SAT collision.

    Pipeline:
    1. Parse semantic plan for constraints and room geometry
    2. Build clearance zones from forbidden zones
    3. Generate candidate positions (grid + jitter + near-target)
    4. DFS + beam search: place objects largest-first, prune infeasible
    5. Gradient-based post-pass to resolve residual overlaps
    """
    rng = random.Random(seed)

    # Adaptive grid step: derive from the smallest footprint in the scene so
    # small objects (chairs, stools) get sub-50 cm sampling resolution while
    # large rooms with only big furniture stay coarse for speed.
    if grid_step is None:
        min_footprint = float("inf")
        for m in full_placed_models:
            sz = m.get("size") or [1.0, 1.0, 1.0]
            if not isinstance(sz, (list, tuple)) or len(sz) < 3:
                continue
            sx = max(0.1, float(sz[0]))
            sy = max(0.1, float(sz[2]) if len(sz) > 2 else float(sz[0]))
            min_footprint = min(min_footprint, min(sx, sy))
        if min_footprint == float("inf"):
            grid_step = 0.5
        else:
            grid_step = max(0.20, min(0.80, 0.55 * min_footprint))

    # Parse semantic plan
    model_plan: Dict[str, List[Dict[str, Any]]] = {}
    objects = semantic_plan.get("objects", []) if isinstance(semantic_plan, dict) else []
    room = semantic_plan.get("room", {}) if isinstance(semantic_plan, dict) else {}
    room_polygon = room.get("polygon") if isinstance(room, dict) else []
    forbidden_zones = room.get("forbidden_zones") if isinstance(room, dict) else []
    if not isinstance(room_polygon, list):
        room_polygon = []
    if not isinstance(forbidden_zones, list):
        forbidden_zones = []

    if isinstance(objects, list):
        for o in objects:
            if isinstance(o, dict) and o.get("Model"):
                key = str(o["Model"])
                model_plan.setdefault(key, []).append(o.get("constraints") or [])

    # Build clearance zones from forbidden zones (SceneSmith-inspired)
    clearance_zones: List[ClearanceZone] = []
    for z in forbidden_zones:
        polygon = z.get("polygon")
        if isinstance(polygon, list) and len(polygon) >= 3:
            clearance_zones.append(build_door_clearance(polygon))

    # Pre-arrange grid/row groups (classroom, warehouse layouts)
    pre_arranged, beam_models = apply_arrangements(
        list(full_placed_models), semantic_plan,
        room_half_size=room_half_size,
        grid_threshold=4,
    )

    # Prepare indexed models with constraints (only beam_models need search)
    indexed_models = _prepare_indexed_models(
        beam_models, objects, model_plan,
    )

    # Topological sort: ensure dependencies (near-target anchors) are placed first
    indexed_models = _topological_sort_models(indexed_models)

    # Generate base grid
    grid = _grid_candidates(room_half_size=room_half_size, step=grid_step)
    # Add jittered points for finer placement
    jittered = _jittered_candidates(grid, jitter=grid_step * 0.3, rng=rng, count=1)
    all_candidates = [(0.0, 0.0)] + grid + jittered

    # DFS + Beam search — seed state with pre-arranged objects
    pre_arranged_obbs = [
        model_to_obb(m, inflation=collision_inflation)
        for m in pre_arranged
    ]
    pre_arranged_placed = {
        f"{str(m.get('Model', ''))}__arr_{i}": m
        for i, m in enumerate(pre_arranged)
    }
    states: List[Dict[str, Any]] = [
        {"placed": pre_arranged_placed, "obbs": pre_arranged_obbs, "score": 0.0,
         "history": [], "instance_ids": []}
    ]

    # History snapshot for backtracking
    history_stack: List[Tuple[int, List[Dict[str, Any]]]] = []
    backtracks_used = 0

    def _expand_step(
        m: Dict[str, Any], states_in: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        name = m["Model"]
        instance_id = f"{name}__{m.get('_index', 0)}"
        constraints = m.get("_constraints", [])
        ring_target = str(m.get("_ring_target", ""))
        ring_index = int(m.get("_ring_index", 0))
        ring_total = int(m.get("_ring_total", 1))
        hz = model_half_height(m)

        # Near-target candidates from any "near" constraint with a placed target
        extra_candidates: List[Tuple[float, float]] = []
        for c in constraints:
            if str(c.get("type", "")).lower() == "near":
                tgt_name = str(c.get("target", ""))
                dist_range = c.get("distance", [0.3, 2.0])
                lo, hi = float(dist_range[0]), float(dist_range[1])
                for st in states_in[:1]:
                    for p in st["placed"].values():
                        if str(p.get("Model", "")) == tgt_name:
                            pp = p.get("Pose", {})
                            extra_candidates.extend(
                                _near_target_candidates(
                                    (float(pp.get("x", 0)), float(pp.get("y", 0))),
                                    (lo, hi),
                                    n_samples=24,
                                    rng=rng,
                                )
                            )

        # Ring-around candidates: when N>=2 items orbit the same anchor,
        # sample positions along a ring around the anchor regardless of
        # whether the LLM emitted a `near` constraint. Without this, items
        # whose constraints are only `face_to`/`left_of`/etc. never see
        # candidates around the anchor at all and clump on whatever side
        # the room-grid happens to favour.
        if ring_target and ring_total > 1:
            for st in states_in[:1]:
                for p in st["placed"].values():
                    if str(p.get("Model", "")) != ring_target:
                        continue
                    pp = p.get("Pose", {})
                    ax = float(pp.get("x", 0.0))
                    ay = float(pp.get("y", 0.0))
                    asize = p.get("size") or [1.0, 1.0, 1.0]
                    a_hx = float(asize[0]) / 2.0
                    a_hy = (float(asize[2]) / 2.0
                            if len(asize) > 2 else a_hx)
                    msize = m.get("size") or [0.5, 0.5, 0.5]
                    m_hx = float(msize[0]) / 2.0
                    m_hy = (float(msize[2]) / 2.0
                            if len(msize) > 2 else m_hx)
                    # Place items just outside the anchor footprint.
                    radius_min = max(a_hx, a_hy) + max(m_hx, m_hy) + 0.05
                    radius_max = radius_min + 0.6
                    extra_candidates.extend(
                        _near_target_candidates(
                            (ax, ay),
                            (radius_min, radius_max),
                            n_samples=max(32, ring_total * 8),
                            rng=rng,
                        )
                    )
                    # Plus the exact desired-angle position (helps when
                    # the random ring missed the right slice).
                    desired_ang = (
                        2.0 * math.pi * float(ring_index) / float(ring_total)
                    )
                    r_mid = (radius_min + radius_max) / 2.0
                    extra_candidates.append(
                        (ax + r_mid * math.cos(desired_ang),
                         ay + r_mid * math.sin(desired_ang))
                    )
                    break  # one anchor instance is enough

        candidates = all_candidates + extra_candidates

        next_states: List[Dict[str, Any]] = []
        for st in states_in:
            best_for_state: List[Tuple[float, Dict[str, Any]]] = []

            for x, y in candidates:
                # Get yaw candidates for this position (special handling for edge objects)
                position_yaw_candidates = _get_yaw_candidates_for_position(
                    x, y, room_half_size, constraints, yaw_candidates_deg
                )
                
                for yaw in position_yaw_candidates:
                    obb = model_to_obb(
                        m,
                        inflation=collision_inflation,
                        pos_override=(x, y),
                        yaw_override=yaw,
                    )

                    if room_polygon:
                        if not obb_inside_polygon(obb, room_polygon):
                            continue
                    else:
                        if not obb_inside_room(obb, room_half_size):
                            continue

                    score, loss = _score_candidate(
                        x=x,
                        y=y,
                        yaw_deg=yaw,
                        obb=obb,
                        room_half_size=room_half_size,
                        constraints=constraints,
                        placed=st["placed"],
                        placed_obbs=st["obbs"],
                        clearance_zones=clearance_zones,
                        model_name=name,
                        ring_target=ring_target,
                        ring_index=ring_index,
                        ring_total=ring_total,
                    )

                    if not loss.is_feasible():
                        continue

                    item = dict(m)
                    item["Pose"] = {"x": float(x), "y": float(y), "z": float(hz)}
                    item["yaw_deg"] = float(yaw)

                    best_for_state.append((
                        float(st["score"]) + score,
                        {
                            "placed": dict(st["placed"]),
                            "obbs": list(st["obbs"]),
                            "score": float(st["score"]) + score,
                            "item": item,
                            "obb": obb,
                            "instance_id": instance_id,
                            "x": float(x),
                            "y": float(y),
                            "yaw": float(yaw),
                        },
                    ))

            best_for_state.sort(key=lambda t: t[0], reverse=True)
            for _, ns in best_for_state[:beam_width]:
                new_state = {
                    "placed": dict(ns["placed"]),
                    "obbs": list(ns["obbs"]),
                    "score": ns["score"],
                    "history": list(st.get("history", [])),
                    "instance_ids": list(st.get("instance_ids", [])),
                    "_last_xy": (ns["x"], ns["y"], ns["yaw"]),
                }
                new_state["placed"][ns["instance_id"]] = ns["item"]
                new_state["obbs"].append(ns["obb"])
                new_state["history"].append((ns["instance_id"], ns["item"], ns["obb"]))
                new_state["instance_ids"].append(ns["instance_id"])
                next_states.append(new_state)

        return next_states

    def _diverse_prune(
        nss: List[Dict[str, Any]], k: int, min_dist: float = 0.30,
    ) -> List[Dict[str, Any]]:
        """Keep top-K states with at least `min_dist` separation in last placement.
        Falls back to top-K by score when beam can't be diversified."""
        if not nss:
            return nss
        nss.sort(key=lambda s: s["score"], reverse=True)
        kept: List[Dict[str, Any]] = [nss[0]]
        for s in nss[1:]:
            if len(kept) >= k:
                break
            sx, sy, _ = s.get("_last_xy", (0.0, 0.0, 0.0))
            ok = True
            for kk in kept:
                kx, ky, _ = kk.get("_last_xy", (0.0, 0.0, 0.0))
                if math.hypot(sx - kx, sy - ky) < min_dist:
                    ok = False
                    break
            if ok:
                kept.append(s)
        # Fill remaining slots with top-score regardless of diversity
        if len(kept) < k:
            seen = {id(s) for s in kept}
            for s in nss:
                if id(s) not in seen:
                    kept.append(s)
                    if len(kept) >= k:
                        break
        return kept

    step_idx = 0
    n_models = len(indexed_models)
    while step_idx < n_models:
        m = indexed_models[step_idx]
        next_states = _expand_step(m, states)

        if not next_states:
            # Backtrack: pop the last placement from each state and retry with
            # different yaw/position diversity. After max_backtracks we give up
            # and place at the room centre clamped to nearest free spot.
            if backtracks_used < max_backtracks and step_idx > 0:
                backtracks_used += 1
                rolled_back = []
                for st in states:
                    hist = st.get("history") or []
                    ids = st.get("instance_ids") or []
                    if not hist:
                        rolled_back.append(st)
                        continue
                    last_id, _, _ = hist[-1]
                    new_placed = {
                        k: v for k, v in st["placed"].items() if k != last_id
                    }
                    new_obbs = [
                        model_to_obb(v, inflation=collision_inflation)
                        for k, v in new_placed.items()
                    ]
                    rolled_back.append({
                        "placed": new_placed,
                        "obbs": new_obbs,
                        "score": st["score"] * 0.9,
                        "history": hist[:-1],
                        "instance_ids": ids[:-1],
                    })
                states = rolled_back
                step_idx -= 1
                # Reseed RNG so jitter samples land elsewhere
                rng = random.Random(seed + backtracks_used * 7919)
                continue

            # Last-resort: place near room centre with progressive offset until
            # an OBB-feasible cell is found, never collapse to (0,0).
            name = m["Model"]
            instance_id = f"{name}__{m.get('_index', 0)}"
            hz = model_half_height(m)
            best_state = max(states, key=lambda s: s["score"]) if states else states[0]
            placed_obbs = best_state["obbs"]
            chosen = (0.0, 0.0)
            for radius in (0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0):
                found = False
                samples = 16 if radius > 0 else 1
                for _ in range(samples):
                    if radius == 0:
                        cx, cy = 0.0, 0.0
                    else:
                        ang = rng.uniform(0, 2 * math.pi)
                        cx = radius * math.cos(ang)
                        cy = radius * math.sin(ang)
                    obb = model_to_obb(
                        m, inflation=collision_inflation,
                        pos_override=(cx, cy), yaw_override=0.0,
                    )
                    if not obb_inside_room(obb, room_half_size):
                        continue
                    if not any(
                        obb_overlap_depth(obb, other) > 0 for other in placed_obbs
                    ):
                        chosen = (cx, cy)
                        found = True
                        break
                if found:
                    break
            fallback_item = dict(m)
            fallback_item["Pose"] = {"x": chosen[0], "y": chosen[1], "z": float(hz)}
            fallback_item["yaw_deg"] = 0.0
            fb_obb = model_to_obb(
                fallback_item, inflation=collision_inflation,
                pos_override=chosen, yaw_override=0.0,
            )
            for st in states:
                st["placed"][instance_id] = fallback_item
                st["obbs"].append(fb_obb)
                st.setdefault("history", []).append(
                    (instance_id, fallback_item, fb_obb),
                )
                st.setdefault("instance_ids", []).append(instance_id)
            step_idx += 1
            continue

        states = _diverse_prune(next_states, max(1, beam_width))
        step_idx += 1

    best = max(states, key=lambda s: s["score"])

    # Reconstruct output in original model order
    out = _reconstruct_output(full_placed_models, best["placed"])

    # Gradient-based post-pass to resolve any residual overlaps
    out, _ = gradient_resolve_overlaps(
        out,
        room_half_size=room_half_size,
        iterations=100,
        step_size=0.04,
        collision_margin=collision_inflation,
        semantic_plan=semantic_plan,
    )

    # Orientation search post-pass (ImperativeScene-inspired):
    # for objects with constraint violations, try all 4 rotations and keep best
    out = _orientation_search_postpass(
        out,
        objects=objects,
        room_half_size=room_half_size,
        collision_inflation=collision_inflation,
    )

    for m in out:
        p = m.get("Pose", {})
        print(f"[placed] {m.get('Model')} → x={p.get('x',0):.2f} y={p.get('y',0):.2f} yaw={m.get('yaw_deg',0):.0f}°")

    return out


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


_TARGET_BEARING_TYPES = frozenset({
    "near", "next_to", "beside",
    "on", "on_top_of", "on-top-of", "on top of", "on_top",
    "left_of", "right_of", "in_front_of", "behind",
    "face_to", "face_same_as",
    "center_aligned",
})


def _topological_sort_models(
    indexed_models: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Topological sort so target-anchor objects are placed before dependants.

    Each model instance is keyed by `(Model_name, _index)` so multiple copies
    of the same model don't collapse their dependency graphs. Constraints of
    every target-bearing type pull placement order — `on_top_of`, `near`,
    `face_to`, `left_of`, etc. all require their target to exist first.
    """
    # Per-instance keys keep duplicates independent.
    inst_keys: List[Tuple[str, int]] = []
    for m in indexed_models:
        inst_keys.append((str(m.get("Model", "")), int(m.get("_index", 0))))

    # Group instances by model name so target-by-name resolves to all copies.
    by_name: Dict[str, List[Tuple[str, int]]] = {}
    for k in inst_keys:
        by_name.setdefault(k[0], []).append(k)

    # Build dependency map: instance_key -> set of instance_keys it depends on.
    deps: Dict[Tuple[str, int], set] = {k: set() for k in inst_keys}
    for m in indexed_models:
        my_key = (str(m.get("Model", "")), int(m.get("_index", 0)))
        for c in m.get("_constraints", []):
            if not isinstance(c, dict):
                continue
            ctype = str(c.get("type", "")).lower()
            if ctype not in _TARGET_BEARING_TYPES:
                continue
            target = str(c.get("target", ""))
            target_keys = by_name.get(target, [])
            for tk in target_keys:
                if tk == my_key:
                    continue
                deps[my_key].add(tk)

    # Reverse adjacency for Kahn's algorithm.
    dependants: Dict[Tuple[str, int], List[Tuple[str, int]]] = {k: [] for k in inst_keys}
    for k, ds in deps.items():
        for d in ds:
            if d in dependants:
                dependants[d].append(k)

    in_degree: Dict[Tuple[str, int], int] = {k: len(ds) for k, ds in deps.items()}
    key_to_model = {
        (str(m.get("Model", "")), int(m.get("_index", 0))): m
        for m in indexed_models
    }

    def _footprint(m: Dict[str, Any]) -> float:
        s = m.get("size", [1.0, 1.0, 1.0])
        if not isinstance(s, (list, tuple)) or len(s) < 3:
            return 1.0
        return float(s[0]) * float(s[2] if len(s) > 2 else s[0])

    queue: List[Tuple[str, int]] = [k for k, d in in_degree.items() if d == 0]
    result: List[Dict[str, Any]] = []
    visited: set = set()

    # Helper to check if object is anchor
    def _is_anchor(m: Dict[str, Any]) -> bool:
        for c in m.get("_constraints", []):
            if isinstance(c, dict) and c.get("anchor"):
                return True
        return False

    while queue:
        # Sort by: 1) anchor first, 2) largest footprint
        queue.sort(
            key=lambda k: (
                not _is_anchor(key_to_model.get(k, {})),  # False (anchor) sorts before True
                -_footprint(key_to_model.get(k, {}))
            ),
        )
        k = queue.pop(0)
        if k in visited:
            continue
        visited.add(k)
        m = key_to_model.get(k)
        if m is not None:
            result.append(m)
        for dep_k in dependants.get(k, []):
            in_degree[dep_k] -= 1
            if in_degree[dep_k] <= 0 and dep_k not in visited:
                queue.append(dep_k)

    # Any cycles or stragglers — append in original order
    if len(result) < len(indexed_models):
        for m in indexed_models:
            k = (str(m.get("Model", "")), int(m.get("_index", 0)))
            if k not in visited:
                result.append(m)
                visited.add(k)

    return result


def _orientation_search_postpass(
    out: List[Dict[str, Any]],
    *,
    objects: Any,
    room_half_size: float,
    collision_inflation: float = 0.01,
) -> List[Dict[str, Any]]:
    """For objects with constraint violations after gradient pass, try all 4
    rotations and keep the orientation that minimizes constraint loss.

    Inspired by ImperativeScene orientation search.
    """
    # Build constraint lookup
    plan_constraints: Dict[str, List[Dict[str, Any]]] = {}
    if isinstance(objects, list):
        for o in objects:
            if isinstance(o, dict) and o.get("Model"):
                name = str(o["Model"])
                plan_constraints.setdefault(name, []).extend(
                    c for c in (o.get("constraints") or []) if isinstance(c, dict)
                )

    placed_lookup: Dict[str, List[Dict[str, Any]]] = {}
    for m in out:
        n = str(m.get("Model") or m.get("name") or "")
        placed_lookup.setdefault(n, []).append(m)

    result = list(out)
    # More yaw candidates for finer orientation control
    yaw_candidates = [0.0, 22.5, 45.0, 67.5, 90.0, 112.5, 135.0, 157.5, 
                     180.0, 202.5, 225.0, 247.5, 270.0, 292.5, 315.0, 337.5]

    for i, m in enumerate(result):
        name = str(m.get("Model") or m.get("name") or "")
        constraints = plan_constraints.get(name, [])
        if not constraints:
            continue
        
        # Check if this object has face_to constraints
        has_face_to = any(str(c.get("type", "")).lower() == "face_to" for c in constraints)
        if has_face_to:
            import logging
            logging.info(f"[orientation] Processing {name} with face_to constraint")

        current_yaw = float(m.get("yaw_deg", 0.0))
        pose = m.get("Pose") or {"x": 0.0, "y": 0.0, "z": 0.0}
        x = float(pose.get("x", 0.0))
        y = float(pose.get("y", 0.0))

        # Compute current score
        placed_others = {
            f"{n}__{j}": result[j]
            for j, n in enumerate((str(r.get("Model", "")) for r in result))
            if j != i
        }
        best_yaw = current_yaw
        best_score = _score_orientation(
            x, y, current_yaw, m, constraints, placed_others,
            room_half_size, collision_inflation,
        )

        for yaw in yaw_candidates:
            if abs(yaw - current_yaw) < 1.0:
                continue
            score = _score_orientation(
                x, y, yaw, m, constraints, placed_others,
                room_half_size, collision_inflation,
            )
            if score > best_score:
                best_score = score
                best_yaw = yaw

        if abs(best_yaw - current_yaw) > 1.0:
            print(f"[orientation_postpass] {name}: {current_yaw:.0f}° -> {best_yaw:.0f}° (score {best_score:.2f})")
            result[i] = dict(m)
            result[i]["yaw_deg"] = best_yaw

    return result


def _score_orientation(
    x: float,
    y: float,
    yaw: float,
    m: Dict[str, Any],
    constraints: List[Dict[str, Any]],
    placed: Dict[str, Dict[str, Any]],
    room_half_size: float,
    collision_inflation: float,
) -> float:
    """Score a candidate orientation based on constraint satisfaction."""
    obb = model_to_obb(
        m,
        inflation=collision_inflation,
        pos_override=(x, y),
        yaw_override=yaw,
    )

    # Inbound score
    inbound = compute_inbound_loss(obb, room_half_size)

    # Overlap penalty
    placed_obbs = [model_to_obb(p, inflation=collision_inflation) for p in placed.values()]
    overlap = compute_overlap_loss(obb, placed_obbs)

    # Constraint score (near = minimize distance)
    constraint_score = 0.0
    for c in constraints:
        ctype = str(c.get("type", "")).lower()
        if ctype == "near":
            target_name, _ = parse_instance_target(str(c.get("target", "")))
            dist_range = c.get("distance", [0.3, 2.0])
            lo, hi = float(dist_range[0]), float(dist_range[1])
            best_dist = float("inf")
            for p in placed.values():
                if str(p.get("Model", "")) == target_name:
                    pp = p.get("Pose", {})
                    d = math.sqrt((x - float(pp.get("x", 0))) ** 2 + (y - float(pp.get("y", 0))) ** 2)
                    best_dist = min(best_dist, d)
            if best_dist < float("inf"):
                if lo <= best_dist <= hi:
                    constraint_score += 10.0
                else:
                    constraint_score -= min(10.0, abs(best_dist - (lo + hi) / 2.0) * 2.0)
        
        elif ctype == "face_to":
            target_name, _ = parse_instance_target(str(c.get("target", "")))
            for p in placed.values():
                if str(p.get("Model", "")) == target_name:
                    pp = p.get("Pose", {})
                    tx, ty = float(pp.get("x", 0)), float(pp.get("y", 0))
                    dist = math.sqrt((x - tx) ** 2 + (y - ty) ** 2)
                    if dist > 1e-6:
                        # Calculate angle from object to target
                        target_yaw = math.degrees(math.atan2(ty - y, tx - x))
                        # Normalize to [0, 360)
                        target_yaw = target_yaw % 360.0
                        # Calculate angular difference (shortest path)
                        diff = abs((yaw - target_yaw + 180.0) % 360.0 - 180.0)
                        # Score: 1.0 when perfectly aligned, 0.0 when 180° off
                        face_score = max(0.0, 1.0 - diff / 180.0)
                        weight = float(c.get("weight", 1.0))
                        constraint_score += weight * face_score * 20.0  # Increased weight for importance
                        # Debug logging
                        import logging
                        logging.debug(f"[face_to] {name} @ ({x:.2f},{y:.2f}) yaw={yaw:.0f}° -> {target_name} @ ({tx:.2f},{ty:.2f}): target_yaw={target_yaw:.0f}°, diff={diff:.0f}°, score={face_score:.2f}")

    return constraint_score - 250.0 * inbound - 100.0 * overlap


def _prepare_indexed_models(
    full_placed_models: Sequence[Dict[str, Any]],
    objects: Any,
    model_plan: Dict[str, List[Dict[str, Any]]],
) -> List[Dict[str, Any]]:
    """Prepare and sort models for placement (largest footprint first)."""
    indexed_models = []
    rr_instance_idx: Dict[str, int] = {}
    _ON_TOP_TYPES = frozenset({"on_top_of", "on", "on-top-of", "on top of", "on_top"})

    for i, m in enumerate(full_placed_models):
        mm = dict(m)
        name = str(mm.get("Model") or mm.get("name") or "")
        mm["Model"] = name
        mm["_index"] = i

        # Assign constraints from plan AND copy is_static, size from semantic_plan
        if isinstance(objects, list):
            per_model_rows = [
                o for o in objects
                if isinstance(o, dict) and str(o.get("Model", "")) == name
            ]
            if per_model_rows:
                row_idx = rr_instance_idx.get(name, 0) % len(per_model_rows)
                plan_obj = per_model_rows[row_idx]
                
                # Copy constraints
                row_constraints = plan_obj.get("constraints")
                if isinstance(row_constraints, list):
                    mm["_constraints"] = row_constraints
                
                # CRITICAL FIX: Copy is_static and size from semantic_plan
                if "is_static" in plan_obj:
                    mm["is_static"] = plan_obj["is_static"]
                    print(f"[floor_solver] ✓ Copied is_static={plan_obj['is_static']} for {name}")
                
                if "size" in plan_obj:
                    mm["size"] = plan_obj["size"]
                    print(f"[floor_solver] ✓ Copied size={plan_obj['size']} for {name}")
                
                rr_instance_idx[name] = rr_instance_idx.get(name, 0) + 1

        if "_constraints" not in mm:
            plan_constraints = model_plan.get(name, [[]])
            if plan_constraints and isinstance(plan_constraints[0], list):
                mm["_constraints"] = plan_constraints[0]
            elif isinstance(plan_constraints, list):
                mm["_constraints"] = plan_constraints
            else:
                mm["_constraints"] = []

        # Skip objects that only have on_top_of constraints — they belong
        # to solve_small_object_placements, not the floor beam search.
        cs = mm.get("_constraints", [])
        if cs and all(str(c.get("type", "")).lower() in _ON_TOP_TYPES for c in cs):
            continue

        indexed_models.append(mm)

    # Ring metadata for items orbiting a common anchor.
    # Source: any constraint that references an anchor (`near`, `face_to`,
    # `left_of`, `right_of`, `in_front_of`, `behind`, `center_aligned`).
    # The LLM is inconsistent about which it emits, so collapse all of
    # them to "this item is associated with anchor X" and let the ring
    # spread distribute the group around X.
    _ANCHOR_CONSTRAINT_TYPES = (
        "near", "face_to", "beside", "left_of", "right_of",
        "in_front_of", "behind", "center_aligned",
    )
    groups: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
    for m in indexed_models:
        constraints = m.get("_constraints", [])
        ring_target = ""
        for c in constraints:
            if str(c.get("type", "")).lower() in _ANCHOR_CONSTRAINT_TYPES:
                ring_target = str(c.get("target", ""))
                if ring_target:
                    break
        m["_ring_target"] = ring_target
        key = (m["Model"], ring_target)
        groups.setdefault(key, []).append(m)

    for (_name, target), group in groups.items():
        if not target:
            continue
        total = len(group)
        # If items have beside.side, assign ring_index so desired_ang matches:
        # front=0°, right=90°, back=180°, left=270°
        _side_to_idx = {"front": 0, "right": 1, "back": 2, "left": 3}
        for idx, item in enumerate(group):
            side_idx = None
            for c in item.get("_constraints", []):
                if str(c.get("type", "")).lower() == "beside":
                    side = str(c.get("side", "")).lower()
                    if side in _side_to_idx:
                        side_idx = _side_to_idx[side]
                    break
            item["_ring_index"] = side_idx if side_idx is not None else idx
            item["_ring_total"] = total
            print(f"[ring] {item['Model']} side={side if 'side' in dir() else '?'} → ring_index={item['_ring_index']} / {total}")

    # Sort: largest footprint first to reduce dead-ends
    indexed_models.sort(
        key=lambda x: float(x.get("size", [1.0, 1.0, 1.0])[0])
        * float(x.get("size", [1.0, 1.0, 1.0])[1]),
        reverse=True,
    )

    return indexed_models


def _reconstruct_output(
    full_placed_models: Sequence[Dict[str, Any]],
    placed: Dict[str, Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Reconstruct output maintaining original model order."""
    by_name: Dict[str, List[Dict[str, Any]]] = {}
    for p in placed.values():
        by_name.setdefault(p["Model"], []).append(p)

    out: List[Dict[str, Any]] = []
    for original in full_placed_models:
        name = str(original.get("Model") or original.get("name"))
        if by_name.get(name):
            out.append(by_name[name].pop(0))
        else:
            fallback = dict(original)
            hz = model_half_height(fallback)
            fallback["Pose"] = {"x": 0.0, "y": 0.0, "z": float(hz)}
            fallback["yaw_deg"] = 0.0
            out.append(fallback)

    return out
