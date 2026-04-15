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

    if ctype == "face_to":
        if dist < 1e-6:
            return 1.0
        target_yaw = math.degrees(math.atan2(ty - y, tx - x))
        diff = abs((yaw_deg - target_yaw + 180.0) % 360.0 - 180.0)
        return max(0.0, 1.0 - diff / 180.0)

    if ctype == "face_same_as":
        tyaw = float(target_item.get("yaw_deg", 0.0))
        diff = abs((yaw_deg - tyaw + 180.0) % 360.0 - 180.0)
        return max(0.0, 1.0 - diff / 180.0)

    if ctype == "center_aligned":
        return max(0.0, 1.0 - (abs(x - tx) + abs(y - ty)) * 0.5)

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
    for c in constraints:
        ctype = str(c.get("type", "")).lower()
        w = float(c.get("weight", 1.0))
        if ctype == "region":
            pref = str(c.get("value", "")).lower()
            s = _region_score(x, y, room_half_size, pref)
            if pref == "edge":
                s = (s + _edge_wall_proximity_score(x, y, room_half_size)) * 0.5
            total += w * s
        else:
            total += w * _relative_score(x, y, yaw_deg, c, placed)

    # Ring spread for duplicate objects around their target
    if ring_target and ring_total > 1 and ring_target in placed:
        tpose = placed[ring_target]["Pose"]
        tx, ty = float(tpose["x"]), float(tpose["y"])
        ang = math.atan2(y - ty, x - tx)
        desired_ang = (2.0 * math.pi * float(ring_index)) / float(ring_total)
        diff = abs((ang - desired_ang + math.pi) % (2.0 * math.pi) - math.pi)
        total += 0.8 * max(0.0, 1.0 - diff / math.pi)

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
    grid_step: float = 0.6,
    yaw_candidates_deg: Sequence[float] = (0.0, 90.0, 180.0, 270.0),
    beam_width: int = 12,
    collision_inflation: float = 0.01,
    seed: int = 42,
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

    # Prepare indexed models with constraints
    indexed_models = _prepare_indexed_models(
        full_placed_models, objects, model_plan,
    )

    # Generate base grid
    grid = _grid_candidates(room_half_size=room_half_size, step=grid_step)
    # Add jittered points for finer placement
    jittered = _jittered_candidates(grid, jitter=grid_step * 0.3, rng=rng, count=1)
    all_candidates = grid + jittered

    # DFS + Beam search
    states: List[Dict[str, Any]] = [
        {"placed": {}, "obbs": [], "score": 0.0}
    ]

    for m in indexed_models:
        name = m["Model"]
        instance_id = f"{name}__{m.get('_index', 0)}"
        constraints = m.get("_constraints", [])
        ring_target = str(m.get("_ring_target", ""))
        ring_index = int(m.get("_ring_index", 0))
        ring_total = int(m.get("_ring_total", 1))
        hz = model_half_height(m)

        # Add near-target candidates if there's a "near" constraint with a placed target
        extra_candidates: List[Tuple[float, float]] = []
        for c in constraints:
            if str(c.get("type", "")).lower() == "near":
                tgt_name = str(c.get("target", ""))
                dist_range = c.get("distance", [0.3, 2.0])
                lo, hi = float(dist_range[0]), float(dist_range[1])
                # Check if target is in any current state
                for st in states[:1]:  # just check best state
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

        candidates = all_candidates + extra_candidates

        next_states: List[Dict[str, Any]] = []
        for st in states:
            best_for_state: List[Tuple[float, Dict[str, Any]]] = []

            for x, y in candidates:
                for yaw in yaw_candidates_deg:
                    yaw_rad = math.radians(yaw)
                    obb = model_to_obb(
                        m,
                        inflation=collision_inflation,
                        pos_override=(x, y),
                        yaw_override=yaw,
                    )

                    # Hard check: inside room
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
                        },
                    ))

            # Keep top-K candidates per state
            best_for_state.sort(key=lambda t: t[0], reverse=True)
            for _, ns in best_for_state[:beam_width]:
                new_state = {
                    "placed": dict(ns["placed"]),
                    "obbs": list(ns["obbs"]),
                    "score": ns["score"],
                }
                new_state["placed"][ns["instance_id"]] = ns["item"]
                new_state["obbs"].append(ns["obb"])
                next_states.append(new_state)

        if not next_states:
            # Fallback: place at origin
            fallback_item = dict(m)
            fallback_item["Pose"] = {"x": 0.0, "y": 0.0, "z": float(hz)}
            fallback_item["yaw_deg"] = 0.0
            for st in states:
                st["placed"][instance_id] = fallback_item
            continue

        # Global beam pruning
        next_states.sort(key=lambda s: s["score"], reverse=True)
        states = next_states[:max(1, beam_width)]

    best = max(states, key=lambda s: s["score"])

    # Reconstruct output in original model order
    out = _reconstruct_output(full_placed_models, best["placed"])

    # Gradient-based post-pass to resolve any residual overlaps
    out = gradient_resolve_overlaps(
        out,
        room_half_size=room_half_size,
        iterations=100,
        step_size=0.04,
        collision_margin=collision_inflation,
    )

    return out


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _prepare_indexed_models(
    full_placed_models: Sequence[Dict[str, Any]],
    objects: Any,
    model_plan: Dict[str, List[Dict[str, Any]]],
) -> List[Dict[str, Any]]:
    """Prepare and sort models for placement (largest footprint first)."""
    indexed_models = []
    rr_instance_idx: Dict[str, int] = {}

    for i, m in enumerate(full_placed_models):
        mm = dict(m)
        name = str(mm.get("Model") or mm.get("name") or "")
        mm["Model"] = name
        mm["_index"] = i

        # Assign constraints from plan
        if isinstance(objects, list):
            per_model_rows = [
                o for o in objects
                if isinstance(o, dict) and str(o.get("Model", "")) == name
            ]
            if per_model_rows:
                row_idx = rr_instance_idx.get(name, 0) % len(per_model_rows)
                row_constraints = per_model_rows[row_idx].get("constraints")
                if isinstance(row_constraints, list):
                    mm["_constraints"] = row_constraints
                rr_instance_idx[name] = rr_instance_idx.get(name, 0) + 1

        if "_constraints" not in mm:
            plan_constraints = model_plan.get(name, [[]])
            if plan_constraints and isinstance(plan_constraints[0], list):
                mm["_constraints"] = plan_constraints[0]
            elif isinstance(plan_constraints, list):
                mm["_constraints"] = plan_constraints
            else:
                mm["_constraints"] = []

        indexed_models.append(mm)

    # Ring metadata for near(target) constraints
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
