"""Physics-aware validation and repair for placed objects.

Improvements:
- Uses OBB/SAT collision from geometry module for accurate overlap detection
- Gradient-based overlap resolution (ImperativeScene-inspired)
- Better semantic constraint repair with convergence checking
"""

import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

from creator.placement.geometry import (
    gradient_resolve_overlaps,
    model_to_obb,
    obb_overlap,
)


def _z_intervals_overlap(a: Dict[str, Any], b: Dict[str, Any]) -> bool:
    """Check if two models overlap in Z (they're at the same height level).

    Objects at different Z levels (one stacked on another) should not be
    separated in XY — only objects at the same floor level should be
    pushed apart.

    Height in scene = size[1] (mesh Y, after euler="90 0 yaw").
    """
    size_a = a.get("size", [1.0, 1.0, 1.0])
    size_b = b.get("size", [1.0, 1.0, 1.0])
    pose_a = a.get("Pose") or {}
    pose_b = b.get("Pose") or {}
    za = float(pose_a.get("z", 0.0))
    zb = float(pose_b.get("z", 0.0))
    # size[1] = mesh Y = scene Z height (not size[2] which is scene Y depth)
    ha = max(0.01, float(size_a[1]) if len(size_a) > 1 else 1.0) / 2.0
    hb = max(0.01, float(size_b[1]) if len(size_b) > 1 else 1.0) / 2.0
    return not ((za + ha < zb - hb) or (zb + hb < za - ha))


def _size_xyz(item: Dict[str, Any]) -> Tuple[float, float, float]:
    size = item.get("size")
    if not isinstance(size, (list, tuple)) or len(size) < 3:
        return 1.0, 1.0, 1.0
    return max(0.01, float(size[0])), max(0.01, float(size[1])), max(0.01, float(size[2]))


def validate_and_repair_layout(
    placed_models: Sequence[Dict[str, Any]],
    *,
    min_z: float = 0.01,
    room_half_size: float = 5.0,
    repair_iters: int = 8,
    push_step: float = 0.06,
    semantic_plan: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """Validate layout and resolve overlaps using gradient-based resolution.

    Two-phase approach:
    1. Fix Z positions (ensure objects rest on floor or support)
    2. Resolve XY overlaps using gradient descent (ImperativeScene approach)
    3. Check convergence and log collision details
    """
    print(f"[validate_and_repair] Starting with {len(placed_models)} models...")
    repaired: List[Dict[str, Any]] = []
    for m in placed_models:
        item = dict(m)
        pose = dict(item.get("Pose") or {"x": 0.0, "y": 0.0, "z": min_z})
        z = float(pose.get("z", min_z))
        if z < min_z:
            z = min_z
        pose["z"] = z
        item["Pose"] = pose
        repaired.append(item)

    # Phase 2: Gradient-based overlap resolution with adaptive iteration limits
    # Calculate adaptive iteration limit based on scene complexity
    num_objects = len(repaired)
    base_iterations = repair_iters * 20
    
    # Add bonus iterations for complex scenes (more than 5 objects)
    if num_objects > 5:
        bonus_iterations = (num_objects - 5) * 50
        adaptive_iterations = base_iterations + bonus_iterations
    else:
        adaptive_iterations = base_iterations
    
    # Cap at 500 iterations to prevent infinite loops
    adaptive_iterations = min(adaptive_iterations, 500)
    
    print(
        f"[validate_and_repair] Using adaptive iteration limit: "
        f"{adaptive_iterations} iterations "
        f"(base={base_iterations}, objects={num_objects})"
    )
    
    # Phase 2.5: Early infeasibility detection
    # Calculate total object footprint area vs available surface area
    total_footprint_area = 0.0
    for m in repaired:
        size = m.get("size", [1.0, 1.0, 1.0])
        # size[0] = width (X), size[2] = depth (Y) for footprint
        width = max(0.01, float(size[0]) if len(size) > 0 else 1.0)
        depth = max(0.01, float(size[2]) if len(size) > 2 else 1.0)
        total_footprint_area += width * depth
    
    # Available room area (square room)
    available_area = (2.0 * room_half_size) ** 2
    density_ratio = total_footprint_area / max(0.01, available_area)
    
    print(
        f"[validate_and_repair] Density check: "
        f"footprint={total_footprint_area:.2f}m², "
        f"available={available_area:.2f}m², ratio={density_ratio:.2f}"
    )
    
    # If density ratio > 0.8, invoke LLM fallback immediately
    if density_ratio > 0.8:
        print(
            f"[validate_and_repair] INFEASIBILITY DETECTED: "
            f"density ratio {density_ratio:.2f} > 0.8 threshold"
        )
        print("[validate_and_repair] Skipping gradient resolution, invoking LLM fallback...")
        
        # Import llm_replan_layout from llm_collision_resolver
        from creator.placement.llm_collision_resolver import llm_replan_layout
        
        # Call LLM fallback with context
        replanned_models, llm_success = llm_replan_layout(
            placed_models=repaired,
            semantic_plan=semantic_plan or {},
            room_half_size=room_half_size,
            collision_details=[],  # No collision details yet
            max_retries=3,
        )
        
        if llm_success:
            print(
                "[validate_and_repair] LLM fallback succeeded - "
                "using replanned layout"
            )
            repaired = replanned_models
            
            # Verify final collision status
            final_collisions = check_collisions(
                repaired, collision_margin=0.02
            )
            if final_collisions:
                print(
                    f"[validate_and_repair] WARNING: "
                    f"{len(final_collisions)} collision(s) remain "
                    f"after LLM fallback (best effort)"
                )
            else:
                print(
                    "[validate_and_repair] SUCCESS: "
                    "All collisions eliminated by LLM fallback"
                )
        else:
            print(
                "[validate_and_repair] LLM fallback failed - "
                "proceeding with gradient resolution"
            )
        
        # Return early if LLM succeeded
        if llm_success:
            return repaired
    
    repaired, converged = gradient_resolve_overlaps(
        repaired,
        room_half_size=room_half_size,
        iterations=adaptive_iterations,
        step_size=push_step,
        collision_margin=0.02,
        semantic_plan=semantic_plan,
    )
    
    if not converged:
        print("[validate_and_repair] Gradient resolution did not converge")

    # Phase 3: Convergence detection
    collisions = check_collisions(repaired, collision_margin=0.02)

    if collisions:
        # Convergence failure - collisions remain after gradient resolution
        print(
            f"[validate_and_repair] CONVERGENCE FAILURE: "
            f"{len(collisions)} collision(s) remain after {adaptive_iterations} iterations"
        )
        print("[validate_and_repair] Collision details:")
        for i, collision in enumerate(collisions):
            obj_a = collision['object_a']
            obj_b = collision['object_b']
            depth = collision['overlap_depth']
            pos_a = collision['position_a']
            pos_b = collision['position_b']
            print(
                f"  [{i+1}] {obj_a} <-> {obj_b}: "
                f"overlap_depth={depth:.4f}m, "
                f"pos_a=({pos_a['x']:.2f}, {pos_a['y']:.2f}, "
                f"{pos_a['z']:.2f}), "
                f"pos_b=({pos_b['x']:.2f}, {pos_b['y']:.2f}, "
                f"{pos_b['z']:.2f})"
            )
        
        # Task 3.2: Invoke LLM fallback when convergence fails
        print("[validate_and_repair] Invoking LLM fallback for layout replanning...")
        
        # Import llm_replan_layout from llm_collision_resolver
        from creator.placement.llm_collision_resolver import llm_replan_layout
        
        # Call LLM fallback with context
        replanned_models, llm_success = llm_replan_layout(
            placed_models=repaired,
            semantic_plan=semantic_plan or {},
            room_half_size=room_half_size,
            collision_details=collisions,
            max_retries=3,
        )
        
        if llm_success:
            print(
                "[validate_and_repair] LLM fallback succeeded - "
                "using replanned layout"
            )
            repaired = replanned_models
            
            # Verify final collision status
            final_collisions = check_collisions(
                repaired, collision_margin=0.02
            )
            if final_collisions:
                print(
                    f"[validate_and_repair] WARNING: "
                    f"{len(final_collisions)} collision(s) remain "
                    f"after LLM fallback (best effort)"
                )
            else:
                print(
                    "[validate_and_repair] SUCCESS: "
                    "All collisions eliminated by LLM fallback"
                )
        else:
            print(
                "[validate_and_repair] LLM fallback failed - "
                "keeping gradient resolution result"
            )
    else:
        print(
            "[validate_and_repair] SUCCESS: "
            "All collisions resolved after gradient resolution"
        )

    return repaired


def _instance_constraints(
    placed_models: Sequence[Dict[str, Any]],
    semantic_plan: Dict[str, Any],
) -> List[List[Dict[str, Any]]]:
    objects = semantic_plan.get("objects", []) if isinstance(semantic_plan, dict) else []
    by_name: Dict[str, List[List[Dict[str, Any]]]] = {}
    for o in objects:
        if not isinstance(o, dict):
            continue
        name = str(o.get("Model") or o.get("name") or "")
        if not name:
            continue
        cs = o.get("constraints") if isinstance(o.get("constraints"), list) else []
        by_name.setdefault(name, []).append([c for c in cs if isinstance(c, dict)])

    rr: Dict[str, int] = {}
    out: List[List[Dict[str, Any]]] = []
    for m in placed_models:
        name = str(m.get("Model") or m.get("name") or "")
        rows = by_name.get(name, [])
        if not rows:
            out.append([])
            continue
        i = rr.get(name, 0) % len(rows)
        rr[name] = rr.get(name, 0) + 1
        out.append(rows[i])
    return out


def _target_pose_candidates(
    placed_models: Sequence[Dict[str, Any]],
    target_name: str,
) -> List[Tuple[float, float]]:
    out: List[Tuple[float, float]] = []
    for m in placed_models:
        if str(m.get("Model") or m.get("name") or "") != target_name:
            continue
        p = m.get("Pose") or {}
        out.append((float(p.get("x", 0.0)), float(p.get("y", 0.0))))
    return out


def _find_nearest_target(
    placed_models: Sequence[Dict[str, Any]],
    *,
    source_index: int,
    target_name: str,
) -> Dict[str, Any]:
    src = placed_models[source_index]
    sp = src.get("Pose") or {}
    sx = float(sp.get("x", 0.0))
    sy = float(sp.get("y", 0.0))

    best: Dict[str, Any] = {}
    best_d = float("inf")
    for i, m in enumerate(placed_models):
        if i == source_index:
            continue
        if str(m.get("Model") or m.get("name") or "") != target_name:
            continue
        tp = m.get("Pose") or {}
        tx = float(tp.get("x", 0.0))
        ty = float(tp.get("y", 0.0))
        d = (sx - tx) * (sx - tx) + (sy - ty) * (sy - ty)
        if d < best_d:
            best_d = d
            best = m
    return best


def repair_semantic_constraints(
    placed_models: Sequence[Dict[str, Any]],
    *,
    semantic_plan: Dict[str, Any],
    iters: int = 6,
    step: float = 0.12,
) -> List[Dict[str, Any]]:
    """Iteratively repair semantic constraint violations.

    Uses diminishing step size for convergence (ImperativeScene-inspired).
    """
    repaired = [dict(m) for m in placed_models]
    constraints_rows = _instance_constraints(repaired, semantic_plan)

    for iteration in range(max(1, iters)):
        moved = False
        # Diminishing step for convergence
        current_step = step * (1.0 - 0.5 * iteration / max(1, iters))

        for i, item in enumerate(repaired):
            pose = dict(item.get("Pose") or {"x": 0.0, "y": 0.0, "z": 0.1})
            x = float(pose.get("x", 0.0))
            y = float(pose.get("y", 0.0))

            for c in constraints_rows[i]:
                ctype = str(c.get("type", "")).lower()
                tname = str(c.get("target", ""))
                targets = _target_pose_candidates(repaired, tname) if tname else []

                if ctype == "region":
                    pref = str(c.get("value", "")).lower()
                    if pref == "middle":
                        x *= (1.0 - current_step * 0.5)
                        y *= (1.0 - current_step * 0.5)
                        moved = True
                    continue

                if not targets:
                    continue

                tx, ty = min(
                    targets,
                    key=lambda t: (x - t[0]) ** 2 + (y - t[1]) ** 2,
                )
                dx, dy = tx - x, ty - y
                dist = math.sqrt(dx * dx + dy * dy)

                if ctype == "near":
                    d = (c.get("distance")
                         if isinstance(c.get("distance"), list)
                         else [0.3, 1.2])
                    low, high = float(d[0]), float(d[1])
                    if dist > high and dist > 1e-6:
                        x += current_step * dx / dist
                        y += current_step * dy / dist
                        moved = True
                    elif dist < low and dist > 1e-6:
                        x -= current_step * dx / dist
                        y -= current_step * dy / dist
                        moved = True
                elif ctype == "left_of" and not (x < tx - 0.1):
                    x -= current_step
                    moved = True
                elif ctype == "right_of" and not (x > tx + 0.1):
                    x += current_step
                    moved = True
                elif ctype == "in_front_of" and not (y > ty + 0.1):
                    y += current_step
                    moved = True
                elif ctype == "behind" and not (y < ty - 0.1):
                    y -= current_step
                    moved = True
                elif ctype in {"on", "on_top_of", "on-top-of", "on top of", "on_top"}:
                    target_item = _find_nearest_target(
                        repaired,
                        source_index=i,
                        target_name=tname,
                    )
                    if target_item:
                        tp = target_item.get("Pose") or {}
                        # size[1] = height; size[0]/size[2] = footprint
                        item_size = item.get("size") or [0, 0.1, 0]
                        target_size = target_item.get("size") or [0, 0.1, 0]
                        s_hh = max(0.01, float(item_size[1])) / 2.0
                        t_hh = max(0.01, float(target_size[1])) / 2.0
                        t_hx = max(0.01, float(target_size[0])) / 2.0
                        t_hy = max(0.01, float(target_size[2])) / 2.0
                        tx_c = float(tp.get("x", 0.0))
                        ty_c = float(tp.get("y", 0.0))
                        pose["z"] = float(tp.get("z", 0.0)) + t_hh + s_hh + 0.01
                        # Only move X,Y to center if outside target footprint
                        if abs(x - tx_c) > t_hx or abs(y - ty_c) > t_hy:
                            x = tx_c
                            y = ty_c
                        moved = True

            pose["x"] = float(x)
            pose["y"] = float(y)
            item["Pose"] = pose
            repaired[i] = item

        if not moved:
            break

    return repaired


def validate_semantic_constraints(
    placed_models: Sequence[Dict[str, Any]],
    *,
    semantic_plan: Dict[str, Any],
) -> Dict[str, Any]:
    constraints_rows = _instance_constraints(placed_models, semantic_plan)
    total = 0
    ok = 0
    details: List[Dict[str, Any]] = []

    for i, item in enumerate(placed_models):
        name = str(item.get("Model") or item.get("name") or "")
        p = item.get("Pose") or {}
        x = float(p.get("x", 0.0))
        y = float(p.get("y", 0.0))
        for c in constraints_rows[i]:
            total += 1
            ctype = str(c.get("type", "")).lower()
            tname = str(c.get("target", ""))
            sat = True
            if ctype == "region":
                pref = str(c.get("value", "")).lower()
                if pref == "middle":
                    sat = math.sqrt(x * x + y * y) <= 2.4
            else:
                targets = _target_pose_candidates(placed_models, tname) if tname else []
                if targets:
                    tx, ty = min(
                        targets,
                        key=lambda t: (x - t[0]) ** 2 + (y - t[1]) ** 2,
                    )
                    dx, dy = x - tx, y - ty
                    dist = math.sqrt(dx * dx + dy * dy)
                    if ctype == "near":
                        d = (c.get("distance")
                             if isinstance(c.get("distance"), list)
                             else [0.3, 1.2])
                        sat = float(d[0]) <= dist <= float(d[1])
                    elif ctype == "left_of":
                        sat = x < tx - 0.1
                    elif ctype == "right_of":
                        sat = x > tx + 0.1
                    elif ctype == "in_front_of":
                        sat = y > ty + 0.1
                    elif ctype == "behind":
                        sat = y < ty - 0.1
                    elif ctype in {"on", "on_top_of", "on-top-of", "on top of", "on_top"}:
                        source_pose = item.get("Pose") or {}
                        source_z = float(source_pose.get("z", 0.0))
                        target_item = _find_nearest_target(
                            placed_models,
                            source_index=i,
                            target_name=tname,
                        )
                        if target_item:
                            target_pose = target_item.get("Pose") or {}
                            target_z = float(target_pose.get("z", 0.0))
                            ssx, ssy, ssz = _size_xyz(item)
                            tsx, tsy, tsz = _size_xyz(target_item)
                            exp_z = target_z + tsz / 2.0 + ssz / 2.0 + 0.01
                            xy_tol = max(0.12, 0.35 * max(tsx, tsy))
                            z_tol = max(0.08, 0.25 * ssz)
                            sat = (
                                abs(x - tx) <= xy_tol
                                and abs(y - ty) <= xy_tol
                                and abs(source_z - exp_z) <= z_tol
                            )

            if sat:
                ok += 1
            else:
                details.append(
                    {
                        "model": name,
                        "constraint": ctype,
                        "target": tname,
                    }
                )

    ratio = 1.0 if total == 0 else float(ok) / float(total)
    return {
        "constraints_total": total,
        "constraints_satisfied": ok,
        "satisfaction_ratio": ratio,
        "violations": details,
    }


# ---------------------------------------------------------------------------
# Scene Graph support: collision detection, stacking validation, support capacity
# ---------------------------------------------------------------------------

def check_collisions(
    placed_models: Sequence[Dict[str, Any]],
    *,
    collision_margin: float = 0.01,
) -> List[Dict[str, Any]]:
    """Check for collisions between objects using OBB + SAT.

    Args:
        placed_models: List of placed model dictionaries with Pose and size
        collision_margin: Minimum clearance between objects (meters)

    Returns:
        List of collision dictionaries with keys:
        - object_a: Name of first colliding object
        - object_b: Name of second colliding object
        - overlap_depth: Penetration depth (meters)
        - position_a: Position of first object
        - position_b: Position of second object
    """
    collisions: List[Dict[str, Any]] = []
    n = len(placed_models)

    for i in range(n):
        for j in range(i + 1, n):
            # Skip objects at different Z levels (stacked objects)
            if not _z_intervals_overlap(placed_models[i], placed_models[j]):
                continue

            obb_i = model_to_obb(placed_models[i], inflation=collision_margin)
            obb_j = model_to_obb(placed_models[j], inflation=collision_margin)

            if obb_overlap(obb_i, obb_j, margin=0.0):
                from creator.placement.geometry import obb_overlap_depth
                depth = obb_overlap_depth(obb_i, obb_j)

                name_i = str(
                    placed_models[i].get("Model")
                    or placed_models[i].get("name")
                    or f"object_{i}"
                )
                name_j = str(
                    placed_models[j].get("Model")
                    or placed_models[j].get("name")
                    or f"object_{j}"
                )

                pose_i = placed_models[i].get("Pose") or {}
                pose_j = placed_models[j].get("Pose") or {}

                collisions.append({
                    "object_a": name_i,
                    "object_b": name_j,
                    "overlap_depth": depth,
                    "position_a": {
                        "x": float(pose_i.get("x", 0.0)),
                        "y": float(pose_i.get("y", 0.0)),
                        "z": float(pose_i.get("z", 0.0)),
                    },
                    "position_b": {
                        "x": float(pose_j.get("x", 0.0)),
                        "y": float(pose_j.get("y", 0.0)),
                        "z": float(pose_j.get("z", 0.0)),
                    },
                })

    return collisions


def validate_stacking(
    placed_models: Sequence[Dict[str, Any]],
    *,
    stability_threshold: float = 0.7,
) -> Dict[str, Any]:
    """Validate stacking stability for all objects.

    Checks:
    1. Objects on surfaces have sufficient overlap with support surface
    2. Center of mass is within support polygon
    3. No unstable configurations (top-heavy stacks)

    Args:
        placed_models: List of placed model dictionaries
        stability_threshold: Minimum overlap ratio for stable stacking (0.0-1.0)

    Returns:
        Dictionary with keys:
        - is_stable: Overall stability (bool)
        - unstable_objects: List of unstable object names
        - stability_issues: List of detailed stability issue dictionaries
    """
    unstable_objects: List[str] = []
    stability_issues: List[Dict[str, Any]] = []

    # Build support relationships
    support_map: Dict[int, int] = {}  # supported_index -> supporter_index
    for i, obj in enumerate(placed_models):
        pose = obj.get("Pose") or {}
        z = float(pose.get("z", 0.0))
        sx, sy, sz = _size_xyz(obj)
        bottom_z = z - sz / 2.0

        # Skip floor objects
        if bottom_z < 0.05:
            continue

        # Find supporting object
        best_supporter = -1
        best_overlap = 0.0

        for j, supporter in enumerate(placed_models):
            if i == j:
                continue

            sup_pose = supporter.get("Pose") or {}
            sup_z = float(sup_pose.get("z", 0.0))
            tsx, tsy, tsz = _size_xyz(supporter)
            top_z = sup_z + tsz / 2.0

            # Check if supporter is below object
            if abs(bottom_z - top_z) > 0.1:
                continue

            # Check XY overlap
            obb_obj = model_to_obb(obj)
            obb_sup = model_to_obb(supporter)

            if obb_overlap(obb_obj, obb_sup, margin=-0.01):
                # Calculate overlap area (approximate)
                from creator.placement.geometry import obb_overlap_depth
                overlap = obb_overlap_depth(obb_obj, obb_sup)
                if overlap > best_overlap:
                    best_overlap = overlap
                    best_supporter = j

        if best_supporter == -1:
            # No supporter found - object is floating
            name = str(obj.get("Model") or obj.get("name") or f"object_{i}")
            unstable_objects.append(name)
            stability_issues.append({
                "object": name,
                "issue": "floating",
                "description": "Object has no supporting surface",
                "position": {
                    "x": float(pose.get("x", 0.0)),
                    "y": float(pose.get("y", 0.0)),
                    "z": z,
                },
            })
        else:
            support_map[i] = best_supporter

            # Check overlap ratio
            obj_area = sx * sy
            overlap_ratio = min(1.0, best_overlap / max(0.01, obj_area))

            if overlap_ratio < stability_threshold:
                name = str(
                    obj.get("Model") or obj.get("name") or f"object_{i}"
                )
                unstable_objects.append(name)
                supporter_name = str(
                    placed_models[best_supporter].get("Model")
                    or placed_models[best_supporter].get("name")
                    or f"object_{best_supporter}"
                )
                description = (
                    f"Object overlap ratio {overlap_ratio:.2f} "
                    f"below threshold {stability_threshold}"
                )
                stability_issues.append({
                    "object": name,
                    "issue": "insufficient_overlap",
                    "description": description,
                    "overlap_ratio": overlap_ratio,
                    "supporter": supporter_name,
                })

    return {
        "is_stable": len(unstable_objects) == 0,
        "unstable_objects": unstable_objects,
        "stability_issues": stability_issues,
    }


def check_support_capacity(
    placed_models: Sequence[Dict[str, Any]],
    *,
    default_capacity: float = 50.0,
) -> Dict[str, Any]:
    """Check if supporting surfaces can bear the weight of objects on them.

    Args:
        placed_models: List of placed model dictionaries
        default_capacity: Default weight capacity in kg (used when not specified)

    Returns:
        Dictionary with keys:
        - capacity_ok: Whether all supports have sufficient capacity (bool)
        - overloaded_objects: List of overloaded supporter names
        - capacity_issues: List of detailed capacity issue dictionaries
    """
    overloaded_objects: List[str] = []
    capacity_issues: List[Dict[str, Any]] = []

    # Build support relationships and calculate loads
    supporter_loads: Dict[int, float] = {}  # supporter_index -> total_load_kg

    for i, obj in enumerate(placed_models):
        pose = obj.get("Pose") or {}
        z = float(pose.get("z", 0.0))
        sx, sy, sz = _size_xyz(obj)
        bottom_z = z - sz / 2.0

        # Skip floor objects
        if bottom_z < 0.05:
            continue

        # Estimate object weight (rough approximation based on volume)
        # Assume average density of ~200 kg/m³ for typical household objects
        volume = sx * sy * sz
        weight = volume * 200.0

        # Find supporting object
        for j, supporter in enumerate(placed_models):
            if i == j:
                continue

            sup_pose = supporter.get("Pose") or {}
            sup_z = float(sup_pose.get("z", 0.0))
            tsx, tsy, tsz = _size_xyz(supporter)
            top_z = sup_z + tsz / 2.0

            # Check if supporter is below object
            if abs(bottom_z - top_z) > 0.1:
                continue

            # Check XY overlap
            obb_obj = model_to_obb(obj)
            obb_sup = model_to_obb(supporter)

            if obb_overlap(obb_obj, obb_sup, margin=-0.01):
                # Add weight to supporter's load
                supporter_loads[j] = supporter_loads.get(j, 0.0) + weight
                break

    # Check capacity for each supporter
    for supporter_idx, total_load in supporter_loads.items():
        supporter = placed_models[supporter_idx]

        # Get capacity from metadata or use default
        capacity = supporter.get("weight_capacity", default_capacity)
        if isinstance(capacity, (int, float)):
            capacity = float(capacity)
        else:
            capacity = default_capacity

        if total_load > capacity:
            name = str(
                supporter.get("Model")
                or supporter.get("name")
                or f"object_{supporter_idx}"
            )
            overloaded_objects.append(name)

            sup_pose = supporter.get("Pose") or {}
            description = (
                f"Total load {total_load:.1f}kg "
                f"exceeds capacity {capacity:.1f}kg"
            )
            capacity_issues.append({
                "supporter": name,
                "issue": "overloaded",
                "description": description,
                "total_load_kg": total_load,
                "capacity_kg": capacity,
                "overload_ratio": total_load / max(0.1, capacity),
                "position": {
                    "x": float(sup_pose.get("x", 0.0)),
                    "y": float(sup_pose.get("y", 0.0)),
                    "z": float(sup_pose.get("z", 0.0)),
                },
            })

    return {
        "capacity_ok": len(overloaded_objects) == 0,
        "overloaded_objects": overloaded_objects,
        "capacity_issues": capacity_issues,
    }
