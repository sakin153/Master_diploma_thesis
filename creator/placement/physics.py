"""Physics-aware validation and repair for placed objects.

Improvements:
- Uses OBB/SAT collision from geometry module for accurate overlap detection
- Gradient-based overlap resolution (ImperativeScene-inspired)
- Better semantic constraint repair with convergence checking
"""

import math
from typing import Any, Dict, List, Sequence, Tuple

from creator.placement.geometry import (
    Vec2,
    gradient_resolve_overlaps,
    model_to_obb,
    obb_overlap,
)


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
) -> List[Dict[str, Any]]:
    """Validate layout and resolve overlaps using gradient-based resolution.

    Two-phase approach:
    1. Fix Z positions (ensure objects rest on floor or support)
    2. Resolve XY overlaps using gradient descent (ImperativeScene approach)
    """
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

    # Phase 2: Gradient-based overlap resolution
    repaired = gradient_resolve_overlaps(
        repaired,
        room_half_size=room_half_size,
        iterations=repair_iters * 20,
        step_size=push_step,
        collision_margin=0.01,
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
                    d = c.get("distance") if isinstance(c.get("distance"), list) else [0.3, 1.2]
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
                        _, _, ssz = _size_xyz(item)
                        _, _, tsz = _size_xyz(target_item)
                        pose["z"] = float(tp.get("z", 0.0)) + tsz / 2.0 + ssz / 2.0 + 0.01
                        x = float(tp.get("x", 0.0))
                        y = float(tp.get("y", 0.0))
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
                        d = c.get("distance") if isinstance(c.get("distance"), list) else [0.3, 1.2]
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
