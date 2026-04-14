import math
from typing import Any, Dict, List, Sequence, Tuple


def _half_xy(size: Sequence[float]) -> List[float]:
    return [max(0.01, float(size[0]) / 2.0), max(0.01, float(size[1]) / 2.0)]


def _aabb(item: Dict[str, Any]) -> Dict[str, float]:
    pose = item.get("Pose") or {"x": 0.0, "y": 0.0}
    size = item.get("size")
    if size is None:
        size = [1.0, 1.0, 1.0]
    hx, hy = _half_xy(size)
    x, y = float(pose.get("x", 0.0)), float(pose.get("y", 0.0))
    return {"min_x": x - hx, "max_x": x + hx, "min_y": y - hy, "max_y": y + hy}


def _overlap_xy(a: Dict[str, float], b: Dict[str, float]) -> bool:
    cond1 = a["max_x"] < b["min_x"]
    cond2 = b["max_x"] < a["min_x"]
    cond3 = a["max_y"] < b["min_y"]
    cond4 = b["max_y"] < a["min_y"]
    separated = cond1 or cond2 or cond3 or cond4
    return not separated


def validate_and_repair_layout(
    placed_models: Sequence[Dict[str, Any]],
    *,
    min_z: float = 0.01,
    repair_iters: int = 8,
    push_step: float = 0.06,
) -> List[Dict[str, Any]]:
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

    # Lightweight interpenetration repair in XY using AABB pushes.
    for _ in range(max(1, repair_iters)):
        moved = False
        for i in range(len(repaired)):
            for j in range(i + 1, len(repaired)):
                ai = _aabb(repaired[i])
                aj = _aabb(repaired[j])
                if not _overlap_xy(ai, aj):
                    continue
                pi = repaired[i].get("Pose") or {"x": 0.0, "y": 0.0}
                pj = repaired[j].get("Pose") or {"x": 0.0, "y": 0.0}
                xi, yi = float(pi.get("x", 0.0)), float(pi.get("y", 0.0))
                xj, yj = float(pj.get("x", 0.0)), float(pj.get("y", 0.0))

                # Push apart along dominant axis.
                dx = xi - xj
                dy = yi - yj
                if abs(dx) >= abs(dy):
                    delta = push_step if dx >= 0 else -push_step
                    pi["x"] = xi + delta
                    pj["x"] = xj - delta
                else:
                    delta = push_step if dy >= 0 else -push_step
                    pi["y"] = yi + delta
                    pj["y"] = yj - delta

                repaired[i]["Pose"] = pi
                repaired[j]["Pose"] = pj
                moved = True
        if not moved:
            break

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


def repair_semantic_constraints(
    placed_models: Sequence[Dict[str, Any]],
    *,
    semantic_plan: Dict[str, Any],
    iters: int = 6,
    step: float = 0.12,
) -> List[Dict[str, Any]]:
    repaired = [dict(m) for m in placed_models]
    constraints_rows = _instance_constraints(repaired, semantic_plan)

    for _ in range(max(1, iters)):
        moved = False
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
                        x *= 0.95
                        y *= 0.95
                        moved = True
                    continue

                if not targets:
                    continue

                tx, ty = min(targets, key=lambda t: (x - t[0]) * (x - t[0]) + (y - t[1]) * (y - t[1]))
                dx, dy = tx - x, ty - y
                dist = math.sqrt(dx * dx + dy * dy)

                if ctype == "near":
                    d = c.get("distance") if isinstance(c.get("distance"), list) else [0.3, 1.2]
                    low, high = float(d[0]), float(d[1])
                    if dist > high and dist > 1e-6:
                        x += step * dx / dist
                        y += step * dy / dist
                        moved = True
                    elif dist < low and dist > 1e-6:
                        x -= step * dx / dist
                        y -= step * dy / dist
                        moved = True
                elif ctype == "left_of" and not (x < tx - 0.1):
                    x -= step
                    moved = True
                elif ctype == "right_of" and not (x > tx + 0.1):
                    x += step
                    moved = True
                elif ctype == "in_front_of" and not (y > ty + 0.1):
                    y += step
                    moved = True
                elif ctype == "behind" and not (y < ty - 0.1):
                    y -= step
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
                        key=lambda t: (x - t[0]) * (x - t[0]) + (y - t[1]) * (y - t[1]),
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
