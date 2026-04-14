import math
from typing import Any, Dict, List, Sequence, Tuple

from creator.placement.physics import validate_and_repair_layout


def _clip(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def _dist_xy(a: Dict[str, Any], b: Dict[str, Any]) -> float:
    pa = a.get("Pose") or {}
    pb = b.get("Pose") or {}
    dx = float(pa.get("x", 0.0)) - float(pb.get("x", 0.0))
    dy = float(pa.get("y", 0.0)) - float(pb.get("y", 0.0))
    return math.sqrt(dx * dx + dy * dy)


def _size_xyz(item: Dict[str, Any]) -> Tuple[float, float, float]:
    size = item.get("size")
    if not isinstance(size, (list, tuple)) or len(size) < 3:
        return 1.0, 1.0, 1.0
    return max(0.01, float(size[0])), max(0.01, float(size[1])), max(0.01, float(size[2]))


def _find_target(
    source: Dict[str, Any],
    target_name: str,
    models: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    candidates = [m for m in models if str(m.get("Model") or m.get("name")) == target_name]
    if not candidates:
        return {}
    best = min(candidates, key=lambda t: _dist_xy(source, t))
    return best


def evaluate_constraint_violations(
    placed_models: Sequence[Dict[str, Any]],
    semantic_plan: Dict[str, Any],
    *,
    room_half_size: float = 5.0,
) -> List[str]:
    objects = semantic_plan.get("objects", []) if isinstance(semantic_plan, dict) else []
    if not isinstance(objects, list) or not objects:
        return []

    by_name: Dict[str, List[Dict[str, Any]]] = {}
    for m in placed_models:
        name = str(m.get("Model") or m.get("name") or "")
        if not name:
            continue
        by_name.setdefault(name, []).append(m)

    violations: List[str] = []

    for obj in objects:
        if not isinstance(obj, dict):
            continue
        name = str(obj.get("Model") or obj.get("name") or "")
        if not name or not by_name.get(name):
            continue
        source = by_name[name].pop(0)
        constraints = (
            obj.get("constraints")
            if isinstance(obj.get("constraints"), list)
            else []
        )

        sp = source.get("Pose") or {}
        sx, sy = float(sp.get("x", 0.0)), float(sp.get("y", 0.0))

        for c in constraints:
            if not isinstance(c, dict):
                continue
            ctype = str(c.get("type", "")).lower()

            if ctype == "region":
                pref = str(c.get("value", "")).lower()
                dc = math.sqrt(sx * sx + sy * sy)
                if pref == "middle" and dc > room_half_size * 0.45:
                    violations.append(f"{name}: region=middle violated (d={dc:.2f})")
                if pref == "edge" and dc < room_half_size * 0.60:
                    violations.append(f"{name}: region=edge violated (d={dc:.2f})")
                continue

            target_name = str(c.get("target", ""))
            if not target_name:
                continue
            target = _find_target(source, target_name, placed_models)
            if not target:
                continue
            tp = target.get("Pose") or {}
            tx, ty = float(tp.get("x", 0.0)), float(tp.get("y", 0.0))
            d = math.sqrt((sx - tx) ** 2 + (sy - ty) ** 2)

            if ctype == "near":
                lo, hi = c.get("distance", [0.4, 1.2])
                lo_f, hi_f = float(lo), float(hi)
                if not (lo_f <= d <= hi_f):
                    violations.append(f"{name}: near {target_name} violated (d={d:.2f})")
            elif ctype == "far":
                lo, hi = c.get("distance", [2.0, 8.0])
                lo_f, hi_f = float(lo), float(hi)
                if not (lo_f <= d <= hi_f):
                    violations.append(f"{name}: far {target_name} violated (d={d:.2f})")
            elif ctype == "left_of" and not (sx < tx - 0.1):
                violations.append(f"{name}: left_of {target_name} violated")
            elif ctype == "right_of" and not (sx > tx + 0.1):
                violations.append(f"{name}: right_of {target_name} violated")
            elif ctype == "in_front_of" and not (sy > ty + 0.1):
                violations.append(f"{name}: in_front_of {target_name} violated")
            elif ctype == "behind" and not (sy < ty - 0.1):
                violations.append(f"{name}: behind {target_name} violated")
            elif ctype in {"on", "on_top_of", "on-top-of", "on top of", "on_top"}:
                spz = float(sp.get("z", 0.0))
                tpz = float(tp.get("z", 0.0))
                ssx, ssy, ssz = _size_xyz(source)
                tsx, tsy, tsz = _size_xyz(target)
                exp_z = tpz + tsz / 2.0 + ssz / 2.0 + 0.01
                xy_tol = max(0.12, 0.35 * max(tsx, tsy))
                z_tol = max(0.08, 0.25 * ssz)
                sat_xy = abs(sx - tx) <= xy_tol and abs(sy - ty) <= xy_tol
                sat_z = abs(spz - exp_z) <= z_tol
                if not (sat_xy and sat_z):
                    violations.append(
                        f"{name}: on {target_name} violated "
                        f"(dx={abs(sx - tx):.2f}, dy={abs(sy - ty):.2f}, "
                        f"dz={abs(spz - exp_z):.2f})"
                    )

    return violations


def repair_layout_by_constraints(
    placed_models: Sequence[Dict[str, Any]],
    semantic_plan: Dict[str, Any],
    *,
    room_half_size: float = 5.0,
    iterations: int = 2,
) -> List[Dict[str, Any]]:
    repaired = [dict(m) for m in placed_models]
    objects = semantic_plan.get("objects", []) if isinstance(semantic_plan, dict) else []
    if not isinstance(objects, list) or not objects:
        return repaired

    for _ in range(max(1, iterations)):
        by_name: Dict[str, List[Dict[str, Any]]] = {}
        for m in repaired:
            name = str(m.get("Model") or m.get("name") or "")
            if not name:
                continue
            by_name.setdefault(name, []).append(m)

        for obj in objects:
            if not isinstance(obj, dict):
                continue
            name = str(obj.get("Model") or obj.get("name") or "")
            if not name or not by_name.get(name):
                continue
            source = by_name[name].pop(0)
            constraints = (
                obj.get("constraints")
                if isinstance(obj.get("constraints"), list)
                else []
            )

            # If object is explicitly attached to a support surface (e.g. book on table),
            # keep XY anchored and do not apply generic region drift on top of it.
            anchored_on_target = False
            for c in constraints:
                if not isinstance(c, dict):
                    continue
                ctype_on = str(c.get("type", "")).lower()
                target_on = str(c.get("target", "")).strip()
                on_types = {"on", "on_top_of", "on-top-of", "on top of", "on_top"}
                if ctype_on in on_types and target_on:
                    anchored_on_target = True
                    break

            pose = dict(source.get("Pose") or {"x": 0.0, "y": 0.0, "z": 0.5})
            sx, sy = float(pose.get("x", 0.0)), float(pose.get("y", 0.0))
            sz = float(pose.get("z", 0.5))

            for c in constraints:
                if not isinstance(c, dict):
                    continue
                ctype = str(c.get("type", "")).lower()

                if ctype == "region":
                    if anchored_on_target:
                        continue
                    pref = str(c.get("value", "")).lower()
                    if pref == "middle":
                        sx *= 0.7
                        sy *= 0.7
                    elif pref == "edge":
                        if abs(sx) < room_half_size * 0.55:
                            sx = math.copysign(room_half_size * 0.7, sx if sx != 0 else 1.0)
                        if abs(sy) < room_half_size * 0.55:
                            sy = math.copysign(room_half_size * 0.7, sy if sy != 0 else -1.0)
                    continue

                target_name = str(c.get("target", ""))
                if not target_name:
                    continue
                target = _find_target(source, target_name, repaired)
                if not target:
                    continue
                tp = target.get("Pose") or {}
                tx, ty = float(tp.get("x", 0.0)), float(tp.get("y", 0.0))
                dx, dy = sx - tx, sy - ty
                d = math.sqrt(dx * dx + dy * dy)

                if ctype == "near":
                    lo, hi = c.get("distance", [0.4, 1.2])
                    lo_f, hi_f = float(lo), float(hi)
                    desired = (lo_f + hi_f) * 0.5
                    if d < 1e-6:
                        dx, dy, d = 1.0, 0.0, 1.0
                    scale = desired / d
                    sx = tx + dx * scale
                    sy = ty + dy * scale
                elif ctype == "far":
                    lo, _ = c.get("distance", [2.0, 8.0])
                    lo_f = float(lo)
                    if d < lo_f and d > 1e-6:
                        scale = (lo_f * 1.05) / d
                        sx = tx + dx * scale
                        sy = ty + dy * scale
                elif ctype == "left_of":
                    sx = min(sx, tx - 0.35)
                elif ctype == "right_of":
                    sx = max(sx, tx + 0.35)
                elif ctype == "in_front_of":
                    sy = max(sy, ty + 0.35)
                elif ctype == "behind":
                    sy = min(sy, ty - 0.35)
                elif ctype in {"on", "on_top_of", "on-top-of", "on top of", "on_top"}:
                    _, _, ssz = _size_xyz(source)
                    _, _, tsz = _size_xyz(target)
                    sx = tx
                    sy = ty
                    sz = float(tp.get("z", 0.0)) + tsz / 2.0 + ssz / 2.0 + 0.01

            pose["x"] = _clip(sx, -room_half_size + 0.2, room_half_size - 0.2)
            pose["y"] = _clip(sy, -room_half_size + 0.2, room_half_size - 0.2)
            pose["z"] = max(0.01, sz)
            source["Pose"] = pose

        repaired = validate_and_repair_layout(repaired)

    # Final hard anchoring pass: keep "on" relations exact even if generic
    # overlap/floor repair nudged objects during the physics pass.
    by_name_final: Dict[str, List[Dict[str, Any]]] = {}
    for m in repaired:
        n = str(m.get("Model") or m.get("name") or "")
        if not n:
            continue
        by_name_final.setdefault(n, []).append(m)

    for obj in objects:
        if not isinstance(obj, dict):
            continue
        name = str(obj.get("Model") or obj.get("name") or "")
        if not name or not by_name_final.get(name):
            continue
        source = by_name_final[name].pop(0)
        constraints = (
            obj.get("constraints")
            if isinstance(obj.get("constraints"), list)
            else []
        )
        for c in constraints:
            if not isinstance(c, dict):
                continue
            ctype = str(c.get("type", "")).lower()
            if ctype not in {"on", "on_top_of", "on-top-of", "on top of", "on_top"}:
                continue
            target_name = str(c.get("target", "")).strip()
            if not target_name:
                continue
            target = _find_target(source, target_name, repaired)
            if not target:
                continue

            source_pose = dict(source.get("Pose") or {"x": 0.0, "y": 0.0, "z": 0.01})
            target_pose = target.get("Pose") or {"x": 0.0, "y": 0.0, "z": 0.0}
            _, _, ssz = _size_xyz(source)
            _, _, tsz = _size_xyz(target)
            source_pose["x"] = float(target_pose.get("x", 0.0))
            source_pose["y"] = float(target_pose.get("y", 0.0))
            source_pose["z"] = float(target_pose.get("z", 0.0)) + tsz / 2.0 + ssz / 2.0 + 0.01
            source["Pose"] = source_pose
            break

    return repaired
