import copy
import json
import math
import os
import re
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

from creator.placement import (
    evaluate_constraint_violations,
    repair_layout_by_constraints,
    validate_and_repair_layout,
)

_ON_TYPES = {"on", "on_top_of", "on-top-of", "on top of", "on_top"}


def _env_enabled(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _size_xyz(model: Dict[str, Any]) -> Tuple[float, float, float]:
    size = model.get("size")
    if not isinstance(size, (list, tuple)) or len(size) < 3:
        return 0.4, 0.4, 0.4
    sx = max(0.05, min(5.0, float(size[0])))
    sy = max(0.05, min(5.0, float(size[1])))
    sz = max(0.05, min(5.0, float(size[2])))
    return sx, sy, sz


def _build_proxy_world_xml(placed_models: List[Dict[str, Any]]) -> str:
    root = ET.Element("mujoco", model="proxy_refine")
    ET.SubElement(root, "compiler", angle="degree")
    ET.SubElement(root, "option", timestep="0.01", gravity="0 0 -9.81", iterations="80")

    worldbody = ET.SubElement(root, "worldbody")
    ET.SubElement(
        worldbody,
        "geom",
        name="floor",
        type="plane",
        size="0 0 0.05",
        pos="0 0 0",
        friction="1.0 0.1 0.1",
    )

    for idx, model in enumerate(placed_models):
        sx, sy, sz = _size_xyz(model)
        pose = model.get("Pose") or {}
        px = float(pose.get("x", 0.0))
        py = float(pose.get("y", 0.0))
        pz = max(sz * 0.5, float(pose.get("z", sz * 0.5)))
        yaw_deg = float(model.get("yaw_deg", 0.0))

        body = ET.SubElement(
            worldbody,
            "body",
            name=f"proxy_{idx}",
            pos=f"{px} {py} {pz}",
            euler=f"0 0 {yaw_deg}",
        )
        ET.SubElement(body, "freejoint")
        ET.SubElement(
            body,
            "geom",
            type="box",
            size=f"{sx / 2.0} {sy / 2.0} {sz / 2.0}",
            density="220",
            friction="0.9 0.1 0.1",
        )

    return ET.tostring(root, encoding="unicode")


def _simulate_proxy_settle(
    placed_models: List[Dict[str, Any]],
    *,
    room_half_size: float,
    sim_steps: int,
) -> Tuple[List[Dict[str, Any]], bool, str]:
    try:
        import mujoco  # type: ignore
    except ImportError:
        return placed_models, False, "mujoco is not installed"

    xml_text = _build_proxy_world_xml(placed_models)
    try:
        proxy_model = mujoco.MjModel.from_xml_string(xml_text)
        proxy_data = mujoco.MjData(proxy_model)
    except Exception as exc:
        return placed_models, False, f"proxy model build failed ({exc})"

    for _ in range(max(1, sim_steps)):
        mujoco.mj_step(proxy_model, proxy_data)

    settled = [copy.deepcopy(model) for model in placed_models]
    for idx, model in enumerate(settled):
        body_name = f"proxy_{idx}"
        body_id = mujoco.mj_name2id(proxy_model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        if body_id < 0:
            continue

        xpos = proxy_data.xpos[body_id]
        xmat = proxy_data.xmat[body_id]
        sx, sy, sz = _size_xyz(model)
        pose = dict(model.get("Pose") or {})
        pose["x"] = max(
            -room_half_size + sx * 0.5,
            min(room_half_size - sx * 0.5, float(xpos[0])),
        )
        pose["y"] = max(
            -room_half_size + sy * 0.5,
            min(room_half_size - sy * 0.5, float(xpos[1])),
        )
        pose["z"] = max(sz * 0.5, float(xpos[2]))
        model["Pose"] = pose
        model["yaw_deg"] = math.degrees(math.atan2(float(xmat[3]), float(xmat[0])))

    return settled, True, ""


def _load_semantic_plan(project_root: Path) -> Dict[str, Any]:
    graph_path = project_root / "scene_graph_latest.json"
    if not graph_path.exists():
        return {"objects": []}
    try:
        with open(graph_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return {"objects": []}
    raw_plan = data.get("raw_plan") if isinstance(data, dict) else None
    if isinstance(raw_plan, dict):
        return raw_plan
    return {"objects": []}


def _violation_score(violations: Sequence[str]) -> Tuple[int, float]:
    magnitude = 0.0
    for text in violations:
        nums = re.findall(r"-?\d+(?:\.\d+)?", str(text))
        for raw in nums:
            try:
                magnitude += abs(float(raw))
            except ValueError:
                continue
    return len(violations), magnitude


def _dist_xy(a: Dict[str, Any], b: Dict[str, Any]) -> float:
    pa = a.get("Pose") or {}
    pb = b.get("Pose") or {}
    dx = float(pa.get("x", 0.0)) - float(pb.get("x", 0.0))
    dy = float(pa.get("y", 0.0)) - float(pb.get("y", 0.0))
    return math.sqrt(dx * dx + dy * dy)


def _find_target(
    source: Dict[str, Any],
    target_name: str,
    models: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    candidates = [m for m in models if str(m.get("Model") or m.get("name")) == target_name]
    if not candidates:
        return {}
    return min(candidates, key=lambda t: _dist_xy(source, t))


def _evaluate_hard_constraint_violations(
    placed_models: Sequence[Dict[str, Any]],
    semantic_plan: Dict[str, Any],
    *,
    room_half_size: float,
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
            if not isinstance(c, dict) or not bool(c.get("hard", False)):
                continue
            ctype = str(c.get("type", "")).lower()

            if ctype == "region":
                pref = str(c.get("value", "")).lower()
                dc = math.sqrt(sx * sx + sy * sy)
                if pref == "middle" and dc > room_half_size * 0.45:
                    violations.append(f"{name}: hard region=middle violated")
                elif pref == "edge" and dc < room_half_size * 0.60:
                    violations.append(f"{name}: hard region=edge violated")
                continue

            target_name = str(c.get("target", ""))
            if not target_name:
                continue
            target = _find_target(source, target_name, placed_models)
            if not target:
                violations.append(f"{name}: hard {ctype} target={target_name} missing")
                continue

            tp = target.get("Pose") or {}
            tx, ty = float(tp.get("x", 0.0)), float(tp.get("y", 0.0))
            d = math.sqrt((sx - tx) ** 2 + (sy - ty) ** 2)

            if ctype == "near":
                lo, hi = c.get("distance", [0.4, 1.2])
                if not (float(lo) <= d <= float(hi)):
                    violations.append(f"{name}: hard near {target_name} violated")
            elif ctype == "far":
                lo, hi = c.get("distance", [2.0, 8.0])
                if not (float(lo) <= d <= float(hi)):
                    violations.append(f"{name}: hard far {target_name} violated")
            elif ctype == "left_of" and not (sx < tx - 0.1):
                violations.append(f"{name}: hard left_of {target_name} violated")
            elif ctype == "right_of" and not (sx > tx + 0.1):
                violations.append(f"{name}: hard right_of {target_name} violated")
            elif ctype == "in_front_of" and not (sy > ty + 0.1):
                violations.append(f"{name}: hard in_front_of {target_name} violated")
            elif ctype == "behind" and not (sy < ty - 0.1):
                violations.append(f"{name}: hard behind {target_name} violated")
            elif ctype in _ON_TYPES:
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
                    violations.append(f"{name}: hard on {target_name} violated")

    return violations


def _candidate_score(
    violations: Sequence[str],
    hard_violations: Sequence[str],
) -> Tuple[int, int, float]:
    count, magnitude = _violation_score(violations)
    return len(hard_violations), count, magnitude


def _is_better(a: Tuple[int, int, float], b: Tuple[int, int, float]) -> bool:
    # Priority: fewer hard violations -> fewer total violations -> lower magnitude.
    if a[0] != b[0]:
        return a[0] < b[0]
    if a[1] != b[1]:
        return a[1] < b[1]
    return a[2] + 1e-6 < b[2]


def refine_scene_with_engine(
    *,
    interface: Any,
    world_path: str,
    placed_models: List[Dict[str, Any]],
    room_half_size: float = 5.0,
) -> List[Dict[str, Any]]:
    if not _env_enabled("CIARE_ENGINE_REFINEMENT", default=True):
        return placed_models

    try:
        sim_steps = int(os.getenv("CIARE_ENGINE_REFINEMENT_STEPS", "240"))
    except ValueError:
        sim_steps = 240
    sim_steps = max(40, min(2000, sim_steps))

    project_root = Path(__file__).resolve().parents[2]
    semantic_plan = _load_semantic_plan(project_root)

    before = evaluate_constraint_violations(
        placed_models,
        semantic_plan,
        room_half_size=room_half_size,
    )
    before_hard = _evaluate_hard_constraint_violations(
        placed_models,
        semantic_plan,
        room_half_size=room_half_size,
    )
    before_score = _candidate_score(before, before_hard)

    try:
        passes = int(os.getenv("CIARE_ENGINE_REFINEMENT_PASSES", "2"))
    except ValueError:
        passes = 2
    passes = max(1, min(6, passes))

    best_models = [copy.deepcopy(m) for m in placed_models]
    best_violations = list(before)
    best_hard_violations = list(before_hard)
    best_score = before_score
    search_seed = [copy.deepcopy(m) for m in placed_models]

    for pass_idx in range(passes):
        candidates: List[List[Dict[str, Any]]] = []

        # Always try deterministic graph-repair even if engine step is unavailable.
        for it in (2, 4):
            deterministic = repair_layout_by_constraints(
                search_seed,
                semantic_plan,
                room_half_size=room_half_size,
                iterations=it,
            )
            candidates.append(deterministic)

        settled, ok, reason = _simulate_proxy_settle(
            search_seed,
            room_half_size=room_half_size,
            sim_steps=sim_steps,
        )
        if ok:
            settled = validate_and_repair_layout(settled)
            candidates.append(settled)
            for it in (1, 2, 3):
                repaired = repair_layout_by_constraints(
                    settled,
                    semantic_plan,
                    room_half_size=room_half_size,
                    iterations=it,
                )
                candidates.append(repaired)
        elif pass_idx == 0:
            print(f"Engine refinement skipped: {reason}.")

        for candidate in candidates:
            v = evaluate_constraint_violations(
                candidate,
                semantic_plan,
                room_half_size=room_half_size,
            )
            hard_v = _evaluate_hard_constraint_violations(
                candidate,
                semantic_plan,
                room_half_size=room_half_size,
            )
            score = _candidate_score(v, hard_v)
            if _is_better(score, best_score):
                best_models = [copy.deepcopy(m) for m in candidate]
                best_violations = v
                best_hard_violations = hard_v
                best_score = score

        search_seed = [copy.deepcopy(m) for m in best_models]

    if not _is_better(best_score, before_score):
        print(
            "Engine refinement: rejected"
            f" (hard {len(before_hard)} -> {len(best_hard_violations)},"
            f" total {len(before)} -> {len(best_violations)})."
        )
        return placed_models

    try:
        objects = interface.load_objects(best_models)
        main_root = interface.create_main_root(best_models, objects)
        tree = interface.create_tree(main_root)
        interface.write_tree_to_file(tree, world_path)
        if hasattr(interface, "try_compile_in_mujoco"):
            interface.try_compile_in_mujoco(world_path)
    except Exception as exc:
        print(f"Engine refinement skipped: failed to rewrite world ({exc}).")
        return placed_models

    print(
        "Engine refinement: accepted"
        f" (hard {len(before_hard)} -> {len(best_hard_violations)},"
        f" total {len(before)} -> {len(best_violations)})."
    )
    return best_models
