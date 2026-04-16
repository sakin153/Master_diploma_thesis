"""Scene validation: geometry overlap + MuJoCo contact checks.

Two levels of validation:
1. Pre-assembly (trimesh): fast mesh-level overlap detection.
2. Post-assembly (mujoco): load the XML and check data.contact at t=0.

Usage in the pipeline:
    from creator.scene.validator import validate_placements, validate_mjcf

    # Before assembly — check geometric overlaps
    issues = validate_placements(placed_models)

    # After assembly — check MuJoCo contacts
    contact_report = validate_mjcf(xml_path)
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# 1. Geometry-level validation (pre-assembly)
# ---------------------------------------------------------------------------

def _model_aabb(m: Dict[str, Any]) -> Optional[Tuple[float, float, float, float, float, float]]:
    """Return (xmin, xmax, ymin, ymax, zmin, zmax) for a placed model in scene space.

    After euler="90 0 yaw" rotation (Y-up → Z-up):
      scene X = size[0] (mesh X)
      scene Y = size[2] (mesh Z = scene depth)
      scene Z = size[1] (mesh Y = scene height)
    """
    pose = m.get("Pose")
    size = m.get("size")
    if pose is None or size is None or len(size) < 3:
        return None
    x = float(pose.get("x", 0.0))
    y = float(pose.get("y", 0.0))
    z = float(pose.get("z", 0.0))
    hx = max(0.01, float(size[0])) / 2.0   # scene X half-extent
    hy = max(0.01, float(size[2])) / 2.0   # scene Y half-extent (mesh Z)
    hz = max(0.01, float(size[1])) / 2.0   # scene Z half-extent (mesh Y = height)
    return x - hx, x + hx, y - hy, y + hy, z - hz, z + hz


def _aabb_overlap(a: Tuple, b: Tuple, margin: float = 0.0) -> bool:
    ax0, ax1, ay0, ay1, az0, az1 = a
    bx0, bx1, by0, by1, bz0, bz1 = b
    return not (
        ax1 + margin < bx0 or bx1 + margin < ax0 or
        ay1 + margin < by0 or by1 + margin < ay0 or
        az1 + margin < bz0 or bz1 + margin < az0
    )


def validate_placements(
    placed_models: List[Dict[str, Any]],
    *,
    overlap_margin: float = -0.02,   # allow 2cm penetration (common from gradient pass)
    room_half_size: float = 5.0,
    verbose: bool = True,
) -> List[Dict[str, Any]]:
    """Check for overlapping objects and out-of-bounds placement.

    Returns a list of issue dicts:
      {"type": "overlap"|"oob", "obj_a": name, "obj_b": name|None,
       "depth": float, "message": str}
    """
    issues: List[Dict[str, Any]] = []
    aabbs = [(m, _model_aabb(m)) for m in placed_models]

    # Overlap check — O(n²)
    for i, (ma, aabb_a) in enumerate(aabbs):
        if aabb_a is None:
            continue
        name_a = str(ma.get("Model") or ma.get("name") or f"obj_{i}")

        # Out-of-bounds check
        pose = ma.get("Pose") or {}
        size = ma.get("size") or [0.1, 0.1, 0.1]
        x, y = float(pose.get("x", 0.0)), float(pose.get("y", 0.0))
        hx = float(size[0]) / 2.0
        hy = float(size[2]) / 2.0 if len(size) > 2 else float(size[1]) / 2.0
        if abs(x) + hx > room_half_size + 0.05 or abs(y) + hy > room_half_size + 0.05:
            issue = {
                "type": "oob",
                "obj_a": name_a,
                "obj_b": None,
                "depth": max(abs(x) + hx - room_half_size, abs(y) + hy - room_half_size),
                "message": f"{name_a} is out of room bounds (x={x:.2f}, y={y:.2f})",
            }
            issues.append(issue)
            if verbose:
                print(f"[validator] OOB: {issue['message']}")

        # Overlap with other objects
        for j, (mb, aabb_b) in enumerate(aabbs):
            if j <= i or aabb_b is None:
                continue
            name_b = str(mb.get("Model") or mb.get("name") or f"obj_{j}")

            # Skip "on top of" pairs — they intentionally share XY bounds
            pose_a_z = float((ma.get("Pose") or {}).get("z", 0.0))
            pose_b_z = float((mb.get("Pose") or {}).get("z", 0.0))
            # size[1] = mesh Y = scene height (Z-axis after euler="90 0 yaw")
            hz_a = float((ma.get("size") or [0, 0.1, 0])[1]) / 2.0
            hz_b = float((mb.get("size") or [0, 0.1, 0])[1]) / 2.0
            # If one object is clearly above the other (stacked), skip
            if abs(pose_a_z - pose_b_z) > hz_a + hz_b - 0.01:
                continue

            if _aabb_overlap(aabb_a, aabb_b, margin=overlap_margin):
                # Estimate penetration depth in XY
                ax0, ax1, ay0, ay1, _, _ = aabb_a
                bx0, bx1, by0, by1, _, _ = aabb_b
                dx = min(ax1, bx1) - max(ax0, bx0)
                dy = min(ay1, by1) - max(ay0, by0)
                depth = min(dx, dy)
                issue = {
                    "type": "overlap",
                    "obj_a": name_a,
                    "obj_b": name_b,
                    "depth": round(depth, 4),
                    "message": f"{name_a} ↔ {name_b}: overlap depth={depth:.3f}m",
                }
                issues.append(issue)
                if verbose:
                    print(f"[validator] Overlap: {issue['message']}")

    if verbose and not issues:
        print(f"[validator] Geometry OK — {len(placed_models)} objects, no issues.")

    return issues


# ---------------------------------------------------------------------------
# 2. MuJoCo-level validation (post-assembly)
# ---------------------------------------------------------------------------

def validate_mjcf(
    xml_path: str,
    *,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Load a compiled MuJoCo XML, run kinematics+collision (no integration),
    and report contacts present at t=0.

    Returns:
        {
            "ok": bool,
            "n_contacts": int,
            "contacts": [{"geom1": str, "geom2": str, "dist": float}, ...],
            "error": str | None,
        }
    """
    try:
        import mujoco  # type: ignore
    except ImportError:
        return {"ok": True, "n_contacts": 0, "contacts": [], "error": "mujoco not installed"}

    try:
        model = mujoco.MjModel.from_xml_path(xml_path)
        data = mujoco.MjData(model)
        mujoco.mj_kinematics(model, data)  # compute positions without integration
        mujoco.mj_collision(model, data)   # detect contacts without integration

        contacts = []
        for i in range(data.ncon):
            c = data.contact[i]
            g1 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, c.geom1) or str(c.geom1)
            g2 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, c.geom2) or str(c.geom2)
            contacts.append({
                "geom1": g1,
                "geom2": g2,
                "dist": round(float(c.dist), 5),
            })

        # Penetrating contacts = dist < 0
        penetrating = [c for c in contacts if c["dist"] < -0.005]

        if verbose:
            print(f"[validator] MuJoCo: {data.ncon} contacts at t=0, "
                  f"{len(penetrating)} penetrating.")
            for p in penetrating[:5]:
                print(f"  Penetration: {p['geom1']} ↔ {p['geom2']} dist={p['dist']:.4f}m")

        return {
            "ok": len(penetrating) == 0,
            "n_contacts": int(data.ncon),
            "contacts": contacts,
            "penetrating": penetrating,
            "error": None,
        }

    except Exception as exc:  # noqa: BLE001
        if verbose:
            print(f"[validator] MuJoCo validation error: {exc}")
        return {"ok": False, "n_contacts": 0, "contacts": [], "penetrating": [],
                "error": str(exc)}


# ---------------------------------------------------------------------------
# 3. Scale sanity check
# ---------------------------------------------------------------------------

_SCALE_SANITY: Tuple[Tuple[Tuple[str, ...], float, float], ...] = (
    # (keywords, min_height_m, max_height_m)
    (("chair", "armchair", "stool"), 0.35, 1.20),
    (("table", "desk"), 0.55, 1.00),
    (("sofa", "couch"), 0.60, 1.20),
    (("bed",), 0.30, 0.80),
    (("cabinet", "wardrobe"), 0.80, 2.40),
    (("cup", "mug"), 0.06, 0.25),
    (("bottle",), 0.10, 0.50),
    (("book",), 0.10, 0.45),
    (("lamp",), 0.20, 2.00),
)


def validate_scales(
    placed_models: List[Dict[str, Any]],
    *,
    verbose: bool = True,
) -> List[Dict[str, Any]]:
    """Warn if any model's height is implausibly small or large for its category."""
    issues = []
    for m in placed_models:
        name = str(m.get("Model") or m.get("name") or "")
        size = m.get("size")
        if size is None or len(size) < 3:
            continue
        # Height in scene = size[1] (mesh Y, becomes Z after 90° X rotation)
        height = float(size[1])
        lname = name.lower()
        for keywords, lo, hi in _SCALE_SANITY:
            if any(kw in lname for kw in keywords):
                if height < lo or height > hi:
                    issue = {
                        "type": "scale_warning",
                        "obj": name,
                        "height_m": round(height, 3),
                        "expected_range": [lo, hi],
                        "message": (
                            f"{name}: height={height:.2f}m outside expected "
                            f"[{lo:.2f}–{hi:.2f}]m"
                        ),
                    }
                    issues.append(issue)
                    if verbose:
                        print(f"[validator] Scale warning: {issue['message']}")
                break
    if verbose and not issues:
        print(f"[validator] Scale OK — all {len(placed_models)} objects within expected ranges.")
    return issues
