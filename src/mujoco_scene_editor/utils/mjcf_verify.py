from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco
import numpy as np


@dataclass(frozen=True)
class MJCFVerificationResult:
    ok: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    steps: int = 0
    min_contact_dist: float | None = None
    max_abs_qvel: float | None = None


def _collect_mjcf_assets(xml_text: str, *, mjcf_dir: Path) -> dict[str, bytes]:
    """Collect assets referenced by <mesh file="...">.

    MuJoCo's Python API can load from XML string if we provide an assets dict
    mapping the (relative) filename in MJCF to its bytes.
    """

    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError:
        return {}

    assets: dict[str, bytes] = {}
    for mesh in root.iter("mesh"):
        file_attr = mesh.get("file")
        if not file_attr:
            continue
        key = file_attr.strip()
        if not key or key in assets:
            continue

        path = (mjcf_dir / key).expanduser().resolve()
        try:
            if not path.exists() or not path.is_file():
                continue
            assets[key] = path.read_bytes()
        except OSError:
            continue

    return assets


def verify_mjcf_physics(
    xml_text: str,
    *,
    mjcf_dir: Path,
    steps: int = 200,
    max_abs_qvel_threshold: float = 80.0,
    max_allowed_penetration_m: float = 0.02,
) -> MJCFVerificationResult:
    """Best-effort physical sanity checks by simulating a few steps."""

    errors: list[str] = []
    warnings: list[str] = []

    mjcf_dir = Path(mjcf_dir)
    assets = _collect_mjcf_assets(xml_text, mjcf_dir=mjcf_dir)

    try:
        model = mujoco.MjModel.from_xml_string(xml_text, assets=assets)
    except Exception as e:
        return MJCFVerificationResult(ok=False, errors=[f"MuJoCo load failed: {e}"])

    data = mujoco.MjData(model)

    min_contact_dist: float | None = None
    max_abs_qvel: float | None = None

    def _update_metrics() -> None:
        nonlocal min_contact_dist, max_abs_qvel

        if data.ncon:
            dists = [float(data.contact[i].dist) for i in range(int(data.ncon))]
            step_min = min(dists) if dists else None
            if step_min is not None:
                min_contact_dist = step_min if min_contact_dist is None else min(min_contact_dist, step_min)

        if data.qvel.size:
            step_max = float(np.max(np.abs(data.qvel)))
            max_abs_qvel = step_max if max_abs_qvel is None else max(max_abs_qvel, step_max)

    # Initial check.
    _update_metrics()

    for _ in range(int(max(0, steps))):
        mujoco.mj_step(model, data)

        if not np.isfinite(data.qpos).all() or not np.isfinite(data.qvel).all():
            errors.append("NaN/Inf detected in state during simulation")
            break

        _update_metrics()

        if max_abs_qvel is not None and max_abs_qvel > max_abs_qvel_threshold:
            errors.append(
                f"Unstable simulation: max |qvel|={max_abs_qvel:.3g} exceeds threshold {max_abs_qvel_threshold:.3g}"
            )
            break

    if min_contact_dist is not None and min_contact_dist < -max_allowed_penetration_m:
        warnings.append(
            f"Deep penetration detected: min contact dist {min_contact_dist:.4f} m"
        )

    ok = len(errors) == 0

    return MJCFVerificationResult(
        ok=ok,
        errors=errors,
        warnings=warnings,
        steps=int(max(0, steps)),
        min_contact_dist=min_contact_dist,
        max_abs_qvel=max_abs_qvel,
    )
