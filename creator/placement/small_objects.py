from typing import Any, Dict, List, Optional, Sequence


def _volume(size: Sequence[float]) -> float:
    return float(size[0]) * float(size[1]) * float(size[2])


def _get_size(model: Dict[str, Any]) -> Sequence[float]:
    size = model.get("size")
    if size is None:
        return [1.0, 1.0, 1.0]
    return size


def _find_best_receptacle(
    item: Dict[str, Any],
    receptacles: Sequence[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    if not receptacles:
        return None
    # Place near the closest large object in XY.
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


def solve_small_object_placements(
    *,
    placed_models: Sequence[Dict[str, Any]],
    small_threshold_volume: float = 0.06,
) -> List[Dict[str, Any]]:
    out = [dict(m) for m in placed_models]

    receptacles = []
    for m in out:
        size = _get_size(m)
        if _volume(size) >= small_threshold_volume:
            receptacles.append(m)

    for i, m in enumerate(out):
        size = _get_size(m)
        if _volume(size) >= small_threshold_volume:
            continue

        receptacle = _find_best_receptacle(m, receptacles)
        if receptacle is None:
            continue

        rp = receptacle.get("Pose") or {"x": 0.0, "y": 0.0, "z": 0.0}
        rsize = _get_size(receptacle)

        pose = dict(m.get("Pose") or {"x": 0.0, "y": 0.0, "z": 0.0})
        pose["x"] = float(rp.get("x", 0.0))
        pose["y"] = float(rp.get("y", 0.0))
        z_base = float(rp.get("z", 0.0))
        z_receptacle = float(rsize[2]) / 2.0
        z_item = float(size[2]) / 2.0
        pose["z"] = z_base + z_receptacle + z_item + 0.01
        out[i]["Pose"] = pose

    return out
