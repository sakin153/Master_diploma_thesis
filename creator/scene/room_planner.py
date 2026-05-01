"""Intelligent room sizing from object footprints.

Given a list of placed models (with sizes), compute the appropriate
room half-size so all objects fit comfortably with circulation space.

Based on area-packing heuristic from LayoutGPT research:
  room_area = sum(obj_footprints) * circulation_multiplier
  room_half_size = sqrt(room_area) / 2
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence, Tuple


# ---------------------------------------------------------------------------
# Circulation multipliers per room type
# These account for walkways, doors, empty space between furniture.
# INCREASED: Much larger multipliers to give objects more space and avoid
# walls constraining scene formation.
# ---------------------------------------------------------------------------
_CIRCULATION: Dict[str, float] = {
    "bedroom":      8.0,   # was 3.0
    "office":       8.0,   # was 3.5
    "classroom":    8.0,   # was 3.0
    "kitchen":      10.0,  # was 4.0
    "living_room":  8.0,   # was 3.5
    "warehouse":    6.0,   # was 2.5
    "lab":          8.0,   # was 3.5
    "dining":       10.0,  # was 4.0
    "restaurant":   12.0,  # was 4.5
    "other":        10.0,  # was 4.0
}

# Minimum room half-size per type (metres)
_MIN_ROOM_HALF: Dict[str, float] = {
    "bedroom":      2.0,
    "office":       2.5,
    "classroom":    3.0,
    "kitchen":      1.8,
    "living_room":  2.5,
    "warehouse":    3.0,
    "lab":          2.5,
    "dining":       3.0,
    "restaurant":   4.0,
    "other":        3.0,
}

# Maximum sensible half-size (increased to allow much larger rooms)
_MAX_ROOM_HALF = 25.0  # was 12.0

# Room aspect ratios — rooms wider than this are unusual for indoor scenes
_PREFERRED_ASPECT = 1.4  # max width/depth ratio


def compute_room_half_size(
    placed_models: Sequence[Dict[str, Any]],
    room_type: str = "other",
    *,
    min_override: Optional[float] = None,
    verbose: bool = True,
) -> float:
    """Compute the room half-size in metres that fits all objects.

    Uses:
      1. Sum of floor footprints (size[0] × size[2], after Y-up rotation)
      2. Circulation multiplier for the room type
      3. Minimum per room type

    Returns a single half-size value (room is square).
    """
    total_footprint = 0.0
    for m in placed_models:
        size = m.get("size")
        if not size or len(size) < 3:
            total_footprint += 0.25  # 0.5×0.5 default
            continue
        # size[0]=width, size[2]=depth (after 90° rotation)
        w = max(0.05, float(size[0]))
        d = max(0.05, float(size[2]))
        total_footprint += w * d

    multiplier = _CIRCULATION.get(room_type, 3.0)
    room_area = total_footprint * multiplier
    half_from_objects = math.sqrt(room_area) / 2.0

    minimum = min_override if min_override is not None else _MIN_ROOM_HALF.get(room_type, 2.0)
    result = max(minimum, min(_MAX_ROOM_HALF, half_from_objects))

    # Round to nearest 0.5m for clean room sizes
    result = round(result * 2) / 2.0

    if verbose:
        print(
            f"[room_planner] {room_type}: "
            f"footprint={total_footprint:.2f}m², "
            f"×{multiplier} → area={room_area:.1f}m², "
            f"room={result*2:.0f}m×{result*2:.0f}m (half={result}m)"
        )
    return result


def room_half_from_spec(
    room_type: str,
    estimated_objects: Optional[List[Any]] = None,
) -> float:
    """Quick estimate before models are loaded, using ObjectHints.

    Used in Stage 0 (before actual mesh sizes are known).
    """
    if not estimated_objects:
        return _MIN_ROOM_HALF.get(room_type, 2.5)

    # Estimate footprints from typical sizes per category
    _TYPICAL_FOOTPRINT: Dict[str, Tuple[float, float]] = {
        "chair": (0.5, 0.5),
        "desk": (1.2, 0.6),
        "table": (1.2, 0.8),
        "sofa": (2.0, 0.9),
        "bed": (1.6, 2.0),
        "wardrobe": (1.2, 0.6),
        "cabinet": (0.9, 0.5),
        "shelf": (1.0, 0.3),
        "bookshelf": (1.0, 0.3),
        "monitor": (0.5, 0.2),
        "lamp": (0.3, 0.3),
        "default": (0.6, 0.6),
    }

    total = 0.0
    for hint in estimated_objects:
        name = (hint.name if hasattr(hint, "name") else str(hint)).lower()
        fp = _TYPICAL_FOOTPRINT.get("default")
        for key, val in _TYPICAL_FOOTPRINT.items():
            if key in name:
                fp = val
                break
        qty = hint.quantity if hasattr(hint, "quantity") else 1
        total += fp[0] * fp[1] * qty

    multiplier = _CIRCULATION.get(room_type, 3.0)
    room_area = total * multiplier
    half = max(
        _MIN_ROOM_HALF.get(room_type, 2.0),
        min(_MAX_ROOM_HALF, math.sqrt(room_area) / 2.0),
    )
    return round(half * 2) / 2.0


# ---------------------------------------------------------------------------
# Wall layout helpers
# ---------------------------------------------------------------------------

def get_wall_bodies(room_half_size: float) -> List[Dict[str, Any]]:
    """Return descriptions of the 4 room walls as body specs.

    Each wall is a box body that acts as parent for wall-mounted objects.
    Wall bodies have no joint → rigidly attached to worldbody.
    """
    rhs = room_half_size
    wall_h = 3.0
    wall_t = 0.05  # half-thickness
    return [
        {
            "name": "wall_north",
            "pos": (0.0, rhs, wall_h / 2),
            "size": (rhs, wall_t, wall_h / 2),
            "normal_dir": "south",   # faces into the room
            "euler": (0.0, 0.0, 0.0),
        },
        {
            "name": "wall_south",
            "pos": (0.0, -rhs, wall_h / 2),
            "size": (rhs, wall_t, wall_h / 2),
            "normal_dir": "north",
            "euler": (0.0, 0.0, 180.0),
        },
        {
            "name": "wall_east",
            "pos": (rhs, 0.0, wall_h / 2),
            "size": (wall_t, rhs, wall_h / 2),
            "normal_dir": "west",
            "euler": (0.0, 0.0, 270.0),
        },
        {
            "name": "wall_west",
            "pos": (-rhs, 0.0, wall_h / 2),
            "size": (wall_t, rhs, wall_h / 2),
            "normal_dir": "east",
            "euler": (0.0, 0.0, 90.0),
        },
    ]
