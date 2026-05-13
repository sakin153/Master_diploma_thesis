"""Room Size Calculator
Computes room dimensions based on object footprints
"""

import math

# Fixed circulation multiplier (accounts for walking space)
_CIRCULATION_MULTIPLIER = 8.0

# Fixed minimum room half-size (meters)
_MIN_ROOM_HALF = 2.5

_MAX_ROOM_HALF = 25.0


def compute_room_half_size(models):
    """Compute room half-size based on total object footprint.
    
    Args:
        models: List of models with 'size' field
        
    Returns:
        float: Room half-size in meters
    """
    total_footprint = 0.0
    for m in models:
        size = m.get("size")
        if not size or len(size) < 3:
            total_footprint += 0.25
            continue
        w = max(0.05, float(size[0]))
        d = max(0.05, float(size[2]))
        total_footprint += w * d

    room_area = total_footprint * _CIRCULATION_MULTIPLIER
    half = math.sqrt(room_area) / 2.0

    result = max(_MIN_ROOM_HALF, min(_MAX_ROOM_HALF, half))
    result = round(result * 2) / 2.0

    print(
        f"[room_scaler] "
        f"площадь объектов={total_footprint:.2f}м², "
        f"×{_CIRCULATION_MULTIPLIER} → {room_area:.1f}м², "
        f"комната={result*2:.0f}м×{result*2:.0f}м"
    )
    return result
