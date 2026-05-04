# Вычисляет размер комнаты по площади объектов.

import math

_CIRCULATION = {
    "bedroom":     8.0,
    "office":      8.0,
    "classroom":   8.0,
    "kitchen":    10.0,
    "living_room": 8.0,
    "warehouse":   6.0,
    "lab":         8.0,
    "other":      10.0,
}

_MIN_ROOM_HALF = {
    "bedroom":     2.0,
    "office":      2.5,
    "classroom":   3.0,
    "kitchen":     1.8,
    "living_room": 2.5,
    "warehouse":   3.0,
    "lab":         2.5,
    "other":       3.0,
}

_MAX_ROOM_HALF = 25.0


def compute_room_half_size(models, room_type="other"):
    """Считает полуразмер комнаты (в метрах) по суммарной площади объектов.

    Принимает список моделей с полем 'size', возвращает число (half_size).
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

    multiplier = _CIRCULATION.get(room_type, 3.0)
    room_area = total_footprint * multiplier
    half = math.sqrt(room_area) / 2.0

    minimum = _MIN_ROOM_HALF.get(room_type, 2.0)
    result = max(minimum, min(_MAX_ROOM_HALF, half))
    result = round(result * 2) / 2.0

    print(
        f"[room_scaler] {room_type}: "
        f"площадь объектов={total_footprint:.2f}м², "
        f"×{multiplier} → {room_area:.1f}м², "
        f"комната={result*2:.0f}м×{result*2:.0f}м"
    )
    return result
