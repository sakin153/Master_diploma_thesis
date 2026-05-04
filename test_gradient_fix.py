#!/usr/bin/env python3
"""Тест для проверки, что gradient_resolve_overlaps не двигает объекты на поверхности."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from project.scene_planner import gradient_resolve_overlaps

# Создаем тестовые данные
models = [
    # Стол (level 0)
    {
        "Model": "table",
        "size": [1.0, 0.8, 1.4],
        "Pose": {"x": 0.0, "y": 0.0, "z": 0.4},
        "yaw_deg": 0.0
    },
    # Тарелка 1 (level 1, on_surface)
    {
        "Model": "plate",
        "size": [0.32, 0.02, 0.32],
        "Pose": {"x": 0.3, "y": 0.0, "z": 0.8},
        "yaw_deg": 0.0
    },
    # Тарелка 2 (level 1, on_surface)
    {
        "Model": "plate",
        "size": [0.32, 0.02, 0.32],
        "Pose": {"x": -0.3, "y": 0.0, "z": 0.8},
        "yaw_deg": 0.0
    },
    # Банан 1 (level 2, on_surface)
    {
        "Model": "banana",
        "size": [0.12, 0.06, 0.20],
        "Pose": {"x": 0.3, "y": 0.0, "z": 0.82},
        "yaw_deg": 0.0
    },
    # Банан 2 (level 2, on_surface)
    {
        "Model": "banana",
        "size": [0.12, 0.06, 0.20],
        "Pose": {"x": -0.3, "y": 0.0, "z": 0.82},
        "yaw_deg": 0.0
    },
]

# Scene graph с иерархией
scene_graph = {
    "nodes": [
        {
            "id": "table_1",
            "model_name": "table",
            "level": 0,
            "relationship": None
        },
        {
            "id": "plate_group",
            "model_name": "plate",
            "level": 1,
            "parent_id": "table_1",
            "relationship": {
                "type": "on_surface",
                "offset_x": 0,
                "offset_y": 0.4,
                "offset_z": 0
            }
        },
        {
            "id": "banana_group",
            "model_name": "banana",
            "level": 2,
            "parent_id": "plate_group",
            "relationship": {
                "type": "on_surface",
                "offset_x": 0,
                "offset_y": 0.02,
                "offset_z": 0
            }
        },
    ]
}

print("=" * 60)
print("ТЕСТ: gradient_resolve_overlaps с защитой surface objects")
print("=" * 60)

print("\nИсходные позиции:")
for m in models:
    pose = m["Pose"]
    print(f"  {m['Model']}: x={pose['x']:.2f}, y={pose['y']:.2f}, z={pose['z']:.2f}")

print("\nЗапуск gradient_resolve_overlaps...")
resolved, converged = gradient_resolve_overlaps(
    models,
    room_half_size=3.0,
    iterations=50,
    scene_graph=scene_graph
)

print(f"\nСошлось: {converged}")
print("\nФинальные позиции:")
for m in resolved:
    pose = m["Pose"]
    print(f"  {m['Model']}: x={pose['x']:.2f}, y={pose['y']:.2f}, z={pose['z']:.2f}")

print("\n" + "=" * 60)
print("ПРОВЕРКА:")
print("=" * 60)

# Проверяем, что тарелки и бананы не сдвинулись в XY
plates_ok = True
bananas_ok = True

for i, m in enumerate(resolved):
    if m["Model"] == "plate":
        original_x = models[i]["Pose"]["x"]
        final_x = m["Pose"]["x"]
        if abs(final_x - original_x) > 0.01:
            print(f"❌ Тарелка сдвинулась: {original_x:.2f} → {final_x:.2f}")
            plates_ok = False
        else:
            print(f"✅ Тарелка осталась на месте: x={final_x:.2f}")
    
    if m["Model"] == "banana":
        original_x = models[i]["Pose"]["x"]
        final_x = m["Pose"]["x"]
        if abs(final_x - original_x) > 0.01:
            print(f"❌ Банан сдвинулся: {original_x:.2f} → {final_x:.2f}")
            bananas_ok = False
        else:
            print(f"✅ Банан остался на месте: x={final_x:.2f}")

print("\n" + "=" * 60)
if plates_ok and bananas_ok:
    print("✅ ТЕСТ ПРОЙДЕН: Объекты на поверхности не двигались!")
else:
    print("❌ ТЕСТ ПРОВАЛЕН: Объекты на поверхности сдвинулись!")
print("=" * 60)
