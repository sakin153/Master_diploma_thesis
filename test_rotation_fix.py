"""
Тестовый скрипт для проверки исправления поворотов стульев.
Запускает только Stage 5 (MuJoCo Assembly) с последними данными.
"""

import os
os.environ.setdefault("MUJOCO_GL", "egl")

import json
from project.mujoco_assembler import assemble_mujoco_scene
from project.preview_renderer import render_scene_preview

# Загружаем последний stage4 (финальный layout)
with open("project/output/stage4_final_layout_20260503_183902.json", "r") as f:
    data = json.load(f)

models = data["models"]
room_half = data["room_half_size"]

print("="*60)
print("ТЕСТ ИСПРАВЛЕНИЯ ПОВОРОТОВ СТУЛЬЕВ")
print("="*60)
print(f"Загружено моделей: {len(models)}")
print(f"Комната: {room_half*2:.1f}m x {room_half*2:.1f}m")
print()

# Показываем углы стульев
print("Углы стульев (yaw_deg):")
for i, model in enumerate(models):
    if "chair" in model.get("Model", "").lower():
        pose = model.get("Pose", {})
        yaw = model.get("yaw_deg", 0)
        print(f"  Стул {i-3}: pos=({pose.get('x', 0):.1f}, {pose.get('y', 0):.1f}), yaw={yaw:.0f}°")

print()
print("Собираем MuJoCo сцену с ИСПРАВЛЕННЫМИ поворотами...")

# Stage 5: MuJoCo XML Assembly (с исправлением)
world_path = assemble_mujoco_scene(
    models,
    room_half,
    output_path=".cache/worlds/scene_rotation_fixed.xml",
    cache_dir=".cache"
)

print()
print(f"✓ MuJoCo сцена сохранена: {world_path}")

# Stage 6: Preview Rendering
print()
print("Рендерим превью...")
preview_path = render_scene_preview(
    world_path,
    output_path="output/preview_rotation_fixed.png",
    width=800,
    height=600
)

print()
print("="*60)
print("РЕЗУЛЬТАТ:")
print("="*60)
print(f"MuJoCo XML: {world_path}")
if preview_path:
    print(f"Превью: {preview_path}")
print()
print("Проверьте превью - стулья теперь должны смотреть на столы!")
print("="*60)
