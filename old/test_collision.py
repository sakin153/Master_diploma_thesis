#!/usr/bin/env python3
import os
import sys

# Добавим project в путь
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

os.environ.setdefault("MUJOCO_GL", "egl")

from project.catalog import load_catalog
from project.prompt_expander import expand_prompt
from project.model_picker import pick_models
from project.model_loader import load_and_scale_models
from project.room_scaler import compute_room_half_size
from project.scene_planner import generate_placement_plan

print("=" * 60)
print("ТЕСТ COLLISION RESOLUTION")
print("=" * 60)

catalog = load_catalog()
spec = expand_prompt("Стол и 4 стула, на столе по 4 тарелки, на каждой тарелке по банану")
models = pick_models(spec, catalog)
models = load_and_scale_models(models)
room_half = compute_room_half_size(models, spec.room_type)

print("\nЗапуск generate_placement_plan с save_outputs=True...")
placement = generate_placement_plan(
    models, room_half, spec.room_type, spec.expanded_description, save_outputs=True
)

print("\n" + "=" * 60)
print("РЕЗУЛЬТАТ:")
print("=" * 60)
print(f"Размещено объектов: {len(placement)}")
print("\nПозиции тарелок и бананов:")
for obj in placement:
    model = obj.get("Model", "")
    if model in ["plate", "banana"]:
        pose = obj.get("Pose", {})
        print(
            f"  {model}: pos=({pose.get('x', 0):.2f}, {pose.get('y', 0):.2f}, "
            f"{pose.get('z', 0):.2f})"
        )
print("=" * 60)
