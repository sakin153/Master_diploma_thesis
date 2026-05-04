import os
os.environ.setdefault("MUJOCO_GL", "egl")

from project.catalog import load_catalog
from project.prompt_expander import expand_prompt
from project.model_picker import pick_models
from project.model_loader import load_and_scale_models
from project.room_scaler import compute_room_half_size
from project.scene_planner import generate_placement_plan
from project.mujoco_assembler import assemble_mujoco_scene
# from project.preview_renderer import render_scene_preview  # Временно отключено

catalog = load_catalog()

# spec = expand_prompt("""A simple kitchen table scene with 5 objects. 
# Three identical plain ceramic coffee mugs are arranged in a row. 
# One ripe yellow banana is placed to the left of the mugs. 
# One green apple is placed to the right of the mugs. 
# The lighting is bright and natural, highlighting the smooth surfaces of the ceramics and fruits.""")
# A close-up view of a desk surface with 5 office items. 

# Two red rectangular notebooks are stacked slightly offset from each other. 
# Two sleek black pens are lying parallel to each other next to the notebooks. 
# One sleek black ergonomic computer mouse is positioned to the right of the pens.
# 2 стола у каждого стола по 4 стуля. На каждом столе лежит две коробки, в каждой коробке по 3 яблока 
spec = expand_prompt("""Только стол и 4 стула. и 4 тарелки на столе напротив каждого стула и ваза""")
models = pick_models(spec, catalog)
models = load_and_scale_models(models)
room_half = compute_room_half_size(models, spec.room_type)
placement = generate_placement_plan(
    models, room_half, spec.room_type, spec.expanded_description, save_outputs=True
)

# Stage 5: MuJoCo XML Assembly
world_path = assemble_mujoco_scene(
    placement,
    room_half,
    output_path=".cache/worlds/scene_latest.xml",
    cache_dir=".cache"
)

# Stage 6.5: Preview Rendering (пропускаем, т.к. нет XML)
# preview_path = render_scene_preview(
#     world_path,
#     output_path=None,  # Auto-generate path
#     width=640,
#     height=480
# )
preview_path = None

print("\n" + "="*60)
print("ИТОГОВЫЙ РЕЗУЛЬТАТ:")
print("="*60)
print(f"Комната: {room_half*2:.1f}m x {room_half*2:.1f}m ({spec.room_type})")
print(f"Размещено объектов: {len(placement)}")
if world_path:
    print(f"MuJoCo сцена: {world_path}")
else:
    print("MuJoCo сцена: не создана (Stage 5 отключен)")
if preview_path:
    print(f"Превью: {preview_path}")
print("\nПозиции объектов:")
for i, obj in enumerate(placement, 1):
    pose = obj.get("Pose", {})
    print(
        f"  {i}. {obj.get('Model', 'unknown')}: "
        f"pos=({pose.get('x', 0):.2f}, {pose.get('y', 0):.2f}, "
        f"{pose.get('z', 0):.2f}), "
        f"yaw={obj.get('yaw_deg', 0):.0f}°"
    )
print("="*60)
print("\nJSON выводы сохранены в: project/output/")
print("Последний layout: project/output/stage4_final_layout_*.json")

# Опционально: запустить viewer
# Раскомментируй эти строки, чтобы автоматически открыть viewer после генерации
# import mujoco
# import mujoco.viewer
# print("\nЗапуск MuJoCo viewer...")
# model = mujoco.MjModel.from_xml_path(world_path)
# data = mujoco.MjData(model)
# mujoco.viewer.launch(model, data)
