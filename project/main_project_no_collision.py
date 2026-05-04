import os
os.environ.setdefault("MUJOCO_GL", "egl")

from project.catalog import load_catalog
from project.prompt_expander import expand_prompt
from project.model_picker import pick_models
from project.model_loader import load_and_scale_models
from project.room_scaler import compute_room_half_size
from project.mujoco_assembler import assemble_mujoco_scene
from project.preview_renderer import render_scene_preview

# Импортируем внутренние функции для обхода validate_and_repair_layout
from project.scene_planner import (
    _build_scene_graph,
    make_world_state,
    _place_all,
    _merge_to_output,
    _save_stage_output,
    _llm_request,
    DEFAULT_MODEL
)
import json

catalog = load_catalog()

spec = expand_prompt("4 стола по 4 стула у каждого")
models = pick_models(spec, catalog)
models = load_and_scale_models(models)
room_half = compute_room_half_size(models, spec.room_type)

# Копируем логику из generate_placement_plan, но БЕЗ validate_and_repair_layout
print(f"[scene_planner] Stage 4: {len(models)} objects, room={room_half*2:.1f}m, type={spec.room_type}")

model_id = DEFAULT_MODEL

def llm_fn(prompt, query_text):
    raw = _llm_request(system=prompt, user=query_text, model=model_id)
    if isinstance(raw, str):
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return raw
    return raw

# Фаза 1: Построение графа
graph = _build_scene_graph(models, room_half, spec.expanded_description, llm_fn)

_save_stage_output({
    "stage": "1_scene_graph",
    "query": spec.expanded_description,
    "room_half_size": room_half,
    "room_type": spec.room_type,
    "graph": graph,
}, "stage1_scene_graph_no_collision")

# Инъекция size_hint в узлы графа
for node in graph["nodes"]:
    for m in models:
        if m.get("Model") == node["model_name"]:
            node["size_hint"] = m.get("size", [1.0, 1.0, 1.0])
            break

# Инициализировать world_state с remaining
world_state = make_world_state(room_half)
type_counts = {}
for m in models:
    name = m.get("Model", "")
    type_counts[name] = type_counts.get(name, 0) + 1
world_state["remaining"] = [
    {"model_name": name, "count": cnt, "size": next(
        (m.get("size", [1.0, 1.0, 1.0]) for m in models if m.get("Model") == name), [1.0, 1.0, 1.0]
    )}
    for name, cnt in type_counts.items()
]

# Фаза 2: Иерархическое размещение
_place_all(graph, world_state, spec.expanded_description, llm_fn)
print(f"[scene_planner] Placed {len(world_state['placed'])} objects")

_save_stage_output({
    "stage": "2_hierarchical_placement",
    "world_state": world_state,
    "graph": graph,
}, "stage2_world_state_no_collision")

# Merge
result = _merge_to_output(world_state, models, graph)
print(f"[scene_planner] Stage 4 complete: {len(result)} models with placement")

_save_stage_output({
    "stage": "3_merged_output",
    "models": result,
    "room_half_size": room_half,
}, "stage3_merged_output_no_collision")

# ПРОПУСКАЕМ validate_and_repair_layout!
print("[scene_planner] SKIPPING validate_and_repair_layout for testing")

_save_stage_output({
    "stage": "4_final_layout_NO_COLLISION_FIX",
    "models": result,
    "room_half_size": room_half,
    "query": spec.expanded_description,
    "room_type": spec.room_type,
}, "stage4_final_layout_no_collision")

placement = result

# Stage 5: MuJoCo XML Assembly
world_path = assemble_mujoco_scene(
    placement,
    room_half,
    output_path=".cache/worlds/scene_no_collision.xml",
    cache_dir=".cache"
)

# Stage 6.5: Preview Rendering
preview_path = render_scene_preview(
    world_path,
    output_path="output/preview_no_collision.png",
    width=640,
    height=480
)

print("\n" + "="*60)
print("ИТОГОВЫЙ РЕЗУЛЬТАТ (БЕЗ КОРРЕКЦИИ КОЛЛИЗИЙ):")
print("="*60)
print(f"Комната: {room_half*2:.1f}m x {room_half*2:.1f}m ({spec.room_type})")
print(f"Размещено объектов: {len(placement)}")
print(f"MuJoCo сцена: {world_path}")
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
