import os
import sys

# Environment configuration
os.environ.setdefault("MUJOCO_GL", "egl")
# Disable orientation detection: text-only LLM unreliably determines mesh "front"
# which can cause incorrect object placement on surfaces
os.environ.setdefault("WORLD_CREATOR_DISABLE_ORIENTATION", "1")

from project.catalog import load_catalog
from project.prompt_expander import expand_prompt
from project.model_picker import pick_models
from project.model_loader import load_and_scale_models
from project.physics_classifier import classify_physics
from project.room_scaler import compute_room_half_size
from project.scene_planner import generate_placement_plan
from project.mujoco_assembler import assemble_mujoco_scene

# Robot integration
from project.robot_picker import pick_robots
from creator.robots.catalog import load_robot_catalog
from creator.robots.placer import place_robots

# Get user query from command line or use default
if len(sys.argv) > 1:
    user_query = " ".join(sys.argv[1:])
else:
    user_query = "A robotic manipulation workspace with a rectangular wooden table placed in the center of a modern laboratory. Various everyday objects are scattered naturally across the tabletop: a red hardcover notebook, a blue ceramic mug, a yellow lemon, a green apple, a black computer mouse, a compact black remote control, a sleek black pen, a pair of black metal scissors, a cylindrical pen holder with multiple pens, and a wooden cutting board. A modern desk lamp illuminates the workspace from the left side, creating realistic shadows across the objects. In the background, a robotic manipulator arm is positioned near the table, ready to interact with the items. Clean robotics lab environment, realistic object arrangement for grasping and pick-and-place tasks, photorealistic materials, soft ambient lighting, highly detailed, cinematic rendering, computer vision dataset style."

# Load model catalog
catalog = load_catalog()

# Expand user prompt into scene specification
spec = expand_prompt(user_query)

# Stage 0.5: Robot Detection and Selection (NEW)
print("\n" + "="*60)
print("STAGE 0.5: ROBOT DETECTION")
print("="*60)
robot_catalog = load_robot_catalog()
detected_robots = pick_robots(spec.original_query, robot_catalog)

if detected_robots:
    print(f"✓ Detected {len(detected_robots)} robot(s):")
    for robot in detected_robots:
        robot_info = robot.get("robot_info", {})
        print(f"  - {robot.get('robot_id')}: {robot_info.get('name', 'unknown')}")
        print(f"    Type: {robot_info.get('type', 'unknown')}, Placement: {robot.get('placement_hint', 'unspecified')}")
    
    # Filter robots from estimated_objects to avoid duplicate processing
    robot_keywords = {'robot', 'робот', 'franka', 'panda', 'ur5', 'ur10', 'ur3',
                     'manipulator', 'манипулятор', 'роборука', 'arm', 'unitree',
                     'spot', 'quadruped', 'четвероногий', 'kinova', 'fetch',
                     'turtlebot', 'mobile', 'мобильный', 'robotic', 'роботический'}
    
    robot_mentions = {r.get('mentioned_as', '').lower() for r in detected_robots}
    robot_ids = {r.get('robot_id', '').lower() for r in detected_robots}
    robot_names = {r.get('robot_info', {}).get('name', '').lower() for r in detected_robots}
    
    def is_robot_object(obj_name):
        """Check if object name refers to a robot."""
        name_lower = obj_name.lower()
        # Check exact matches
        if name_lower in robot_mentions or name_lower in robot_ids or name_lower in robot_names:
            return True
        # Check if any robot keyword is in the name
        return any(kw in name_lower for kw in robot_keywords)
    
    original_count = len(spec.estimated_objects)
    spec.estimated_objects = [
        obj for obj in spec.estimated_objects
        if not is_robot_object(obj.name)
    ]
    filtered_count = original_count - len(spec.estimated_objects)
    if filtered_count > 0:
        print(f"✓ Filtered {filtered_count} robot object(s) from scene graph (will be placed separately)")
    
    # Update expanded_description to note robots are handled separately
    robot_names_str = ", ".join([r.get('robot_info', {}).get('name', r.get('robot_id', 'robot')) for r in detected_robots])
    spec.expanded_description += f" Note: {robot_names_str} will be placed separately via robot placement system."
else:
    print("✓ No robots detected in query")

# Stage 1-3: Object selection and room sizing
models = pick_models(spec, catalog)
models = load_and_scale_models(models, scene_desc=spec.expanded_description)
models = classify_physics(models, spec, spec.expanded_description)
room_half = compute_room_half_size(models)

# Stage 4: Object placement
placement = generate_placement_plan(
    models, room_half, spec.expanded_description, save_outputs=True
)

# Stage 4.5: Robot Placement (NEW)
robot_placements = []
if detected_robots:
    print("\n" + "="*60)
    print("STAGE 4.5: ROBOT PLACEMENT")
    print("="*60)
    robot_ids = [r["robot_id"] for r in detected_robots]
    robot_placements = place_robots(robot_ids, placement, room_half)
    print(f"✓ Placed {len(robot_placements)} robot(s)")
    for rp in robot_placements:
        pos = rp.get("pos", [0, 0, 0])
        print(f"  - {rp.get('robot_id')}: pos=({pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f}), yaw={rp.get('yaw', 0):.0f}°")

# Stage 5: MuJoCo XML Assembly
world_path = assemble_mujoco_scene(
    placement,
    room_half,
    output_path=".cache/worlds/scene_latest.xml",
    cache_dir=".cache",
    robots=robot_placements  # Pass robots to assembler
)

# Stage 6: Preview Rendering (currently disabled)
preview_path = None

print("\n" + "="*60)
print("ИТОГОВЫЙ РЕЗУЛЬТАТ:")
print("="*60)
print(f"Комната: {room_half*2:.1f}m x {room_half*2:.1f}m")
print(f"Размещено объектов: {len(placement)}")
if robot_placements:
    print(f"Размещено роботов: {len(robot_placements)}")
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
if robot_placements:
    print("\nПозиции роботов:")
    for i, robot in enumerate(robot_placements, 1):
        pos = robot.get("pos", [0, 0, 0])
        print(
            f"  {i}. {robot.get('robot_id', 'unknown')}: "
            f"pos=({pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f}), "
            f"yaw={robot.get('yaw', 0):.0f}°"
        )
print("="*60)
print("\nJSON outputs saved to: project/output/")
print("Latest layout: project/output/stage4_final_layout_*.json")
