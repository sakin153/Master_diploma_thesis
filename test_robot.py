"""Quick test: load a robot XML directly in MuJoCo viewer."""
import sys
import os

try:
    import mujoco
    from mujoco import viewer
except ImportError:
    print("Run with: .venv/bin/python test_robot.py")
    sys.exit(1)

# Default: franka_fr3. Pass robot name as argv[1] (franka_fr3 / kuka_iiwa_14 / unitree_a1)
ROBOT = sys.argv[1] if len(sys.argv) > 1 else "franka_fr3"

XML_MAP = {
    "franka_fr3":   "robot_assets/franka_fr3/fr3.xml",
    "kuka_iiwa_14": "robot_assets/kuka_iiwa_14/iiwa14.xml",
    "unitree_a1":   "robot_assets/unitree_a1/a1.xml",
}

xml_path = XML_MAP.get(ROBOT, ROBOT)  # also accept a direct path
if not os.path.exists(xml_path):
    print(f"Not found: {xml_path}")
    sys.exit(1)

print(f"Loading: {xml_path}")
m = mujoco.MjModel.from_xml_path(xml_path)
print(f"OK — {m.nbody} bodies, {m.njnt} joints, {m.nu} actuators")
viewer.launch(m)
