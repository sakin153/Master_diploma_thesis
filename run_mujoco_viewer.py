import os
import sys

try:
    from mujoco import viewer
    import mujoco
except ImportError:
    print(
        "MuJoCo is not installed in this environment. "
        "Activate your .venv and install mujoco."
    )
    sys.exit(1)

SCENE_PATH = "/var/tmp/ciare/worlds/scene_latest.xml"

if not os.path.exists(SCENE_PATH):
    print(f"File not found: {SCENE_PATH}")
    sys.exit(1)

try:
    m = mujoco.MjModel.from_xml_path(SCENE_PATH)
except Exception as e:
    print(f"Failed to load scene as MJCF/XML: {e}\n"
          f"If your file is SDF/URDF, convert it to MJCF first.")
    sys.exit(1)

print(f"Launching MuJoCo viewer for: {SCENE_PATH}")
viewer.launch(m)
