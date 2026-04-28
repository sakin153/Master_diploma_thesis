import glob
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


def _find_latest_scene() -> str:
    """Pick the most recently generated scene_latest.xml.

    Searches the per-run /tmp/ciare_run_*/worlds/ caches first (current
    main.py default), falling back to the legacy stable caches.
    Pass a path as argv[1] to override.
    """
    if len(sys.argv) > 1:
        return sys.argv[1]

    candidates: list[str] = []
    for pattern in (
        "/tmp/ciare_*/worlds/scene_latest.xml",
        os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            ".cache_ciare", "worlds", "scene_latest.xml",
        ),
        "/var/tmp/ciare/worlds/scene_latest.xml",
    ):
        candidates.extend(glob.glob(pattern))

    candidates = [p for p in candidates if os.path.exists(p)]
    if not candidates:
        return ""
    candidates.sort(key=os.path.getmtime, reverse=True)
    return candidates[0]


SCENE_PATH = _find_latest_scene()
if not SCENE_PATH:
    print(
        "No scene_latest.xml found. Run `python main.py` first, "
        "or pass an explicit path: python run_mujoco_viewer.py <path>.xml"
    )
    sys.exit(1)

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
