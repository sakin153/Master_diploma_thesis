from creator.runner import generate_world
from pathlib import Path

# --- Configure here ---
SIMULATOR = "mujoco"  # "gazebo" | "mujoco"
QUERY = "Большой стол, на нем 2 ящика по центру стола, а в ящике 10 яблок разного цвета."  # World generation query
import time
# Per-run cache to dodge stale root-owned converted/ files left behind by
# obj2mjcf on prior failures. Once those are cleaned (sudo rm -rf .cache_ciare
# /var/tmp/ciare/converted), this can become a stable directory.
CACHE_DIR = f"/tmp/ciare_run_{int(time.time())}"
ASSETS_DIR = str((Path(__file__).resolve().parent / "assets").resolve())


def main() -> int:
    world_path = generate_world(
        simulator=SIMULATOR,
        query=QUERY,
        cache_dir=CACHE_DIR,
        assets_dir=ASSETS_DIR,
        vlm_validation=True,  # Whether to use VLM-based validation of generated objects and relations.
    )
    print(f"Generated world at: {world_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
