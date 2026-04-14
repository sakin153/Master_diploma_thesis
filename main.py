from creator.runner import generate_world

# --- Configure here ---
SIMULATOR = "mujoco"  # "gazebo" | "mujoco"
QUERY = "table and book on it"  # World generation query
CACHE_DIR = None  # e.g. "/var/tmp/ciare" or custom path


def main() -> int:
    world_path = generate_world(simulator=SIMULATOR, query=QUERY, cache_dir=CACHE_DIR)
    print(f"Generated world at: {world_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
