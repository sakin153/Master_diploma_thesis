from creator.runner import generate_world
import sys


def main() -> int:
    # Get query from command line arguments
    if len(sys.argv) > 1:
        query = " ".join(sys.argv[1:])
    else:
        # Default query if no arguments provided
        query = "стол с  4 стульями вокруг и книга, ваза, ноутбук на столе"
    #стол с  двумя коробками и в коробках по 4 яблока в каждом
    # Огромный стол для ужина с  4 стульями вокруг и книга, ваза, ноутбук на столе
    import time
    # Per-run cache to dodge stale root-owned converted/ files left behind by
    # obj2mjcf on prior failures. Once those are cleaned (sudo rm -rf .cache_ciare
    # /var/tmp/ciare/converted), this can become a stable directory.
    cache_dir = f"/tmp/ciare_fresh_{int(time.time())}"

    # When None, EmbodiedGenLoader uses its DEFAULT_DATASET_DIR (the HuggingFace
    # snapshot under ~/.cache/huggingface/hub/). Override here for a custom path.
    assets_dir = None

    world_path = generate_world(
        query=query,
        cache_dir=cache_dir,
        assets_dir=assets_dir,
        vlm_validation=False,
        seed=42,
    )
    print(f"Generated world at: {world_path}")
    
    # Save preview renders for analysis
    try:
        from creator.scene.vlm_validator import save_scene_preview
        import os
        base = os.path.splitext(world_path)[0]
        
        # Top-down view
        save_scene_preview(world_path, f"{base}_top.png", 
                          camera_distance=3.5, camera_azimuth=0, camera_elevation=-89)
        # Perspective view
        save_scene_preview(world_path, f"{base}_perspective.png",
                          camera_distance=3.0, camera_azimuth=45, camera_elevation=-25)
        # Side view
        save_scene_preview(world_path, f"{base}_side.png",
                          camera_distance=3.0, camera_azimuth=90, camera_elevation=-20)
        print(f"Saved renders: {base}_{{top,perspective,side}}.png")
    except Exception as e:
        print(f"Render failed: {e}")
    
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
