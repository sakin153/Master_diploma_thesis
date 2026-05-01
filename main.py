from creator.runner import generate_world
import sys
import io
from contextlib import redirect_stdout, redirect_stderr


def main() -> int:
    # Get query from command line arguments
    if len(sys.argv) > 1:
        query = " ".join(sys.argv[1:])
    else:
        # Default query if no arguments provided
        query = "столовая"
    # стол с  двумя коробками и в коробках по 4 яблока в каждом
    # Огромный стол для ужина с  4 стульями вокруг и книга, ваза, ноутбук на столе
    import time
    # Per-run cache to dodge stale root-owned converted/ files left behind by
    # obj2mjcf on prior failures. Once those are cleaned (sudo rm -rf .cache_ciare
    # /var/tmp/ciare/converted), this can become a stable directory.
    cache_dir = f"/tmp/ciare_fresh_{int(time.time())}"

    # When None, EmbodiedGenLoader uses its DEFAULT_DATASET_DIR (the HuggingFace
    # snapshot under ~/.cache/huggingface/hub/). Override here for a custom path.
    assets_dir = None
    
    seed = 42

    # Capture logs
    log_buffer = io.StringIO()
    
    try:
        # Redirect stdout and stderr to capture logs
        with redirect_stdout(log_buffer), redirect_stderr(log_buffer):
            world_path = generate_world(
                query=query,
                cache_dir=cache_dir,
                assets_dir=assets_dir,
                vlm_validation=False,
                seed=seed,
            )
        
        # Get captured logs
        logs = log_buffer.getvalue()
        
        # Print to console
        print(logs)
        print(f"Generated world at: {world_path}")
        
        # Save scene for analysis with metadata and logs
        try:
            from save_scene_for_analysis import save_scene_for_analysis
            import re
            import os
            
            # Create safe filename from query
            safe_name = re.sub(r'[^\w\s-]', '', query).strip().replace(' ', '_')[:50]
            
            # Collect metadata
            metadata = {
                "query": query,
                "seed": seed,
                "cache_dir": cache_dir,
                "assets_dir": assets_dir,
                "vlm_validation": False,
                "world_path": world_path,
            }
            
            # Try to extract additional info from logs
            if "Room type:" in logs:
                import re
                match = re.search(r'Room type: (\w+)', logs)
                if match:
                    metadata["room_type"] = match.group(1)
            
            if "Final room:" in logs:
                match = re.search(r'Final room: (\d+)m', logs)
                if match:
                    metadata["room_size"] = f"{match.group(1)}m × {match.group(1)}m"
            
            if "Universal System placed" in logs:
                match = re.search(r'Universal System placed (\d+) objects', logs)
                if match:
                    metadata["objects_count"] = int(match.group(1))
            
            # Check file size
            if os.path.exists(world_path):
                metadata["file_size"] = os.path.getsize(world_path)
            
            saved_dir = save_scene_for_analysis(world_path, safe_name, metadata, logs)
            if saved_dir:
                print(f"\n✅ Scene saved for analysis: {saved_dir}")
        except Exception as e:
            print(f"⚠️  Could not save scene for analysis: {e}")
            import traceback
            traceback.print_exc()
    
    except Exception as e:
        # Get logs even on error
        logs = log_buffer.getvalue()
        print(logs)
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
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
