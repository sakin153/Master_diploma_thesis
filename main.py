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
        query = """A simple kitchen table scene with 5 objects. 
Three identical plain ceramic coffee mugs are arranged in a row. 
One ripe yellow banana is placed to the left of the mugs. 
One green apple is placed to the right of the mugs. 
The lighting is bright and natural, highlighting the smooth surfaces of the ceramics and fruits."
    # стол с  двумя коробками и в коробках по 4 яблока в каждом
    # Огромный стол для ужина с  4 стульями вокруг и книга, ваза, ноутбук на столе"""
    import os
    
    # Use stable output directory instead of temporary
    cache_dir = os.path.join("output", "cache")
    
    # Create output directory if it doesn't exist
    os.makedirs(cache_dir, exist_ok=True)

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
    
    except Exception as e:
        # Get logs even on error
        logs = log_buffer.getvalue()
        print(logs)
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    # Save preview renders for analysis - DISABLED (vlm_validator removed)
    # try:
    #     from creator.scene.vlm_validator import save_scene_preview
    #     import os
    #     base = os.path.splitext(world_path)[0]
    #     
    #     # Top-down view
    #     save_scene_preview(world_path, f"{base}_top.png", 
    #                       camera_distance=3.5, camera_azimuth=0, camera_elevation=-89)
    #     # Perspective view
    #     save_scene_preview(world_path, f"{base}_perspective.png",
    #                       camera_distance=3.0, camera_azimuth=45, camera_elevation=-25)
    #     # Side view
    #     save_scene_preview(world_path, f"{base}_side.png",
    #                       camera_distance=3.0, camera_azimuth=90, camera_elevation=-20)
    #     print(f"Saved renders: {base}_{{top,perspective,side}}.png")
    # except Exception as e:
    #     print(f"Render failed: {e}")
    
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
