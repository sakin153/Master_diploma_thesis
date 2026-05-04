# Stage 6.5 - Preview Rendering
# Renders MuJoCo scene to PNG preview image

import base64
import io
import os


def render_scene_preview(world_path, output_path=None, width=640, height=480):
    """Stage 6.5: Render MuJoCo scene to PNG preview.
    
    Args:
        world_path: Path to MuJoCo XML file
        output_path: Path to save PNG (default: world_path with _preview.png suffix)
        width: Image width in pixels (default: 640)
        height: Image height in pixels (default: 480)
    
    Returns:
        str: Path to saved PNG file, or None if rendering failed
    """
    print(f"[preview_renderer] Rendering scene preview")
    print(f"[preview_renderer] Input: {world_path}")
    print(f"[preview_renderer] Resolution: {width}x{height}")
    
    # Determine output path
    if output_path is None:
        output_path = os.path.splitext(world_path)[0] + "_preview.png"
    
    # Render scene
    b64_image = _render_mujoco_scene(
        world_path,
        width=width,
        height=height,
        camera_distance=8.0,
        camera_azimuth=45.0,
        camera_elevation=-15.0,
    )
    
    if b64_image is None:
        print(f"[preview_renderer] ✗ Rendering failed")
        return None
    
    # Save PNG
    try:
        raw_bytes = base64.b64decode(b64_image)
        with open(output_path, "wb") as f:
            f.write(raw_bytes)
        print(f"[preview_renderer] ✓ Preview saved: {output_path}")
        return output_path
    except Exception as e:
        print(f"[preview_renderer] ✗ Failed to save preview: {e}")
        return None


def _render_mujoco_scene(world_path, width, height, camera_distance, camera_azimuth, camera_elevation):
    """Render MuJoCo XML scene to base64-encoded PNG string.
    
    Returns None if rendering fails (mujoco not installed, bad XML, etc.).
    Camera is placed at isometric angle to see full room layout.
    """
    try:
        import mujoco
        import numpy as np
        from PIL import Image
        
        # Load model
        model = mujoco.MjModel.from_xml_path(world_path)
        data = mujoco.MjData(model)
        
        # Step physics to resolve initial penetrations
        for _ in range(100):
            mujoco.mj_step(model, data)
        
        # Create renderer
        renderer = mujoco.Renderer(model, height=height, width=width)
        
        # Setup camera
        cam = mujoco.MjvCamera()
        cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        cam.distance = camera_distance
        cam.azimuth = camera_azimuth
        cam.elevation = camera_elevation
        cam.lookat[:] = [0.0, 0.0, 0.5]  # Look at table height
        
        # Update scene
        renderer.update_scene(data, camera=cam)
        
        # Enable shadows for cleaner preview
        renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = True
        renderer.scene.flags[mujoco.mjtRndFlag.mjRND_REFLECTION] = False
        
        # Render to pixels
        pixels = renderer.render()  # uint8 H×W×3
        
        # Convert to PNG and encode as base64
        img = Image.fromarray(pixels.astype("uint8"))
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode("utf-8")
        
    except Exception as e:
        print(f"[preview_renderer] Render error: {e}")
        return None
