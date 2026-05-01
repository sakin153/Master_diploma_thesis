#!/usr/bin/env python3
"""Simple MuJoCo scene viewer."""
import sys
import mujoco
import mujoco.viewer

def main():
    if len(sys.argv) < 2:
        print("Usage: python view_scene.py <scene.xml>")
        print("Opening latest scene from /tmp/ciare_fresh_*/worlds/scene_latest.xml")
        import glob
        import os
        scenes = glob.glob("/tmp/ciare_fresh_*/worlds/scene_latest.xml")
        if not scenes:
            print("No scenes found!")
            return 1
        # Sort by modification time (newest first)
        scenes.sort(key=lambda x: os.path.getmtime(x), reverse=True)
        scene_path = scenes[0]  # Most recently modified
        print(f"Opening: {scene_path}")
    else:
        scene_path = sys.argv[1]
    
    try:
        model = mujoco.MjModel.from_xml_path(scene_path)
        data = mujoco.MjData(model)
        
        print(f"✓ Loaded scene: {scene_path}")
        print(f"  Objects: {model.nbody - 1}")  # -1 for world body
        print(f"  Geoms: {model.ngeom}")
        print("\nControls:")
        print("  Left mouse: Rotate camera")
        print("  Right mouse: Move camera")
        print("  Scroll: Zoom")
        print("  Space: Pause/Resume")
        print("  Backspace: Reset")
        print("  ESC: Exit")
        print("\nOpening viewer...")
        
        mujoco.viewer.launch(model, data)
        return 0
        
    except Exception as e:
        print(f"❌ Error loading scene: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
