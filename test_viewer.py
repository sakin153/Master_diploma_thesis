#!/usr/bin/env python3
"""Test if scene file exists and can be loaded."""
import glob
import os

# Find latest scene
scenes = sorted(glob.glob("/tmp/ciare_fresh_*/worlds/scene_latest.xml"))
print(f"Found {len(scenes)} scene(s)")

if scenes:
    latest = scenes[-1]
    print(f"\nLatest scene: {latest}")
    print(f"File exists: {os.path.exists(latest)}")
    print(f"File size: {os.path.getsize(latest)} bytes")
    
    # Try to load with MuJoCo
    try:
        import mujoco
        model = mujoco.MjModel.from_xml_path(latest)
        print(f"\n✓ Scene loaded successfully!")
        print(f"  Bodies: {model.nbody}")
        print(f"  Geoms: {model.ngeom}")
        print(f"  Meshes: {model.nmesh}")
        
        # Try to launch viewer
        print("\nLaunching viewer...")
        import mujoco.viewer
        data = mujoco.MjData(model)
        mujoco.viewer.launch(model, data)
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
else:
    print("\n❌ No scenes found!")
    print("Run: python main.py \"стол и стул\"")
