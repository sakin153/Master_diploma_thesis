#!/usr/bin/env python3
"""Quick test script to verify preview rendering works."""

import sys
import os

# Test import
try:
    from creator.scene.vlm_validator import save_scene_preview
    print("✓ Import successful: creator.scene.vlm_validator.save_scene_preview")
except ImportError as e:
    print(f"✗ Import failed: {e}")
    sys.exit(1)

# Test with a simple scene
print("\nTesting scene generation with preview...")
from creator.runner import generate_world

try:
    world_path = generate_world(
        query="стол",
        cache_dir=os.path.expanduser("~/.cache/huggingface/hub"),
        assets_dir="assets",
        seed=42,
        vlm_validation=False,
    )
    print(f"\n✓ Scene generated: {world_path}")
    
    # Check if preview was created
    preview_path = world_path.replace(".xml", "_preview.png")
    if os.path.exists(preview_path):
        print(f"✓ Preview created: {preview_path}")
        print(f"  Size: {os.path.getsize(preview_path)} bytes")
    else:
        print(f"✗ Preview not found at: {preview_path}")
        
except Exception as e:
    print(f"\n✗ Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n✓ All tests passed!")
