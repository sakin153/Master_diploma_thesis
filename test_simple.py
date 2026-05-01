#!/usr/bin/env python3
"""Simple test script."""

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from creator.runner import generate_world

def main():
    query = "стол"
    print(f"Testing: {query}")
    print("=" * 80)
    
    try:
        world_path = generate_world(
            query=query,
            cache_dir=None,
            vlm_validation=False,
            max_vlm_iters=0,
            assets_dir=None,
            seed=42,
        )
        
        print("=" * 80)
        print(f"✅ SUCCESS: Scene generated at {world_path}")
        
        # Try to view the scene
        print("\nTrying to view scene...")
        os.system(f"python view_scene.py {world_path}")
        
        return 0
        
    except Exception as e:
        print("=" * 80)
        print(f"❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())
