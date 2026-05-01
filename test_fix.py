#!/usr/bin/env python3
"""Test script to verify the model_loc fix."""

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from creator.runner import generate_world

def test_simple_scene():
    """Test generating a simple scene with table and chairs."""
    print("=" * 80)
    print("Testing: стол и 3 стула")
    print("=" * 80)
    
    try:
        world_path = generate_world(
            query="стол и 3 стула",
            cache_dir=None,
            vlm_validation=False,
            max_vlm_iters=0,
            assets_dir=None,
            seed=42,
        )
        
        print("\n" + "=" * 80)
        print(f"✓ SUCCESS: Scene generated at {world_path}")
        print("=" * 80)
        
        # Check if file exists
        if os.path.exists(world_path):
            file_size = os.path.getsize(world_path)
            print(f"✓ File exists: {file_size} bytes")
            
            # Read first few lines
            with open(world_path, 'r') as f:
                lines = f.readlines()[:10]
                print(f"✓ First 10 lines of XML:")
                for line in lines:
                    print(f"  {line.rstrip()}")
        else:
            print(f"✗ ERROR: File does not exist at {world_path}")
            return False
        
        return True
        
    except Exception as e:
        print("\n" + "=" * 80)
        print(f"✗ ERROR: {e}")
        print("=" * 80)
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_simple_scene()
    sys.exit(0 if success else 1)
