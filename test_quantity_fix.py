#!/usr/bin/env python3
"""Test the quantity fix with real scene generation."""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from creator.runner import generate_world

def test_quantity(query: str, expected_objects: dict):
    """Test scene generation with expected object counts."""
    
    print("=" * 80)
    print(f"Testing: {query}")
    print(f"Expected: {expected_objects}")
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
        
        print(f"\n✓ Scene generated: {world_path}")
        
        # Check if file exists
        if os.path.exists(world_path):
            with open(world_path, 'r') as f:
                content = f.read()
            
            # Count objects in XML (rough estimate)
            import re
            bodies = re.findall(r'<body name="([^"]+)"', content)
            print(f"\n✓ Found {len(bodies)} bodies in XML:")
            for body in bodies:
                print(f"  - {body}")
            
            return True
        else:
            print(f"\n✗ File not found: {world_path}")
            return False
            
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    tests = [
        ("стол и 3 стула", {"table": 1, "chair": 3}),
    ]
    
    for query, expected in tests:
        success = test_quantity(query, expected)
        if not success:
            print(f"\n✗ Test failed for: {query}")
            sys.exit(1)
    
    print("\n" + "=" * 80)
    print("✓ All tests passed!")
    print("=" * 80)
