#!/usr/bin/env python3
"""Detailed test for orientation resolver."""

import sys
import math

sys.path.insert(0, '.')

from creator.placement.semantic_enforcement.orientation_resolver import OrientationResolver

def test_scenario(name, table_pos, table_yaw, chair_pos, facing_dir, facing_away, expected_yaw):
    """Test a specific scenario."""
    print(f"\nTest: {name}")
    print(f"  Table at {table_pos}, yaw={table_yaw}°")
    print(f"  Chair at {chair_pos}, facing {facing_dir}, away={facing_away}")
    
    resolver = OrientationResolver()
    
    plan = {
        "schema_version": "1.0",
        "room_size": {"width": 10.0, "length": 10.0, "height": 3.0},
        "objects": [
            {
                "id": "table_1",
                "Model": "dining_table",
                "type": "furniture",
                "size": {"width": 1.5, "length": 0.8, "height": 0.75},
                "is_static": True,
                "model_loc": "models/tables/dining_table_01.xml",
                "position": {"absolute": {"x": table_pos[0], "y": table_pos[1], "z": 0.375}},
                "orientation": {"absolute": {"yaw_deg": table_yaw, "pitch_deg": 0.0, "roll_deg": 0.0}},
            },
            {
                "id": "chair_1",
                "Model": "dining_chair",
                "type": "furniture",
                "size": {"width": 0.5, "length": 0.5, "height": 0.9},
                "is_static": True,
                "model_loc": "models/chairs/dining_chair_01.xml",
                "position": {"absolute": {"x": chair_pos[0], "y": chair_pos[1], "z": 0.45}},
                "orientation": {
                    "absolute": None,
                    "relative": {
                        "facing": "table_1",
                        "facing_direction": facing_dir,
                        "facing_away": facing_away
                    }
                },
            }
        ]
    }
    
    try:
        resolved = resolver.resolve_orientations(plan)
        chair = resolved["objects"][1]
        orient = chair["orientation"]["absolute"]
        actual_yaw = orient['yaw_deg']
        
        print(f"  Result: yaw={actual_yaw:.2f}° (expected ~{expected_yaw}°)")
        
        # Check if within tolerance
        diff = abs(actual_yaw - expected_yaw)
        if diff > 180:
            diff = 360 - diff
        
        if diff < 10.0:
            print(f"  ✅ PASS (diff={diff:.2f}°)")
            return True
        else:
            print(f"  ❌ FAIL (diff={diff:.2f}°)")
            return False
            
    except Exception as e:
        print(f"  ❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    print("=" * 70)
    print("Detailed Orientation Resolver Tests")
    print("=" * 70)
    
    results = []
    
    # Test 1: Chair in front of table (at +Y), facing front of table
    # Table at origin facing +Y (yaw=0), chair at (0, 1)
    # Chair should face back towards table center = 180°
    results.append(test_scenario(
        "Chair in front, facing front",
        table_pos=(0, 0),
        table_yaw=0,
        chair_pos=(0, 1),
        facing_dir="front",
        facing_away=False,
        expected_yaw=180
    ))
    
    # Test 2: Chair to the right of table (at +X), facing left side
    # Table at origin facing +Y (yaw=0), chair at (1, 0)
    # Chair should face left towards table = 270°
    results.append(test_scenario(
        "Chair to right, facing left side",
        table_pos=(0, 0),
        table_yaw=0,
        chair_pos=(1, 0),
        facing_dir="left_side",
        facing_away=False,
        expected_yaw=270
    ))
    
    # Test 3: Chair behind table (at -Y), facing back
    # Table at origin facing +Y (yaw=0), chair at (0, -1)
    # Chair should face forward towards table = 0°
    results.append(test_scenario(
        "Chair behind, facing back",
        table_pos=(0, 0),
        table_yaw=0,
        chair_pos=(0, -1),
        facing_dir="back",
        facing_away=False,
        expected_yaw=0
    ))
    
    # Test 4: Chair to left of table (at -X), facing right side
    # Table at origin facing +Y (yaw=0), chair at (-1, 0)
    # Chair should face right towards table = 90°
    results.append(test_scenario(
        "Chair to left, facing right side",
        table_pos=(0, 0),
        table_yaw=0,
        chair_pos=(-1, 0),
        facing_dir="right_side",
        facing_away=False,
        expected_yaw=90
    ))
    
    # Test 5: Test facing_away modifier
    # Chair at (1, 0), facing front with facing_away=True
    # Should be opposite of facing front normally
    results.append(test_scenario(
        "Chair with facing_away=True",
        table_pos=(0, 0),
        table_yaw=0,
        chair_pos=(1, 0),
        facing_dir="front",
        facing_away=True,
        expected_yaw=90  # Should face away from the front
    ))
    
    # Test 6: Rotated table
    # Table at origin facing +X (yaw=90), chair at (1, 0)
    # Chair facing front of table (which is now +X direction)
    results.append(test_scenario(
        "Rotated table (yaw=90)",
        table_pos=(0, 0),
        table_yaw=90,
        chair_pos=(1, 0),
        facing_dir="front",
        facing_away=False,
        expected_yaw=180  # Should face back towards table
    ))
    
    print("\n" + "=" * 70)
    passed = sum(results)
    total = len(results)
    print(f"Results: {passed}/{total} tests passed")
    print("=" * 70)
    
    return 0 if passed == total else 1

if __name__ == "__main__":
    sys.exit(main())
