#!/usr/bin/env python3
"""Simple test for orientation resolver."""

import sys
import math

# Add the project root to the path
sys.path.insert(0, '.')

from creator.placement.semantic_enforcement.orientation_resolver import OrientationResolver

def test_basic():
    """Test basic orientation resolution."""
    print("Testing basic orientation resolution...")
    
    resolver = OrientationResolver()
    
    # Create a simple plan: table at origin facing +Y (yaw=0), chair at (0, 1) should face back towards table
    plan = {
        "schema_version": "1.0",
        "room_size": {"width": 6.0, "length": 6.0, "height": 3.0},
        "objects": [
            {
                "id": "table_1",
                "Model": "dining_table",
                "type": "furniture",
                "size": {"width": 1.5, "length": 0.8, "height": 0.75},
                "is_static": True,
                "model_loc": "models/tables/dining_table_01.xml",
                "position": {"absolute": {"x": 0.0, "y": 0.0, "z": 0.375}},
                "orientation": {"absolute": {"yaw_deg": 0.0, "pitch_deg": 0.0, "roll_deg": 0.0}},
            },
            {
                "id": "chair_1",
                "Model": "dining_chair",
                "type": "furniture",
                "size": {"width": 0.5, "length": 0.5, "height": 0.9},
                "is_static": True,
                "model_loc": "models/chairs/dining_chair_01.xml",
                "position": {"absolute": {"x": 0.0, "y": 1.0, "z": 0.45}},
                "orientation": {
                    "absolute": None,
                    "relative": {
                        "facing": "table_1",
                        "facing_direction": "front",
                        "facing_away": False
                    }
                },
            }
        ]
    }
    
    try:
        resolved = resolver.resolve_orientations(plan)
        chair = resolved["objects"][1]
        orient = chair["orientation"]["absolute"]
        print(f"Chair orientation: yaw={orient['yaw_deg']:.2f}°")
        print("✅ Test passed!")
        return True
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_basic()
    sys.exit(0 if success else 1)
