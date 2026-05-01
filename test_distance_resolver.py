#!/usr/bin/env python3
"""Quick test for distance resolver."""

from creator.placement.semantic_enforcement.distance_resolver import (
    DistanceResolver,
    CircularDependencyError,
)


def test_basic_resolution():
    """Test basic relative position resolution."""
    print("Testing basic position resolution...")
    
    resolver = DistanceResolver()
    
    # Create a simple plan: table with absolute position, chair relative to table
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
                "position": {
                    "absolute": None,
                    "relative": {
                        "relative_to": "table_1",
                        "direction": "front",
                        "distance": 0.5,
                        "reference_point": "center"
                    }
                },
                "orientation": {"absolute": {"yaw_deg": 0.0, "pitch_deg": 0.0, "roll_deg": 0.0}},
            }
        ]
    }
    
    resolved = resolver.resolve_positions(plan)
    
    # Check that chair has absolute position now
    chair = resolved["objects"][1]
    assert chair["position"]["absolute"] is not None, "Chair should have absolute position"
    
    pos = chair["position"]["absolute"]
    print(f"✓ Chair position resolved to: x={pos['x']:.2f}, y={pos['y']:.2f}, z={pos['z']:.2f}")
    
    # Chair should be in front of table (positive Y direction)
    assert pos["y"] > 0, "Chair should be in front of table (positive Y)"
    assert abs(pos["x"]) < 0.01, "Chair should be centered on X axis"
    
    print("✅ Basic resolution test passed!\n")


def test_multi_level_dependencies():
    """Test multi-level dependencies (A depends on B, B depends on C)."""
    print("Testing multi-level dependencies...")
    
    resolver = DistanceResolver()
    
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
                "position": {
                    "absolute": None,
                    "relative": {
                        "relative_to": "table_1",
                        "direction": "front",
                        "distance": 0.5,
                        "reference_point": "center"
                    }
                },
                "orientation": {"absolute": {"yaw_deg": 0.0, "pitch_deg": 0.0, "roll_deg": 0.0}},
            },
            {
                "id": "lamp_1",
                "Model": "desk_lamp",
                "type": "small_object",
                "size": {"width": 0.2, "length": 0.2, "height": 0.3},
                "is_static": False,
                "model_loc": "models/lamps/desk_lamp_01.xml",
                "position": {
                    "absolute": None,
                    "relative": {
                        "relative_to": "chair_1",
                        "direction": "left",
                        "distance": 0.3,
                        "reference_point": "center"
                    }
                },
                "orientation": {"absolute": {"yaw_deg": 0.0, "pitch_deg": 0.0, "roll_deg": 0.0}},
            }
        ]
    }
    
    resolved = resolver.resolve_positions(plan)
    
    # Check all objects have absolute positions
    for obj in resolved["objects"]:
        assert obj["position"]["absolute"] is not None, f"{obj['id']} should have absolute position"
        pos = obj["position"]["absolute"]
        print(f"✓ {obj['id']}: x={pos['x']:.2f}, y={pos['y']:.2f}, z={pos['z']:.2f}")
    
    print("✅ Multi-level dependencies test passed!\n")


def test_circular_dependency_detection():
    """Test circular dependency detection."""
    print("Testing circular dependency detection...")
    
    resolver = DistanceResolver()
    
    # Create a plan with circular dependency: A depends on B, B depends on A
    plan = {
        "schema_version": "1.0",
        "room_size": {"width": 6.0, "length": 6.0, "height": 3.0},
        "objects": [
            {
                "id": "chair_1",
                "Model": "dining_chair",
                "type": "furniture",
                "size": {"width": 0.5, "length": 0.5, "height": 0.9},
                "is_static": True,
                "model_loc": "models/chairs/dining_chair_01.xml",
                "position": {
                    "absolute": None,
                    "relative": {
                        "relative_to": "chair_2",
                        "direction": "front",
                        "distance": 0.5,
                        "reference_point": "center"
                    }
                },
                "orientation": {"absolute": {"yaw_deg": 0.0, "pitch_deg": 0.0, "roll_deg": 0.0}},
            },
            {
                "id": "chair_2",
                "Model": "dining_chair",
                "type": "furniture",
                "size": {"width": 0.5, "length": 0.5, "height": 0.9},
                "is_static": True,
                "model_loc": "models/chairs/dining_chair_01.xml",
                "position": {
                    "absolute": None,
                    "relative": {
                        "relative_to": "chair_1",
                        "direction": "back",
                        "distance": 0.5,
                        "reference_point": "center"
                    }
                },
                "orientation": {"absolute": {"yaw_deg": 0.0, "pitch_deg": 0.0, "roll_deg": 0.0}},
            }
        ]
    }
    
    try:
        resolver.resolve_positions(plan)
        print("✗ Should have detected circular dependency")
        return False
    except CircularDependencyError as e:
        print(f"✓ Correctly detected circular dependency: {e}")
    
    print("✅ Circular dependency detection test passed!\n")
    return True


def test_all_directions():
    """Test all direction types."""
    print("Testing all direction types...")
    
    resolver = DistanceResolver()
    
    directions = [
        "front", "back", "left", "right",
        "front_left", "front_right", "back_left", "back_right"
    ]
    
    for direction in directions:
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
                    "position": {"absolute": {"x": 0.0, "y": 0.0, "z": 0.375}},
                    "orientation": {"absolute": {"yaw_deg": 0.0, "pitch_deg": 0.0, "roll_deg": 0.0}},
                },
                {
                    "id": f"chair_{direction}",
                    "Model": "dining_chair",
                    "type": "furniture",
                    "size": {"width": 0.5, "length": 0.5, "height": 0.9},
                    "is_static": True,
                    "model_loc": "models/chairs/dining_chair_01.xml",
                    "position": {
                        "absolute": None,
                        "relative": {
                            "relative_to": "table_1",
                            "direction": direction,
                            "distance": 1.0,
                            "reference_point": "center"
                        }
                    },
                    "orientation": {"absolute": {"yaw_deg": 0.0, "pitch_deg": 0.0, "roll_deg": 0.0}},
                }
            ]
        }
        
        resolved = resolver.resolve_positions(plan)
        chair = resolved["objects"][1]
        pos = chair["position"]["absolute"]
        print(f"✓ Direction '{direction}': x={pos['x']:.2f}, y={pos['y']:.2f}")
    
    print("✅ All directions test passed!\n")


def main():
    """Run all tests."""
    print("=" * 60)
    print("Testing Distance Resolver")
    print("=" * 60 + "\n")
    
    try:
        test_basic_resolution()
        test_multi_level_dependencies()
        test_circular_dependency_detection()
        test_all_directions()
        
        print("=" * 60)
        print("✅ ALL TESTS PASSED!")
        print("=" * 60)
        return 0
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
