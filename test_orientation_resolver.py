#!/usr/bin/env python3
"""Quick test for orientation resolver."""

from creator.placement.semantic_enforcement.orientation_resolver import (
    OrientationResolver,
)


def test_basic_orientation():
    """Test basic relative orientation resolution."""
    print("Testing basic orientation resolution...")
    
    resolver = OrientationResolver()
    
    # Create a simple plan: table at origin, chair in front facing the table
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
                "position": {"absolute": {"x": 0.0, "y": 1.0, "z": 0.45}},  # In front of table
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
    
    resolved = resolver.resolve_orientations(plan)
    
    # Check that chair has absolute orientation now
    chair = resolved["objects"][1]
    assert chair["orientation"]["absolute"] is not None, "Chair should have absolute orientation"
    
    orient = chair["orientation"]["absolute"]
    print(f"✓ Chair orientation resolved to: yaw={orient['yaw_deg']:.2f}°")
    
    # Chair is at (0, 1) facing table at (0, 0)
    # Should face towards -Y direction (180°)
    expected_yaw = 180.0
    tolerance = 5.0
    assert abs(orient["yaw_deg"] - expected_yaw) < tolerance, \
        f"Chair should face table (expected ~{expected_yaw}°, got {orient['yaw_deg']:.2f}°)"
    
    print("✅ Basic orientation test passed!\n")


def test_facing_directions():
    """Test all facing directions."""
    print("Testing all facing directions...")
    
    resolver = OrientationResolver()
    
    facing_directions = ["front", "back", "left_side", "right_side"]
    
    for facing_dir in facing_directions:
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
                    "id": f"chair_{facing_dir}",
                    "Model": "dining_chair",
                    "type": "furniture",
                    "size": {"width": 0.5, "length": 0.5, "height": 0.9},
                    "is_static": True,
                    "model_loc": "models/chairs/dining_chair_01.xml",
                    "position": {"absolute": {"x": 2.0, "y": 0.0, "z": 0.45}},  # To the right of table
                    "orientation": {
                        "absolute": None,
                        "relative": {
                            "facing": "table_1",
                            "facing_direction": facing_dir,
                            "facing_away": False
                        }
                    },
                }
            ]
        }
        
        resolved = resolver.resolve_orientations(plan)
        chair = resolved["objects"][1]
        orient = chair["orientation"]["absolute"]
        print(f"✓ Facing direction '{facing_dir}': yaw={orient['yaw_deg']:.2f}°")
    
    print("✅ All facing directions test passed!\n")


def test_facing_away():
    """Test facing_away modifier."""
    print("Testing facing_away modifier...")
    
    resolver = OrientationResolver()
    
    # Test with facing_away=False
    plan_towards = {
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
                "position": {"absolute": {"x": 1.0, "y": 0.0, "z": 0.45}},
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
    
    resolved_towards = resolver.resolve_orientations(plan_towards)
    yaw_towards = resolved_towards["objects"][1]["orientation"]["absolute"]["yaw_deg"]
    
    # Test with facing_away=True
    plan_away = {
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
                "position": {"absolute": {"x": 1.0, "y": 0.0, "z": 0.45}},
                "orientation": {
                    "absolute": None,
                    "relative": {
                        "facing": "table_1",
                        "facing_direction": "front",
                        "facing_away": True
                    }
                },
            }
        ]
    }
    
    resolved_away = resolver.resolve_orientations(plan_away)
    yaw_away = resolved_away["objects"][1]["orientation"]["absolute"]["yaw_deg"]
    
    print(f"✓ Facing towards: yaw={yaw_towards:.2f}°")
    print(f"✓ Facing away: yaw={yaw_away:.2f}°")
    
    # The difference should be 180° (or close to it, accounting for normalization)
    diff = abs(yaw_away - yaw_towards)
    if diff > 180:
        diff = 360 - diff
    
    assert abs(diff - 180.0) < 5.0, \
        f"Facing away should be 180° different (got {diff:.2f}°)"
    
    print("✅ Facing away test passed!\n")


def test_angle_normalization():
    """Test angle normalization to [0, 360) range."""
    print("Testing angle normalization...")
    
    resolver = OrientationResolver()
    
    # Test various angles
    test_cases = [
        (0.0, 0.0),
        (90.0, 90.0),
        (180.0, 180.0),
        (270.0, 270.0),
        (360.0, 0.0),
        (450.0, 90.0),
        (-90.0, 270.0),
        (-180.0, 180.0),
    ]
    
    for input_angle, expected in test_cases:
        normalized = resolver._normalize_angle(input_angle)
        assert abs(normalized - expected) < 0.01, \
            f"Normalize {input_angle}° should be {expected}°, got {normalized:.2f}°"
        print(f"✓ {input_angle}° → {normalized:.2f}°")
    
    print("✅ Angle normalization test passed!\n")


def main():
    """Run all tests."""
    print("=" * 60)
    print("Testing Orientation Resolver")
    print("=" * 60 + "\n")
    
    try:
        test_basic_orientation()
        test_facing_directions()
        test_facing_away()
        test_angle_normalization()
        
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
