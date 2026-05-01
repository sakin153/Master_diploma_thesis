#!/usr/bin/env python3
"""Quick test for semantic enforcement components."""

from creator.placement.semantic_enforcement.models import (
    Position,
    Orientation,
    SemanticObject,
    SemanticPlan,
)
from creator.placement.semantic_enforcement.schema_validator import SchemaValidator


def test_data_models():
    """Test that data models can be instantiated."""
    print("Testing data models...")
    
    # Test Position
    pos_abs = Position(absolute={"x": 1.0, "y": 2.0, "z": 0.0})
    print(f"✓ Absolute position: {pos_abs}")
    
    pos_rel = Position(relative={
        "relative_to": "table_1",
        "direction": "front",
        "distance": 0.5,
        "reference_point": "center"
    })
    print(f"✓ Relative position: {pos_rel}")
    
    # Test Orientation
    orient_abs = Orientation(absolute={"yaw_deg": 90.0, "pitch_deg": 0.0, "roll_deg": 0.0})
    print(f"✓ Absolute orientation: {orient_abs}")
    
    orient_rel = Orientation(relative={
        "facing": "table_1",
        "facing_direction": "front",
        "facing_away": False
    })
    print(f"✓ Relative orientation: {orient_rel}")
    
    # Test SemanticObject
    obj = SemanticObject(
        id="chair_1",
        Model="dining_chair",
        type="furniture",
        size={"width": 0.5, "length": 0.5, "height": 0.9},
        is_static=True,
        model_loc="models/chairs/dining_chair_01.xml",
        position=pos_rel,
        orientation=orient_rel,
    )
    print(f"✓ Semantic object: {obj.id} ({obj.Model})")
    
    # Test SemanticPlan
    plan = SemanticPlan(
        schema_version="1.0",
        room_size={"width": 6.0, "length": 6.0, "height": 3.0},
        objects=[obj],
    )
    print(f"✓ Semantic plan with {len(plan.objects)} objects")
    
    print("✅ Data models test passed!\n")


def test_schema_validator():
    """Test schema validator with valid and invalid plans."""
    print("Testing schema validator...")
    
    validator = SchemaValidator()
    
    # Test 1: Valid plan
    valid_plan = {
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
                "position": {"absolute": {"x": 0.0, "y": 0.0, "z": 0.0}},
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
                    "relative": {
                        "relative_to": "table_1",
                        "direction": "front",
                        "distance": 0.5,
                        "reference_point": "center"
                    }
                },
                "orientation": {
                    "relative": {
                        "facing": "table_1",
                        "facing_direction": "front",
                        "facing_away": False
                    }
                },
            }
        ]
    }
    
    result = validator.validate_plan(valid_plan)
    if result.success:
        print("✓ Valid plan passed validation")
    else:
        print(f"✗ Valid plan failed: {result.errors}")
        return False
    
    # Test 2: Missing required field
    invalid_plan_1 = {
        "schema_version": "1.0",
        "objects": []  # Missing room_size
    }
    
    result = validator.validate_plan(invalid_plan_1)
    if not result.success and "room_size" in str(result.errors):
        print("✓ Correctly detected missing room_size")
    else:
        print(f"✗ Failed to detect missing room_size")
        return False
    
    # Test 3: Invalid reference
    invalid_plan_2 = {
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
                    "relative": {
                        "relative_to": "table_999",  # Non-existent reference
                        "direction": "front",
                        "distance": 0.5,
                        "reference_point": "center"
                    }
                },
                "orientation": {"absolute": {"yaw_deg": 0.0, "pitch_deg": 0.0, "roll_deg": 0.0}},
            }
        ]
    }
    
    result = validator.validate_plan(invalid_plan_2)
    if not result.success and "non-existent" in str(result.errors).lower():
        print("✓ Correctly detected invalid reference")
    else:
        print(f"✗ Failed to detect invalid reference: {result.errors}")
        return False
    
    # Test 4: Both absolute and relative position
    invalid_plan_3 = {
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
                    "absolute": {"x": 1.0, "y": 2.0, "z": 0.0},
                    "relative": {"relative_to": "table_1", "direction": "front", "distance": 0.5}
                },
                "orientation": {"absolute": {"yaw_deg": 0.0, "pitch_deg": 0.0, "roll_deg": 0.0}},
            }
        ]
    }
    
    result = validator.validate_plan(invalid_plan_3)
    if not result.success and "both" in str(result.errors).lower():
        print("✓ Correctly detected both absolute and relative position")
    else:
        print(f"✗ Failed to detect both absolute and relative: {result.errors}")
        return False
    
    print("✅ Schema validator test passed!\n")
    return True


def main():
    """Run all tests."""
    print("=" * 60)
    print("Testing Semantic Plan Enforcement Components")
    print("=" * 60 + "\n")
    
    try:
        test_data_models()
        test_schema_validator()
        
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
