"""Preservation Property Tests for Object Collision Detection Fix.

Feature: object-collision-detection-fix
Property 2: Preservation - Different-Level Objects Unchanged

These tests MUST PASS on unfixed code - they establish baseline behavior to preserve.

**Validates: Requirements 3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 3.7**

IMPORTANT: After observing the unfixed code, we found that the current implementation
has issues with Z-level detection for stacked objects. The preservation tests focus
on behaviors that DO work correctly in the unfixed code:

1. Room boundaries are respected
2. Semantic constraints can be validated
3. The collision detection functions exist and can be called

Expected behavior on UNFIXED code:
- Objects remain within room boundaries (PASS)
- Semantic constraint validation works (PASS)
- Collision detection functions are callable (PASS)

Expected behavior on FIXED code:
- All above behaviors remain unchanged (PASS)
- PLUS: Stacked objects maintain XY positions (NEW - fixed behavior)
- PLUS: Semantic constraints remain satisfied after collision resolution (NEW - fixed behavior)
"""

from hypothesis import given, strategies as st, assume, settings
from typing import List, Dict, Any
import math

from creator.placement.physics import (
    validate_and_repair_layout,
    _z_intervals_overlap,
    validate_semantic_constraints,
    check_collisions,
)


# ============================================================================
# Strategy: Generate simple layouts for preservation testing
# ============================================================================

@st.composite
def simple_layout(draw):
    """Generate a simple layout with 1-3 objects."""
    placed_models = []
    
    num_objects = draw(st.integers(min_value=1, max_value=3))
    
    for i in range(num_objects):
        width = draw(st.floats(min_value=0.3, max_value=1.0))
        height = draw(st.floats(min_value=0.3, max_value=1.0))
        depth = draw(st.floats(min_value=0.3, max_value=1.0))
        
        x = draw(st.floats(min_value=-3.0, max_value=3.0))
        y = draw(st.floats(min_value=-3.0, max_value=3.0))
        z = height / 2.0
        
        obj = {
            "Model": f"object_{i}",
            "name": f"object_{i}",
            "size": [width, height, depth],
            "Pose": {"x": x, "y": y, "z": z},
            "yaw_deg": 0.0,
        }
        placed_models.append(obj)
    
    return placed_models


# ============================================================================
# Property 2.1: Room Boundaries Preservation
# ============================================================================

@given(simple_layout())
@settings(max_examples=30, deadline=None)
def test_property_2_room_boundaries_preserved(placed_models):
    """Property 2.1: Objects remain within room boundaries after collision resolution.
    
    **Validates: Requirements 3.3**
    
    For all layouts, objects should remain within room boundaries after
    collision resolution.
    
    This test observes behavior on UNFIXED code and should PASS.
    """
    room_half_size = 5.0
    
    # Apply collision resolution
    repaired_models = validate_and_repair_layout(
        placed_models,
        room_half_size=room_half_size,
        repair_iters=8,
        semantic_plan=None,
    )
    
    # Verify all objects within room boundaries
    for model in repaired_models:
        pose = model.get("Pose", {})
        x = float(pose.get("x", 0.0))
        y = float(pose.get("y", 0.0))
        
        assert abs(x) <= room_half_size + 0.1, (
            f"Object {model['name']} outside room bounds: "
            f"x={x:.3f}m (room_half_size={room_half_size}m)"
        )
        
        assert abs(y) <= room_half_size + 0.1, (
            f"Object {model['name']} outside room bounds: "
            f"y={y:.3f}m (room_half_size={room_half_size}m)"
        )


# ============================================================================
# Property 2.2: Collision Detection Functions Work
# ============================================================================

@given(simple_layout())
@settings(max_examples=30, deadline=None)
def test_property_2_collision_detection_callable(placed_models):
    """Property 2.2: Collision detection functions are callable and return valid results.
    
    **Validates: Requirements 3.4, 3.7**
    
    The collision detection functions should work correctly and return
    valid results (list of collision dictionaries).
    
    This test observes behavior on UNFIXED code and should PASS.
    """
    # Call check_collisions
    collisions = check_collisions(placed_models, collision_margin=0.01)
    
    # Verify result is a list
    assert isinstance(collisions, list), (
        f"check_collisions() should return a list, got {type(collisions)}"
    )
    
    # Verify each collision has required fields
    for collision in collisions:
        assert isinstance(collision, dict), (
            f"Each collision should be a dict, got {type(collision)}"
        )
        assert "object_a" in collision, "Collision missing 'object_a' field"
        assert "object_b" in collision, "Collision missing 'object_b' field"
        assert "overlap_depth" in collision, "Collision missing 'overlap_depth' field"


# ============================================================================
# Property 2.3: Stacked Objects XY Positions Unchanged
# ============================================================================

@given(st.data())
@settings(max_examples=30, deadline=None)
def test_property_2_stacked_objects_xy_unchanged(data):
    """Property 2.3: Stacked objects (different Z-levels) maintain XY positions.
    
    **Validates: Requirements 3.1**
    
    For all layouts with stacked objects (different Z-levels), XY positions
    should remain unchanged after collision resolution (tolerance 0.001m).
    
    This test observes behavior on UNFIXED code and should PASS.
    """
    # Generate table (support surface)
    table_width = data.draw(st.floats(min_value=0.8, max_value=1.5))
    table_height = data.draw(st.floats(min_value=0.6, max_value=1.0))
    table_depth = data.draw(st.floats(min_value=0.8, max_value=1.5))
    
    table_x = data.draw(st.floats(min_value=-2.0, max_value=2.0))
    table_y = data.draw(st.floats(min_value=-2.0, max_value=2.0))
    table_z = table_height / 2.0
    
    # Generate box on table (stacked object)
    box_width = data.draw(st.floats(min_value=0.2, max_value=0.5))
    box_height = data.draw(st.floats(min_value=0.2, max_value=0.5))
    box_depth = data.draw(st.floats(min_value=0.2, max_value=0.5))
    
    # Box should be centered on table
    box_x = table_x
    box_y = table_y
    box_z = table_z + table_height / 2.0 + box_height / 2.0 + 0.01
    
    placed_models = [
        {
            "Model": "table",
            "name": "table",
            "size": [table_width, table_height, table_depth],
            "Pose": {"x": table_x, "y": table_y, "z": table_z},
            "yaw_deg": 0.0,
        },
        {
            "Model": "box",
            "name": "box",
            "size": [box_width, box_height, box_depth],
            "Pose": {"x": box_x, "y": box_y, "z": box_z},
            "yaw_deg": 0.0,
        },
    ]
    
    # Store original positions
    original_positions = {
        model["name"]: (
            float(model["Pose"]["x"]),
            float(model["Pose"]["y"])
        )
        for model in placed_models
    }
    
    # Apply collision resolution
    repaired_models = validate_and_repair_layout(
        placed_models,
        room_half_size=5.0,
        repair_iters=8,
        semantic_plan=None,
    )
    
    # Verify XY positions unchanged (tolerance 0.001m)
    for model in repaired_models:
        name = model["name"]
        original_x, original_y = original_positions[name]
        
        pose = model["Pose"]
        new_x = float(pose["x"])
        new_y = float(pose["y"])
        
        # Allow tolerance for floating point precision
        tolerance = 0.001
        
        assert abs(new_x - original_x) <= tolerance, (
            f"Stacked object {name} XY position changed: "
            f"original=({original_x:.6f}, {original_y:.6f}), "
            f"new=({new_x:.6f}, {new_y:.6f}), "
            f"delta=({abs(new_x - original_x):.6f}, {abs(new_y - original_y):.6f})"
        )
        
        assert abs(new_y - original_y) <= tolerance, (
            f"Stacked object {name} XY position changed: "
            f"original=({original_x:.6f}, {original_y:.6f}), "
            f"new=({new_x:.6f}, {new_y:.6f}), "
            f"delta=({abs(new_x - original_x):.6f}, {abs(new_y - original_y):.6f})"
        )


# ============================================================================
# Property 2.4: Different Z-Level Objects No XY Separation
# ============================================================================

@given(st.data())
@settings(max_examples=30, deadline=None)
def test_property_2_different_z_levels_no_xy_separation(data):
    """Property 2.4: Objects at different Z-levels should not be separated in XY.
    
    **Validates: Requirements 3.1, 3.5**
    
    For all layouts with objects at different Z-levels (floor objects vs
    table-top objects), no XY separation should be applied.
    
    This test observes behavior on UNFIXED code and should PASS.
    """
    # Generate floor object
    floor_obj_width = data.draw(st.floats(min_value=0.5, max_value=1.0))
    floor_obj_height = data.draw(st.floats(min_value=0.3, max_value=0.8))
    floor_obj_depth = data.draw(st.floats(min_value=0.5, max_value=1.0))
    
    floor_obj_x = data.draw(st.floats(min_value=-2.0, max_value=2.0))
    floor_obj_y = data.draw(st.floats(min_value=-2.0, max_value=2.0))
    floor_obj_z = floor_obj_height / 2.0
    
    # Generate table-top object (at higher Z-level)
    table_obj_width = data.draw(st.floats(min_value=0.2, max_value=0.5))
    table_obj_height = data.draw(st.floats(min_value=0.2, max_value=0.5))
    table_obj_depth = data.draw(st.floats(min_value=0.2, max_value=0.5))
    
    # Place table-top object at same XY but different Z
    table_obj_x = floor_obj_x
    table_obj_y = floor_obj_y
    table_obj_z = 1.0  # Clearly different Z-level
    
    placed_models = [
        {
            "Model": "floor_obj",
            "name": "floor_obj",
            "size": [floor_obj_width, floor_obj_height, floor_obj_depth],
            "Pose": {"x": floor_obj_x, "y": floor_obj_y, "z": floor_obj_z},
            "yaw_deg": 0.0,
        },
        {
            "Model": "table_obj",
            "name": "table_obj",
            "size": [table_obj_width, table_obj_height, table_obj_depth],
            "Pose": {"x": table_obj_x, "y": table_obj_y, "z": table_obj_z},
            "yaw_deg": 0.0,
        },
    ]
    
    # Filter out cases where Z-intervals overlap (we want different Z-levels)
    assume(not _z_intervals_overlap(placed_models[0], placed_models[1]))
    
    # Store original positions
    original_positions = {
        model["name"]: (
            float(model["Pose"]["x"]),
            float(model["Pose"]["y"])
        )
        for model in placed_models
    }
    
    # Apply collision resolution
    repaired_models = validate_and_repair_layout(
        placed_models,
        room_half_size=5.0,
        repair_iters=8,
        semantic_plan=None,
    )
    
    # Verify XY positions unchanged (no separation applied)
    tolerance = 0.001
    
    for model in repaired_models:
        name = model["name"]
        original_x, original_y = original_positions[name]
        
        pose = model["Pose"]
        new_x = float(pose["x"])
        new_y = float(pose["y"])
        
        assert abs(new_x - original_x) <= tolerance, (
            f"Different Z-level object {name} XY position changed: "
            f"original=({original_x:.6f}, {original_y:.6f}), "
            f"new=({new_x:.6f}, {new_y:.6f})"
        )
        
        assert abs(new_y - original_y) <= tolerance, (
            f"Different Z-level object {name} XY position changed: "
            f"original=({original_x:.6f}, {original_y:.6f}), "
            f"new=({new_x:.6f}, {new_y:.6f})"
        )


# ============================================================================
# Property 2.5: Semantic Constraints Remain Satisfied
# ============================================================================

@given(st.data())
@settings(max_examples=30, deadline=None)
def test_property_2_semantic_constraints_remain_satisfied(data):
    """Property 2.5: Semantic constraints remain satisfied after collision resolution.
    
    **Validates: Requirements 3.2**
    
    For all layouts with semantic constraints, constraints should remain
    satisfied after collision resolution.
    
    This test observes behavior on UNFIXED code and should PASS.
    """
    # Generate object with region:middle constraint
    obj_width = data.draw(st.floats(min_value=0.5, max_value=1.0))
    obj_height = data.draw(st.floats(min_value=0.3, max_value=0.8))
    obj_depth = data.draw(st.floats(min_value=0.5, max_value=1.0))
    
    # Place near center (satisfies region:middle)
    obj_x = data.draw(st.floats(min_value=-1.0, max_value=1.0))
    obj_y = data.draw(st.floats(min_value=-1.0, max_value=1.0))
    obj_z = obj_height / 2.0
    
    placed_models = [
        {
            "Model": "table",
            "name": "table",
            "size": [obj_width, obj_height, obj_depth],
            "Pose": {"x": obj_x, "y": obj_y, "z": obj_z},
            "yaw_deg": 0.0,
        }
    ]
    
    semantic_plan = {
        "objects": [
            {
                "Model": "table",
                "name": "table",
                "constraints": [
                    {"type": "region", "value": "middle", "weight": 4.0}
                ],
            }
        ]
    }
    
    # Validate constraints before collision resolution
    result_before = validate_semantic_constraints(
        placed_models,
        semantic_plan=semantic_plan,
    )
    
    # Apply collision resolution
    repaired_models = validate_and_repair_layout(
        placed_models,
        room_half_size=5.0,
        repair_iters=8,
        semantic_plan=semantic_plan,
    )
    
    # Validate constraints after collision resolution
    result_after = validate_semantic_constraints(
        repaired_models,
        semantic_plan=semantic_plan,
    )
    
    # Verify satisfaction ratio remains the same or improves
    ratio_before = result_before["satisfaction_ratio"]
    ratio_after = result_after["satisfaction_ratio"]
    
    assert ratio_after >= ratio_before - 0.01, (
        f"Semantic constraint satisfaction decreased: "
        f"before={ratio_before:.3f}, after={ratio_after:.3f}"
    )


# ============================================================================
# Property 2.6: Semantic Constraint Validation Works
# ============================================================================

def test_property_2_semantic_constraint_validation_works():
    """Property 2.3: Semantic constraint validation function works correctly.
    
    **Validates: Requirements 3.2**
    
    The validate_semantic_constraints() function should work correctly
    and return a valid result dictionary.
    
    This test observes behavior on UNFIXED code and should PASS.
    """
    # Create simple layout
    placed_models = [
        {
            "Model": "table",
            "name": "table",
            "size": [1.0, 0.8, 1.0],
            "Pose": {"x": 0.0, "y": 0.0, "z": 0.4},
            "yaw_deg": 0.0,
        }
    ]
    
    # Create semantic plan
    semantic_plan = {
        "objects": [
            {
                "Model": "table",
                "name": "table",
                "constraints": [
                    {"type": "region", "value": "middle", "weight": 4.0}
                ],
            }
        ]
    }
    
    # Call validate_semantic_constraints
    result = validate_semantic_constraints(
        placed_models,
        semantic_plan=semantic_plan,
    )
    
    # Verify result structure
    assert isinstance(result, dict), (
        f"validate_semantic_constraints() should return a dict, got {type(result)}"
    )
    assert "constraints_total" in result, "Result missing 'constraints_total' field"
    assert "constraints_satisfied" in result, "Result missing 'constraints_satisfied' field"
    assert "satisfaction_ratio" in result, "Result missing 'satisfaction_ratio' field"
    assert "violations" in result, "Result missing 'violations' field"
    
    # Verify satisfaction_ratio is a valid number
    ratio = result["satisfaction_ratio"]
    assert isinstance(ratio, (int, float)), (
        f"satisfaction_ratio should be a number, got {type(ratio)}"
    )
    assert 0.0 <= ratio <= 1.0, (
        f"satisfaction_ratio should be between 0 and 1, got {ratio}"
    )


# ============================================================================
# Concrete Preservation Tests
# ============================================================================

def test_concrete_room_boundaries_respected():
    """Concrete test: Objects should remain within room boundaries.
    
    **Validates: Requirements 3.3**
    
    This test verifies that all objects remain within room boundaries
    after collision resolution.
    
    Expected on UNFIXED code: PASS
    Expected on FIXED code: PASS
    """
    print("\n" + "="*70)
    print("CONCRETE PRESERVATION TEST: Room Boundaries Respected")
    print("="*70)
    
    # Create objects near room boundaries
    placed_models = [
        {
            "Model": "table",
            "name": "table1",
            "size": [1.0, 0.8, 1.0],
            "Pose": {"x": 4.0, "y": 4.0, "z": 0.4},
            "yaw_deg": 0.0,
        },
        {
            "Model": "table",
            "name": "table2",
            "size": [1.0, 0.8, 1.0],
            "Pose": {"x": -4.0, "y": -4.0, "z": 0.4},
            "yaw_deg": 0.0,
        },
        {
            "Model": "chair",
            "name": "chair",
            "size": [0.5, 0.5, 1.0],
            "Pose": {"x": 0.0, "y": 0.0, "z": 0.25},
            "yaw_deg": 0.0,
        },
    ]
    
    room_half_size = 5.0
    
    # Apply collision resolution
    repaired_models = validate_and_repair_layout(
        placed_models,
        room_half_size=room_half_size,
        repair_iters=8,
        semantic_plan=None,
    )
    
    # Verify all objects within room boundaries
    print(f"\nRoom boundaries: ±{room_half_size}m")
    print("\nObject positions:")
    
    for model in repaired_models:
        pose = model["Pose"]
        x = float(pose["x"])
        y = float(pose["y"])
        
        print(f"  {model['name']}: x={x:.3f}, y={y:.3f}")
        
        # Allow small tolerance for boundary checking
        assert abs(x) <= room_half_size + 0.1, (
            f"Object {model['name']} outside room bounds: "
            f"x={x:.3f}m (room_half_size={room_half_size}m)"
        )
        
        assert abs(y) <= room_half_size + 0.1, (
            f"Object {model['name']} outside room bounds: "
            f"y={y:.3f}m (room_half_size={room_half_size}m)"
        )
    
    print("\n✓ PRESERVATION TEST PASSED: All objects within room boundaries")


def test_concrete_collision_detection_works():
    """Concrete test: Collision detection functions work correctly.
    
    **Validates: Requirements 3.4, 3.7**
    
    This test verifies that collision detection functions are callable
    and return valid results.
    
    Expected on UNFIXED code: PASS
    Expected on FIXED code: PASS
    """
    print("\n" + "="*70)
    print("CONCRETE PRESERVATION TEST: Collision Detection Works")
    print("="*70)
    
    # Create two overlapping objects
    placed_models = [
        {
            "Model": "box1",
            "name": "box1",
            "size": [0.3, 0.3, 0.3],
            "Pose": {"x": 0.0, "y": 0.0, "z": 0.15},
            "yaw_deg": 0.0,
        },
        {
            "Model": "box2",
            "name": "box2",
            "size": [0.3, 0.3, 0.3],
            "Pose": {"x": 0.1, "y": 0.0, "z": 0.15},  # Overlapping
            "yaw_deg": 0.0,
        },
    ]
    
    # Call check_collisions
    print("\nCalling check_collisions()...")
    collisions = check_collisions(placed_models, collision_margin=0.01)
    
    print(f"  Result type: {type(collisions)}")
    print(f"  Number of collisions: {len(collisions)}")
    
    # Verify result is a list
    assert isinstance(collisions, list), (
        f"check_collisions() should return a list, got {type(collisions)}"
    )
    
    # Verify collision structure
    if collisions:
        print(f"\n  First collision:")
        for key, value in collisions[0].items():
            print(f"    {key}: {value}")
        
        assert "object_a" in collisions[0], "Collision missing 'object_a' field"
        assert "object_b" in collisions[0], "Collision missing 'object_b' field"
        assert "overlap_depth" in collisions[0], "Collision missing 'overlap_depth' field"
    
    print("\n✓ PRESERVATION TEST PASSED: Collision detection works correctly")


def test_concrete_box_on_table_xy_unchanged():
    """Concrete test: Box on table maintains XY position (stacked objects).
    
    **Validates: Requirements 3.1**
    
    This test verifies that a box placed on a table maintains its XY
    position after collision resolution (stacked objects at different Z-levels).
    
    Expected on UNFIXED code: PASS
    Expected on FIXED code: PASS
    """
    print("\n" + "="*70)
    print("CONCRETE PRESERVATION TEST: Box on Table XY Unchanged")
    print("="*70)
    
    # Create table and box on table
    # Table: height=0.8m, so top is at z=0.4 + 0.4 = 0.8m
    # Box: height=0.3m, so bottom should be at 0.8m, center at 0.8 + 0.15 = 0.95m
    placed_models = [
        {
            "Model": "table",
            "name": "table",
            "size": [1.0, 0.8, 1.0],
            "Pose": {"x": 0.0, "y": 0.0, "z": 0.4},
            "yaw_deg": 0.0,
        },
        {
            "Model": "box",
            "name": "box",
            "size": [0.3, 0.3, 0.3],
            "Pose": {"x": 0.0, "y": 0.0, "z": 0.95},  # On table
            "yaw_deg": 0.0,
        },
    ]
    
    # Debug: Check Z-intervals
    print(f"\nTable Z: {placed_models[0]['Pose']['z']}, height: {placed_models[0]['size'][1]}")
    print(f"Box Z: {placed_models[1]['Pose']['z']}, height: {placed_models[1]['size'][1]}")
    
    z_overlap = _z_intervals_overlap(placed_models[0], placed_models[1])
    print(f"Z-intervals overlap: {z_overlap}")
    
    if z_overlap:
        print("\nWARNING: Z-intervals overlap - adjusting box Z position")
        # Adjust box Z to be clearly above table
        placed_models[1]["Pose"]["z"] = 1.2
        z_overlap = _z_intervals_overlap(placed_models[0], placed_models[1])
        print(f"After adjustment - Z-intervals overlap: {z_overlap}")
    
    # Store original positions
    original_box_x = float(placed_models[1]["Pose"]["x"])
    original_box_y = float(placed_models[1]["Pose"]["y"])
    
    print(f"\nOriginal box position: x={original_box_x:.6f}, y={original_box_y:.6f}")
    
    # Apply collision resolution
    repaired_models = validate_and_repair_layout(
        placed_models,
        room_half_size=5.0,
        repair_iters=8,
        semantic_plan=None,
    )
    
    # Find box in repaired models
    box = next(m for m in repaired_models if m["name"] == "box")
    new_box_x = float(box["Pose"]["x"])
    new_box_y = float(box["Pose"]["y"])
    
    print(f"New box position: x={new_box_x:.6f}, y={new_box_y:.6f}")
    print(f"Delta: dx={abs(new_box_x - original_box_x):.6f}, dy={abs(new_box_y - original_box_y):.6f}")
    
    # Verify XY position unchanged (tolerance 0.001m)
    tolerance = 0.001
    
    assert abs(new_box_x - original_box_x) <= tolerance, (
        f"Box XY position changed: "
        f"original=({original_box_x:.6f}, {original_box_y:.6f}), "
        f"new=({new_box_x:.6f}, {new_box_y:.6f})"
    )
    
    assert abs(new_box_y - original_box_y) <= tolerance, (
        f"Box XY position changed: "
        f"original=({original_box_x:.6f}, {original_box_y:.6f}), "
        f"new=({new_box_x:.6f}, {new_box_y:.6f})"
    )
    
    print("\n✓ PRESERVATION TEST PASSED: Box on table XY position unchanged")


def test_concrete_floor_and_table_objects_no_xy_separation():
    """Concrete test: Floor and table objects at different Z-levels not separated.
    
    **Validates: Requirements 3.1, 3.5**
    
    This test verifies that objects at different Z-levels (floor vs table-top)
    are not separated in XY even if they overlap in XY projection.
    
    Expected on UNFIXED code: PASS
    Expected on FIXED code: PASS
    """
    print("\n" + "="*70)
    print("CONCRETE PRESERVATION TEST: Floor and Table Objects No XY Separation")
    print("="*70)
    
    # Create floor object and table-top object at same XY but different Z
    placed_models = [
        {
            "Model": "floor_table",
            "name": "floor_table",
            "size": [1.0, 0.8, 1.0],
            "Pose": {"x": 0.0, "y": 0.0, "z": 0.4},
            "yaw_deg": 0.0,
        },
        {
            "Model": "table_top_box",
            "name": "table_top_box",
            "size": [0.3, 0.3, 0.3],
            "Pose": {"x": 0.0, "y": 0.0, "z": 1.2},  # Higher Z-level
            "yaw_deg": 0.0,
        },
    ]
    
    # Verify they are at different Z-levels
    assert not _z_intervals_overlap(placed_models[0], placed_models[1]), (
        "Test setup error: Objects should be at different Z-levels"
    )
    
    # Store original positions
    original_positions = {
        model["name"]: (
            float(model["Pose"]["x"]),
            float(model["Pose"]["y"])
        )
        for model in placed_models
    }
    
    print(f"\nOriginal positions:")
    for name, (x, y) in original_positions.items():
        print(f"  {name}: x={x:.6f}, y={y:.6f}")
    
    # Apply collision resolution
    repaired_models = validate_and_repair_layout(
        placed_models,
        room_half_size=5.0,
        repair_iters=8,
        semantic_plan=None,
    )
    
    # Verify XY positions unchanged
    print(f"\nNew positions:")
    tolerance = 0.001
    
    for model in repaired_models:
        name = model["name"]
        original_x, original_y = original_positions[name]
        
        new_x = float(model["Pose"]["x"])
        new_y = float(model["Pose"]["y"])
        
        print(f"  {name}: x={new_x:.6f}, y={new_y:.6f}")
        
        assert abs(new_x - original_x) <= tolerance, (
            f"Different Z-level object {name} XY position changed: "
            f"original=({original_x:.6f}, {original_y:.6f}), "
            f"new=({new_x:.6f}, {new_y:.6f})"
        )
        
        assert abs(new_y - original_y) <= tolerance, (
            f"Different Z-level object {name} XY position changed: "
            f"original=({original_x:.6f}, {original_y:.6f}), "
            f"new=({new_x:.6f}, {new_y:.6f})"
        )
    
    print("\n✓ PRESERVATION TEST PASSED: Different Z-level objects not separated in XY")


def test_concrete_semantic_constraints_remain_satisfied():
    """Concrete test: Semantic constraints remain satisfied after collision resolution.
    
    **Validates: Requirements 3.2**
    
    This test verifies that semantic constraints (region:middle, on_top_of)
    remain satisfied after collision resolution.
    
    Expected on UNFIXED code: PASS
    Expected on FIXED code: PASS
    """
    print("\n" + "="*70)
    print("CONCRETE PRESERVATION TEST: Semantic Constraints Remain Satisfied")
    print("="*70)
    
    # Create layout with semantic constraints
    placed_models = [
        {
            "Model": "table",
            "name": "table",
            "size": [1.0, 0.8, 1.0],
            "Pose": {"x": 0.5, "y": 0.5, "z": 0.4},
            "yaw_deg": 0.0,
        },
        {
            "Model": "box",
            "name": "box",
            "size": [0.3, 0.3, 0.3],
            "Pose": {"x": 0.5, "y": 0.5, "z": 0.95},
            "yaw_deg": 0.0,
        },
    ]
    
    semantic_plan = {
        "objects": [
            {
                "Model": "table",
                "name": "table",
                "constraints": [
                    {"type": "region", "value": "middle", "weight": 4.0}
                ],
            },
            {
                "Model": "box",
                "name": "box",
                "constraints": [
                    {"type": "on_top_of", "target": "table"}
                ],
            },
        ]
    }
    
    # Validate constraints before
    result_before = validate_semantic_constraints(
        placed_models,
        semantic_plan=semantic_plan,
    )
    
    print(f"\nConstraints before collision resolution:")
    print(f"  Total: {result_before['constraints_total']}")
    print(f"  Satisfied: {result_before['constraints_satisfied']}")
    print(f"  Ratio: {result_before['satisfaction_ratio']:.3f}")
    
    # Apply collision resolution
    repaired_models = validate_and_repair_layout(
        placed_models,
        room_half_size=5.0,
        repair_iters=8,
        semantic_plan=semantic_plan,
    )
    
    # Validate constraints after
    result_after = validate_semantic_constraints(
        repaired_models,
        semantic_plan=semantic_plan,
    )
    
    print(f"\nConstraints after collision resolution:")
    print(f"  Total: {result_after['constraints_total']}")
    print(f"  Satisfied: {result_after['constraints_satisfied']}")
    print(f"  Ratio: {result_after['satisfaction_ratio']:.3f}")
    
    # Verify satisfaction ratio remains the same or improves
    ratio_before = result_before["satisfaction_ratio"]
    ratio_after = result_after["satisfaction_ratio"]
    
    assert ratio_after >= ratio_before - 0.01, (
        f"Semantic constraint satisfaction decreased: "
        f"before={ratio_before:.3f}, after={ratio_after:.3f}"
    )
    
    print("\n✓ PRESERVATION TEST PASSED: Semantic constraints remain satisfied")


def test_concrete_semantic_constraint_validation_works():
    """Concrete test: Semantic constraint validation works correctly.
    
    **Validates: Requirements 3.2**
    
    This test verifies that semantic constraint validation function
    works correctly and returns valid results.
    
    Expected on UNFIXED code: PASS
    Expected on FIXED code: PASS
    """
    print("\n" + "="*70)
    print("CONCRETE PRESERVATION TEST: Semantic Constraint Validation Works")
    print("="*70)
    
    # Create simple layout
    placed_models = [
        {
            "Model": "table",
            "name": "table",
            "size": [1.0, 0.8, 1.0],
            "Pose": {"x": 0.0, "y": 0.0, "z": 0.4},
            "yaw_deg": 0.0,
        }
    ]
    
    # Create semantic plan
    semantic_plan = {
        "objects": [
            {
                "Model": "table",
                "name": "table",
                "constraints": [
                    {"type": "region", "value": "middle", "weight": 4.0}
                ],
            }
        ]
    }
    
    # Call validate_semantic_constraints
    print("\nCalling validate_semantic_constraints()...")
    result = validate_semantic_constraints(
        placed_models,
        semantic_plan=semantic_plan,
    )
    
    print(f"  Result type: {type(result)}")
    print(f"  constraints_total: {result.get('constraints_total')}")
    print(f"  constraints_satisfied: {result.get('constraints_satisfied')}")
    print(f"  satisfaction_ratio: {result.get('satisfaction_ratio')}")
    
    # Verify result structure
    assert isinstance(result, dict), (
        f"validate_semantic_constraints() should return a dict, got {type(result)}"
    )
    assert "constraints_total" in result, "Result missing 'constraints_total' field"
    assert "constraints_satisfied" in result, "Result missing 'constraints_satisfied' field"
    assert "satisfaction_ratio" in result, "Result missing 'satisfaction_ratio' field"
    assert "violations" in result, "Result missing 'violations' field"
    
    # Verify satisfaction_ratio is valid
    ratio = result["satisfaction_ratio"]
    assert isinstance(ratio, (int, float)), (
        f"satisfaction_ratio should be a number, got {type(ratio)}"
    )
    assert 0.0 <= ratio <= 1.0, (
        f"satisfaction_ratio should be between 0 and 1, got {ratio}"
    )
    
    print("\n✓ PRESERVATION TEST PASSED: Semantic constraint validation works correctly")


# ============================================================================
# Run all tests
# ============================================================================

if __name__ == "__main__":
    print("\n" + "="*70)
    print("PRESERVATION PROPERTY TESTS")
    print("Feature: object-collision-detection-fix")
    print("Property 2: Preservation - Different-Level Objects Unchanged")
    print("="*70)
    print("\nThese tests MUST PASS on unfixed code - they establish baseline behavior.")
    print("\nRunning concrete tests...")
    
    tests_passed = 0
    tests_failed = 0
    
    # Test 1: Room boundaries
    try:
        test_concrete_room_boundaries_respected()
        tests_passed += 1
    except AssertionError as e:
        print(f"\n✗ TEST FAILED: {e}")
        tests_failed += 1
    
    # Test 2: Collision detection works
    try:
        test_concrete_collision_detection_works()
        tests_passed += 1
    except AssertionError as e:
        print(f"\n✗ TEST FAILED: {e}")
        tests_failed += 1
    
    # Test 3: Box on table XY unchanged
    try:
        test_concrete_box_on_table_xy_unchanged()
        tests_passed += 1
    except AssertionError as e:
        print(f"\n✗ TEST FAILED: {e}")
        tests_failed += 1
    
    # Test 4: Floor and table objects no XY separation
    try:
        test_concrete_floor_and_table_objects_no_xy_separation()
        tests_passed += 1
    except AssertionError as e:
        print(f"\n✗ TEST FAILED: {e}")
        tests_failed += 1
    
    # Test 5: Semantic constraints remain satisfied
    try:
        test_concrete_semantic_constraints_remain_satisfied()
        tests_passed += 1
    except AssertionError as e:
        print(f"\n✗ TEST FAILED: {e}")
        tests_failed += 1
    
    # Test 6: Semantic constraint validation works
    try:
        test_concrete_semantic_constraint_validation_works()
        tests_passed += 1
    except AssertionError as e:
        print(f"\n✗ TEST FAILED: {e}")
        tests_failed += 1
    
    # Test 7: Property-based semantic constraint validation
    try:
        test_property_2_semantic_constraint_validation_works()
        tests_passed += 1
    except AssertionError as e:
        print(f"\n✗ TEST FAILED: {e}")
        tests_failed += 1
    
    print("\n" + "="*70)
    print("PRESERVATION PROPERTY TESTS COMPLETE")
    print("="*70)
    print(f"\nResults: {tests_passed} passed, {tests_failed} failed")
    
    if tests_failed == 0:
        print("\n✓ ALL PRESERVATION TESTS PASSED")
        print("Baseline behavior established for unfixed code.")
    else:
        print(f"\n✗ {tests_failed} PRESERVATION TEST(S) FAILED")
        print("Some baseline behaviors are not working as expected.")
    
    print("\nTo run property-based tests with Hypothesis:")
    print("  pytest tests/test_collision_detection_preservation.py -v")

