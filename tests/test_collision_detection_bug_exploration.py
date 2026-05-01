"""Bug Condition Exploration Test for Object Collision Detection.

Feature: object-collision-detection-fix
Property 1: Bug Condition - Same-Level Objects Overlap Detection

This test MUST FAIL on unfixed code - failure confirms the bug exists.

**Validates: Requirements 1.1, 1.2, 1.3, 1.4, 2.1, 2.2, 2.3, 2.4, 2.5**

The test creates scenarios where multiple objects are placed on the same surface
(same Z-level) and verifies that they do NOT overlap after collision resolution.

Expected behavior on UNFIXED code:
- Two boxes on table overlap by ~0.1m (FAIL)
- Three apples in box all overlap (FAIL)
- Multiple objects on floor have overlaps (FAIL)
- Small surface with many objects fails to converge (FAIL)

Expected behavior on FIXED code:
- check_collisions() returns empty list (no overlaps) (PASS)
- OR LLM fallback was invoked to replan layout (PASS)
- All objects remain within room boundaries (PASS)
- Semantic constraints are satisfied (PASS)
"""

import pytest
from typing import List, Dict, Any

from creator.placement.physics import (
    check_collisions,
    validate_and_repair_layout,
    _z_intervals_overlap,
)
from creator.placement.geometry import (
    model_to_obb,
    obb_overlap,
)


# Tolerance for numerical precision issues (1mm)
COLLISION_TOLERANCE = 0.001


def assert_no_significant_collisions(final_collisions: List[Dict[str, Any]]) -> None:
    """Assert that no significant collisions remain after repair.
    
    Filters out negligible overlaps (< 1mm) due to numerical precision.
    """
    significant_collisions = [
        c for c in final_collisions if c['overlap_depth'] >= COLLISION_TOLERANCE
    ]
    
    print("\n" + "="*70)
    print("ASSERTION: No significant collisions should remain after repair")
    print("="*70)
    
    if len(significant_collisions) < len(final_collisions):
        negligible_count = len(final_collisions) - len(significant_collisions)
        print(f"Note: {negligible_count} negligible collision(s) "
              f"(overlap < {COLLISION_TOLERANCE}m) ignored due to numerical precision")
    
    assert len(significant_collisions) == 0, (
        f"Expected no collisions after repair, but found {len(significant_collisions)} "
        f"significant collision(s). This confirms the bug exists in unfixed code."
    )
    
    print("\n✓ TEST PASSED: No significant collisions detected after repair")


# ============================================================================
# Test Case 1: Two Boxes on Table
# ============================================================================

def test_two_boxes_on_table_no_overlap():
    """Test Case 1: Two boxes on table should not overlap.
    
    **Validates: Requirements 1.1, 2.1, 2.2, 2.3, 2.4, 2.5**
    
    Setup:
    - Table at z=0.4m (half-height of 0.8m table)
    - Box1 at (0.5, 0.0, 0.9) - on table
    - Box2 at (0.6, 0.0, 0.9) - on table, overlapping with Box1
    - Box sizes: 0.3m × 0.3m × 0.3m each
    
    Expected on UNFIXED code:
    - Boxes overlap by ~0.1m in X direction (FAIL)
    
    Expected on FIXED code:
    - check_collisions() returns empty list (PASS)
    - OR LLM fallback was invoked (PASS)
    """
    print("\n" + "="*70)
    print("TEST CASE 1: Two Boxes on Table")
    print("="*70)
    
    # Create table
    table = {
        "Model": "table",
        "name": "table",
        "size": [1.0, 0.8, 1.0],  # width, height, depth
        "Pose": {"x": 0.0, "y": 0.0, "z": 0.4},  # z = half-height
        "yaw_deg": 0.0,
    }
    
    # Create two boxes on table - positioned to overlap
    box1 = {
        "Model": "cardboard_box",
        "name": "box1",
        "size": [0.3, 0.3, 0.3],  # width, height, depth
        "Pose": {"x": 0.5, "y": 0.0, "z": 0.9},  # z = table_top + half-height
        "yaw_deg": 0.0,
    }
    
    box2 = {
        "Model": "cardboard_box",
        "name": "box2",
        "size": [0.3, 0.3, 0.3],
        "Pose": {"x": 0.6, "y": 0.0, "z": 0.9},  # Only 0.1m apart, should overlap
        "yaw_deg": 0.0,
    }
    
    placed_models = [table, box1, box2]
    
    # Verify boxes are at same Z-level
    assert _z_intervals_overlap(box1, box2), "Boxes should be at same Z-level"
    
    # Check initial collision state (before repair)
    print("\nInitial collision check (before repair):")
    initial_collisions = check_collisions(placed_models, collision_margin=0.01)
    print(f"  Initial collisions: {len(initial_collisions)}")
    for collision in initial_collisions:
        print(f"    - {collision['object_a']} <-> {collision['object_b']}: "
              f"overlap_depth={collision['overlap_depth']:.4f}m")
    
    # Apply collision resolution
    print("\nApplying collision resolution...")
    repaired_models = validate_and_repair_layout(
        placed_models,
        room_half_size=5.0,
        repair_iters=8,
        semantic_plan=None,
    )
    
    # Check final collision state (after repair)
    print("\nFinal collision check (after repair):")
    final_collisions = check_collisions(repaired_models, collision_margin=0.01)
    print(f"  Final collisions: {len(final_collisions)}")
    for collision in final_collisions:
        print(f"    - {collision['object_a']} <-> {collision['object_b']}: "
              f"overlap_depth={collision['overlap_depth']:.4f}m")
    
    # Verify room boundaries
    print("\nRoom boundary check:")
    for model in repaired_models:
        pose = model.get("Pose", {})
        x, y = pose.get("x", 0.0), pose.get("y", 0.0)
        print(f"  {model['name']}: x={x:.3f}, y={y:.3f}")
        assert abs(x) <= 5.0, f"{model['name']} outside room bounds (x={x})"
        assert abs(y) <= 5.0, f"{model['name']} outside room bounds (y={y})"
    
    # CRITICAL ASSERTION: No significant collisions should remain
    assert_no_significant_collisions(final_collisions)


# ============================================================================
# Test Case 2: Three Apples in Box
# ============================================================================

def test_three_apples_in_box_no_overlap():
    """Test Case 2: Three apples in box should not overlap.
    
    **Validates: Requirements 1.1, 2.1, 2.2, 2.3, 2.4, 2.5**
    
    Setup:
    - Box at z=0.9m (on table)
    - Apple1 at (0.5, 0.0, 1.1) - in box
    - Apple2 at (0.52, 0.0, 1.1) - in box, overlapping with Apple1
    - Apple3 at (0.48, 0.0, 1.1) - in box, overlapping with Apple1
    - Apple sizes: 0.08m diameter each
    
    Expected on UNFIXED code:
    - All three apples overlap at center of box (FAIL)
    
    Expected on FIXED code:
    - check_collisions() returns empty list (PASS)
    - OR LLM fallback was invoked (PASS)
    """
    print("\n" + "="*70)
    print("TEST CASE 2: Three Apples in Box")
    print("="*70)
    
    # Create box (container)
    box = {
        "Model": "cardboard_box",
        "name": "box",
        "size": [0.4, 0.3, 0.4],  # width, height, depth
        "Pose": {"x": 0.5, "y": 0.0, "z": 0.9},
        "yaw_deg": 0.0,
    }
    
    # Create three apples in box - positioned to overlap
    apple1 = {
        "Model": "apple2",
        "name": "apple1",
        "size": [0.08, 0.08, 0.08],
        "Pose": {"x": 0.5, "y": 0.0, "z": 1.1},  # Center of box
        "yaw_deg": 0.0,
    }
    
    apple2 = {
        "Model": "apple2",
        "name": "apple2",
        "size": [0.08, 0.08, 0.08],
        "Pose": {"x": 0.52, "y": 0.0, "z": 1.1},  # Only 0.02m apart, should overlap
        "yaw_deg": 0.0,
    }
    
    apple3 = {
        "Model": "apple2",
        "name": "apple3",
        "size": [0.08, 0.08, 0.08],
        "Pose": {"x": 0.48, "y": 0.0, "z": 1.1},  # Only 0.02m apart, should overlap
        "yaw_deg": 0.0,
    }
    
    placed_models = [box, apple1, apple2, apple3]
    
    # Verify apples are at same Z-level
    assert _z_intervals_overlap(apple1, apple2), "Apple1 and Apple2 should be at same Z-level"
    assert _z_intervals_overlap(apple1, apple3), "Apple1 and Apple3 should be at same Z-level"
    assert _z_intervals_overlap(apple2, apple3), "Apple2 and Apple3 should be at same Z-level"
    
    # Check initial collision state
    print("\nInitial collision check (before repair):")
    initial_collisions = check_collisions(placed_models, collision_margin=0.01)
    print(f"  Initial collisions: {len(initial_collisions)}")
    for collision in initial_collisions:
        print(f"    - {collision['object_a']} <-> {collision['object_b']}: "
              f"overlap_depth={collision['overlap_depth']:.4f}m")
    
    # Apply collision resolution
    print("\nApplying collision resolution...")
    repaired_models = validate_and_repair_layout(
        placed_models,
        room_half_size=5.0,
        repair_iters=8,
        semantic_plan=None,
    )
    
    # Check final collision state
    print("\nFinal collision check (after repair):")
    final_collisions = check_collisions(repaired_models, collision_margin=0.01)
    print(f"  Final collisions: {len(final_collisions)}")
    for collision in final_collisions:
        print(f"    - {collision['object_a']} <-> {collision['object_b']}: "
              f"overlap_depth={collision['overlap_depth']:.4f}m")
    
    # Verify room boundaries
    print("\nRoom boundary check:")
    for model in repaired_models:
        pose = model.get("Pose", {})
        x, y = pose.get("x", 0.0), pose.get("y", 0.0)
        print(f"  {model['name']}: x={x:.3f}, y={y:.3f}")
        assert abs(x) <= 5.0, f"{model['name']} outside room bounds (x={x})"
        assert abs(y) <= 5.0, f"{model['name']} outside room bounds (y={y})"
    
    # CRITICAL ASSERTION: No significant collisions should remain
    assert_no_significant_collisions(final_collisions)


# ============================================================================
# Test Case 3: Dense Floor Placement
# ============================================================================

def test_dense_floor_placement_no_overlap():
    """Test Case 3: Multiple objects on floor should not overlap.
    
    **Validates: Requirements 1.1, 2.1, 2.2, 2.3, 2.4, 2.5**
    
    Setup:
    - 8 objects placed on floor in 2m × 2m region
    - Objects at same Z-level (floor level)
    - Some objects positioned to overlap
    
    Expected on UNFIXED code:
    - Multiple overlaps between floor objects (FAIL)
    
    Expected on FIXED code:
    - check_collisions() returns empty list (PASS)
    - OR LLM fallback was invoked (PASS)
    """
    print("\n" + "="*70)
    print("TEST CASE 3: Dense Floor Placement")
    print("="*70)
    
    # Create 8 objects on floor in dense arrangement
    placed_models = []
    
    # Grid positions in 2m × 2m region (will cause overlaps)
    positions = [
        (0.3, 0.3), (0.5, 0.3), (0.7, 0.3),
        (0.3, 0.5), (0.5, 0.5), (0.7, 0.5),
        (0.3, 0.7), (0.5, 0.7),
    ]
    
    for i, (x, y) in enumerate(positions):
        obj = {
            "Model": "cardboard_box",
            "name": f"box{i+1}",
            "size": [0.3, 0.3, 0.3],  # 0.3m boxes
            "Pose": {"x": x, "y": y, "z": 0.15},  # z = half-height
            "yaw_deg": 0.0,
        }
        placed_models.append(obj)
    
    # Verify objects are at same Z-level
    print("\nVerifying objects are at same Z-level:")
    for i in range(len(placed_models)):
        for j in range(i + 1, len(placed_models)):
            if _z_intervals_overlap(placed_models[i], placed_models[j]):
                print(f"  {placed_models[i]['name']} <-> {placed_models[j]['name']}: same Z-level ✓")
    
    # Check initial collision state
    print("\nInitial collision check (before repair):")
    initial_collisions = check_collisions(placed_models, collision_margin=0.01)
    print(f"  Initial collisions: {len(initial_collisions)}")
    for collision in initial_collisions:
        print(f"    - {collision['object_a']} <-> {collision['object_b']}: "
              f"overlap_depth={collision['overlap_depth']:.4f}m")
    
    # Apply collision resolution
    print("\nApplying collision resolution...")
    repaired_models = validate_and_repair_layout(
        placed_models,
        room_half_size=5.0,
        repair_iters=8,
        semantic_plan=None,
    )
    
    # Check final collision state
    print("\nFinal collision check (after repair):")
    final_collisions = check_collisions(repaired_models, collision_margin=0.01)
    print(f"  Final collisions: {len(final_collisions)}")
    for collision in final_collisions:
        print(f"    - {collision['object_a']} <-> {collision['object_b']}: "
              f"overlap_depth={collision['overlap_depth']:.4f}m")
    
    # Verify room boundaries
    print("\nRoom boundary check:")
    for model in repaired_models:
        pose = model.get("Pose", {})
        x, y = pose.get("x", 0.0), pose.get("y", 0.0)
        print(f"  {model['name']}: x={x:.3f}, y={y:.3f}")
        assert abs(x) <= 5.0, f"{model['name']} outside room bounds (x={x})"
        assert abs(y) <= 5.0, f"{model['name']} outside room bounds (y={y})"
    
    # CRITICAL ASSERTION: No significant collisions should remain
    assert_no_significant_collisions(final_collisions)


# ============================================================================
# Test Case 4: Small Surface Infeasibility
# ============================================================================

def test_small_surface_infeasibility():
    """Test Case 4: Small surface with many objects should converge or invoke LLM fallback.
    
    **Validates: Requirements 1.1, 2.1, 2.2, 2.3, 2.4, 2.5, 2.6**
    
    Setup:
    - Small table (0.6m × 0.6m)
    - 4 boxes (0.3m × 0.3m each)
    - Total object footprint: 4 × 0.09 = 0.36 m²
    - Available surface: 0.36 m²
    - Ratio: 1.0 (100% coverage, infeasible with margins)
    
    Expected on UNFIXED code:
    - Gradient resolution runs for max iterations without convergence (FAIL)
    - Produces overlapping layout (FAIL)
    
    Expected on FIXED code:
    - check_collisions() returns empty list (PASS)
    - OR LLM fallback was invoked (PASS)
    """
    print("\n" + "="*70)
    print("TEST CASE 4: Small Surface Infeasibility")
    print("="*70)
    
    # Create small table
    table = {
        "Model": "table",
        "name": "small_table",
        "size": [0.6, 0.8, 0.6],  # Small 0.6m × 0.6m table
        "Pose": {"x": 0.0, "y": 0.0, "z": 0.4},
        "yaw_deg": 0.0,
    }
    
    # Create 4 boxes on small table - positioned to overlap
    boxes = []
    positions = [(0.15, 0.15), (0.15, -0.15), (-0.15, 0.15), (-0.15, -0.15)]
    
    for i, (x, y) in enumerate(positions):
        box = {
            "Model": "cardboard_box",
            "name": f"box{i+1}",
            "size": [0.3, 0.3, 0.3],
            "Pose": {"x": x, "y": y, "z": 0.95},  # z = table_top + half-height
            "yaw_deg": 0.0,
        }
        boxes.append(box)
    
    placed_models = [table] + boxes
    
    # Calculate surface coverage
    table_area = 0.6 * 0.6
    box_area = 0.3 * 0.3
    total_box_area = 4 * box_area
    coverage_ratio = total_box_area / table_area
    
    print(f"\nSurface coverage analysis:")
    print(f"  Table area: {table_area:.3f} m²")
    print(f"  Total box area: {total_box_area:.3f} m²")
    print(f"  Coverage ratio: {coverage_ratio:.2f} (100% = infeasible with margins)")
    
    # Check initial collision state
    print("\nInitial collision check (before repair):")
    initial_collisions = check_collisions(placed_models, collision_margin=0.01)
    print(f"  Initial collisions: {len(initial_collisions)}")
    for collision in initial_collisions:
        print(f"    - {collision['object_a']} <-> {collision['object_b']}: "
              f"overlap_depth={collision['overlap_depth']:.4f}m")
    
    # Apply collision resolution
    print("\nApplying collision resolution...")
    repaired_models = validate_and_repair_layout(
        placed_models,
        room_half_size=5.0,
        repair_iters=8,
        semantic_plan=None,
    )
    
    # Check final collision state
    print("\nFinal collision check (after repair):")
    final_collisions = check_collisions(repaired_models, collision_margin=0.01)
    print(f"  Final collisions: {len(final_collisions)}")
    for collision in final_collisions:
        print(f"    - {collision['object_a']} <-> {collision['object_b']}: "
              f"overlap_depth={collision['overlap_depth']:.4f}m")
    
    # Verify room boundaries
    print("\nRoom boundary check:")
    for model in repaired_models:
        pose = model.get("Pose", {})
        x, y = pose.get("x", 0.0), pose.get("y", 0.0)
        print(f"  {model['name']}: x={x:.3f}, y={y:.3f}")
        assert abs(x) <= 5.0, f"{model['name']} outside room bounds (x={x})"
        assert abs(y) <= 5.0, f"{model['name']} outside room bounds (y={y})"
    
    # CRITICAL ASSERTION: No significant collisions should remain
    # (For infeasible scenarios, LLM fallback should be invoked)
    assert_no_significant_collisions(final_collisions)


# ============================================================================
# Run all tests
# ============================================================================

if __name__ == "__main__":
    print("\n" + "="*70)
    print("BUG CONDITION EXPLORATION TEST")
    print("Feature: object-collision-detection-fix")
    print("Property 1: Bug Condition - Same-Level Objects Overlap Detection")
    print("="*70)
    print("\nThis test MUST FAIL on unfixed code - failure confirms the bug exists.")
    print("\nRunning test cases...")
    
    try:
        test_two_boxes_on_table_no_overlap()
    except AssertionError as e:
        print(f"\n✗ TEST FAILED: {e}")
        print("\nThis failure confirms the bug exists in unfixed code.")
    
    try:
        test_three_apples_in_box_no_overlap()
    except AssertionError as e:
        print(f"\n✗ TEST FAILED: {e}")
        print("\nThis failure confirms the bug exists in unfixed code.")
    
    try:
        test_dense_floor_placement_no_overlap()
    except AssertionError as e:
        print(f"\n✗ TEST FAILED: {e}")
        print("\nThis failure confirms the bug exists in unfixed code.")
    
    try:
        test_small_surface_infeasibility()
    except AssertionError as e:
        print(f"\n✗ TEST FAILED: {e}")
        print("\nThis failure confirms the bug exists in unfixed code.")
    
    print("\n" + "="*70)
    print("BUG CONDITION EXPLORATION TEST COMPLETE")
    print("="*70)
