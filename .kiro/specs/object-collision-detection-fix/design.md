# Object Collision Detection Fix - Bugfix Design

## Overview

This design addresses the collision detection bug where objects placed on the same surface (e.g., two boxes on a table) intersect/overlap with each other. The current implementation has gaps in the collision detection and resolution pipeline that allow objects at the same Z-level to penetrate each other, violating physical realism.

The fix uses a two-phase approach:
1. **Algorithmic Resolution**: Use OBB (Oriented Bounding Box) collision detection with iterative gradient-based overlap resolution
2. **LLM Fallback**: When algorithmic methods fail (too many objects, small surface, convergence issues), invoke LLM to determine optimal placement

This follows the universal mechanism pattern: algorithmic methods as primary solution, LLM as intelligent fallback.

## Glossary

- **Bug_Condition (C)**: Objects at the same Z-level (same surface) that have overlapping OBBs after placement
- **Property (P)**: Objects at the same Z-level should have non-overlapping OBBs with minimum clearance margin
- **Preservation**: Existing behavior for objects at different Z-levels (stacked objects), semantic constraints, and room boundaries must remain unchanged
- **OBB (Oriented Bounding Box)**: 2D bounding box in XY plane that accounts for object rotation (yaw)
- **SAT (Separating Axis Theorem)**: Algorithm for detecting overlap between oriented bounding boxes
- **gradient_resolve_overlaps()**: Function in `geometry.py` that iteratively separates overlapping objects using gradient descent
- **check_collisions()**: Function in `physics.py` that detects collisions between objects using OBB+SAT
- **validate_and_repair_layout()**: Function in `physics.py` that orchestrates collision detection and resolution
- **Same Z-level**: Objects whose Z-intervals overlap (determined by `_z_intervals_overlap()`)
- **LLM Fallback**: Universal mechanism where LLM is invoked when algorithmic methods cannot find a solution

## Bug Details

### Bug Condition

The bug manifests when multiple objects are placed on the same surface (same Z-level) and their OBBs overlap in the XY plane. The collision detection system either fails to detect these overlaps or the gradient-based resolution fails to separate them within the iteration limit.

**Formal Specification:**
```
FUNCTION isBugCondition(input)
  INPUT: input of type PlacementLayout (list of placed objects)
  OUTPUT: boolean
  
  FOR each pair (obj_i, obj_j) in input WHERE i < j DO
    IF _z_intervals_overlap(obj_i, obj_j) THEN
      obb_i := model_to_obb(obj_i, inflation=collision_margin)
      obb_j := model_to_obb(obj_j, inflation=collision_margin)
      
      IF obb_overlap(obb_i, obb_j, margin=0.0) THEN
        RETURN true  // Bug condition detected
      END IF
    END IF
  END FOR
  
  RETURN false  // No overlaps at same Z-level
END FUNCTION
```

### Examples

**Example 1: Two boxes on table**
- Input: Table at z=0.4m, Box1 at (0.5, 0.0, 0.9), Box2 at (0.6, 0.0, 0.9)
- Box sizes: 0.3m × 0.3m × 0.3m each
- Expected: Boxes should be separated by at least collision_margin (0.01m)
- Actual: Boxes overlap by ~0.1m in X direction

**Example 2: Three apples in box**
- Input: Box at z=0.9m, Apple1 at (0.5, 0.0, 1.1), Apple2 at (0.52, 0.0, 1.1), Apple3 at (0.48, 0.0, 1.1)
- Apple sizes: 0.08m diameter each
- Expected: Apples should be separated with clearance
- Actual: All three apples overlap at center of box

**Example 3: Multiple objects on floor**
- Input: 5 objects placed on floor (z ≈ half-height) in middle region
- Expected: Objects distributed without overlap
- Actual: Objects cluster and overlap in center

**Edge Case: Small surface with many objects**
- Input: Small table (0.6m × 0.6m) with 4 boxes (0.3m × 0.3m each)
- Expected: System should detect infeasibility and invoke LLM fallback
- Actual: Gradient resolution runs for max iterations without convergence, produces overlapping layout

## Expected Behavior

### Preservation Requirements

**Unchanged Behaviors:**
- Objects at different Z-levels (stacked objects) must NOT be separated in XY plane
- Semantic constraints (near, on_top_of, inside, region) must continue to be satisfied
- Objects must remain within room boundaries after collision resolution
- Anchored objects (with on_top_of constraints) must preserve their surface-relative positions
- The `_z_intervals_overlap()` function must continue to correctly identify objects at same vs different levels

**Scope:**
All inputs that do NOT involve objects at the same Z-level should be completely unaffected by this fix. This includes:
- Stacked objects (box on table, table on floor) - no XY separation
- Objects with large Z-separation (floor objects vs table-top objects)
- Single object placements (no collision possible)
- Empty scenes or scenes with only one object per surface

## Hypothesized Root Cause

Based on the bug description and code analysis, the most likely issues are:

1. **Incomplete Collision Detection**: The `check_collisions()` function correctly implements OBB+SAT detection, but may not be called at the right points in the pipeline, or its results may be ignored

2. **Gradient Resolution Convergence Failure**: The `gradient_resolve_overlaps()` function has a fixed iteration limit (repair_iters * 20 = 160 iterations by default). For dense placements or small surfaces, this may be insufficient to fully separate all objects

3. **Anchored Objects Blocking Resolution**: Objects with `on_top_of` constraints are marked as anchored and excluded from gradient updates. If multiple anchored objects overlap, they cannot be separated by the gradient method

4. **Missing LLM Fallback**: When algorithmic resolution fails (convergence not reached, too many objects, small surface), there is no fallback mechanism to invoke LLM for replanning the layout

5. **Incorrect Collision Margin**: The collision_margin parameter may be too small (0.01m) for the gradient step size (0.06m), causing objects to be pushed apart but still within collision distance

## Correctness Properties

Property 1: Bug Condition - Same-Level Objects Non-Overlapping

_For any_ placement layout where objects are at the same Z-level (isBugCondition returns true before fix), the fixed collision resolution system SHALL detect all OBB overlaps and either (a) successfully separate the objects using gradient-based resolution with non-overlapping OBBs and minimum clearance margin, or (b) invoke LLM fallback to replan the layout when algorithmic resolution fails to converge.

**Validates: Requirements 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8**

Property 2: Preservation - Different-Level Objects Unchanged

_For any_ placement layout where objects are at different Z-levels (stacked objects where isBugCondition returns false), the fixed collision resolution system SHALL produce exactly the same XY positions as the original system, preserving the stacking relationships and semantic constraints without applying XY separation.

**Validates: Requirements 3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 3.7**

## Fix Implementation

### Changes Required

Assuming our root cause analysis is correct:

**File**: `creator/placement/physics.py`

**Function**: `validate_and_repair_layout()`

**Specific Changes**:
1. **Add Convergence Detection**: After calling `gradient_resolve_overlaps()`, check if overlaps still exist using `check_collisions()`
   - If collisions remain after max iterations, mark as convergence failure
   - Log the number of remaining collisions and their details

2. **Add LLM Fallback Invocation**: When convergence fails, invoke LLM to replan layout
   - Pass context: list of objects with sizes, current positions, semantic constraints, room boundaries
   - Request: "Reposition these objects to eliminate overlaps while preserving constraints"
   - Validate LLM response and apply new positions

3. **Add Iteration Limit Tuning**: Make iteration limit adaptive based on scene complexity
   - Base iterations: repair_iters * 20
   - Add bonus iterations: +50 per object beyond 5 objects
   - Cap at 500 iterations to prevent infinite loops

4. **Improve Collision Margin**: Increase collision_margin from 0.01m to 0.02m to provide better clearance buffer

5. **Add Fallback Detection Heuristics**: Detect infeasible scenarios early
   - Calculate total object footprint area vs available surface area
   - If ratio > 0.8, invoke LLM fallback immediately (skip gradient resolution)

**File**: `creator/placement/geometry.py`

**Function**: `gradient_resolve_overlaps()`

**Specific Changes**:
1. **Add Convergence Tracking**: Track overlap count per iteration
   - If overlap count doesn't decrease for 20 consecutive iterations, return early with convergence failure flag
   - Return tuple: (resolved_models, converged: bool)

2. **Improve Gradient Scaling**: Adjust gradient scale based on overlap depth
   - For deep overlaps (depth > 0.1m), use larger step multiplier (6.0 instead of 4.0)
   - For shallow overlaps (depth < 0.02m), use smaller step multiplier (2.0) for fine-tuning

3. **Add Anchored Object Handling**: When both objects in a collision are anchored, mark as unresolvable
   - Track unresolvable collisions separately
   - If any unresolvable collisions exist, return convergence failure

**File**: `creator/placement/universal_system.py` (or new file `creator/placement/llm_fallback.py`)

**Function**: New function `llm_replan_layout()`

**Specific Changes**:
1. **Create LLM Fallback Function**: Implement LLM-based layout replanning
   - Input: placed_models, semantic_plan, room_half_size, collision_details
   - Prompt: Structured prompt with object list, constraints, collision info
   - Output: New positions for objects (validated for feasibility)

2. **Add Validation**: Validate LLM response before applying
   - Check all objects have valid positions
   - Check positions are within room boundaries
   - Check semantic constraints are still satisfied
   - If validation fails, retry with modified prompt (up to 3 attempts)

3. **Add Fallback Logging**: Log when LLM fallback is invoked
   - Log reason: convergence failure, infeasible density, unresolvable anchored collisions
   - Log LLM response and validation results
   - Log final collision status after applying LLM positions

## Testing Strategy

### Validation Approach

The testing strategy follows a two-phase approach: first, surface counterexamples that demonstrate the bug on unfixed code, then verify the fix works correctly and preserves existing behavior.

### Exploratory Bug Condition Checking

**Goal**: Surface counterexamples that demonstrate the bug BEFORE implementing the fix. Confirm or refute the root cause analysis. If we refute, we will need to re-hypothesize.

**Test Plan**: Create test scenes with known collision scenarios and run collision detection on UNFIXED code. Observe failures and measure overlap depths to understand the root cause.

**Test Cases**:
1. **Two Boxes on Table Test**: Place two 0.3m boxes at (0.5, 0.0, 0.9) and (0.6, 0.0, 0.9) on a table at z=0.4m (will fail on unfixed code - boxes overlap by ~0.1m)
2. **Three Apples in Box Test**: Place three 0.08m apples at (0.5, 0.0, 1.1), (0.52, 0.0, 1.1), (0.48, 0.0, 1.1) in a box (will fail on unfixed code - all apples overlap)
3. **Dense Floor Placement Test**: Place 8 objects on floor in 2m × 2m region (will fail on unfixed code - multiple overlaps)
4. **Small Surface Infeasibility Test**: Place 4 boxes (0.3m each) on small table (0.6m × 0.6m) (will fail on unfixed code - gradient resolution doesn't converge)

**Expected Counterexamples**:
- `check_collisions()` returns non-empty list of collisions after `gradient_resolve_overlaps()`
- Overlap depths range from 0.05m to 0.15m for typical cases
- Possible causes: insufficient iterations, anchored objects blocking resolution, missing LLM fallback

### Fix Checking

**Goal**: Verify that for all inputs where the bug condition holds, the fixed function produces the expected behavior.

**Pseudocode:**
```
FOR ALL layout WHERE isBugCondition(layout) DO
  result := validate_and_repair_layout_fixed(layout)
  collisions := check_collisions(result)
  
  ASSERT collisions is empty OR llm_fallback_was_invoked
  ASSERT all objects remain within room boundaries
  ASSERT semantic constraints are satisfied
END FOR
```

**Test Implementation**:
- Use property-based testing to generate random layouts with known collisions
- Apply fixed `validate_and_repair_layout()`
- Verify no collisions remain (or LLM fallback was invoked)
- Verify all preservation requirements are met

### Preservation Checking

**Goal**: Verify that for all inputs where the bug condition does NOT hold, the fixed function produces the same result as the original function.

**Pseudocode:**
```
FOR ALL layout WHERE NOT isBugCondition(layout) DO
  result_original := validate_and_repair_layout_original(layout)
  result_fixed := validate_and_repair_layout_fixed(layout)
  
  ASSERT positions_equal(result_original, result_fixed, tolerance=0.001)
END FOR
```

**Testing Approach**: Property-based testing is recommended for preservation checking because:
- It generates many test cases automatically across the input domain
- It catches edge cases that manual unit tests might miss
- It provides strong guarantees that behavior is unchanged for all non-buggy inputs

**Test Plan**: Observe behavior on UNFIXED code first for stacked objects and different-level placements, then write property-based tests capturing that behavior.

**Test Cases**:
1. **Stacked Objects Preservation**: Observe that box-on-table placement keeps box centered on table on unfixed code, then write test to verify this continues after fix
2. **Different Z-Level Preservation**: Observe that floor objects and table-top objects are not separated in XY on unfixed code, then write test to verify this continues after fix
3. **Semantic Constraints Preservation**: Observe that "near", "on_top_of", "region:middle" constraints are satisfied on unfixed code, then write test to verify this continues after fix
4. **Room Boundaries Preservation**: Observe that objects stay within room boundaries on unfixed code, then write test to verify this continues after fix

### Unit Tests

- Test `check_collisions()` with known overlapping and non-overlapping OBB pairs
- Test `gradient_resolve_overlaps()` convergence detection with various overlap scenarios
- Test `_z_intervals_overlap()` with objects at same and different Z-levels
- Test LLM fallback invocation conditions (convergence failure, infeasible density)
- Test LLM response validation (valid positions, within bounds, constraints satisfied)

### Property-Based Tests

- Generate random layouts with 2-10 objects at same Z-level, verify all collisions are resolved
- Generate random layouts with stacked objects (different Z-levels), verify XY positions unchanged
- Generate random semantic constraints, verify they remain satisfied after collision resolution
- Generate random room sizes and object counts, verify objects stay within boundaries

### Integration Tests

- Test full pipeline: semantic plan → layout solving → collision resolution → validation
- Test LLM fallback integration: trigger fallback conditions and verify LLM is invoked
- Test edge cases: empty scene, single object, all objects at different Z-levels
- Test performance: measure time for gradient resolution vs LLM fallback on various scene complexities
