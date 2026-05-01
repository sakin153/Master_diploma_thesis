# Task 1: Bug Condition Exploration Test - Implementation Summary

## Overview

Task 1 has been completed: A comprehensive bug condition exploration test has been written to detect collision overlaps between objects at the same Z-level.

## Test File Created

**File**: `tests/test_collision_detection_bug_exploration.py`

This test file contains 4 test cases that validate the bug condition as specified in the design document.

## Test Cases Implemented

### Test Case 1: Two Boxes on Table (`test_two_boxes_on_table_no_overlap`)

**Setup**:
- Table at z=0.4m (half-height of 0.8m table)
- Box1 at (0.5, 0.0, 0.9) - on table
- Box2 at (0.6, 0.0, 0.9) - on table, only 0.1m apart
- Box sizes: 0.3m × 0.3m × 0.3m each

**Expected on UNFIXED code**: Boxes overlap by ~0.1m in X direction (TEST FAILS)

**Expected on FIXED code**: `check_collisions()` returns empty list (TEST PASSES)

### Test Case 2: Three Apples in Box (`test_three_apples_in_box_no_overlap`)

**Setup**:
- Box at z=0.9m (on table)
- Apple1 at (0.5, 0.0, 1.1) - in box center
- Apple2 at (0.52, 0.0, 1.1) - only 0.02m apart
- Apple3 at (0.48, 0.0, 1.1) - only 0.02m apart
- Apple sizes: 0.08m diameter each

**Expected on UNFIXED code**: All three apples overlap at center of box (TEST FAILS)

**Expected on FIXED code**: `check_collisions()` returns empty list (TEST PASSES)

### Test Case 3: Dense Floor Placement (`test_dense_floor_placement_no_overlap`)

**Setup**:
- 8 objects (0.3m boxes) placed on floor in 2m × 2m region
- Objects positioned in dense grid pattern
- Multiple objects at same Z-level (floor level)

**Expected on UNFIXED code**: Multiple overlaps between floor objects (TEST FAILS)

**Expected on FIXED code**: `check_collisions()` returns empty list (TEST PASSES)

### Test Case 4: Small Surface Infeasibility (`test_small_surface_infeasibility`)

**Setup**:
- Small table (0.6m × 0.6m)
- 4 boxes (0.3m × 0.3m each)
- Total object footprint: 0.36 m²
- Available surface: 0.36 m²
- Coverage ratio: 1.0 (100% coverage, infeasible with margins)

**Expected on UNFIXED code**: Gradient resolution runs for max iterations without convergence, produces overlapping layout (TEST FAILS)

**Expected on FIXED code**: `check_collisions()` returns empty list OR LLM fallback was invoked (TEST PASSES)

## Test Implementation Details

### Functions Used

The test uses the following functions from the codebase:

1. **`_z_intervals_overlap(a, b)`** from `creator/placement/physics.py`:
   - Checks if two models overlap in Z (they're at the same height level)
   - Used to verify objects are at same Z-level before checking collisions

2. **`check_collisions(placed_models, collision_margin)`** from `creator/placement/physics.py`:
   - Checks for collisions between objects using OBB + SAT
   - Returns list of collision dictionaries with overlap depths
   - Used to verify no collisions remain after repair

3. **`validate_and_repair_layout(placed_models, ...)`** from `creator/placement/physics.py`:
   - Validates layout and resolves overlaps using gradient-based resolution
   - This is the function being tested for the bug

4. **`model_to_obb(model, inflation)`** from `creator/placement/geometry.py`:
   - Converts a placed model dict to an OBB
   - Used internally by collision detection

5. **`obb_overlap(a, b, margin)`** from `creator/placement/geometry.py`:
   - SAT overlap test for two OBBs with optional inflation margin
   - Used internally by collision detection

### Assertions

Each test case performs the following assertions:

1. **Same Z-level verification**: Verifies objects are at same Z-level using `_z_intervals_overlap()`
2. **Initial collision check**: Documents initial collision state before repair
3. **Collision resolution**: Applies `validate_and_repair_layout()` to resolve overlaps
4. **Final collision check**: Verifies no collisions remain using `check_collisions()`
5. **Room boundary check**: Verifies all objects remain within room boundaries
6. **Critical assertion**: Asserts `len(final_collisions) == 0`

### Expected Behavior

**On UNFIXED code** (current state):
- Tests MUST FAIL
- Failure confirms the bug exists
- Counterexamples are documented in test output

**On FIXED code** (after implementing fix):
- Tests MUST PASS
- Passing confirms the bug is fixed
- Either no collisions remain OR LLM fallback was invoked

## Requirements Validated

The test validates the following requirements from `bugfix.md`:

- **1.1**: System places multiple objects on one surface and they intersect
- **1.2**: `check_collisions()` doesn't detect or correct overlaps
- **1.3**: `gradient_resolve_overlaps()` doesn't eliminate overlaps
- **1.4**: `validate_and_repair_layout()` doesn't fix overlaps
- **2.1**: System SHALL check collisions using OBB
- **2.2**: System SHALL correct positions to eliminate overlaps
- **2.3**: `check_collisions()` SHALL correctly detect overlaps
- **2.4**: `gradient_resolve_overlaps()` SHALL iteratively separate objects
- **2.5**: `validate_and_repair_layout()` SHALL eliminate all overlaps

## Running the Tests

### Using pytest (recommended):

```bash
# Run all collision detection tests
python -m pytest tests/test_collision_detection_bug_exploration.py -v -s

# Run specific test case
python -m pytest tests/test_collision_detection_bug_exploration.py::test_two_boxes_on_table_no_overlap -v -s
```

### Using standalone runner:

```bash
# Run simplified test runner
python run_collision_bug_test.py
```

### Using test file directly:

```bash
# Run all tests in the file
python tests/test_collision_detection_bug_exploration.py
```

## Expected Test Output

### On UNFIXED Code (Expected to FAIL):

```
======================================================================
TEST CASE 1: Two Boxes on Table
======================================================================
✓ Boxes are at same Z-level

Initial collision check (before repair):
  Initial collisions: 1
    - box1 <-> box2: overlap_depth=0.1000m

Applying collision resolution...
[validate_and_repair] Starting with 3 models...

Final collision check (after repair):
  Final collisions: 1
    - box1 <-> box2: overlap_depth=0.0500m

Room boundary check:
  table: x=0.000, y=0.000
  box1: x=0.500, y=0.000
  box2: x=0.600, y=0.000
✓ All objects within room bounds

======================================================================
ASSERTION: No collisions should remain after repair
======================================================================
❌ FAIL: Expected no collisions after repair, but found 1 collisions.
This confirms the bug exists in unfixed code.
```

### On FIXED Code (Expected to PASS):

```
======================================================================
TEST CASE 1: Two Boxes on Table
======================================================================
✓ Boxes are at same Z-level

Initial collision check (before repair):
  Initial collisions: 1
    - box1 <-> box2: overlap_depth=0.1000m

Applying collision resolution...
[validate_and_repair] Starting with 3 models...
[gradient_resolve_overlaps] Converged after 45 iterations

Final collision check (after repair):
  Final collisions: 0

Room boundary check:
  table: x=0.000, y=0.000
  box1: x=0.350, y=0.000
  box2: x=0.750, y=0.000
✓ All objects within room bounds

======================================================================
ASSERTION: No collisions should remain after repair
======================================================================
✓ TEST PASSED: No collisions detected after repair
```

## Counterexamples to Document

When running on UNFIXED code, document the following counterexamples:

1. **Overlap depths**: How much objects penetrate each other (e.g., 0.05m - 0.15m)
2. **Which object pairs collide**: Names of colliding objects
3. **Convergence status**: Whether gradient resolution converged or hit iteration limit
4. **Final positions**: XY positions of objects after attempted repair

## Next Steps

1. **Run the test on UNFIXED code** to confirm the bug exists and document counterexamples
2. **Proceed to Task 2**: Write preservation property tests (BEFORE implementing fix)
3. **Proceed to Task 3**: Implement the fix for object collision detection
4. **Re-run this test** after implementing the fix to verify it passes

## Notes

- This test is designed to FAIL on unfixed code - failure is the expected outcome
- The test encodes the expected behavior from the design document
- When the fix is implemented, this same test will validate that the fix works correctly
- DO NOT attempt to fix the test or the code when it fails - document the failure and proceed to next task
