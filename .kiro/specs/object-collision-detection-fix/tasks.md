# Implementation Plan

- [x] 1. Write bug condition exploration test
  - **Property 1: Bug Condition** - Same-Level Objects Overlap Detection
  - **CRITICAL**: This test MUST FAIL on unfixed code - failure confirms the bug exists
  - **DO NOT attempt to fix the test or the code when it fails**
  - **NOTE**: This test encodes the expected behavior - it will validate the fix when it passes after implementation
  - **GOAL**: Surface counterexamples that demonstrate objects at same Z-level overlap
  - **Scoped PBT Approach**: Test concrete failing cases (two boxes on table, three apples in box, dense floor placement, small surface infeasibility)
  - Test implementation details from Bug Condition in design:
    - For each pair of objects at same Z-level (using `_z_intervals_overlap()`), check if their OBBs overlap using `obb_overlap()`
    - Test cases: Two boxes on table (overlap ~0.1m), three apples in box (all overlap), 8 objects on floor (multiple overlaps), 4 boxes on small table (convergence failure)
  - The test assertions should match the Expected Behavior Properties from design:
    - Assert that `check_collisions()` returns empty list (no overlaps) OR LLM fallback was invoked
    - Assert all objects remain within room boundaries
    - Assert semantic constraints are satisfied
  - Run test on UNFIXED code
  - **EXPECTED OUTCOME**: Test FAILS (this is correct - it proves the bug exists)
  - Document counterexamples found: overlap depths, which object pairs collide, whether gradient resolution converged
  - Mark task complete when test is written, run, and failure is documented
  - _Requirements: 1.1, 1.2, 1.3, 1.4, 2.1, 2.2, 2.3, 2.4, 2.5_

- [x] 2. Write preservation property tests (BEFORE implementing fix)
  - **Property 2: Preservation** - Different-Level Objects Unchanged
  - **IMPORTANT**: Follow observation-first methodology
  - Observe behavior on UNFIXED code for non-buggy inputs (objects at different Z-levels):
    - Stacked objects (box on table): observe that box stays centered on table in XY plane
    - Different Z-level objects (floor objects vs table-top objects): observe no XY separation applied
    - Semantic constraints ("near", "on_top_of", "region:middle"): observe constraints are satisfied
    - Room boundaries: observe objects stay within room boundaries
  - Write property-based tests capturing observed behavior patterns from Preservation Requirements:
    - For all layouts with stacked objects (different Z-levels), XY positions should remain unchanged (tolerance 0.001m)
    - For all layouts with semantic constraints, constraints should remain satisfied after collision resolution
    - For all layouts, objects should remain within room boundaries after collision resolution
  - Property-based testing generates many test cases for stronger guarantees
  - Run tests on UNFIXED code
  - **EXPECTED OUTCOME**: Tests PASS (this confirms baseline behavior to preserve)
  - Mark task complete when tests are written, run, and passing on unfixed code
  - _Requirements: 3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 3.7_

- [x] 3. Fix for object collision detection between same-level objects

  - [x] 3.1 Implement convergence detection in `validate_and_repair_layout()` (physics.py)
    - After calling `gradient_resolve_overlaps()`, check if overlaps still exist using `check_collisions()`
    - If collisions remain after max iterations, mark as convergence failure
    - Log the number of remaining collisions and their details
    - _Bug_Condition: isBugCondition(input) - objects at same Z-level with overlapping OBBs_
    - _Expected_Behavior: collisions is empty OR llm_fallback_was_invoked_
    - _Preservation: Objects at different Z-levels unchanged (3.1), semantic constraints satisfied (3.2), room boundaries respected (3.3)_
    - _Requirements: 2.1, 2.2, 2.5_

  - [x] 3.2 Implement LLM fallback invocation in `validate_and_repair_layout()` (physics.py)
    - When convergence fails, invoke LLM to replan layout
    - Pass context: list of objects with sizes, current positions, semantic constraints, room boundaries
    - Request: "Reposition these objects to eliminate overlaps while preserving constraints"
    - Validate LLM response and apply new positions
    - _Bug_Condition: isBugCondition(input) AND gradient resolution failed to converge_
    - _Expected_Behavior: LLM provides valid non-overlapping positions_
    - _Preservation: Semantic constraints satisfied (3.2), room boundaries respected (3.3), LLM only used as fallback (3.6)_
    - _Requirements: 2.6, 2.7, 2.8_

  - [x] 3.3 Implement adaptive iteration limits in `validate_and_repair_layout()` (physics.py)
    - Base iterations: repair_iters * 20
    - Add bonus iterations: +50 per object beyond 5 objects
    - Cap at 500 iterations to prevent infinite loops
    - _Bug_Condition: isBugCondition(input) with many objects_
    - _Expected_Behavior: More iterations for complex scenes, convergence more likely_
    - _Preservation: Existing iteration behavior for simple scenes unchanged_
    - _Requirements: 2.4, 2.5_

  - [x] 3.4 Improve collision margin in `validate_and_repair_layout()` (physics.py)
    - Increase collision_margin from 0.01m to 0.02m
    - Provides better clearance buffer between objects
    - _Bug_Condition: isBugCondition(input) with shallow overlaps_
    - _Expected_Behavior: Objects separated with minimum clearance margin_
    - _Preservation: Existing collision detection logic unchanged (3.7)_
    - _Requirements: 2.2, 2.3_

  - [x] 3.5 Implement early infeasibility detection in `validate_and_repair_layout()` (physics.py)
    - Calculate total object footprint area vs available surface area
    - If ratio > 0.8, invoke LLM fallback immediately (skip gradient resolution)
    - _Bug_Condition: isBugCondition(input) with high object density_
    - _Expected_Behavior: LLM fallback invoked early for infeasible scenarios_
    - _Preservation: LLM only used as fallback (3.6)_
    - _Requirements: 2.6_

  - [x] 3.6 Implement convergence tracking in `gradient_resolve_overlaps()` (geometry.py)
    - Track overlap count per iteration
    - If overlap count doesn't decrease for 20 consecutive iterations, return early with convergence failure flag
    - Return tuple: (resolved_models, converged: bool)
    - _Bug_Condition: isBugCondition(input) with gradient resolution stuck_
    - _Expected_Behavior: Early detection of convergence failure_
    - _Preservation: Existing gradient resolution logic unchanged for converging cases_
    - _Requirements: 2.4, 2.5_

  - [x] 3.7 Improve gradient scaling in `gradient_resolve_overlaps()` (geometry.py)
    - For deep overlaps (depth > 0.1m), use larger step multiplier (6.0 instead of 4.0)
    - For shallow overlaps (depth < 0.02m), use smaller step multiplier (2.0) for fine-tuning
    - _Bug_Condition: isBugCondition(input) with varying overlap depths_
    - _Expected_Behavior: Faster convergence for deep overlaps, more precise for shallow overlaps_
    - _Preservation: Existing gradient scaling logic improved, not replaced_
    - _Requirements: 2.4_

  - [x] 3.8 Implement anchored object handling in `gradient_resolve_overlaps()` (geometry.py)
    - When both objects in a collision are anchored, mark as unresolvable
    - Track unresolvable collisions separately
    - If any unresolvable collisions exist, return convergence failure
    - _Bug_Condition: isBugCondition(input) with anchored objects overlapping_
    - _Expected_Behavior: Convergence failure triggers LLM fallback_
    - _Preservation: Anchored objects continue to respect on_top_of constraints (3.2)_
    - _Requirements: 2.4, 2.5, 2.6_

  - [x] 3.9 Create LLM fallback function `llm_replan_layout()` (universal_system.py or new file)
    - Input: placed_models, semantic_plan, room_half_size, collision_details
    - Prompt: Structured prompt with object list, constraints, collision info
    - Output: New positions for objects (validated for feasibility)
    - Validation: Check all objects have valid positions, within room boundaries, semantic constraints satisfied
    - If validation fails, retry with modified prompt (up to 3 attempts)
    - Add logging: reason for invocation, LLM response, validation results, final collision status
    - _Bug_Condition: isBugCondition(input) AND gradient resolution failed_
    - _Expected_Behavior: LLM provides valid non-overlapping positions_
    - _Preservation: Semantic constraints satisfied (3.2), room boundaries respected (3.3)_
    - _Requirements: 2.6, 2.7, 2.8_

  - [x] 3.10 Verify bug condition exploration test now passes
    - **Property 1: Expected Behavior** - Same-Level Objects Non-Overlapping
    - **IMPORTANT**: Re-run the SAME test from task 1 - do NOT write a new test
    - The test from task 1 encodes the expected behavior
    - When this test passes, it confirms the expected behavior is satisfied
    - Run bug condition exploration test from step 1
    - **EXPECTED OUTCOME**: Test PASSES (confirms bug is fixed)
    - Verify: `check_collisions()` returns empty list OR LLM fallback was invoked
    - Verify: All objects remain within room boundaries
    - Verify: Semantic constraints are satisfied
    - _Requirements: 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8_

  - [x] 3.11 Verify preservation tests still pass
    - **Property 2: Preservation** - Different-Level Objects Unchanged
    - **IMPORTANT**: Re-run the SAME tests from task 2 - do NOT write new tests
    - Run preservation property tests from step 2
    - **EXPECTED OUTCOME**: Tests PASS (confirms no regressions)
    - Verify: Stacked objects (different Z-levels) have unchanged XY positions
    - Verify: Semantic constraints remain satisfied
    - Verify: Objects remain within room boundaries
    - Confirm all tests still pass after fix (no regressions)
    - _Requirements: 3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 3.7_

- [x] 4. Checkpoint - Ensure all tests pass
  - Run full test suite including bug condition test, preservation tests, unit tests, and integration tests
  - Verify all tests pass
  - If any test fails, investigate and fix before proceeding
  - Ask the user if questions arise
