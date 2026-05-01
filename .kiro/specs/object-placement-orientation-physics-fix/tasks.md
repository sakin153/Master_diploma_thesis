# Implementation Plan

## Overview

This implementation plan fixes two critical bugs in the Universal Placement System:
1. **Orientation Bug**: Objects with `region:edge` constraints face walls instead of room center
2. **Physics Bug**: Small objects are incorrectly marked as static instead of dynamic

The root cause is data loss in the pipeline: size and is_static metadata are not preserved when converting from semantic_plan → scene_graph → layout_solver → floor_solver.

---

## Tasks

- [x] 1. Write bug condition exploration test
  - **Property 1: Bug Condition** - Size and Physics Metadata Loss in Pipeline
  - **CRITICAL**: This test MUST FAIL on unfixed code - failure confirms the bug exists
  - **DO NOT attempt to fix the test or the code when it fails**
  - **NOTE**: This test encodes the expected behavior - it will validate the fix when it passes after implementation
  - **GOAL**: Surface counterexamples that demonstrate metadata is lost during pipeline data flow
  - **Scoped PBT Approach**: Scope the property to concrete failing cases - objects with size in semantic_plan but missing size in objects_to_place
  - Test implementation details from Bug Condition in design:
    - Generate semantic_plan with objects that have size and is_static defined
    - Trace through _build_scene_graph() and verify scene_graph nodes have size in metadata
    - Trace through _solve_with_dfs() and verify objects_to_place includes size and is_static
    - Assert that size exists in semantic_plan.objects[i].size
    - Assert that size exists in scene_graph.nodes[i].metadata.size (will fail on unfixed code)
    - Assert that size exists in objects_to_place[i]["size"] (will fail on unfixed code)
    - Assert that is_static exists in objects_to_place[i]["is_static"] (will fail on unfixed code)
  - The test assertions should match the Expected Behavior Properties from design:
    - Property 1: Size and physics metadata SHALL be preserved through entire pipeline
    - For edge objects: verify yaw != 0° (face-center calculation triggered)
    - For small objects: verify is_static == False
  - Run test on UNFIXED code
  - **EXPECTED OUTCOME**: Test FAILS (this is correct - it proves the bug exists)
  - Document counterexamples found:
    - scene_graph nodes missing size field in metadata
    - objects_to_place list missing size and is_static fields
    - Edge objects placed with yaw=0° instead of face-center angles
    - Small objects marked as is_static=true instead of false
  - Mark task complete when test is written, run, and failure is documented
  - _Requirements: 2.1, 2.2, 2.3, 2.4, 2.5, 2.6_

- [x] 2. Write preservation property tests (BEFORE implementing fix)
  - **Property 2: Preservation** - Non-Edge Object and Large Furniture Behavior
  - **IMPORTANT**: Follow observation-first methodology
  - Observe behavior on UNFIXED code for non-buggy inputs:
    - Objects without region:edge constraint (interior placement)
    - Large objects with volume ≥ 0.1 m³ (furniture)
    - Various constraint types (near, on, inside, beside)
  - Write property-based tests capturing observed behavior patterns from Preservation Requirements:
    - For non-edge objects: verify yaw candidates remain [0°, 90°, 180°, 270°]
    - For large furniture: verify is_static remains true
    - For other constraints: verify constraint resolution unchanged
    - For export format: verify MuJoCo XML structure unchanged
  - Property-based testing generates many test cases for stronger guarantees
  - Test cases:
    - Generate scenes with objects without region:edge constraint
    - Generate scenes with large furniture (volume ≥ 0.1 m³)
    - Generate scenes with various constraint combinations
    - Verify placement results match observed patterns
  - Run tests on UNFIXED code
  - **EXPECTED OUTCOME**: Tests PASS (this confirms baseline behavior to preserve)
  - Mark task complete when tests are written, run, and passing on unfixed code
  - _Requirements: 3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 3.7, 3.8_

- [x] 3. Fix metadata preservation in Universal Placement System

  - [x] 3.1 Fix _build_scene_graph() to preserve size in scene_graph nodes
    - Modify `_build_scene_graph()` in `creator/placement/universal_system.py`
    - After creating scene_graph nodes from parsed_plan, look up size from model_sizes dict
    - Add size to node.metadata: `node.metadata["size"] = model_sizes.get(model_name, [1.0, 1.0, 1.0])`
    - Ensure size is available for downstream layout_solver
    - _Bug_Condition: isBugCondition(input) where semantic_plan has size but scene_graph.nodes[i].metadata.size NOT EXISTS_
    - _Expected_Behavior: scene_graph.nodes[i].metadata.size SHALL exist and match semantic_plan size_
    - _Preservation: Non-edge objects and large furniture placement logic unchanged_
    - _Requirements: 2.1, 2.2, 2.3, 3.5_

  - [x] 3.2 Fix _build_scene_graph() to preserve is_static flag
    - Modify `_build_scene_graph()` in `creator/placement/universal_system.py`
    - When creating nodes, copy is_static from semantic_plan objects
    - Add to node.metadata: `node.metadata["is_static"] = obj.get("is_static", True)`
    - Verify flag is preserved through pipeline
    - _Bug_Condition: isBugCondition(input) where semantic_plan has is_static but scene_graph.nodes[i].metadata.is_static NOT EXISTS_
    - _Expected_Behavior: scene_graph.nodes[i].metadata.is_static SHALL exist and match semantic_plan is_static_
    - _Preservation: Large furniture remains static (is_static=true)_
    - _Requirements: 2.4, 2.5, 2.6, 3.3, 3.4_

  - [x] 3.3 Fix _solve_with_dfs() to include size in objects_to_place
    - Modify `_solve_with_dfs()` in `creator/placement/layout_solver.py`
    - Verify that `obj_dict = dict(node.metadata)` includes size field from node.metadata
    - If size is missing, log warning and use default [1.0, 1.0, 1.0]
    - Ensure complete metadata is passed to floor_solver
    - _Bug_Condition: isBugCondition(input) where scene_graph has size but objects_to_place[i]["size"] NOT EXISTS_
    - _Expected_Behavior: objects_to_place[i]["size"] SHALL exist and enable face-center yaw calculation_
    - _Preservation: DFS + beam search placement algorithm unchanged_
    - _Requirements: 2.1, 2.2, 2.3, 3.5_

  - [x] 3.4 Fix _solve_with_dfs() to include is_static in objects_to_place
    - Modify `_solve_with_dfs()` in `creator/placement/layout_solver.py`
    - Copy is_static from node.metadata to obj_dict
    - Ensure floor_solver receives physics information
    - _Bug_Condition: isBugCondition(input) where scene_graph has is_static but objects_to_place[i]["is_static"] NOT EXISTS_
    - _Expected_Behavior: objects_to_place[i]["is_static"] SHALL exist and control physics behavior_
    - _Preservation: Large furniture physics behavior unchanged_
    - _Requirements: 2.4, 2.5, 2.6, 3.3, 3.4_

  - [x] 3.5 Verify orientation logic invocation in floor_solver
    - Check that `_get_yaw_candidates_for_position()` in `creator/placement/floor_solver.py` is called when size is available
    - Verify edge objects trigger face-center yaw calculation
    - Add logging to show yaw candidates for debugging
    - Test with "гостиная" prompt to verify sofas face center
    - _Bug_Condition: isBugCondition(input) where edge objects have size but yaw=0° is used_
    - _Expected_Behavior: Edge objects SHALL use face-center yaw from _compute_face_center_yaw(x, y)_
    - _Preservation: Non-edge objects continue using default yaw candidates_
    - _Requirements: 2.1, 2.2, 2.3, 3.1, 3.2_

  - [x] 3.6 Verify bug condition exploration test now passes
    - **Property 1: Expected Behavior** - Size and Physics Metadata Preserved
    - **IMPORTANT**: Re-run the SAME test from task 1 - do NOT write a new test
    - The test from task 1 encodes the expected behavior
    - When this test passes, it confirms the expected behavior is satisfied
    - Run bug condition exploration test from step 1
    - Verify assertions pass:
      - scene_graph.nodes[i].metadata.size exists
      - objects_to_place[i]["size"] exists
      - objects_to_place[i]["is_static"] exists
      - Edge objects have yaw != 0° (face-center calculation triggered)
      - Small objects have is_static == False
    - **EXPECTED OUTCOME**: Test PASSES (confirms bug is fixed)
    - _Requirements: 2.1, 2.2, 2.3, 2.4, 2.5, 2.6_

  - [x] 3.7 Verify preservation tests still pass
    - **Property 2: Preservation** - Non-Edge Object and Large Furniture Behavior
    - **IMPORTANT**: Re-run the SAME tests from task 2 - do NOT write new tests
    - Run preservation property tests from step 2
    - Verify all preservation tests pass:
      - Non-edge objects use default yaw candidates
      - Large furniture remains static
      - Constraint resolution unchanged
      - Export format unchanged
    - **EXPECTED OUTCOME**: Tests PASS (confirms no regressions)
    - Confirm all tests still pass after fix (no regressions)

- [x] 4. Checkpoint - Ensure all tests pass
  - Run complete test suite including:
    - Bug condition exploration test (should now pass)
    - Preservation property tests (should still pass)
    - Unit tests for metadata preservation
    - Integration tests with "гостиная" and "яблоки в коробках" prompts
  - Verify test results:
    - All exploration tests pass (bug is fixed)
    - All preservation tests pass (no regressions)
    - Integration tests show correct orientations and physics
  - If any tests fail, investigate and fix before proceeding
  - Ask the user if questions arise or if manual verification is needed
