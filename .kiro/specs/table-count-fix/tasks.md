# Implementation Plan

- [x] 1. Write bug condition exploration test
  - **Property 1: Bug Condition** - LLM Missing Objects in Semantic Plan
  - **CRITICAL**: This test MUST FAIL on unfixed code - failure confirms the bug exists
  - **DO NOT attempt to fix the test or the code when it fails**
  - **NOTE**: This test encodes the expected behavior - it will validate the fix when it passes after implementation
  - **GOAL**: Surface counterexamples that demonstrate the bug exists
  - **Scoped PBT Approach**: For deterministic bugs, scope the property to the concrete failing case(s) to ensure reproducibility
  - Test implementation details from Bug Condition in design:
    - Test case 1: Request "4 tables in 2x2 grid" with `chosen_models` = 4 × "table"
    - Test case 2: Request "4 tables and 4 chairs around each" with `chosen_models` = 4 × "table", 16 × "chair"
    - Test case 3: Request "4 chairs around table" with `chosen_models` = 1 × "table", 4 × "chair"
    - Test case 4: Request "8 tables" with `chosen_models` = 8 × "table"
  - The test assertions should match the Expected Behavior Properties from design:
    - ASSERT: For all object types in `chosen_models` with count >= 4, LLM plan contains exactly that count
    - ASSERT: For "chairs around tables" requests, each chair has `beside` constraint targeting specific table
    - ASSERT: Chairs are evenly distributed across tables (e.g., 4 chairs per table for 16 chairs and 4 tables)
  - Run test on UNFIXED code
  - **EXPECTED OUTCOME**: Test FAILS (this is correct - it proves the bug exists)
  - Document counterexamples found to understand root cause:
    - Example: LLM returns only 3 tables instead of 4
    - Example: Chairs have `region: middle` constraint instead of `beside: table_N`
  - Mark task complete when test is written, run, and failure is documented
  - _Requirements: 1.1, 1.2, 1.3, 1.4, 1.5_

- [x] 2. Write preservation property tests (BEFORE implementing fix)
  - **Property 2: Preservation** - Small Object Count and Existing Behavior
  - **IMPORTANT**: Follow observation-first methodology
  - Observe behavior on UNFIXED code for non-buggy inputs:
    - Test case 1: Requests with 1-3 objects (< grid_threshold)
    - Test case 2: Objects with explicit positional constraints (beside, face_to, on_top_of)
    - Test case 3: Requests without "around" keyword
    - Test case 4: `_normalize_name` function behavior with suffixes
  - Write property-based tests capturing observed behavior patterns from Preservation Requirements:
    - Property: For all requests with object count < 4, system uses beam search (not grid)
    - Property: For all plans where all `chosen_models` are present, `_ensure_plan_covers_chosen_models` returns plan unchanged
    - Property: For all objects with explicit positional constraints, system uses beam search
    - Property: For all requests without "around" keyword, system uses standard grid/beam placement
    - Property: For all model names with `_N` suffix, `_normalize_name` correctly groups instances
  - Property-based testing generates many test cases for stronger guarantees
  - Run tests on UNFIXED code
  - **EXPECTED OUTCOME**: Tests PASS (this confirms baseline behavior to preserve)
  - Mark task complete when tests are written, run, and passing on unfixed code
  - _Requirements: 3.1, 3.2, 3.3, 3.4, 3.5_

- [x] 3. Fix for LLM missing objects in semantic plan

  - [x] 3.1 Strengthen prompt instructions in `creator/contexts_prompts/constraints.py` (fmt_constraints_plan_tmpl)
    - Replace "Output EXACTLY the objects listed above — no more, no less" with explicit instruction:
      ```
      CRITICAL RULE: You MUST include ALL N instances of each object type shown above.
      - If the list shows "table x4 → table_1, table_2, table_3, table_4", your output MUST contain all 4 tables
      - If the list shows "chair x16 → chair_1 ... chair_16", your output MUST contain all 16 chairs
      - Missing even one instance is a critical error
      ```
    - Add verification checklist before JSON example:
      ```
      Before generating JSON, verify:
      ✓ Count of each object type in your output matches the count shown in "Objects (N total)" above
      ✓ All instances are numbered sequentially: object_1, object_2, ..., object_N
      ```
    - _Bug_Condition: isBugCondition(input) where LLM returns fewer objects than in chosen_models for count >= 4_
    - _Expected_Behavior: LLM generates semantic plan with all N instances for any object type with count >= 4_
    - _Preservation: Small object count (< 4) behavior unchanged, existing constraint handling unchanged_
    - _Requirements: 2.1, 2.2, 2.3_

  - [x] 3.2 Improve `_build_models_str` in `creator/placement/plan.py`
    - Change formatting logic to explicitly list all instances for count >= 4:
      ```python
      if count == 1:
          lines.append(f"  {name} x1  → {name}_1")
      elif count <= 3:
          # Small count: show all explicitly
          instances = ", ".join(f"{name}_{i}" for i in range(1, count + 1))
          lines.append(f"  {name} x{count}  → {instances}")
      else:
          # Large count: show first 2, last 2, and total
          lines.append(f"  {name} x{count}  → {name}_1, {name}_2, ..., {name}_{count-1}, {name}_{count}")
          lines.append(f"    (CRITICAL: ALL {count} instances must be included)")
      ```
    - _Bug_Condition: isBugCondition(input) where "..." notation may be interpreted as optional by LLM_
    - _Expected_Behavior: Explicit enumeration with critical reminder for large counts_
    - _Preservation: Small count formatting unchanged_
    - _Requirements: 2.2, 2.3_

  - [x] 3.3 Add explicit distribution in `creator/contexts_prompts/constraints.py` (fmt_seating_plan_tmpl)
    - Add distribution requirement section:
      ```
      Distribution requirement:
      - If there are N anchors and M children, distribute children evenly
      - Example: 4 tables and 16 chairs → 4 chairs per table
      - Explicitly assign: chair_1..chair_4 to table_1, chair_5..chair_8 to table_2, etc.
      ```
    - Expand JSON example with explicit distribution:
      ```json
      // Example: 2 tables, 8 chairs (4 per table)
      {
        "objects": [
          // Chairs for table_1
          {"Model": "chair", "constraints": [{"type": "beside", "target": "table_1", "side": "front", ...}]},
          {"Model": "chair", "constraints": [{"type": "beside", "target": "table_1", "side": "back", ...}]},
          {"Model": "chair", "constraints": [{"type": "beside", "target": "table_1", "side": "left", ...}]},
          {"Model": "chair", "constraints": [{"type": "beside", "target": "table_1", "side": "right", ...}]},
          // Chairs for table_2
          {"Model": "chair", "constraints": [{"type": "beside", "target": "table_2", "side": "front", ...}]},
          ...
        ]
      }
      ```
    - _Bug_Condition: isBugCondition(input) where chairs get region constraints instead of beside constraints_
    - _Expected_Behavior: Chairs explicitly distributed with beside constraints to specific tables_
    - _Preservation: Non-seating arrangements unchanged_
    - _Requirements: 2.4, 2.5_

  - [x] 3.4 Improve `_ensure_plan_covers_chosen_models` in `creator/placement/plan.py` (fallback mechanism)
    - Add explicit chair distribution logic:
      ```python
      # For chairs specifically, ensure even distribution across tables
      if "chair" in base.lower() or "stool" in base.lower():
          # Calculate target count per anchor
          target_per_anchor = len([m for m in chosen_models 
                                  if _normalize_name(_safe_str(m.get("Model"))) == base]) // len(targets)
          # Distribute to reach target_per_anchor for each anchor
      ```
    - Add post-validation in `_build_plan_single`:
      ```python
      # Validate that all chosen_models are present
      from collections import Counter
      expected_counts = Counter(_normalize_name(_safe_str(m.get("Model") or m.get("name"))) 
                                for m in chosen_models)
      actual_counts = Counter(_normalize_name(_safe_str(obj.get("Model"))) 
                             for obj in objects)
      
      for obj_type, expected_count in expected_counts.items():
          if actual_counts.get(obj_type, 0) < expected_count:
              # Log warning but continue - _ensure_plan_covers_chosen_models should have fixed this
              pass
      ```
    - _Bug_Condition: isBugCondition(input) where fallback doesn't correctly distribute missing objects_
    - _Expected_Behavior: Fallback correctly adds missing objects with proper distribution_
    - _Preservation: Existing fallback behavior for non-chair objects unchanged_
    - _Requirements: 2.6_

  - [x] 3.5 Verify bug condition exploration test now passes
    - **Property 1: Expected Behavior** - LLM Generates Complete Object List
    - **IMPORTANT**: Re-run the SAME test from task 1 - do NOT write a new test
    - The test from task 1 encodes the expected behavior
    - When this test passes, it confirms the expected behavior is satisfied
    - Run bug condition exploration test from step 1
    - **EXPECTED OUTCOME**: Test PASSES (confirms bug is fixed)
    - Verify all test cases pass:
      - Test case 1: "4 tables" → LLM returns all 4 tables
      - Test case 2: "4 tables and 4 chairs around each" → LLM returns 4 tables + 16 chairs with correct constraints
      - Test case 3: "4 chairs around table" → LLM returns 1 table + 4 chairs with beside constraints
      - Test case 4: "8 tables" → LLM returns all 8 tables
    - _Requirements: 2.1, 2.2, 2.3, 2.4, 2.5_

  - [x] 3.6 Verify preservation tests still pass
    - **Property 2: Preservation** - Small Object Count and Existing Behavior
    - **IMPORTANT**: Re-run the SAME tests from task 2 - do NOT write new tests
    - Run preservation property tests from step 2
    - **EXPECTED OUTCOME**: Tests PASS (confirms no regressions)
    - Confirm all preservation properties still hold:
      - Property: Requests with object count < 4 use beam search
      - Property: Plans with all objects present are unchanged by `_ensure_plan_covers_chosen_models`
      - Property: Objects with explicit positional constraints use beam search
      - Property: Requests without "around" keyword use standard placement
      - Property: `_normalize_name` correctly groups instances with `_N` suffix
    - _Requirements: 3.1, 3.2, 3.3, 3.4, 3.5_

- [x] 4. Checkpoint - Ensure all tests pass
  - Run all unit tests for modified functions:
    - Test `_build_models_str` with various configurations (1, 2, 4, 8, 16 objects)
    - Test `_ensure_plan_covers_chosen_models` for adding missing objects
    - Test prompt validation for presence of critical instructions
    - Test chair distribution around tables for correct constraints
  - Run all property-based tests:
    - Bug condition test (Property 1) - should PASS
    - Preservation test (Property 2) - should PASS
  - Run integration tests:
    - Full flow: request → model selection → plan generation → scene placement for "4 tables and 4 chairs around each"
    - Test beam search vs grid arrangement switching based on object count
    - Visual verification: generate scene and verify all objects placed correctly
  - Ensure all tests pass, ask the user if questions arise
