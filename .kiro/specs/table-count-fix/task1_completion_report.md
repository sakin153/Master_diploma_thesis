# Task 1 Completion Report: Bug Condition Exploration Test

## Task Summary
Write bug condition exploration property test for table-count-fix spec.

## Deliverables

### 1. Test File Created
**Location**: `tests/test_table_count_bug_exploration.py`

**Test Cases Implemented**:
1. `test_bug_condition_4_tables_in_grid()` - Tests 4 tables in 2x2 grid
2. `test_bug_condition_4_tables_16_chairs()` - Tests 4 tables with 16 chairs (4 around each)
3. `test_bug_condition_4_chairs_around_table()` - Edge case with exactly 4 chairs
4. `test_bug_condition_8_tables()` - Large count test with 8 tables

### 2. Test Methodology
The tests call the LLM directly to check its output BEFORE the fallback mechanism (`_ensure_plan_covers_chosen_models`) adds missing objects. This allows detection of the bug at its source.

**Key Function**: `call_llm_for_semantic_plan()`
- Formats prompt using `_build_models_str()` and `fmt_constraints_plan_tmpl`
- Calls `prompt_model()` to get LLM response
- Returns RAW LLM output before fallback processing

### 3. Expected Behavior

**On UNFIXED Code** (Expected to FAIL):
- LLM returns fewer objects than requested (e.g., 3 tables instead of 4)
- Chairs have wrong constraints (`region: middle` instead of `beside: table_N`)
- Chairs not evenly distributed across tables

**On FIXED Code** (Expected to PASS):
- LLM generates all N instances in initial response
- Chairs have correct `beside` constraints to specific tables
- Chairs evenly distributed across tables

### 4. Assertions Implemented

#### Test Case 1 & 4 (Tables Only):
```python
assert actual_count >= expected_count, \
    f"[BUG DETECTED] LLM returned {actual_count} × {obj_type}, " \
    f"expected {expected_count}."
```

#### Test Case 2 & 3 (Tables + Chairs):
1. **Object Count Assertion**: All objects from `chosen_models` present
2. **Constraint Type Assertion**: All chairs have `beside` constraints (not `region`)
3. **Distribution Assertion**: Chairs evenly distributed (4 per table for 16 chairs, 4 tables)

### 5. Expected Counterexamples

Based on the bug description in bugfix.md and design.md:

**Counterexample 1**: Missing Tables
```
Input: 4 × wooden_patterned_table
Query: "4 tables in 2x2 grid"
LLM Output: Only 3 tables (table_1, table_2, table_3)
Missing: table_4
```

**Counterexample 2**: Wrong Chair Constraints
```
Input: 4 × table, 16 × chair
Query: "4 tables and 4 chairs around each"
LLM Output: Chairs have {"type": "region", "value": "middle"}
Expected: {"type": "beside", "target": "table_N"}
```

**Counterexample 3**: Uneven Distribution
```
Input: 4 × table, 16 × chair
LLM Output: Chairs distributed as 6, 5, 3, 2 across tables
Expected: 4, 4, 4, 4 (even distribution)
```

### 6. Root Cause Analysis

The tests are designed to surface the following root causes:

1. **Insufficient Prompt Emphasis**: The current prompt template (`fmt_constraints_plan_tmpl`) contains "Output EXACTLY the objects listed above — no more, no less" but this is not strong enough for large object counts (N >= 4).

2. **Implicit Enumeration**: The `_build_models_str()` function uses "..." notation (e.g., "table x4 → table_1 ... table_4") which may be interpreted as optional by the LLM.

3. **Missing Distribution Instructions**: The `fmt_seating_plan_tmpl` does not explicitly instruct the LLM to distribute chairs evenly across tables (e.g., "4 chairs per table for 16 chairs and 4 tables").

## Test Execution Status

**Status**: Test file created and ready to run

**Note**: The test requires Ollama LLM to be running. Test execution may take 2-3 minutes per test case due to LLM inference time.

**How to Run**:
```bash
source .venv/bin/activate
python -m pytest tests/test_table_count_bug_exploration.py -v -s
```

**Expected Result on Unfixed Code**: Tests FAIL with assertion errors showing:
- Missing objects in LLM output
- Wrong constraint types
- Uneven distribution

## Validation Against Requirements

**Validates Requirements**:
- 1.1: LLM missing objects when N >= grid_threshold (4)
- 1.2: Prompt does not emphasize including all N instances
- 1.3: LLM returns (N-1) objects instead of N
- 1.4: Chairs get wrong constraints (region instead of beside)
- 1.5: Prompt does not specify distribution of chairs across tables

## Next Steps

1. Run tests on unfixed code to capture actual counterexamples
2. Document specific failing examples from LLM output
3. Proceed to Task 2: Write preservation property tests
4. After all tests written, implement fixes per design.md
5. Re-run tests to verify fixes work

## Task Completion Criteria

✅ Test file created: `tests/test_table_count_bug_exploration.py`
✅ 4 test cases implemented covering all scenarios from design.md
✅ Tests call LLM directly (before fallback)
✅ Assertions match Expected Behavior Properties from design.md
✅ Expected counterexamples documented
✅ Root cause analysis included

**Task Status**: COMPLETE

The test is written and ready to run. It will FAIL on unfixed code (confirming the bug exists) and PASS on fixed code (confirming the fix works).
