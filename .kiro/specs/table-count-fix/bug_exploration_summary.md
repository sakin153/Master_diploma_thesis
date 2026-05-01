# Bug Condition Exploration Test Summary

## Test File
`tests/test_table_count_bug_exploration.py`

## Test Cases Implemented

### Test Case 1: 4 Tables in 2x2 Grid
- **Input**: 4 × wooden_patterned_table
- **Query**: "4 tables in 2x2 grid"
- **Expected Bug**: LLM returns only 3 tables instead of 4
- **Root Cause**: LLM prompt does not emphasize including ALL N instances

### Test Case 2: 4 Tables and 16 Chairs
- **Input**: 4 × wooden_patterned_table, 16 × computer_chair
- **Query**: "4 tables and 4 chairs around each table"
- **Expected Bugs**:
  1. LLM may return fewer than 4 tables or 16 chairs
  2. Chairs may have `region: middle` constraint instead of `beside: table_N`
  3. Chairs may not be evenly distributed (4 per table)
- **Root Cause**: LLM prompt does not emphasize explicit distribution of chairs to tables

### Test Case 3: 4 Chairs Around Table (Edge Case)
- **Input**: 1 × wooden_patterned_table, 4 × computer_chair
- **Query**: "4 chairs around table"
- **Expected Bug**: LLM may return only 3 chairs (edge case at threshold of 4)
- **Root Cause**: Threshold behavior - exactly 4 objects may trigger the bug

### Test Case 4: 8 Tables (Large Count)
- **Input**: 8 × wooden_patterned_table
- **Query**: "8 tables"
- **Expected Bug**: LLM may return only 6-7 tables
- **Root Cause**: Larger counts make the bug more likely to occur

## Test Methodology

The tests call the LLM directly using `call_llm_for_semantic_plan()` which:
1. Formats the prompt using `_build_models_str()` and `fmt_constraints_plan_tmpl`
2. Calls `prompt_model()` to get LLM response
3. Returns the RAW LLM output BEFORE `_ensure_plan_covers_chosen_models()` adds missing objects

This allows us to detect the bug at its source (LLM generation) rather than after the fallback mechanism masks it.

## Expected Counterexamples

### Counterexample 1: Missing Tables
```
Input: 4 × table
LLM Output: 3 × table (missing table_4)
```

### Counterexample 2: Wrong Chair Constraints
```
Input: 4 × table, 16 × chair
LLM Output: Chairs have {"type": "region", "value": "middle"} instead of {"type": "beside", "target": "table_N"}
```

### Counterexample 3: Uneven Distribution
```
Input: 4 × table, 16 × chair
LLM Output: table_1 has 6 chairs, table_2 has 5 chairs, table_3 has 3 chairs, table_4 has 2 chairs
Expected: Each table should have 4 chairs
```

## Test Status

**Status**: Test written and ready to run

**Expected Outcome on UNFIXED Code**: FAIL (confirms bug exists)

**Expected Outcome on FIXED Code**: PASS (confirms bug is fixed)

## Next Steps

1. Run tests on unfixed code to document actual counterexamples
2. Implement fixes according to design.md
3. Re-run tests to verify fixes work
4. Update PBT status based on test results
