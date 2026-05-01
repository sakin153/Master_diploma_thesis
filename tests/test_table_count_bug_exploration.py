"""
Bug Condition Exploration Test for Table Count Fix

This test MUST FAIL on unfixed code - failure confirms the bug exists.
DO NOT attempt to fix the test or the code when it fails.

The test encodes the expected behavior - it will validate the fix when it passes after implementation.
"""

import pytest
from collections import Counter
from creator.placement.plan import build_semantic_plan, _normalize_name, _safe_str


def _count_objects_in_plan(plan, object_type):
    """Count how many instances of object_type are in the plan."""
    count = 0
    for obj in plan.get("objects", []):
        model = _safe_str(obj.get("Model"))
        # Normalize to base name (remove _N suffix)
        import re
        base = re.sub(r'_\d+$', '', model)
        if base.lower() == object_type.lower():
            count += 1
    return count


def _get_chair_constraints(plan):
    """Extract all chair constraints from the plan."""
    chair_constraints = []
    for obj in plan.get("objects", []):
        model = _safe_str(obj.get("Model"))
        import re
        base = re.sub(r'_\d+$', '', model)
        if "chair" in base.lower():
            chair_constraints.append({
                "model": model,
                "constraints": obj.get("constraints", [])
            })
    return chair_constraints


class MockPromptModel:
    """Mock LLM that simulates the bug: returns fewer objects than requested."""
    
    def __init__(self, bug_mode=True):
        self.bug_mode = bug_mode
        self.call_count = 0
    
    def __call__(self, content, query, *args, **kwargs):
        self.call_count += 1
        
        # Simulate the bug: LLM returns fewer objects than requested
        if self.bug_mode and "table x4" in content:
            # Bug: Return only 3 tables instead of 4
            return {
                "objects": [
                    {"Model": "table_1", "is_static": True, "constraints": [{"type": "region", "value": "middle"}]},
                    {"Model": "table_2", "is_static": True, "constraints": [{"type": "beside", "target": "table_1", "side": "left"}]},
                    {"Model": "table_3", "is_static": True, "constraints": [{"type": "beside", "target": "table_1", "side": "right"}]},
                    # table_4 is MISSING - this is the bug
                ]
            }
        
        if self.bug_mode and "table x8" in content:
            # Bug: Return only 6 tables instead of 8
            return {
                "objects": [
                    {"Model": "table_1", "is_static": True, "constraints": [{"type": "region", "value": "middle"}]},
                    {"Model": "table_2", "is_static": True, "constraints": [{"type": "beside", "target": "table_1"}]},
                    {"Model": "table_3", "is_static": True, "constraints": [{"type": "beside", "target": "table_1"}]},
                    {"Model": "table_4", "is_static": True, "constraints": [{"type": "beside", "target": "table_1"}]},
                    {"Model": "table_5", "is_static": True, "constraints": [{"type": "beside", "target": "table_1"}]},
                    {"Model": "table_6", "is_static": True, "constraints": [{"type": "beside", "target": "table_1"}]},
                    # table_7 and table_8 are MISSING
                ]
            }
        
        # For chairs, simulate wrong constraints (region instead of beside)
        if self.bug_mode and "chair" in content.lower():
            # Bug: Chairs get region:middle instead of beside:table_N
            return {
                "objects": [
                    {"Model": "computer chair_1", "is_static": True, "constraints": [{"type": "region", "value": "middle"}]},
                    {"Model": "computer chair_2", "is_static": True, "constraints": [{"type": "region", "value": "middle"}]},
                    {"Model": "computer chair_3", "is_static": True, "constraints": [{"type": "region", "value": "middle"}]},
                    {"Model": "computer chair_4", "is_static": True, "constraints": [{"type": "region", "value": "middle"}]},
                ]
            }
        
        # Default fallback
        return {"objects": []}


# Reduced test cases for faster execution
@pytest.mark.parametrize("test_case", [
    {
        "name": "4_tables",
        "query": "4 tables in 2x2 grid",
        "chosen_models": [
            {"Model": "table"}, {"Model": "table"}, {"Model": "table"}, {"Model": "table"}
        ],
        "expected_count": 4,
        "object_type": "table"
    },
    {
        "name": "8_tables",
        "query": "8 tables",
        "chosen_models": [{"Model": "table"}] * 8,
        "expected_count": 8,
        "object_type": "table"
    },
])
def test_bug_condition_llm_missing_objects(test_case):
    """
    Property 1: Bug Condition - LLM Missing Objects in Semantic Plan
    
    EXPECTED OUTCOME: Test FAILS on unfixed code (proves bug exists)
    
    This test checks that for object types with count >= 4, the LLM plan
    contains exactly that count. On unfixed code, the LLM may return fewer
    objects than requested.
    """
    mock_llm = MockPromptModel(bug_mode=True)
    
    plan = build_semantic_plan(
        prompt_model=mock_llm,
        prompt_template="{models_str}\n\n{query}",
        query=test_case["query"],
        chosen_model="table",
        chosen_models=test_case["chosen_models"],
        context_models=[],
    )
    
    actual_count = _count_objects_in_plan(plan, test_case["object_type"])
    
    # ASSERTION: For object types with count >= 4, plan must contain exactly that count
    assert actual_count == test_case["expected_count"], (
        f"Bug detected: Expected {test_case['expected_count']} {test_case['object_type']}s, "
        f"but LLM returned {actual_count}. "
        f"This confirms the bug exists in the unfixed code."
    )


def test_bug_condition_chairs_around_tables():
    """
    Property 1: Bug Condition - Chairs with Wrong Constraints
    
    EXPECTED OUTCOME: Test FAILS on unfixed code (proves bug exists)
    
    This test checks that chairs around tables have 'beside' constraints
    targeting specific tables, not 'region:middle' constraints.
    """
    mock_llm = MockPromptModel(bug_mode=True)
    
    chosen_models = (
        [{"Model": "table"}] * 4 +
        [{"Model": "computer chair"}] * 16
    )
    
    plan = build_semantic_plan(
        prompt_model=mock_llm,
        prompt_template="{models_str}\n\n{query}",
        query="4 tables and 4 chairs around each",
        chosen_model="table",
        chosen_models=chosen_models,
        context_models=[],
    )
    
    chair_constraints = _get_chair_constraints(plan)
    
    # ASSERTION: Each chair should have 'beside' constraint targeting a specific table
    chairs_with_beside = 0
    chairs_with_region = 0
    
    for chair_data in chair_constraints:
        has_beside = False
        has_region = False
        
        for constraint in chair_data["constraints"]:
            if isinstance(constraint, dict):
                ctype = _safe_str(constraint.get("type")).lower()
                if ctype == "beside":
                    has_beside = True
                    target = _safe_str(constraint.get("target"))
                    # Check that target is a specific table (table_1, table_2, etc.)
                    assert "table" in target.lower(), (
                        f"Chair {chair_data['model']} has beside constraint but target is not a table: {target}"
                    )
                elif ctype == "region":
                    has_region = True
        
        if has_beside:
            chairs_with_beside += 1
        if has_region:
            chairs_with_region += 1
    
    # ASSERTION: All chairs should have beside constraints, not region constraints
    assert chairs_with_beside == len(chair_constraints), (
        f"Bug detected: Only {chairs_with_beside}/{len(chair_constraints)} chairs have 'beside' constraints. "
        f"{chairs_with_region} chairs have 'region' constraints instead. "
        f"This confirms the bug exists in the unfixed code."
    )


if __name__ == "__main__":
    print("Running bug condition exploration tests...")
    print("EXPECTED: Tests should FAIL on unfixed code (this proves the bug exists)")
    print()
    
    # Run test case 1: 4 tables
    try:
        test_bug_condition_llm_missing_objects({
            "name": "4_tables",
            "query": "4 tables in 2x2 grid",
            "chosen_models": [{"Model": "table"}] * 4,
            "expected_count": 4,
            "object_type": "table"
        })
        print("✓ Test 1 (4 tables) PASSED")
    except AssertionError as e:
        print(f"✗ Test 1 (4 tables) FAILED: {e}")
    
    # Run test case 2: 8 tables
    try:
        test_bug_condition_llm_missing_objects({
            "name": "8_tables",
            "query": "8 tables",
            "chosen_models": [{"Model": "table"}] * 8,
            "expected_count": 8,
            "object_type": "table"
        })
        print("✓ Test 2 (8 tables) PASSED")
    except AssertionError as e:
        print(f"✗ Test 2 (8 tables) FAILED: {e}")
    
    # Run test case 3: chairs around tables
    try:
        test_bug_condition_chairs_around_tables()
        print("✓ Test 3 (chairs around tables) PASSED")
    except AssertionError as e:
        print(f"✗ Test 3 (chairs around tables) FAILED: {e}")
