"""
Preservation Property Tests for Table Count Fix

These tests capture the baseline behavior that must be preserved after the fix.
Tests should PASS on unfixed code (confirms baseline behavior to preserve).
"""

import pytest
import re
from creator.placement.plan import _safe_str, _build_models_str


def _normalize_name(name: str) -> str:
    """Remove _N suffix from model names for grouping."""
    return re.sub(r'_\d+$', '', name)


def test_normalize_name_preserves_suffix_removal():
    """
    Property: For all model names with _N suffix, _normalize_name correctly groups instances
    
    EXPECTED OUTCOME: Test PASSES on unfixed code (confirms baseline behavior)
    """
    # Test cases
    assert _normalize_name("table_1") == "table"
    assert _normalize_name("table_2") == "table"
    assert _normalize_name("chair_10") == "chair"
    assert _normalize_name("computer chair_5") == "computer chair"
    assert _normalize_name("table") == "table"  # No suffix
    assert _normalize_name("my_table_3") == "my_table"  # Only removes trailing _N


def test_build_models_str_small_counts():
    """
    Property: For requests with object count < 4, formatting is explicit
    
    EXPECTED OUTCOME: Test PASSES on unfixed code (confirms baseline behavior)
    """
    # Test with 1 object
    models_1 = [{"Model": "table"}]
    result_1 = _build_models_str(models_1)
    assert "table x1" in result_1
    assert "table_1" in result_1
    
    # Test with 2 objects
    models_2 = [{"Model": "chair"}, {"Model": "chair"}]
    result_2 = _build_models_str(models_2)
    assert "chair x2" in result_2
    
    # Test with 3 objects
    models_3 = [{"Model": "table"}] * 3
    result_3 = _build_models_str(models_3)
    assert "table x3" in result_3


def test_build_models_str_large_counts():
    """
    Property: For requests with object count >= 4, formatting uses ellipsis
    
    EXPECTED OUTCOME: Test PASSES on unfixed code (confirms current behavior)
    """
    # Test with 4 objects
    models_4 = [{"Model": "table"}] * 4
    result_4 = _build_models_str(models_4)
    assert "table x4" in result_4
    assert "..." in result_4 or "table_1" in result_4  # Current behavior uses ...
    
    # Test with 8 objects
    models_8 = [{"Model": "chair"}] * 8
    result_8 = _build_models_str(models_8)
    assert "chair x8" in result_8


def test_build_models_str_mixed_types():
    """
    Property: For mixed object types, each type is listed separately
    
    EXPECTED OUTCOME: Test PASSES on unfixed code (confirms baseline behavior)
    """
    models = [
        {"Model": "table"},
        {"Model": "table"},
        {"Model": "chair"},
        {"Model": "chair"},
        {"Model": "chair"},
    ]
    result = _build_models_str(models)
    
    # Should list both types
    assert "table x2" in result
    assert "chair x3" in result
    
    # Should show total count
    assert "5 total" in result


if __name__ == "__main__":
    print("Running preservation property tests...")
    print("EXPECTED: Tests should PASS on unfixed code (confirms baseline behavior)")
    print()
    
    # Run all tests
    try:
        test_normalize_name_preserves_suffix_removal()
        print("✓ Test 1 (normalize_name) PASSED")
    except AssertionError as e:
        print(f"✗ Test 1 (normalize_name) FAILED: {e}")
    
    try:
        test_build_models_str_small_counts()
        print("✓ Test 2 (small counts) PASSED")
    except AssertionError as e:
        print(f"✗ Test 2 (small counts) FAILED: {e}")
    
    try:
        test_build_models_str_large_counts()
        print("✓ Test 3 (large counts) PASSED")
    except AssertionError as e:
        print(f"✗ Test 3 (large counts) FAILED: {e}")
    
    try:
        test_build_models_str_mixed_types()
        print("✓ Test 4 (mixed types) PASSED")
    except AssertionError as e:
        print(f"✗ Test 4 (mixed types) FAILED: {e}")
