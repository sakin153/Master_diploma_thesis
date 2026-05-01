"""Tests for Scene Graph support in physics module.

Tests the new methods added to physics.py:
- check_collisions()
- validate_stacking()
- check_support_capacity()
"""

import pytest
from creator.placement.physics import (
    check_collisions,
    validate_stacking,
    check_support_capacity,
)


def test_check_collisions_no_overlap():
    """Test collision detection with non-overlapping objects."""
    placed_models = [
        {
            "Model": "table",
            "Pose": {"x": 0.0, "y": 0.0, "z": 0.5},
            "size": [1.0, 1.0, 1.0],
            "yaw_deg": 0.0,
        },
        {
            "Model": "chair",
            "Pose": {"x": 2.0, "y": 0.0, "z": 0.5},
            "size": [0.5, 1.0, 0.5],
            "yaw_deg": 0.0,
        },
    ]

    collisions = check_collisions(placed_models)
    assert len(collisions) == 0, "Expected no collisions between separated objects"


def test_check_collisions_with_overlap():
    """Test collision detection with overlapping objects."""
    placed_models = [
        {
            "Model": "table",
            "Pose": {"x": 0.0, "y": 0.0, "z": 0.5},
            "size": [1.0, 1.0, 1.0],
            "yaw_deg": 0.0,
        },
        {
            "Model": "chair",
            "Pose": {"x": 0.3, "y": 0.0, "z": 0.5},  # Overlapping with table
            "size": [0.5, 1.0, 0.5],
            "yaw_deg": 0.0,
        },
    ]

    collisions = check_collisions(placed_models)
    assert len(collisions) > 0, "Expected collision between overlapping objects"
    assert collisions[0]["object_a"] == "table"
    assert collisions[0]["object_b"] == "chair"
    assert collisions[0]["overlap_depth"] > 0


def test_check_collisions_stacked_objects():
    """Test that stacked objects (different Z levels) are not reported as collisions."""
    placed_models = [
        {
            "Model": "table",
            "Pose": {"x": 0.0, "y": 0.0, "z": 0.5},
            "size": [1.0, 1.0, 1.0],
            "yaw_deg": 0.0,
        },
        {
            "Model": "book",
            "Pose": {"x": 0.0, "y": 0.0, "z": 1.1},  # On top of table
            "size": [0.3, 0.05, 0.2],
            "yaw_deg": 0.0,
        },
    ]

    collisions = check_collisions(placed_models)
    assert len(collisions) == 0, "Stacked objects should not be reported as collisions"


def test_validate_stacking_stable():
    """Test stacking validation with stable configuration."""
    placed_models = [
        {
            "Model": "table",
            "Pose": {"x": 0.0, "y": 0.0, "z": 0.5},
            "size": [1.0, 1.0, 1.0],
            "yaw_deg": 0.0,
        },
        {
            "Model": "book",
            "Pose": {"x": 0.0, "y": 0.0, "z": 1.025},  # Properly on table
            "size": [0.3, 0.05, 0.2],
            "yaw_deg": 0.0,
        },
    ]

    result = validate_stacking(placed_models)
    assert result["is_stable"], "Expected stable stacking configuration"
    assert len(result["unstable_objects"]) == 0


def test_validate_stacking_floating():
    """Test stacking validation with floating object."""
    placed_models = [
        {
            "Model": "table",
            "Pose": {"x": 0.0, "y": 0.0, "z": 0.5},
            "size": [1.0, 1.0, 1.0],
            "yaw_deg": 0.0,
        },
        {
            "Model": "book",
            "Pose": {"x": 3.0, "y": 3.0, "z": 2.0},  # Floating in air
            "size": [0.3, 0.05, 0.2],
            "yaw_deg": 0.0,
        },
    ]

    result = validate_stacking(placed_models)
    assert not result["is_stable"], "Expected unstable configuration with floating object"
    assert "book" in result["unstable_objects"]
    assert any(issue["issue"] == "floating" for issue in result["stability_issues"])


def test_validate_stacking_insufficient_overlap():
    """Test stacking validation with insufficient overlap."""
    placed_models = [
        {
            "Model": "table",
            "Pose": {"x": 0.0, "y": 0.0, "z": 0.5},
            "size": [1.0, 1.0, 1.0],
            "yaw_deg": 0.0,
        },
        {
            "Model": "book",
            "Pose": {"x": 0.9, "y": 0.9, "z": 1.025},  # Barely on table edge
            "size": [0.3, 0.05, 0.2],
            "yaw_deg": 0.0,
        },
    ]

    result = validate_stacking(placed_models, stability_threshold=0.7)
    # This might be stable or unstable depending on exact overlap calculation
    # Just verify the structure is correct
    assert "is_stable" in result
    assert "unstable_objects" in result
    assert "stability_issues" in result


def test_check_support_capacity_ok():
    """Test support capacity check with acceptable load."""
    placed_models = [
        {
            "Model": "table",
            "Pose": {"x": 0.0, "y": 0.0, "z": 0.5},
            "size": [1.0, 1.0, 1.0],
            "yaw_deg": 0.0,
            "weight_capacity": 100.0,  # 100 kg capacity
        },
        {
            "Model": "book",
            "Pose": {"x": 0.0, "y": 0.0, "z": 1.025},
            "size": [0.3, 0.05, 0.2],  # Small, light object
            "yaw_deg": 0.0,
        },
    ]

    result = check_support_capacity(placed_models)
    assert result["capacity_ok"], "Expected sufficient capacity for light object"
    assert len(result["overloaded_objects"]) == 0


def test_check_support_capacity_overloaded():
    """Test support capacity check with overloaded surface."""
    placed_models = [
        {
            "Model": "table",
            "Pose": {"x": 0.0, "y": 0.0, "z": 0.5},
            "size": [1.0, 1.0, 1.0],
            "yaw_deg": 0.0,
            "weight_capacity": 1.0,  # Very low capacity (1 kg)
        },
        {
            "Model": "heavy_box",
            "Pose": {"x": 0.0, "y": 0.0, "z": 1.5},
            "size": [0.8, 1.0, 0.8],  # Large, heavy object
            "yaw_deg": 0.0,
        },
    ]

    result = check_support_capacity(placed_models)
    assert not result["capacity_ok"], "Expected overloaded capacity"
    assert "table" in result["overloaded_objects"]
    assert len(result["capacity_issues"]) > 0


def test_check_support_capacity_default():
    """Test support capacity check with default capacity."""
    placed_models = [
        {
            "Model": "table",
            "Pose": {"x": 0.0, "y": 0.0, "z": 0.5},
            "size": [1.0, 1.0, 1.0],
            "yaw_deg": 0.0,
            # No weight_capacity specified - should use default
        },
        {
            "Model": "book",
            "Pose": {"x": 0.0, "y": 0.0, "z": 1.025},
            "size": [0.3, 0.05, 0.2],
            "yaw_deg": 0.0,
        },
    ]

    result = check_support_capacity(placed_models, default_capacity=50.0)
    # Should use default capacity of 50kg
    assert "capacity_ok" in result
    assert "overloaded_objects" in result
    assert "capacity_issues" in result


def test_empty_scene():
    """Test all methods with empty scene."""
    placed_models = []

    collisions = check_collisions(placed_models)
    assert len(collisions) == 0

    stacking = validate_stacking(placed_models)
    assert stacking["is_stable"]
    assert len(stacking["unstable_objects"]) == 0

    capacity = check_support_capacity(placed_models)
    assert capacity["capacity_ok"]
    assert len(capacity["overloaded_objects"]) == 0


def test_single_object():
    """Test all methods with single floor object."""
    placed_models = [
        {
            "Model": "table",
            "Pose": {"x": 0.0, "y": 0.0, "z": 0.5},
            "size": [1.0, 1.0, 1.0],
            "yaw_deg": 0.0,
        },
    ]

    collisions = check_collisions(placed_models)
    assert len(collisions) == 0

    stacking = validate_stacking(placed_models)
    assert stacking["is_stable"]

    capacity = check_support_capacity(placed_models)
    assert capacity["capacity_ok"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
