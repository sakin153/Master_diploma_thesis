"""Unit tests for PipelineTracker.

Tests the pipeline tracking functionality for semantic plan enforcement.
"""

import json
import tempfile
from pathlib import Path

from creator.placement.semantic_enforcement import (
    PipelineTracker,
    SemanticObject,
    PlacedObject,
    Position,
    Orientation,
)


def test_log_stage_with_dicts():
    """Test logging stages with dictionary objects."""
    tracker = PipelineTracker(output_dir=tempfile.mkdtemp())
    
    objects = [
        {"id": "obj1", "Model": "table", "position": {"x": 0, "y": 0, "z": 0}},
        {"id": "obj2", "Model": "chair", "position": {"x": 1, "y": 0, "z": 0}},
    ]
    
    tracker.log_stage("test_stage", objects)
    
    assert len(tracker.stages) == 1
    assert tracker.stages[0].name == "test_stage"
    assert tracker.stages[0].object_count == 2


def test_log_stage_with_dataclasses():
    """Test logging stages with dataclass objects."""
    tracker = PipelineTracker(output_dir=tempfile.mkdtemp())
    
    objects = [
        SemanticObject(
            id="obj1",
            Model="table",
            type="furniture",
            size={"width": 1.0, "length": 1.0, "height": 0.75},
            is_static=True,
            model_loc="models/table.xml",
            position=Position(absolute={"x": 0, "y": 0, "z": 0}),
            orientation=Orientation(absolute={"yaw_deg": 0, "pitch_deg": 0, "roll_deg": 0}),
        ),
        PlacedObject(
            id="obj2",
            Model="chair",
            type="furniture",
            size={"width": 0.5, "length": 0.5, "height": 0.9},
            is_static=False,
            model_loc="models/chair.xml",
            position={"x": 1, "y": 0, "z": 0},
            orientation={"yaw_deg": 90, "pitch_deg": 0, "roll_deg": 0},
        ),
    ]
    
    tracker.log_stage("test_stage", objects)
    
    assert len(tracker.stages) == 1
    assert tracker.stages[0].name == "test_stage"
    assert tracker.stages[0].object_count == 2


def test_detect_changes_added_objects():
    """Test detection of added objects between stages."""
    tracker = PipelineTracker(output_dir=tempfile.mkdtemp())
    
    stage1_objects = [
        {"id": "obj1", "Model": "table"},
    ]
    
    stage2_objects = [
        {"id": "obj1", "Model": "table"},
        {"id": "obj2", "Model": "chair"},
    ]
    
    tracker.log_stage("stage1", stage1_objects)
    tracker.log_stage("stage2", stage2_objects)
    
    change = tracker.detect_changes("stage1", "stage2")
    
    assert change is not None
    assert len(change.added) == 1
    assert "obj2" in change.added
    assert len(change.removed) == 0


def test_detect_changes_removed_objects():
    """Test detection of removed objects between stages."""
    tracker = PipelineTracker(output_dir=tempfile.mkdtemp())
    
    stage1_objects = [
        {"id": "obj1", "Model": "table"},
        {"id": "obj2", "Model": "chair"},
    ]
    
    stage2_objects = [
        {"id": "obj1", "Model": "table"},
    ]
    
    tracker.log_stage("stage1", stage1_objects)
    tracker.log_stage("stage2", stage2_objects)
    
    change = tracker.detect_changes("stage1", "stage2")
    
    assert change is not None
    assert len(change.removed) == 1
    assert "obj2" in change.removed
    assert len(change.added) == 0


def test_detect_changes_modified_objects():
    """Test detection of modified objects between stages."""
    tracker = PipelineTracker(output_dir=tempfile.mkdtemp())
    
    stage1_objects = [
        {"id": "obj1", "Model": "table", "position": {"x": 0, "y": 0, "z": 0}},
    ]
    
    stage2_objects = [
        {"id": "obj1", "Model": "table", "position": {"x": 1, "y": 0, "z": 0}},
    ]
    
    tracker.log_stage("stage1", stage1_objects)
    tracker.log_stage("stage2", stage2_objects)
    
    change = tracker.detect_changes("stage1", "stage2")
    
    assert change is not None
    assert len(change.modified) == 1
    assert change.modified[0]["id"] == "obj1"
    assert change.modified[0]["field"] == "position"


def test_generate_trace():
    """Test generation of complete pipeline trace."""
    tracker = PipelineTracker(output_dir=tempfile.mkdtemp())
    
    stage1_objects = [{"id": "obj1", "Model": "table"}]
    stage2_objects = [{"id": "obj1", "Model": "table"}, {"id": "obj2", "Model": "chair"}]
    
    tracker.log_stage("stage1", stage1_objects)
    tracker.log_stage("stage2", stage2_objects)
    
    trace = tracker.generate_trace()
    
    assert "pipeline_trace" in trace
    assert trace["pipeline_trace"]["total_stages"] == 2
    assert len(trace["pipeline_trace"]["stages"]) == 2
    assert len(trace["pipeline_trace"]["changes"]) == 1


def test_save_trace():
    """Test saving trace to JSON file."""
    temp_dir = tempfile.mkdtemp()
    tracker = PipelineTracker(output_dir=temp_dir)
    
    objects = [{"id": "obj1", "Model": "table"}]
    tracker.log_stage("test_stage", objects)
    
    trace = tracker.generate_trace()
    output_path = tracker.save_trace(trace, filename="test_trace.json")
    
    assert output_path.exists()
    assert output_path.name == "test_trace.json"
    
    # Verify file content
    with open(output_path, "r") as f:
        loaded_trace = json.load(f)
    
    assert loaded_trace == trace


def test_get_stage_summary():
    """Test getting summary for a specific stage."""
    tracker = PipelineTracker(output_dir=tempfile.mkdtemp())
    
    objects = [
        {"id": "obj1", "Model": "table"},
        {"id": "obj2", "Model": "chair"},
    ]
    
    tracker.log_stage("test_stage", objects)
    
    summary = tracker.get_stage_summary("test_stage")
    
    assert summary is not None
    assert summary["name"] == "test_stage"
    assert summary["object_count"] == 2
    assert "obj1" in summary["object_ids"]
    assert "obj2" in summary["object_ids"]


def test_get_stage_summary_nonexistent():
    """Test getting summary for nonexistent stage."""
    tracker = PipelineTracker(output_dir=tempfile.mkdtemp())
    
    summary = tracker.get_stage_summary("nonexistent_stage")
    
    assert summary is None


def test_multiple_stages_tracking():
    """Test tracking objects through multiple pipeline stages."""
    tracker = PipelineTracker(output_dir=tempfile.mkdtemp())
    
    # Simulate full pipeline
    semantic_plan = [
        {"id": "table_1", "Model": "table", "position": {"relative": {"relative_to": "origin"}}},
        {"id": "chair_1", "Model": "chair", "position": {"relative": {"relative_to": "table_1"}}},
    ]
    
    distance_resolved = [
        {"id": "table_1", "Model": "table", "position": {"x": 0, "y": 0, "z": 0}},
        {"id": "chair_1", "Model": "chair", "position": {"x": 1, "y": 0, "z": 0}},
    ]
    
    orientation_resolved = [
        {"id": "table_1", "Model": "table", "position": {"x": 0, "y": 0, "z": 0}, "orientation": {"yaw_deg": 0}},
        {"id": "chair_1", "Model": "chair", "position": {"x": 1, "y": 0, "z": 0}, "orientation": {"yaw_deg": 180}},
    ]
    
    tracker.log_stage("semantic_plan", semantic_plan)
    tracker.log_stage("distance_resolver", distance_resolved)
    tracker.log_stage("orientation_resolver", orientation_resolved)
    
    trace = tracker.generate_trace()
    
    # Verify all stages logged
    assert trace["pipeline_trace"]["total_stages"] == 3
    
    # Verify changes detected
    changes = trace["pipeline_trace"]["changes"]
    assert len(changes) == 2
    
    # First change: position modified
    assert changes[0]["from_stage"] == "semantic_plan"
    assert changes[0]["to_stage"] == "distance_resolver"
    assert len(changes[0]["modified"]) > 0
    
    # Second change: orientation added
    assert changes[1]["from_stage"] == "distance_resolver"
    assert changes[1]["to_stage"] == "orientation_resolver"


if __name__ == "__main__":
    # Run all tests
    test_log_stage_with_dicts()
    print("✓ test_log_stage_with_dicts passed")
    
    test_log_stage_with_dataclasses()
    print("✓ test_log_stage_with_dataclasses passed")
    
    test_detect_changes_added_objects()
    print("✓ test_detect_changes_added_objects passed")
    
    test_detect_changes_removed_objects()
    print("✓ test_detect_changes_removed_objects passed")
    
    test_detect_changes_modified_objects()
    print("✓ test_detect_changes_modified_objects passed")
    
    test_generate_trace()
    print("✓ test_generate_trace passed")
    
    test_save_trace()
    print("✓ test_save_trace passed")
    
    test_get_stage_summary()
    print("✓ test_get_stage_summary passed")
    
    test_get_stage_summary_nonexistent()
    print("✓ test_get_stage_summary_nonexistent passed")
    
    test_multiple_stages_tracking()
    print("✓ test_multiple_stages_tracking passed")
    
    print("\n✅ All tests passed!")
