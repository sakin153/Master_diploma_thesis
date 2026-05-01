#!/usr/bin/env python3
"""Test script for Semantic Enforcement Pipeline integration.

This script tests the integration of SemanticEnforcementPipeline into runner.py.
"""

import json
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from creator.placement.semantic_enforcement.pipeline import SemanticEnforcementPipeline
from creator.placement.semantic_enforcement.schema_validator import SchemaValidator


def test_schema_validator():
    """Test that schema validator works correctly."""
    print("[Test] Testing SchemaValidator...")
    
    validator = SchemaValidator()
    
    # Valid plan
    valid_plan = {
        "schema_version": "1.0",
        "room_size": {"width": 10.0, "length": 10.0, "height": 3.0},
        "objects": [
            {
                "id": "table_1",
                "Model": "dining_table",
                "type": "furniture",
                "size": {"width": 1.5, "length": 0.8, "height": 0.75},
                "is_static": True,
                "model_loc": "models/tables/dining_table_01.xml",
                "position": {
                    "absolute": {"x": 0.0, "y": 0.0, "z": 0.0}
                },
                "orientation": {
                    "absolute": {"yaw_deg": 0.0, "pitch_deg": 0.0, "roll_deg": 0.0}
                }
            }
        ]
    }
    
    result = validator.validate_plan(valid_plan)
    assert result.success, f"Valid plan should pass validation: {result.errors}"
    print("[Test] ✓ SchemaValidator works correctly")


def test_pipeline_initialization():
    """Test that pipeline can be initialized."""
    print("[Test] Testing SemanticEnforcementPipeline initialization...")
    
    pipeline = SemanticEnforcementPipeline(
        workspace_root=".",
        output_dir="output/test",
        enable_tracking=True,
        enable_collision_check=True,
    )
    
    assert pipeline is not None
    assert pipeline.schema_validator is not None
    assert pipeline.distance_resolver is not None
    assert pipeline.orientation_resolver is not None
    assert pipeline.placement_executor is not None
    assert pipeline.scene_assembly is not None
    
    print("[Test] ✓ SemanticEnforcementPipeline initialized successfully")


def test_semantic_plan_format():
    """Test that the new semantic plan format is correct."""
    print("[Test] Testing semantic plan format...")
    
    # Example semantic plan in new format
    semantic_plan = {
        "schema_version": "1.0",
        "room_size": {"width": 10.0, "length": 10.0, "height": 3.0},
        "objects": [
            {
                "id": "table_1",
                "Model": "dining_table",
                "type": "furniture",
                "size": {"width": 1.5, "length": 0.8, "height": 0.75},
                "is_static": True,
                "model_loc": "assets/table/wooden_patterned_table.obj",
                "position": {
                    "absolute": {"x": 0.0, "y": 0.0, "z": 0.0}
                },
                "orientation": {
                    "absolute": {"yaw_deg": 0.0, "pitch_deg": 0.0, "roll_deg": 0.0}
                }
            },
            {
                "id": "chair_1",
                "Model": "dining_chair",
                "type": "furniture",
                "size": {"width": 0.5, "length": 0.5, "height": 0.9},
                "is_static": True,
                "model_loc": "assets/chair/chair.obj",
                "position": {
                    "relative": {
                        "relative_to": "table_1",
                        "direction": "front",
                        "distance": 0.6,
                        "reference_point": "center"
                    }
                },
                "orientation": {
                    "relative": {
                        "facing": "table_1",
                        "facing_direction": "front"
                    }
                }
            }
        ]
    }
    
    # Validate schema
    validator = SchemaValidator()
    result = validator.validate_plan(semantic_plan)
    
    assert result.success, f"Semantic plan should be valid: {result.errors}"
    print("[Test] ✓ Semantic plan format is correct")


def main():
    """Run all tests."""
    print("=" * 60)
    print("Testing Semantic Enforcement Pipeline Integration")
    print("=" * 60)
    
    try:
        test_schema_validator()
        test_pipeline_initialization()
        test_semantic_plan_format()
        
        print("\n" + "=" * 60)
        print("✓ All tests passed!")
        print("=" * 60)
        return 0
        
    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
