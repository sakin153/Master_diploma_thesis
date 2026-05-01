"""Quick integration test for SemanticEnforcementPipeline."""

from creator.placement.semantic_enforcement import SemanticEnforcementPipeline

# Create a minimal semantic plan
semantic_plan = {
    "schema_version": "1.0",
    "room_size": {"width": 10.0, "length": 10.0, "height": 3.0},
    "objects": [
        {
            "id": "table_1",
            "Model": "table",
            "type": "furniture",
            "size": {"width": 1.5, "length": 0.8, "height": 0.75},
            "is_static": True,
            "model_loc": "assets/table/wooden_patterned_table.obj",
            "position": {"absolute": {"x": 0.0, "y": 0.0, "z": 0.375}, "relative": None},
            "orientation": {"absolute": {"yaw_deg": 0.0, "pitch_deg": 0.0, "roll_deg": 0.0}, "relative": None},
        }
    ],
    "metadata": {},
}

# Initialize pipeline
print("Initializing pipeline...")
pipeline = SemanticEnforcementPipeline(
    workspace_root=".",
    enable_tracking=True,
    enable_collision_check=True,
)

# Execute pipeline
print("Executing pipeline...")
try:
    mujoco_xml, placement_solution, pipeline_trace = pipeline.execute_pipeline(
        semantic_plan, output_filename="test_scene"
    )
    
    print(f"\n✓ Pipeline executed successfully!")
    print(f"  - Placed {len(placement_solution.objects)} objects")
    print(f"  - Generated MuJoCo XML ({len(mujoco_xml)} characters)")
    
    if pipeline_trace:
        print(f"  - Pipeline trace: {pipeline_trace['pipeline_trace']['total_stages']} stages")
    
    # Verify object was placed
    assert len(placement_solution.objects) == 1
    assert placement_solution.objects[0].id == "table_1"
    assert placement_solution.objects[0].position["x"] == 0.0
    assert placement_solution.objects[0].position["y"] == 0.0
    assert placement_solution.objects[0].position["z"] == 0.375
    
    print("\n✓ All assertions passed!")
    
except Exception as e:
    print(f"\n✗ Pipeline failed: {e}")
    import traceback
    traceback.print_exc()
    exit(1)
