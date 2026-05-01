"""Example usage of PipelineTracker in semantic enforcement pipeline.

This example demonstrates how to integrate PipelineTracker into the
semantic plan enforcement pipeline to track objects through all stages.
"""

from creator.placement.semantic_enforcement import (
    PipelineTracker,
    SemanticObject,
    PlacedObject,
    Position,
    Orientation,
)


def example_pipeline_with_tracking():
    """Example of using PipelineTracker in a complete pipeline."""
    
    # Initialize tracker
    tracker = PipelineTracker(output_dir="analysis_logs")
    
    # Stage 1: Semantic Plan (from LLM)
    semantic_plan_objects = [
        SemanticObject(
            id="table_1",
            Model="dining_table",
            type="furniture",
            size={"width": 1.5, "length": 0.8, "height": 0.75},
            is_static=True,
            model_loc="models/tables/dining_table_01.xml",
            position=Position(absolute={"x": 5.0, "y": 5.0, "z": 0.0}),
            orientation=Orientation(absolute={"yaw_deg": 0.0, "pitch_deg": 0.0, "roll_deg": 0.0}),
        ),
        SemanticObject(
            id="chair_1",
            Model="dining_chair",
            type="furniture",
            size={"width": 0.5, "length": 0.5, "height": 0.9},
            is_static=False,
            model_loc="models/chairs/dining_chair_01.xml",
            position=Position(
                relative={
                    "relative_to": "table_1",
                    "direction": "front",
                    "distance": 0.5,
                    "reference_point": "front_edge",
                }
            ),
            orientation=Orientation(
                relative={
                    "facing": "table_1",
                    "facing_direction": "front",
                }
            ),
        ),
    ]
    
    tracker.log_stage("semantic_plan", semantic_plan_objects)
    
    # Stage 2: After Schema Validation
    # (objects unchanged, just validated)
    tracker.log_stage("schema_validation", semantic_plan_objects)
    
    # Stage 3: After Distance Resolution
    # (relative positions converted to absolute)
    distance_resolved_objects = [
        SemanticObject(
            id="table_1",
            Model="dining_table",
            type="furniture",
            size={"width": 1.5, "length": 0.8, "height": 0.75},
            is_static=True,
            model_loc="models/tables/dining_table_01.xml",
            position=Position(absolute={"x": 5.0, "y": 5.0, "z": 0.0}),
            orientation=Orientation(absolute={"yaw_deg": 0.0, "pitch_deg": 0.0, "roll_deg": 0.0}),
        ),
        SemanticObject(
            id="chair_1",
            Model="dining_chair",
            type="furniture",
            size={"width": 0.5, "length": 0.5, "height": 0.9},
            is_static=False,
            model_loc="models/chairs/dining_chair_01.xml",
            position=Position(absolute={"x": 5.0, "y": 5.9, "z": 0.0}),  # Resolved to absolute
            orientation=Orientation(
                relative={
                    "facing": "table_1",
                    "facing_direction": "front",
                }
            ),
        ),
    ]
    
    tracker.log_stage("distance_resolver", distance_resolved_objects)
    
    # Stage 4: After Orientation Resolution
    # (relative orientations converted to absolute)
    orientation_resolved_objects = [
        SemanticObject(
            id="table_1",
            Model="dining_table",
            type="furniture",
            size={"width": 1.5, "length": 0.8, "height": 0.75},
            is_static=True,
            model_loc="models/tables/dining_table_01.xml",
            position=Position(absolute={"x": 5.0, "y": 5.0, "z": 0.0}),
            orientation=Orientation(absolute={"yaw_deg": 0.0, "pitch_deg": 0.0, "roll_deg": 0.0}),
        ),
        SemanticObject(
            id="chair_1",
            Model="dining_chair",
            type="furniture",
            size={"width": 0.5, "length": 0.5, "height": 0.9},
            is_static=False,
            model_loc="models/chairs/dining_chair_01.xml",
            position=Position(absolute={"x": 5.0, "y": 5.9, "z": 0.0}),
            orientation=Orientation(absolute={"yaw_deg": 180.0, "pitch_deg": 0.0, "roll_deg": 0.0}),  # Resolved
        ),
    ]
    
    tracker.log_stage("orientation_resolver", orientation_resolved_objects)
    
    # Stage 5: After Placement Execution
    placed_objects = [
        PlacedObject(
            id="table_1",
            Model="dining_table",
            type="furniture",
            size={"width": 1.5, "length": 0.8, "height": 0.75},
            is_static=True,
            model_loc="models/tables/dining_table_01.xml",
            position={"x": 5.0, "y": 5.0, "z": 0.0},
            orientation={"yaw_deg": 0.0, "pitch_deg": 0.0, "roll_deg": 0.0},
        ),
        PlacedObject(
            id="chair_1",
            Model="dining_chair",
            type="furniture",
            size={"width": 0.5, "length": 0.5, "height": 0.9},
            is_static=False,
            model_loc="models/chairs/dining_chair_01.xml",
            position={"x": 5.0, "y": 5.9, "z": 0.0},
            orientation={"yaw_deg": 180.0, "pitch_deg": 0.0, "roll_deg": 0.0},
        ),
    ]
    
    tracker.log_stage("placement_executor", placed_objects)
    
    # Stage 6: After Scene Assembly
    # (same objects, now in MuJoCo XML)
    tracker.log_stage("scene_assembly", placed_objects)
    
    # Generate and save trace
    trace = tracker.generate_trace()
    output_path = tracker.save_trace(trace)
    
    print(f"\n✅ Pipeline trace saved to: {output_path}")
    
    # Print summary
    print("\n📊 Pipeline Summary:")
    print(f"Total stages: {trace['pipeline_trace']['total_stages']}")
    print(f"Total changes detected: {len(trace['pipeline_trace']['changes'])}")
    
    # Print stage summaries
    print("\n📋 Stage Summaries:")
    for stage_name in ["semantic_plan", "distance_resolver", "orientation_resolver", "placement_executor"]:
        summary = tracker.get_stage_summary(stage_name)
        if summary:
            print(f"  {stage_name}: {summary['object_count']} objects - {summary['object_ids']}")
    
    # Print detected changes
    print("\n🔍 Detected Changes:")
    for change in trace['pipeline_trace']['changes']:
        print(f"  {change['from_stage']} → {change['to_stage']}:")
        if change['added']:
            print(f"    Added: {change['added']}")
        if change['removed']:
            print(f"    Removed: {change['removed']}")
        if change['modified']:
            print(f"    Modified: {len(change['modified'])} fields")
            for mod in change['modified'][:3]:  # Show first 3 modifications
                print(f"      - {mod['id']}.{mod['field']}: {mod['old']} → {mod['new']}")


if __name__ == "__main__":
    example_pipeline_with_tracking()
