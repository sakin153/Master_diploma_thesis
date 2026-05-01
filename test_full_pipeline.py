#!/usr/bin/env python3
"""Test full semantic enforcement pipeline end-to-end."""

from creator.placement.semantic_enforcement import (
    SemanticEnforcementPipeline,
)


def test_simple_scene():
    """Test simple scene with table and chair."""
    print("=" * 60)
    print("Testing Full Semantic Enforcement Pipeline")
    print("=" * 60 + "\n")
    
    # Create pipeline
    pipeline = SemanticEnforcementPipeline()
    
    # Simple semantic plan: table at origin, chair in front
    semantic_plan = {
        "schema_version": "1.0",
        "room_size": {"width": 6.0, "length": 6.0, "height": 3.0},
        "objects": [
            {
                "id": "table_1",
                "Model": "dining_table",
                "type": "furniture",
                "size": {"width": 1.5, "length": 0.8, "height": 0.75},
                "is_static": True,
                "model_loc": "assets/table/wooden_patterned_table.obj",
                "position": {"absolute": {"x": 0.0, "y": 0.0, "z": 0.375}},
                "orientation": {"absolute": {"yaw_deg": 0.0, "pitch_deg": 0.0, "roll_deg": 0.0}},
            },
            {
                "id": "chair_1",
                "Model": "dining_chair",
                "type": "furniture",
                "size": {"width": 0.5, "length": 0.5, "height": 0.9},
                "is_static": False,
                "model_loc": "assets/cardboard_box/box.obj",  # Using box as chair placeholder
                "position": {
                    "relative": {
                        "relative_to": "table_1",
                        "direction": "front",
                        "distance": 0.5,
                        "reference_point": "front_edge"
                    }
                },
                "orientation": {
                    "relative": {
                        "facing": "table_1",
                        "facing_direction": "front",
                        "facing_away": False
                    }
                },
            }
        ]
    }
    
    print("Input semantic plan:")
    print(f"  - Objects: {len(semantic_plan['objects'])}")
    print(f"  - Room size: {semantic_plan['room_size']}")
    print()
    
    # Execute pipeline
    try:
        result = pipeline.execute_pipeline(semantic_plan)
        
        if result["success"]:
            print("✅ Pipeline executed successfully!")
            print()
            print("Results:")
            print(f"  - Placed objects: {len(result['placement_solution'].objects)}")
            print(f"  - MuJoCo XML generated: {len(result['mujoco_xml'])} characters")
            print()
            
            # Print object positions
            print("Object positions:")
            for obj in result["placement_solution"].objects:
                pos = obj.position
                orient = obj.orientation
                print(f"  - {obj.id}:")
                print(f"      Position: ({pos['x']:.2f}, {pos['y']:.2f}, {pos['z']:.2f})")
                print(f"      Orientation: yaw={orient['yaw_deg']:.1f}°")
            
            # Save XML to file
            output_path = "test_scene_output.xml"
            with open(output_path, "w") as f:
                f.write(result["mujoco_xml"])
            print(f"\n✅ MuJoCo XML saved to: {output_path}")
            
            return True
            
        else:
            print("❌ Pipeline failed!")
            print(f"Errors: {result.get('errors', [])}")
            return False
            
    except Exception as e:
        print(f"❌ Pipeline error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    import sys
    success = test_simple_scene()
    sys.exit(0 if success else 1)
