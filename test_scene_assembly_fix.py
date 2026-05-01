#!/usr/bin/env python3
"""Quick test to verify SceneAssembly XML generation fix."""

import sys
from pathlib import Path

# Add creator to path
sys.path.insert(0, str(Path(__file__).parent))

from creator.placement.semantic_enforcement.models import (
    PlacementSolution,
    PlacedObject,
    Position,
    Orientation,
)
from creator.placement.semantic_enforcement.scene_assembly import SceneAssembly

def test_scene_assembly():
    """Test that SceneAssembly generates valid MuJoCo XML."""
    
    # Create a simple placement solution with one object
    objects = [
        PlacedObject(
            id="table_1",
            name="table",
            model_loc="assets/table/wooden_patterned_table.glb",
            position=Position(x=0.0, y=0.0, z=0.0),
            orientation=Orientation(yaw_deg=0.0, pitch_deg=0.0, roll_deg=0.0),
            is_static=True,
        )
    ]
    
    solution = PlacementSolution(
        objects=objects,
        room_size={"width": 10.0, "length": 10.0, "height": 3.0},
    )
    
    # Create scene assembly
    assembly = SceneAssembly(workspace_root=".")
    
    # Generate XML
    xml_str = assembly.assemble_scene(solution)
    
    # Check that XML contains expected elements
    checks = [
        ("<mujoco", "Root mujoco element"),
        ("<asset>", "Asset section"),
        ('<mesh name="mesh_table_1"', "Mesh asset definition"),
        ("<worldbody>", "Worldbody"),
        ('<geom name="floor"', "Floor"),
        ('<body name="table_1"', "Object body"),
        ('type="mesh"', "Mesh geom type"),
        ('mesh="mesh_table_1"', "Mesh reference"),
    ]
    
    print("Testing SceneAssembly XML generation...")
    print("=" * 60)
    
    all_passed = True
    for check_str, description in checks:
        if check_str in xml_str:
            print(f"✓ {description}")
        else:
            print(f"✗ {description} - NOT FOUND")
            all_passed = False
    
    print("=" * 60)
    
    # Check that old <include> tag is NOT present
    if "<include" in xml_str:
        print("✗ ERROR: Old <include> tag still present!")
        all_passed = False
    else:
        print("✓ No <include> tags (correct)")
    
    if all_passed:
        print("\n✓ All checks passed!")
        print("\nGenerated XML preview (first 1000 chars):")
        print("-" * 60)
        print(xml_str[:1000])
        print("-" * 60)
        return 0
    else:
        print("\n✗ Some checks failed!")
        print("\nGenerated XML:")
        print(xml_str)
        return 1

if __name__ == "__main__":
    sys.exit(test_scene_assembly())
