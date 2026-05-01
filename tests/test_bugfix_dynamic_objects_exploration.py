"""Bug Condition Exploration Test for Dynamic Objects Incorrectly Marked as Static.

Feature: fixed-object-position-investigation
Property 1: Bug Condition - Dynamic Objects Incorrectly Marked as Static

This test MUST FAIL on unfixed code - failure confirms the bug exists.

**Validates: Requirements 1.1, 1.2, 1.3, 2.1, 2.2, 2.3, 2.4**

The test creates objects that should be dynamic (small objects with volume < 0.3 m³
and containers) and asserts that they have is_static=false after processing through
_generate_semantic_plan().

Expected behavior on UNFIXED code:
- Small objects (apples) have is_static=true instead of false (FAIL)
- Container objects (boxes) have is_static=true instead of false (FAIL)
- LLM or heuristic incorrectly sets is_static=true (FAIL)

Expected behavior on FIXED code:
- All small objects have is_static=false (PASS)
- All container objects have is_static=false (PASS)
- LLM-based dynamics determination correctly sets is_static (PASS)
"""

import pytest
from hypothesis import given, strategies as st, assume, settings
from typing import List, Dict, Any

from creator.placement.universal_system import UniversalPlacementSystem
from creator.placement.command_interpreter import CommandInterpreter


# ============================================================================
# Hypothesis Strategies for generating test data
# ============================================================================

@st.composite
def small_object_data(draw):
    """Generate small object data (volume < 0.3 m³).
    
    Returns a dictionary with:
    - Model: object type name
    - size: [width, depth, height] with volume < 0.3 m³
    - expected_is_static: False (small objects should be dynamic)
    """
    # Small objects: apples, books, cups, etc.
    object_types = ['apple2', 'book', 'cup', 'ball', 'toy']
    model_name = draw(st.sampled_from(object_types))
    
    # Generate size with volume < 0.3 m³
    # For very small objects like apples: ~0.0005 m³
    # For medium objects like books: ~0.006 m³
    # For larger small objects: up to 0.29 m³
    width = draw(st.floats(min_value=0.05, max_value=0.6))
    depth = draw(st.floats(min_value=0.05, max_value=0.6))
    height = draw(st.floats(min_value=0.05, max_value=0.6))
    
    size = [width, depth, height]
    volume = width * depth * height
    
    # Ensure volume < 0.3 m³
    assume(volume < 0.3)
    
    return {
        "Model": model_name,
        "size": size,
        "volume": volume,
        "expected_is_static": False,
        "reason": f"small object (volume={volume:.4f} m³ < 0.3 m³)",
    }


@st.composite
def container_object_data(draw):
    """Generate container object data.
    
    Returns a dictionary with:
    - Model: container type name (box, basket, crate, etc.)
    - size: [width, depth, height]
    - expected_is_static: False (containers should be dynamic)
    """
    # Container types
    container_types = ['cardboard_box', 'basket', 'crate', 'bin', 'bowl', 'cup']
    model_name = draw(st.sampled_from(container_types))
    
    # Generate size - containers can be various sizes
    width = draw(st.floats(min_value=0.2, max_value=0.8))
    depth = draw(st.floats(min_value=0.2, max_value=0.8))
    height = draw(st.floats(min_value=0.2, max_value=0.8))
    
    size = [width, depth, height]
    volume = width * depth * height
    
    return {
        "Model": model_name,
        "size": size,
        "volume": volume,
        "expected_is_static": False,
        "reason": f"container ('{model_name}' contains container keyword)",
    }


@st.composite
def large_static_object_data(draw):
    """Generate large static object data (volume >= 0.3 m³).
    
    Returns a dictionary with:
    - Model: furniture type name
    - size: [width, depth, height] with volume >= 0.3 m³
    - expected_is_static: True (large furniture should be static)
    """
    # Large furniture
    furniture_types = ['wooden_patterned_table', 'chair', 'sofa', 'cabinet', 'desk']
    model_name = draw(st.sampled_from(furniture_types))
    
    # Generate size with volume >= 0.3 m³
    width = draw(st.floats(min_value=0.8, max_value=2.0))
    depth = draw(st.floats(min_value=0.6, max_value=1.5))
    height = draw(st.floats(min_value=0.5, max_value=1.2))
    
    size = [width, depth, height]
    volume = width * depth * height
    
    # Ensure volume >= 0.3 m³
    assume(volume >= 0.3)
    
    return {
        "Model": model_name,
        "size": size,
        "volume": volume,
        "expected_is_static": True,
        "reason": f"large furniture (volume={volume:.4f} m³ >= 0.3 m³)",
    }


# ============================================================================
# Property 1: Bug Condition - Dynamic Objects Incorrectly Marked as Static
# ============================================================================

@given(small_object_data())
@settings(max_examples=20, deadline=None)
def test_property_1_small_objects_should_be_dynamic(small_obj_data):
    """Property 1a: Small objects (volume < 0.3 m³) SHALL have is_static=false.
    
    **Validates: Requirements 1.1, 1.2, 1.3, 2.1, 2.2, 2.3, 2.4**
    
    This test creates a small object (volume < 0.3 m³) and verifies that
    _generate_semantic_plan() sets is_static=false for it.
    
    Expected on UNFIXED code: FAIL (small objects have is_static=true)
    Expected on FIXED code: PASS (small objects have is_static=false)
    """
    print(f"\n[TEST] Testing small object: {small_obj_data['Model']}")
    print(f"  Size: {small_obj_data['size']}")
    print(f"  Volume: {small_obj_data['volume']:.4f} m³")
    print(f"  Expected is_static: {small_obj_data['expected_is_static']}")
    print(f"  Reason: {small_obj_data['reason']}")
    
    # Create system
    system = UniversalPlacementSystem()
    command_interpreter = CommandInterpreter()
    
    # Create chosen_models with the small object
    chosen_models = [small_obj_data]
    
    # Parse spatial command (simple placement)
    description = f"Place a {small_obj_data['Model']} on the floor"
    spatial_command = command_interpreter.parse_spatial_command(description)
    
    # Call _generate_semantic_plan (this is where the bug occurs)
    semantic_plan = system._generate_semantic_plan(
        description=description,
        chosen_models=chosen_models,
        spatial_command=spatial_command,
        prompt_model=None,  # Use fallback (heuristic-based)
        prompt_template=None,
    )
    
    # Verify semantic_plan has objects
    assert "objects" in semantic_plan, "semantic_plan missing 'objects' key"
    assert len(semantic_plan["objects"]) > 0, "semantic_plan has no objects"
    
    # Find the small object in semantic_plan
    small_obj = None
    for obj in semantic_plan["objects"]:
        model_name = obj.get("Model", "")
        if small_obj_data["Model"] in model_name or model_name in small_obj_data["Model"]:
            small_obj = obj
            break
    
    assert small_obj is not None, \
        f"Could not find {small_obj_data['Model']} in semantic_plan.objects"
    
    print(f"\n[TEST] Found object in semantic_plan:")
    print(f"  id: {small_obj.get('id')}")
    print(f"  Model: {small_obj.get('Model')}")
    print(f"  size: {small_obj.get('size')}")
    print(f"  is_static: {small_obj.get('is_static')}")
    
    # CRITICAL ASSERTION: Small objects MUST have is_static=false
    # This will FAIL on unfixed code because:
    # - LLM might set is_static=true incorrectly
    # - Heuristic might not be applied or might be wrong
    # - System might default to is_static=true
    assert "is_static" in small_obj, \
        f"[BUG DETECTED] Small object {small_obj['id']} missing is_static field"
    
    assert small_obj["is_static"] == False, \
        f"[BUG DETECTED] Small object {small_obj['id']} " \
        f"(volume={small_obj_data['volume']:.4f} m³ < 0.3 m³) " \
        f"has is_static={small_obj['is_static']}, expected False. " \
        f"Small objects should be dynamic and fall under gravity."
    
    print(f"\n[TEST] ✅ Small object correctly marked as dynamic (is_static=false)")


@given(container_object_data())
@settings(max_examples=20, deadline=None)
def test_property_1_containers_should_be_dynamic(container_data):
    """Property 1b: Container objects SHALL have is_static=false.
    
    **Validates: Requirements 1.1, 1.2, 1.3, 2.1, 2.2, 2.3, 2.4**
    
    This test creates a container object (box, basket, crate, etc.) and verifies
    that _generate_semantic_plan() sets is_static=false for it.
    
    Expected on UNFIXED code: FAIL (containers have is_static=true)
    Expected on FIXED code: PASS (containers have is_static=false)
    """
    print(f"\n[TEST] Testing container: {container_data['Model']}")
    print(f"  Size: {container_data['size']}")
    print(f"  Volume: {container_data['volume']:.4f} m³")
    print(f"  Expected is_static: {container_data['expected_is_static']}")
    print(f"  Reason: {container_data['reason']}")
    
    # Create system
    system = UniversalPlacementSystem()
    command_interpreter = CommandInterpreter()
    
    # Create chosen_models with the container
    chosen_models = [container_data]
    
    # Parse spatial command
    description = f"Place a {container_data['Model']} on the floor"
    spatial_command = command_interpreter.parse_spatial_command(description)
    
    # Call _generate_semantic_plan
    semantic_plan = system._generate_semantic_plan(
        description=description,
        chosen_models=chosen_models,
        spatial_command=spatial_command,
        prompt_model=None,  # Use fallback (heuristic-based)
        prompt_template=None,
    )
    
    # Verify semantic_plan has objects
    assert "objects" in semantic_plan, "semantic_plan missing 'objects' key"
    assert len(semantic_plan["objects"]) > 0, "semantic_plan has no objects"
    
    # Find the container in semantic_plan
    container_obj = None
    for obj in semantic_plan["objects"]:
        model_name = obj.get("Model", "")
        if container_data["Model"] in model_name or model_name in container_data["Model"]:
            container_obj = obj
            break
    
    assert container_obj is not None, \
        f"Could not find {container_data['Model']} in semantic_plan.objects"
    
    print(f"\n[TEST] Found object in semantic_plan:")
    print(f"  id: {container_obj.get('id')}")
    print(f"  Model: {container_obj.get('Model')}")
    print(f"  size: {container_obj.get('size')}")
    print(f"  is_static: {container_obj.get('is_static')}")
    
    # CRITICAL ASSERTION: Containers MUST have is_static=false
    # This will FAIL on unfixed code because:
    # - LLM might not recognize containers as dynamic
    # - Heuristic might not check for container keywords
    # - System might default to is_static=true
    assert "is_static" in container_obj, \
        f"[BUG DETECTED] Container {container_obj['id']} missing is_static field"
    
    assert container_obj["is_static"] == False, \
        f"[BUG DETECTED] Container {container_obj['id']} " \
        f"(Model='{container_data['Model']}') " \
        f"has is_static={container_obj['is_static']}, expected False. " \
        f"Containers should be dynamic for robot manipulation training."
    
    print(f"\n[TEST] ✅ Container correctly marked as dynamic (is_static=false)")


@given(large_static_object_data())
@settings(max_examples=10, deadline=None)
def test_property_1_large_objects_should_be_static(large_obj_data):
    """Property 1c: Large objects (volume >= 0.3 m³) SHALL have is_static=true.
    
    **Validates: Requirements 3.1, 3.2**
    
    This test creates a large furniture object (volume >= 0.3 m³) and verifies
    that _generate_semantic_plan() sets is_static=true for it.
    
    This test should PASS on both unfixed and fixed code (preservation check).
    
    Expected on UNFIXED code: PASS (large objects are static)
    Expected on FIXED code: PASS (large objects remain static)
    """
    print(f"\n[TEST] Testing large object: {large_obj_data['Model']}")
    print(f"  Size: {large_obj_data['size']}")
    print(f"  Volume: {large_obj_data['volume']:.4f} m³")
    print(f"  Expected is_static: {large_obj_data['expected_is_static']}")
    print(f"  Reason: {large_obj_data['reason']}")
    
    # Create system
    system = UniversalPlacementSystem()
    command_interpreter = CommandInterpreter()
    
    # Create chosen_models with the large object
    chosen_models = [large_obj_data]
    
    # Parse spatial command
    description = f"Place a {large_obj_data['Model']} in the room"
    spatial_command = command_interpreter.parse_spatial_command(description)
    
    # Call _generate_semantic_plan
    semantic_plan = system._generate_semantic_plan(
        description=description,
        chosen_models=chosen_models,
        spatial_command=spatial_command,
        prompt_model=None,  # Use fallback (heuristic-based)
        prompt_template=None,
    )
    
    # Verify semantic_plan has objects
    assert "objects" in semantic_plan, "semantic_plan missing 'objects' key"
    assert len(semantic_plan["objects"]) > 0, "semantic_plan has no objects"
    
    # Find the large object in semantic_plan
    large_obj = None
    for obj in semantic_plan["objects"]:
        model_name = obj.get("Model", "")
        if large_obj_data["Model"] in model_name or model_name in large_obj_data["Model"]:
            large_obj = obj
            break
    
    assert large_obj is not None, \
        f"Could not find {large_obj_data['Model']} in semantic_plan.objects"
    
    print(f"\n[TEST] Found object in semantic_plan:")
    print(f"  id: {large_obj.get('id')}")
    print(f"  Model: {large_obj.get('Model')}")
    print(f"  size: {large_obj.get('size')}")
    print(f"  is_static: {large_obj.get('is_static')}")
    
    # ASSERTION: Large objects SHOULD have is_static=true
    # This should PASS on both unfixed and fixed code
    assert "is_static" in large_obj, \
        f"Large object {large_obj['id']} missing is_static field"
    
    assert large_obj["is_static"] == True, \
        f"Large object {large_obj['id']} " \
        f"(volume={large_obj_data['volume']:.4f} m³ >= 0.3 m³) " \
        f"has is_static={large_obj['is_static']}, expected True. " \
        f"Large furniture should be static."
    
    print(f"\n[TEST] ✅ Large object correctly marked as static (is_static=true)")


# ============================================================================
# Concrete unit tests with specific objects from the bug report
# ============================================================================

def test_concrete_apple_should_be_dynamic():
    """Concrete test: Apple (size=[0.08, 0.08, 0.08]) should have is_static=false.
    
    **Validates: Requirements 1.1, 1.2, 1.3, 2.1, 2.2, 2.3, 2.4**
    
    This is a concrete test case from the bug report examples.
    
    Expected on UNFIXED code: FAIL
    Expected on FIXED code: PASS
    """
    print("\n[TEST] Concrete test: Apple")
    
    # Apple from bug report
    apple_data = {
        "Model": "apple2",
        "size": [0.08, 0.08, 0.08],
    }
    
    volume = 0.08 * 0.08 * 0.08
    print(f"  Size: {apple_data['size']}")
    print(f"  Volume: {volume:.6f} m³ (< 0.3 m³)")
    
    # Create system
    system = UniversalPlacementSystem()
    command_interpreter = CommandInterpreter()
    
    # Parse spatial command
    description = "Place an apple on the floor"
    spatial_command = command_interpreter.parse_spatial_command(description)
    
    # Call _generate_semantic_plan
    semantic_plan = system._generate_semantic_plan(
        description=description,
        chosen_models=[apple_data],
        spatial_command=spatial_command,
        prompt_model=None,
        prompt_template=None,
    )
    
    # Find apple in semantic_plan
    apple_obj = semantic_plan["objects"][0]
    
    print(f"\n[TEST] Apple in semantic_plan:")
    print(f"  id: {apple_obj.get('id')}")
    print(f"  Model: {apple_obj.get('Model')}")
    print(f"  size: {apple_obj.get('size')}")
    print(f"  is_static: {apple_obj.get('is_static')}")
    
    # CRITICAL ASSERTION
    assert apple_obj["is_static"] == False, \
        f"[BUG DETECTED] Apple has is_static={apple_obj['is_static']}, expected False"
    
    print(f"\n[TEST] ✅ Apple correctly marked as dynamic")


def test_concrete_cardboard_box_should_be_dynamic():
    """Concrete test: Cardboard box should have is_static=false.
    
    **Validates: Requirements 1.1, 1.2, 1.3, 2.1, 2.2, 2.3, 2.4**
    
    This is a concrete test case from the bug report examples.
    
    Expected on UNFIXED code: FAIL
    Expected on FIXED code: PASS
    """
    print("\n[TEST] Concrete test: Cardboard box")
    
    # Cardboard box from bug report
    box_data = {
        "Model": "cardboard_box",
        "size": [0.4, 0.3, 0.4],
    }
    
    volume = 0.4 * 0.3 * 0.4
    print(f"  Size: {box_data['size']}")
    print(f"  Volume: {volume:.4f} m³ (< 0.3 m³)")
    print(f"  Is container: True (contains 'box')")
    
    # Create system
    system = UniversalPlacementSystem()
    command_interpreter = CommandInterpreter()
    
    # Parse spatial command
    description = "Place a cardboard box on the floor"
    spatial_command = command_interpreter.parse_spatial_command(description)
    
    # Call _generate_semantic_plan
    semantic_plan = system._generate_semantic_plan(
        description=description,
        chosen_models=[box_data],
        spatial_command=spatial_command,
        prompt_model=None,
        prompt_template=None,
    )
    
    # Find box in semantic_plan
    box_obj = semantic_plan["objects"][0]
    
    print(f"\n[TEST] Box in semantic_plan:")
    print(f"  id: {box_obj.get('id')}")
    print(f"  Model: {box_obj.get('Model')}")
    print(f"  size: {box_obj.get('size')}")
    print(f"  is_static: {box_obj.get('is_static')}")
    
    # CRITICAL ASSERTION
    assert box_obj["is_static"] == False, \
        f"[BUG DETECTED] Cardboard box has is_static={box_obj['is_static']}, expected False"
    
    print(f"\n[TEST] ✅ Cardboard box correctly marked as dynamic")


def test_concrete_table_should_be_static():
    """Concrete test: Table should have is_static=true (preservation check).
    
    **Validates: Requirements 3.1, 3.2**
    
    This is a concrete test case from the bug report examples.
    
    Expected on UNFIXED code: PASS
    Expected on FIXED code: PASS
    """
    print("\n[TEST] Concrete test: Table (preservation)")
    
    # Table from bug report
    table_data = {
        "Model": "wooden_patterned_table",
        "size": [1.5, 0.75, 0.8],
    }
    
    volume = 1.5 * 0.75 * 0.8
    print(f"  Size: {table_data['size']}")
    print(f"  Volume: {volume:.4f} m³ (>= 0.3 m³)")
    
    # Create system
    system = UniversalPlacementSystem()
    command_interpreter = CommandInterpreter()
    
    # Parse spatial command
    description = "Place a table in the room"
    spatial_command = command_interpreter.parse_spatial_command(description)
    
    # Call _generate_semantic_plan
    semantic_plan = system._generate_semantic_plan(
        description=description,
        chosen_models=[table_data],
        spatial_command=spatial_command,
        prompt_model=None,
        prompt_template=None,
    )
    
    # Find table in semantic_plan
    table_obj = semantic_plan["objects"][0]
    
    print(f"\n[TEST] Table in semantic_plan:")
    print(f"  id: {table_obj.get('id')}")
    print(f"  Model: {table_obj.get('Model')}")
    print(f"  size: {table_obj.get('size')}")
    print(f"  is_static: {table_obj.get('is_static')}")
    
    # ASSERTION (should pass on both unfixed and fixed code)
    assert table_obj["is_static"] == True, \
        f"Table has is_static={table_obj['is_static']}, expected True"
    
    print(f"\n[TEST] ✅ Table correctly marked as static (preservation)")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
