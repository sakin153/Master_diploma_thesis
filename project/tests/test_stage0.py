#!/usr/bin/env python3
"""Test Stage 0: Prompt Expansion

This script demonstrates how to use the prompt_expander module.
"""

from project.prompt_expander import expand_prompt


def mock_llm_function(system_prompt, user_query, model_name):
    """Mock LLM function for testing.
    
    In production, replace this with actual LLM API call.
    """
    # Simulate LLM response
    return {
        "expanded_description": f"A detailed scene based on: {user_query}",
        "room_type": "office",
        "room_style": "modern",
        "anchor_objects": ["desk"],
        "estimated_objects": [
            {"name": "desk", "quantity": 5, "notes": "primary workspace"},
            {"name": "chair", "quantity": 5, "notes": "beside desk, facing it"},
            {"name": "lamp", "quantity": 1, "notes": "on_top_of desk"},
        ],
        "room_dimensions_hint": "large (8x8m)"
    }


def main():
    """Test prompt expansion."""
    print("=" * 60)
    print("Stage 0: Prompt Expansion Test")
    print("=" * 60)
    
    # Test query
    query = "офис с 5 столами"
    
    print(f"\nInput query: '{query}'")
    print("\nExpanding...")
    
    # Expand prompt
    scene_spec = expand_prompt(
        query=query,
        prompt_model_fn=mock_llm_function,
        llm_model="test-model",
        verbose=True
    )
    
    # Display results
    print("\n" + "=" * 60)
    print("Results:")
    print("=" * 60)
    print(f"\nOriginal query: {scene_spec.original_query}")
    print(f"Expanded description: {scene_spec.expanded_description}")
    print(f"Room type: {scene_spec.room_type}")
    print(f"Room style: {scene_spec.room_style}")
    print(f"Room size: {scene_spec.room_half_size * 2}m × {scene_spec.room_half_size * 2}m")
    print(f"\nAnchor objects: {scene_spec.anchor_objects}")
    print(f"\nEstimated objects:")
    for obj in scene_spec.estimated_objects:
        print(f"  - {obj.name} × {obj.quantity}")
        if obj.notes:
            print(f"    Notes: {obj.notes}")
    
    print("\n" + "=" * 60)
    print("✓ Stage 0 test completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
