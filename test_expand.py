#!/usr/bin/env python3
"""Test prompt expansion to see what objects are extracted."""
import sys
sys.path.insert(0, '.')

from creator.llm.model import prompt_model
from creator.scene.prompt_expander import expand_prompt

# Test queries
queries = [
    "стол и стул",
    "стол и 3 стула",
    "стол и 4 стула",
    "2 стола и 8 стульев",
]

print("Testing prompt expansion:\n")
print("=" * 80)

for query in queries:
    print(f"\nQuery: '{query}'")
    print("-" * 80)
    
    try:
        spec = expand_prompt(
            query=query,
            prompt_model_fn=prompt_model,
            llm_model="deepseek-v3.1:671b-cloud",
            verbose=False,
        )
        
        print(f"Room type: {spec.room_type}")
        print(f"Room size: {spec.room_half_size}m")
        print(f"Expanded: {spec.expanded_description}")
        print(f"\nObjects:")
        for obj in spec.estimated_objects:
            print(f"  - {obj.name} x{obj.quantity} ({obj.notes})")
        
        total = sum(o.quantity for o in spec.estimated_objects)
        print(f"\nTotal objects: {total}")
        
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
    
    print("=" * 80)
