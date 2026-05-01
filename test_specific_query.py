#!/usr/bin/env python3
"""Test specific query to see what's extracted."""
import sys
sys.path.insert(0, '.')

from creator.llm.model import prompt_model
from creator.scene.prompt_expander import expand_prompt

query = "стол и две коробки на нем, в каждой коробке по 3 яблока"

print(f"Testing query: '{query}'")
print("=" * 80)

spec = expand_prompt(
    query=query,
    prompt_model_fn=prompt_model,
    llm_model="deepseek-v3.1:671b-cloud",
    verbose=False,
)

print(f"\nExpanded: {spec.expanded_description}")
print(f"\nObjects extracted:")
for obj in spec.estimated_objects:
    print(f"  - {obj.name} x{obj.quantity} ({obj.notes})")

total = sum(o.quantity for o in spec.estimated_objects)
print(f"\nTotal objects: {total}")

print("\n" + "=" * 80)
print("Expected:")
print("  - table x1")
print("  - cardboard_box x2")
print("  - apple x6 (3 per box)")
print("Total: 9 objects")
