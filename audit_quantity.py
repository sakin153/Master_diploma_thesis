#!/usr/bin/env python3
"""Audit script to trace object quantities through the pipeline."""

import sys
import os
import json

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from creator.scene.prompt_expander import expand_prompt
from creator.llm.model import prompt_model

def audit_quantity_flow(query: str):
    """Trace object quantities through each stage."""
    
    print("=" * 80)
    print(f"AUDIT: Tracing object quantities for query: '{query}'")
    print("=" * 80)
    
    chosen_model = "deepseek-v3.1:671b-cloud"
    
    # Stage 0: Prompt expansion
    print("\n[STAGE 0] Prompt Expansion")
    print("-" * 80)
    scene_spec = expand_prompt(
        query,
        prompt_model_fn=prompt_model,
        llm_model=chosen_model,
        verbose=True,
    )
    
    print(f"\nExpanded query: {scene_spec.effective_query}")
    print(f"Room type: {scene_spec.room_type}")
    print(f"Estimated objects:")
    
    total_objects = 0
    for obj in (scene_spec.estimated_objects or []):
        name = str(getattr(obj, "name", "")).strip()
        qty = int(getattr(obj, "quantity", 1))
        notes = str(getattr(obj, "notes", "")).strip()
        print(f"  - {name} x{qty} ({notes})")
        total_objects += qty
    
    print(f"\nTotal objects from expansion: {total_objects}")
    
    # Stage 1: Object list building
    print("\n[STAGE 1] Object List Building")
    print("-" * 80)
    objects = []
    for h in (scene_spec.estimated_objects or []):
        name = str(getattr(h, "name", "")).strip()
        qty = max(1, min(int(getattr(h, "quantity", 1)), 20))
        objects.extend([name] * qty)
    
    print(f"Objects list (after extend): {objects}")
    print(f"Total objects in list: {len(objects)}")
    
    # Count by type
    from collections import Counter
    counts = Counter(objects)
    print(f"\nObject counts:")
    for obj_name, count in counts.items():
        print(f"  - {obj_name}: {count}")
    
    return {
        "query": query,
        "expanded_query": scene_spec.effective_query,
        "estimated_objects": scene_spec.estimated_objects,
        "objects_list": objects,
        "total_count": len(objects),
        "counts_by_type": dict(counts)
    }

if __name__ == "__main__":
    test_queries = [
        "стол и 3 стула",
        "2 стола и 8 стульев",
        "4 стола и 16 стульев",
    ]
    
    results = []
    for query in test_queries:
        result = audit_quantity_flow(query)
        results.append(result)
        print("\n" + "=" * 80)
        print(f"SUMMARY for '{query}':")
        print(f"  Expected: {query}")
        print(f"  Got: {result['total_count']} objects total")
        print(f"  Breakdown: {result['counts_by_type']}")
        print("=" * 80)
        print("\n\n")
    
    # Save results
    with open("audit_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False, default=str)
    
    print("\n✓ Audit complete. Results saved to audit_results.json")
