#!/usr/bin/env python3
"""Test real scene generation with detailed logging."""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Patch print to capture all output
original_print = print
log_lines = []

def logged_print(*args, **kwargs):
    line = " ".join(str(arg) for arg in args)
    log_lines.append(line)
    original_print(*args, **kwargs)

import builtins
builtins.print = logged_print

def test_generation():
    query = "стол и 3 стула"
    
    original_print("=" * 80)
    original_print(f"REAL GENERATION TEST: {query}")
    original_print("=" * 80)
    
    try:
        from creator.runner import generate_world
        
        world_path = generate_world(
            query=query,
            cache_dir=None,
            vlm_validation=False,
            max_vlm_iters=0,
            assets_dir=None,
            seed=42,
        )
        
        original_print("\n" + "=" * 80)
        original_print(f"✓ Scene generated: {world_path}")
        original_print("=" * 80)
        
        # Analyze logs
        original_print("\n" + "=" * 80)
        original_print("LOG ANALYSIS")
        original_print("=" * 80)
        
        # Count Stage1 lines
        stage1_lines = [l for l in log_lines if "[Stage1]" in l]
        original_print(f"\n[Stage1] lines: {len(stage1_lines)}")
        for line in stage1_lines:
            original_print(f"  {line}")
        
        # Count semantic plan objects
        semantic_plan_lines = [l for l in log_lines if "Generated semantic plan with" in l]
        for line in semantic_plan_lines:
            original_print(f"\n{line}")
        
        # Count placed objects
        placed_lines = [l for l in log_lines if "✓ Placed" in l and "objects" in l]
        for line in placed_lines:
            original_print(f"{line}")
        
        # Check XML
        if os.path.exists(world_path):
            with open(world_path, 'r') as f:
                content = f.read()
            
            import re
            bodies = re.findall(r'<body name="([^"]+)"', content)
            original_print(f"\n✓ Bodies in XML: {len(bodies)}")
            for body in bodies:
                original_print(f"  - {body}")
        
        return True
        
    except Exception as e:
        original_print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_generation()
    sys.exit(0 if success else 1)
