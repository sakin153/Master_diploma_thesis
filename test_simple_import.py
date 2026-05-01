#!/usr/bin/env python3
"""Simple import test."""

print("Testing imports...")

try:
    from creator.placement.semantic_enforcement import SemanticEnforcementPipeline
    print("✓ SemanticEnforcementPipeline imported")
    
    pipeline = SemanticEnforcementPipeline()
    print("✓ Pipeline created")
    print(f"  Components: {dir(pipeline)}")
    
except Exception as e:
    print(f"✗ Error: {e}")
    import traceback
    traceback.print_exc()
