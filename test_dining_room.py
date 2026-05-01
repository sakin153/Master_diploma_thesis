#!/usr/bin/env python3
"""Test the semantic enforcement pipeline with the dining room query."""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from creator.runner import generate_world

def main():
    """Test with the Russian dining room query."""
    print("=" * 60)
    print("Testing Semantic Enforcement Pipeline")
    print("Query: Простая столовая где есть 4 стола и 16 стульев")
    print("       у каждого стола по 4 стула")
    print("=" * 60)
    
    try:
        world_path = generate_world(
            query="Простая столовая где есть 4 стола и 16 стульев у каждого стола по 4 стула",
            cache_dir=".cache",
            vlm_validation=False,
            seed=42,
        )
        
        print("\n" + "=" * 60)
        print(f"✓ Scene generated successfully!")
        print(f"✓ Scene saved to: {world_path}")
        print("=" * 60)
        return 0
        
    except Exception as e:
        print(f"\n✗ Scene generation failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
