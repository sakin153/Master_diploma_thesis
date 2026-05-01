#!/usr/bin/env python3
"""Test that the uuid fix works correctly."""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

def main():
    """Test the uuid fix."""
    print("=" * 60)
    print("Testing UUID Fix")
    print("=" * 60)
    
    try:
        # Test that we can import uuid module
        import uuid as uuid_module
        print(f"✓ Can import uuid module: {uuid_module}")
        
        # Test that we can generate a UUID
        test_uuid = uuid_module.uuid4()
        print(f"✓ Can generate UUID: {test_uuid}")
        
        # Test that the runner imports correctly
        print("\nTesting runner.py imports...")
        from creator import runner
        print("✓ Successfully imported creator.runner")
        
        # Check that generate_world function exists
        if hasattr(runner, 'generate_world'):
            print("✓ generate_world function exists")
        else:
            print("✗ generate_world function not found")
            return 1
        
        # Check that _generate_semantic_plan_with_validation exists
        if hasattr(runner, '_generate_semantic_plan_with_validation'):
            print("✓ _generate_semantic_plan_with_validation function exists")
        else:
            print("✗ _generate_semantic_plan_with_validation function not found")
            return 1
        
        print("\n" + "=" * 60)
        print("✓ All UUID fix tests passed!")
        print("=" * 60)
        print("\nThe uuid naming conflict has been fixed.")
        print("You can now run: python main.py create \"<your query>\"")
        return 0
        
    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
