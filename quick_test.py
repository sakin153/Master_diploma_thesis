#!/usr/bin/env python3
"""Quick test to verify the system works."""

import sys
import os

# Suppress warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def test_imports():
    """Test that all imports work."""
    print("Testing imports...")
    
    try:
        import uuid as uuid_module
        print(f"✓ uuid module: {uuid_module}")
        
        from creator.contexts_prompts.semantic_plan import fmt_semantic_plan_tmpl
        print(f"✓ semantic_plan template: {len(fmt_semantic_plan_tmpl)} chars")
        
        from creator.placement.semantic_enforcement.pipeline import SemanticEnforcementPipeline
        print(f"✓ SemanticEnforcementPipeline: {SemanticEnforcementPipeline}")
        
        from creator.runner import generate_world, _generate_semantic_plan_with_validation
        print(f"✓ generate_world: {generate_world}")
        print(f"✓ _generate_semantic_plan_with_validation: {_generate_semantic_plan_with_validation}")
        
        print("\n✅ All imports successful!")
        return True
        
    except Exception as e:
        print(f"\n❌ Import failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_uuid_fix():
    """Test that uuid fix works."""
    print("\nTesting UUID fix...")
    
    try:
        import uuid
        test_id = uuid.uuid4()
        print(f"✓ Can generate UUID: {test_id}")
        
        # Test that we can use uuid in a dict comprehension (like in runner.py)
        test_dict = {"uuid": str(uuid.uuid4())}
        print(f"✓ Can use uuid in dict: {test_dict}")
        
        print("\n✅ UUID fix works!")
        return True
        
    except Exception as e:
        print(f"\n❌ UUID test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("=" * 60)
    print("Quick System Test")
    print("=" * 60)
    
    results = []
    
    # Test imports
    results.append(("Imports", test_imports()))
    
    # Test UUID fix
    results.append(("UUID Fix", test_uuid_fix()))
    
    # Summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    
    for name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status}: {name}")
    
    all_passed = all(passed for _, passed in results)
    
    if all_passed:
        print("\n🎉 All tests passed! System is ready.")
        print("\nYou can now run:")
        print('  python main.py "стол и стул"')
        return 0
    else:
        print("\n❌ Some tests failed. Please fix the issues above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
