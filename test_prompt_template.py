#!/usr/bin/env python3
"""Test that the semantic plan prompt template exists and is formatted correctly."""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

def main():
    """Test the prompt template."""
    print("=" * 60)
    print("Testing Semantic Plan Prompt Template")
    print("=" * 60)
    
    try:
        # Import the template
        from creator.contexts_prompts.semantic_plan import fmt_semantic_plan_tmpl
        
        print("✓ Successfully imported fmt_semantic_plan_tmpl")
        print(f"✓ Template length: {len(fmt_semantic_plan_tmpl)} characters")
        
        # Check for key elements
        checks = [
            ("schema_version", "schema_version" in fmt_semantic_plan_tmpl),
            ("position", "position" in fmt_semantic_plan_tmpl),
            ("orientation", "orientation" in fmt_semantic_plan_tmpl),
            ("relative", "relative" in fmt_semantic_plan_tmpl),
            ("relative_to", "relative_to" in fmt_semantic_plan_tmpl),
            ("facing", "facing" in fmt_semantic_plan_tmpl),
            ("query placeholder", "{query}" in fmt_semantic_plan_tmpl),
            ("room_width placeholder", "{room_width}" in fmt_semantic_plan_tmpl),
            ("models_str placeholder", "{models_str}" in fmt_semantic_plan_tmpl),
        ]
        
        all_passed = True
        for name, result in checks:
            status = "✓" if result else "✗"
            print(f"{status} Contains '{name}': {result}")
            if not result:
                all_passed = False
        
        if all_passed:
            print("\n" + "=" * 60)
            print("✓ All checks passed!")
            print("=" * 60)
            
            # Try to format the template
            try:
                formatted = fmt_semantic_plan_tmpl.format(
                    query="Test query",
                    room_width=10.0,
                    room_length=10.0,
                    room_height=3.0,
                    models_str="Test models"
                )
                print(f"✓ Template can be formatted successfully")
                print(f"✓ Formatted length: {len(formatted)} characters")
            except Exception as e:
                print(f"✗ Failed to format template: {e}")
                return 1
            
            return 0
        else:
            print("\n" + "=" * 60)
            print("✗ Some checks failed!")
            print("=" * 60)
            return 1
        
    except ImportError as e:
        print(f"✗ Failed to import template: {e}")
        import traceback
        traceback.print_exc()
        return 1
    except Exception as e:
        print(f"✗ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
