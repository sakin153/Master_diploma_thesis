#!/usr/bin/env python3
"""Quick test."""

print("Starting test...")

try:
    from creator.placement.semantic_enforcement.orientation_resolver import OrientationResolver
    print("Import successful")
    
    resolver = OrientationResolver()
    print("Resolver created")
    
    # Simple test
    plan = {
        "schema_version": "1.0",
        "room_size": {"width": 6.0, "length": 6.0, "height": 3.0},
        "objects": [
            {
                "id": "table_1",
                "position": {"absolute": {"x": 0.0, "y": 0.0, "z": 0.0}},
                "orientation": {"absolute": {"yaw_deg": 0.0, "pitch_deg": 0.0, "roll_deg": 0.0}},
            },
        ]
    }
    
    result = resolver.resolve_orientations(plan)
    print("Test passed!")
    
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
