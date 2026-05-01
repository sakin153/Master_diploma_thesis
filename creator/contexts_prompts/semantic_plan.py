"""LLM prompt template for semantic plan generation.

This module contains the prompt template for the LLM to generate semantic plans
in the format required by the SemanticEnforcementPipeline.

The LLM acts as the main scene composer and architect, with full control over:
- Object placement (absolute or relative positions)
- Object orientation (absolute or relative facing directions)
- Spatial relationships between objects
- Distances and reference points
"""

fmt_semantic_plan_tmpl = """
CRITICAL: You MUST respond ONLY in English. Do NOT use any other language (Chinese, Russian, etc.) in your response.
CRITICAL: Your response MUST be valid JSON only. Do NOT include any text, comments, or explanations outside the JSON structure.

You are a professional scene composer and architect for a MuJoCo physics simulator.
Your role is to create detailed placement instructions for every object in the scene.

User query: {query}

Room dimensions:
- Width: {room_width} meters
- Length: {room_length} meters
- Height: {room_height} meters

Available models for this scene:
{models_str}

---

## Your Role as Scene Composer

You have COMPLETE CONTROL over the scene composition:
- Decide exact positions for every object (absolute coordinates or relative to other objects)
- Decide exact orientations for every object (absolute angles or facing directions)
- Specify distances between objects
- Define spatial relationships ("chair faces table", "lamp on table surface")

The system will execute your instructions EXACTLY as specified without modifications.

---

## Output Format

Generate a JSON semantic plan with this structure:

```json
{{
  "schema_version": "1.0",
  "room_size": {{
    "width": {room_width},
    "length": {room_length},
    "height": {room_height}
  }},
  "objects": [
    // Array of objects with placement instructions
  ]
}}
```

---

## Object Specification

Each object MUST have these fields:

**Required fields:**
- `id`: Unique identifier (e.g., "table_1", "chair_1", "chair_2")
- `Model`: Exact model name from the available models list above
- `type`: "furniture" (tables, chairs, sofas, beds) or "small_object" (plates, cups, books, lamps)
- `size`: Object dimensions in meters: {{"width": float, "length": float, "height": float}}
- `is_static`: true for furniture, false for small movable objects
- `position`: Position specification (see below)
- `orientation`: Orientation specification (see below)

**Note**: Do NOT include `model_loc` field - the system will add it automatically based on the Model name.

---

## Position Specification

Choose ONE of these two options for each object:

### Option 1: Absolute Position
Use when you want to specify exact coordinates in the room:

```json
"position": {{
  "absolute": {{
    "x": 5.0,    // meters from room origin (center of room)
    "y": 5.0,    // meters from room origin
    "z": 0.0     // height (0.0 for floor objects)
  }}
}}
```

### Option 2: Relative Position
Use when you want to place object relative to another object:

```json
"position": {{
  "relative": {{
    "relative_to": "table_1",        // ID of target object
    "direction": "front",             // Direction from target
    "distance": 0.5,                  // Distance in meters
    "reference_point": "center"       // Point on target to measure from
  }}
}}
```

**Available directions:**
- `"front"`: In front of target's front face
- `"back"`: Behind target's back face
- `"left"`: To the left of target
- `"right"`: To the right of target
- `"front_left"`: 45° between front and left
- `"front_right"`: 45° between front and right
- `"back_left"`: 45° between back and left
- `"back_right"`: 45° between back and right
- `"above"`: Directly above target
- `"below"`: Directly below target

**Available reference points:**
- `"center"`: Geometric center of target object
- `"front_edge"`: Center of front face
- `"back_edge"`: Center of back face
- `"left_edge"`: Center of left face
- `"right_edge"`: Center of right face
- `"top_surface"`: Center of top face (for placing objects on surfaces)

**Distance guidelines:**
- Chair around dining table: 0.5-0.8 meters
- Small object on table surface: use "above" direction with distance 0.0
- Side table next to sofa: 0.3-0.6 meters
- Objects that should touch: 0.0 meters

---

## Orientation Specification

Choose ONE of these two options for each object:

### Option 1: Absolute Orientation
Use when you want to specify exact angles:

```json
"orientation": {{
  "absolute": {{
    "yaw_deg": 0.0,      // Rotation around vertical axis (0-360)
    "pitch_deg": 0.0,    // Usually 0 for furniture
    "roll_deg": 0.0      // Usually 0 for furniture
  }}
}}
```

**Yaw angle reference:**
- 0°: Facing positive X direction
- 90°: Facing positive Y direction
- 180°: Facing negative X direction
- 270°: Facing negative Y direction

### Option 2: Relative Orientation
Use when you want object to face another object:

```json
"orientation": {{
  "relative": {{
    "facing": "table_1",              // ID of target object to face
    "facing_direction": "front",      // Which side of target to face
    "facing_away": false              // Optional: face away instead of toward
  }}
}}
```

**Available facing directions:**
- `"front"`: Face the front of target object
- `"back"`: Face the back of target object
- `"left_side"`: Face the left side of target object
- `"right_side"`: Face the right side of target object

**facing_away modifier:**
- `false` (default): Face toward the target
- `true`: Face away from the target (back to target)

---

## Complete Examples

### Example 1: Simple dining scene (1 table, 4 chairs)

```json
{{
  "schema_version": "1.0",
  "room_size": {{"width": 10.0, "length": 10.0, "height": 3.0}},
  "objects": [
    {{
      "id": "table_1",
      "Model": "dining_table",
      "type": "furniture",
      "size": {{"width": 1.5, "length": 0.8, "height": 0.75}},
      "is_static": true,
      "position": {{
        "absolute": {{"x": 0.0, "y": 0.0, "z": 0.0}}
      }},
      "orientation": {{
        "absolute": {{"yaw_deg": 0.0, "pitch_deg": 0.0, "roll_deg": 0.0}}
      }}
    }},
    {{
      "id": "chair_1",
      "Model": "dining_chair",
      "type": "furniture",
      "size": {{"width": 0.5, "length": 0.5, "height": 0.9}},
      "is_static": true,
      "position": {{
        "relative": {{
          "relative_to": "table_1",
          "direction": "front",
          "distance": 0.6,
          "reference_point": "center"
        }}
      }},
      "orientation": {{
        "relative": {{
          "facing": "table_1",
          "facing_direction": "front"
        }}
      }}
    }},
    {{
      "id": "chair_2",
      "Model": "dining_chair",
      "type": "furniture",
      "size": {{"width": 0.5, "length": 0.5, "height": 0.9}},
      "is_static": true,
      "position": {{
        "relative": {{
          "relative_to": "table_1",
          "direction": "back",
          "distance": 0.6,
          "reference_point": "center"
        }}
      }},
      "orientation": {{
        "relative": {{
          "facing": "table_1",
          "facing_direction": "front"
        }}
      }}
    }},
    {{
      "id": "chair_3",
      "Model": "dining_chair",
      "type": "furniture",
      "size": {{"width": 0.5, "length": 0.5, "height": 0.9}},
      "is_static": true,
      "position": {{
        "relative": {{
          "relative_to": "table_1",
          "direction": "left",
          "distance": 0.6,
          "reference_point": "center"
        }}
      }},
      "orientation": {{
        "relative": {{
          "facing": "table_1",
          "facing_direction": "front"
        }}
      }}
    }},
    {{
      "id": "chair_4",
      "Model": "dining_chair",
      "type": "furniture",
      "size": {{"width": 0.5, "length": 0.5, "height": 0.9}},
      "is_static": true,
      "position": {{
        "relative": {{
          "relative_to": "table_1",
          "direction": "right",
          "distance": 0.6,
          "reference_point": "center"
        }}
      }},
      "orientation": {{
        "relative": {{
          "facing": "table_1",
          "facing_direction": "front"
        }}
      }}
    }}
  ]
}}
```

### Example 2: Restaurant scene (4 tables, 16 chairs - 4 chairs per table)

```json
{{
  "schema_version": "1.0",
  "room_size": {{"width": 15.0, "length": 15.0, "height": 3.0}},
  "objects": [
    {{
      "id": "table_1",
      "Model": "dining_table",
      "type": "furniture",
      "size": {{"width": 1.5, "length": 0.8, "height": 0.75}},
      "is_static": true,      "position": {{"absolute": {{"x": -3.0, "y": -3.0, "z": 0.0}}}},
      "orientation": {{"absolute": {{"yaw_deg": 0.0, "pitch_deg": 0.0, "roll_deg": 0.0}}}}
    }},
    {{
      "id": "table_2",
      "Model": "dining_table",
      "type": "furniture",
      "size": {{"width": 1.5, "length": 0.8, "height": 0.75}},
      "is_static": true,      "position": {{"absolute": {{"x": 3.0, "y": -3.0, "z": 0.0}}}},
      "orientation": {{"absolute": {{"yaw_deg": 0.0, "pitch_deg": 0.0, "roll_deg": 0.0}}}}
    }},
    {{
      "id": "table_3",
      "Model": "dining_table",
      "type": "furniture",
      "size": {{"width": 1.5, "length": 0.8, "height": 0.75}},
      "is_static": true,      "position": {{"absolute": {{"x": -3.0, "y": 3.0, "z": 0.0}}}},
      "orientation": {{"absolute": {{"yaw_deg": 0.0, "pitch_deg": 0.0, "roll_deg": 0.0}}}}
    }},
    {{
      "id": "table_4",
      "Model": "dining_table",
      "type": "furniture",
      "size": {{"width": 1.5, "length": 0.8, "height": 0.75}},
      "is_static": true,      "position": {{"absolute": {{"x": 3.0, "y": 3.0, "z": 0.0}}}},
      "orientation": {{"absolute": {{"yaw_deg": 0.0, "pitch_deg": 0.0, "roll_deg": 0.0}}}}
    }},
    {{
      "id": "chair_1",
      "Model": "dining_chair",
      "type": "furniture",
      "size": {{"width": 0.5, "length": 0.5, "height": 0.9}},
      "is_static": true,      "position": {{
        "relative": {{
          "relative_to": "table_1",
          "direction": "front",
          "distance": 0.6,
          "reference_point": "center"
        }}
      }},
      "orientation": {{
        "relative": {{"facing": "table_1", "facing_direction": "front"}}
      }}
    }},
    {{
      "id": "chair_2",
      "Model": "dining_chair",
      "type": "furniture",
      "size": {{"width": 0.5, "length": 0.5, "height": 0.9}},
      "is_static": true,      "position": {{
        "relative": {{
          "relative_to": "table_1",
          "direction": "back",
          "distance": 0.6,
          "reference_point": "center"
        }}
      }},
      "orientation": {{
        "relative": {{"facing": "table_1", "facing_direction": "front"}}
      }}
    }},
    {{
      "id": "chair_3",
      "Model": "dining_chair",
      "type": "furniture",
      "size": {{"width": 0.5, "length": 0.5, "height": 0.9}},
      "is_static": true,      "position": {{
        "relative": {{
          "relative_to": "table_1",
          "direction": "left",
          "distance": 0.6,
          "reference_point": "center"
        }}
      }},
      "orientation": {{
        "relative": {{"facing": "table_1", "facing_direction": "front"}}
      }}
    }},
    {{
      "id": "chair_4",
      "Model": "dining_chair",
      "type": "furniture",
      "size": {{"width": 0.5, "length": 0.5, "height": 0.9}},
      "is_static": true,      "position": {{
        "relative": {{
          "relative_to": "table_1",
          "direction": "right",
          "distance": 0.6,
          "reference_point": "center"
        }}
      }},
      "orientation": {{
        "relative": {{"facing": "table_1", "facing_direction": "front"}}
      }}
    }},
    // Chairs 5-8 for table_2 (same pattern, different relative_to)
    {{
      "id": "chair_5",
      "Model": "dining_chair",
      "type": "furniture",
      "size": {{"width": 0.5, "length": 0.5, "height": 0.9}},
      "is_static": true,      "position": {{
        "relative": {{
          "relative_to": "table_2",
          "direction": "front",
          "distance": 0.6,
          "reference_point": "center"
        }}
      }},
      "orientation": {{
        "relative": {{"facing": "table_2", "facing_direction": "front"}}
      }}
    }},
    // ... (chairs 6-8 for table_2, chairs 9-12 for table_3, chairs 13-16 for table_4)
    // Follow the same pattern: front, back, left, right for each table
  ]
}}
```

### Example 3: Objects on table surface

```json
{{
  "schema_version": "1.0",
  "room_size": {{"width": 10.0, "length": 10.0, "height": 3.0}},
  "objects": [
    {{
      "id": "table_1",
      "Model": "dining_table",
      "type": "furniture",
      "size": {{"width": 1.5, "length": 0.8, "height": 0.75}},
      "is_static": true,      "position": {{"absolute": {{"x": 0.0, "y": 0.0, "z": 0.0}}}},
      "orientation": {{"absolute": {{"yaw_deg": 0.0, "pitch_deg": 0.0, "roll_deg": 0.0}}}}
    }},
    {{
      "id": "plate_1",
      "Model": "dinner_plate",
      "type": "small_object",
      "size": {{"width": 0.25, "length": 0.25, "height": 0.02}},
      "is_static": false,      "position": {{
        "relative": {{
          "relative_to": "table_1",
          "direction": "above",
          "distance": 0.0,
          "reference_point": "top_surface"
        }}
      }},
      "orientation": {{"absolute": {{"yaw_deg": 0.0, "pitch_deg": 0.0, "roll_deg": 0.0}}}}
    }},
    {{
      "id": "lamp_1",
      "Model": "desk_lamp",
      "type": "small_object",
      "size": {{"width": 0.15, "length": 0.15, "height": 0.4}},
      "is_static": false,      "position": {{
        "relative": {{
          "relative_to": "table_1",
          "direction": "above",
          "distance": 0.0,
          "reference_point": "top_surface"
        }}
      }},
      "orientation": {{"absolute": {{"yaw_deg": 45.0, "pitch_deg": 0.0, "roll_deg": 0.0}}}}
    }}
  ]
}}
```

---

## Important Rules

1. **Object Count**: You MUST generate EXACTLY one object specification for EACH entry in the "Available models for this scene" list above. Count the number of entries in that list - that's how many objects you must generate. If there are 4 entries (1 table + 3 chairs), generate 4 objects. Even if multiple entries have the same Model name (e.g., "computer chair" appears 3 times), you MUST create 3 separate object specifications with unique IDs (chair_1, chair_2, chair_3).

2. **Unique IDs**: Each object must have a unique ID. Use pattern: `model_type_number` (e.g., "table_1", "chair_1", "chair_2", "chair_3").

3. **Relationships**: When user specifies relationships like "у каждого стола по 4 стула" (4 chairs per table), create explicit relative positioning for each chair to its specific table.

4. **Dependency Order**: Objects with absolute positions can be placed first. Objects with relative positions will be resolved in dependency order automatically.

5. **Z-coordinate**: For floor objects (tables, chairs), use z=0.0 in absolute positioning. For objects on surfaces, use "above" direction with distance 0.0.

6. **Model Names**: Use EXACT model names from the available models list provided above.

7. **Distances**: Be realistic with distances. Chairs around tables: 0.5-0.8m. Objects on surfaces: use "above" with distance 0.0.

8. **Orientation Logic**: Chairs should face their tables. Objects on surfaces can have any orientation. Use relative orientation when objects should face each other.

---

## Your Task

Based on the user query and available models above, generate a complete semantic plan in JSON format.

Remember:
- You are the architect - you have full control
- Be explicit about every object's position and orientation
- Use relative positioning to create logical relationships
- The system will execute your plan exactly as specified

Generate the semantic plan now:
"""
