"""LLM prompt template for determining object placement priorities.

This module contains the prompt template for the LLM to analyze the scene
and determine which objects should be placed first (anchors) and which
should be placed relative to others (dependent objects).
"""

fmt_placement_priority_tmpl = """
CRITICAL: You MUST respond ONLY in English. Do NOT use any other language.
CRITICAL: Your response MUST be valid JSON only.

You are a scene planning expert. Your task is to analyze a scene description
and determine the optimal order for placing objects.

User query: {query}

Available objects to place:
{objects_str}

---

## Your Task

Analyze the scene and divide objects into priority groups:

1. **Anchor objects** - Objects that should be placed first with absolute positions
   - Usually: tables, sofas, beds, large furniture
   - These define the main structure of the scene

2. **Dependent objects** - Objects that should be placed relative to anchors
   - Usually: chairs (around tables), lamps (on tables), cushions (on sofas)
   - These are positioned based on anchor objects

---

## Output Format

Generate a JSON response with this structure:

```json
{{
  "anchor_objects": [
    {{"index": 0, "reason": "Main table - defines dining area"}},
    {{"index": 1, "reason": "Second table - defines workspace"}}
  ],
  "dependent_objects": [
    {{"index": 2, "reason": "Chair - should be placed around table"}},
    {{"index": 3, "reason": "Chair - should be placed around table"}},
    {{"index": 4, "reason": "Lamp - should be placed on table surface"}}
  ]
}}
```

---

## Rules

1. **Use index numbers** from the "Available objects" list above (0-indexed)
2. **Provide brief reason** for each classification
3. **Anchor objects** should be large, structural items that define the scene
4. **Dependent objects** should be items that naturally relate to anchors
5. **Consider semantics**: "4 tables with 3 chairs each" → tables are anchors, chairs are dependent

---

## Examples

### Example 1: Dining scene
Query: "dining table with 4 chairs"
Objects: [table, chair, chair, chair, chair]

Response:
```json
{{
  "anchor_objects": [
    {{"index": 0, "reason": "Dining table - main anchor for the scene"}}
  ],
  "dependent_objects": [
    {{"index": 1, "reason": "Chair - placed around table"}},
    {{"index": 2, "reason": "Chair - placed around table"}},
    {{"index": 3, "reason": "Chair - placed around table"}},
    {{"index": 4, "reason": "Chair - placed around table"}}
  ]
}}
```

### Example 2: Office scene
Query: "2 desks with chairs and lamps"
Objects: [desk, desk, chair, chair, lamp, lamp]

Response:
```json
{{
  "anchor_objects": [
    {{"index": 0, "reason": "First desk - defines workspace area"}},
    {{"index": 1, "reason": "Second desk - defines second workspace"}}
  ],
  "dependent_objects": [
    {{"index": 2, "reason": "Chair - positioned at first desk"}},
    {{"index": 3, "reason": "Chair - positioned at second desk"}},
    {{"index": 4, "reason": "Lamp - placed on first desk surface"}},
    {{"index": 5, "reason": "Lamp - placed on second desk surface"}}
  ]
}}
```

### Example 3: Living room
Query: "sofa with coffee table and 2 cushions"
Objects: [sofa, coffee_table, cushion, cushion]

Response:
```json
{{
  "anchor_objects": [
    {{"index": 0, "reason": "Sofa - main seating area anchor"}},
    {{"index": 1, "reason": "Coffee table - secondary anchor in front of sofa"}}
  ],
  "dependent_objects": [
    {{"index": 2, "reason": "Cushion - placed on sofa"}},
    {{"index": 3, "reason": "Cushion - placed on sofa"}}
  ]
}}
```

---

Now analyze the scene and generate the placement priority plan:
"""
