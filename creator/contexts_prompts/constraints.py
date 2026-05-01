fmt_constraints_plan_tmpl = """
You are a professional scene planner for a MuJoCo physics simulator.

User query: {query}

Chosen models for this scene:
{models_str}

CRITICAL RULE: You MUST include ALL N instances of each object type shown above.
- If the list shows "table x4 → table_1, table_2, table_3, table_4", your output MUST contain all 4 tables
- If the list shows "chair x16 → chair_1 ... chair_16", your output MUST contain all 16 chairs
- Missing even one instance is a critical error

Before generating JSON, verify:
✓ Count of each object type in your output matches the count shown in "Objects (N total)" above
✓ All instances are numbered sequentially: object_1, object_2, ..., object_N

---

## Hierarchical placement graph

Think in THREE layers:

**Layer 1 — Anchors** (1-2 large furniture pieces that define the scene)
- Get `region: middle` or `region: edge`
- Everything else references them

**Layer 2 — Floor furniture** (chairs, stools, secondary tables, shelves)
- Use `beside` + `face_to` relative to the anchor
- NEVER use `on_top_of` for floor furniture

**Layer 3 — Surface items** (plates, cups, books, keyboards, lamps, vases, phones, food)
- ALWAYS use `on_top_of` pointing to their surface (table, shelf, desk, box)
- NEVER use `near` or `region` for surface items — they must have `on_top_of`
- For multiple instances of the same surface: use `_1`, `_2` suffix (first "table" = `table_1`)

---

## Constraint types

**on_top_of** — object rests on a surface
  - `target`: surface name (use `table_1`, `table_2` for multiple tables)
  - Required for: ALL small objects (plates, cups, books, lamps, vases, keyboards, food, phones)
  - FORBIDDEN for: chairs, stools, sofas, beds, cabinets (they stand on the floor)

**beside** — placed at a specific side of target
  - `target`, `side`: "front"|"back"|"left"|"right", `distance`: [min, max] meters
  - For chairs around a table: assign different sides so they spread out
    * 2 chairs → "front", "back"
    * 4 chairs → "front", "back", "left", "right"

**face_to** — object's front faces toward target
  - Always pair with `beside` for chairs/stools facing their table

**region** — room-level placement
  - `value`: "middle" (anchor only) | "edge" (wall furniture)
  - Only ONE object gets `region: middle` — the primary anchor

**near** — loose grouping, any direction
  - Only for floor-level objects with no specific side requirement
  - NEVER for surface items

---

## Distances (center-to-center)

- Chair around dining table (1.2-1.8m long): `[0.8, 1.2]`
- Stool at bar counter: `[0.5, 0.9]`
- Side table beside sofa: `[0.4, 0.8]`

---

## is_static

- `true`: all furniture (tables, chairs, sofas, beds, shelves, cabinets)
- `false`: all small objects (books, cups, plates, keyboards, phones, food, vases, lamps)

---

## Output format

```json
{{
  "objects": [
    {{
      "Model": "dining table",
      "is_static": true,
      "constraints": [
        {{"type": "region", "value": "middle", "hard": false, "weight": 5.0}}
      ]
    }},
    {{
      "Model": "chair",
      "is_static": true,
      "constraints": [
        {{"type": "beside", "target": "dining table", "side": "front", "distance": [0.8, 1.2], "hard": false, "weight": 3.0}},
        {{"type": "face_to", "target": "dining table", "hard": false, "weight": 2.0}}
      ]
    }},
    {{
      "Model": "chair",
      "is_static": true,
      "constraints": [
        {{"type": "beside", "target": "dining table", "side": "back", "distance": [0.8, 1.2], "hard": false, "weight": 3.0}},
        {{"type": "face_to", "target": "dining table", "hard": false, "weight": 2.0}}
      ]
    }},
    {{
      "Model": "plate",
      "is_static": false,
      "constraints": [
        {{"type": "on_top_of", "target": "dining table", "hard": false, "weight": 1.0}}
      ]
    }},
    {{
      "Model": "lamp",
      "is_static": false,
      "constraints": [
        {{"type": "on_top_of", "target": "dining table", "hard": false, "weight": 1.0}}
      ]
    }}
  ]
}}
```

### Example: two boxes on a table, apples in each box

```json
{{
  "objects": [
    {{"Model": "table", "is_static": true, "constraints": [{{"type": "region", "value": "middle", "hard": false, "weight": 5.0}}]}},
    {{"Model": "box", "is_static": true, "constraints": [{{"type": "on_top_of", "target": "table", "hard": false, "weight": 1.0}}]}},
    {{"Model": "box", "is_static": true, "constraints": [{{"type": "on_top_of", "target": "table", "hard": false, "weight": 1.0}}]}},
    {{"Model": "apple", "is_static": false, "constraints": [{{"type": "on_top_of", "target": "box_1", "hard": false, "weight": 1.0}}]}},
    {{"Model": "apple", "is_static": false, "constraints": [{{"type": "on_top_of", "target": "box_1", "hard": false, "weight": 1.0}}]}},
    {{"Model": "apple", "is_static": false, "constraints": [{{"type": "on_top_of", "target": "box_2", "hard": false, "weight": 1.0}}]}},
    {{"Model": "apple", "is_static": false, "constraints": [{{"type": "on_top_of", "target": "box_2", "hard": false, "weight": 1.0}}]}}
  ]
}}
```

### Example: restaurant (multiple tables)

```json
{{
  "objects": [
    {{"Model": "table", "is_static": true, "constraints": [{{"type": "region", "value": "middle", "hard": false, "weight": 5.0}}]}},
    {{"Model": "table", "is_static": true, "constraints": [{{"type": "beside", "target": "table_1", "side": "left", "distance": [2.0, 3.0], "hard": false, "weight": 2.0}}]}},
    {{"Model": "table", "is_static": true, "constraints": [{{"type": "beside", "target": "table_1", "side": "right", "distance": [2.0, 3.0], "hard": false, "weight": 2.0}}]}},
    {{"Model": "chair", "is_static": true, "constraints": [{{"type": "beside", "target": "table_1", "side": "front", "distance": [0.8, 1.2], "hard": false, "weight": 3.0}}, {{"type": "face_to", "target": "table_1", "hard": false, "weight": 2.0}}]}},
    {{"Model": "chair", "is_static": true, "constraints": [{{"type": "beside", "target": "table_1", "side": "back", "distance": [0.8, 1.2], "hard": false, "weight": 3.0}}, {{"type": "face_to", "target": "table_1", "hard": false, "weight": 2.0}}]}},
    {{"Model": "plate", "is_static": false, "constraints": [{{"type": "on_top_of", "target": "table_1", "hard": false, "weight": 1.0}}]}},
    {{"Model": "plate", "is_static": false, "constraints": [{{"type": "on_top_of", "target": "table_2", "hard": false, "weight": 1.0}}]}}
  ]
}}
```
"""

fmt_seating_plan_tmpl = """
You are a scene planner for a MuJoCo physics simulator.

The scene already has these anchor objects placed:
{anchors_str}

Now assign constraints for these remaining objects:
{children_str}

CRITICAL DISTRIBUTION RULE:
- You MUST distribute children EVENLY across ALL anchors listed above
- Count the anchors above and divide children equally
- Example: 4 anchors (table_1, table_2, table_3, table_4) and 16 children → assign EXACTLY 4 children to EACH anchor
- Assign children sequentially: first N to anchor_1, next N to anchor_2, etc.

Rules:
- Each object gets EXACTLY ONE `beside` constraint pointing to one anchor, and ONE `face_to` the same anchor.
- Distribute objects across anchors as the scene requires.
- `side`: "front" | "back" | "left" | "right" — which side of the anchor to place on.
- `offset`: lateral offset in meters along the side edge (0.0 = centered, positive = right along edge, negative = left along edge).
  * 1 chair per side → offset: 0.0
  * 2 chairs per side → offsets: -0.3, +0.3
  * 3 chairs per side → offsets: -0.5, 0.0, +0.5
  * 4 chairs per side → use all 4 sides (front, back, left, right) with offset 0.0
- `distance`: gap from anchor edge to object center, in meters. Chairs: 0.4–0.6.
- `is_static`: true for chairs/stools, false for small objects.

Output JSON only:
```json
{{
  "objects": [
    // Example: 2 tables, 8 chairs (4 per table)
    // Chairs for table_1 (chair_1 through chair_4)
    {{
      "Model": "computer chair",
      "is_static": true,
      "constraints": [
        {{"type": "beside", "target": "table_1", "side": "front", "offset": 0.0, "distance": 0.5, "hard": false, "weight": 3.0}},
        {{"type": "face_to", "target": "table_1", "hard": false, "weight": 2.0}}
      ]
    }},
    {{
      "Model": "computer chair",
      "is_static": true,
      "constraints": [
        {{"type": "beside", "target": "table_1", "side": "back", "offset": 0.0, "distance": 0.5, "hard": false, "weight": 3.0}},
        {{"type": "face_to", "target": "table_1", "hard": false, "weight": 2.0}}
      ]
    }},
    {{
      "Model": "computer chair",
      "is_static": true,
      "constraints": [
        {{"type": "beside", "target": "table_1", "side": "left", "offset": 0.0, "distance": 0.5, "hard": false, "weight": 3.0}},
        {{"type": "face_to", "target": "table_1", "hard": false, "weight": 2.0}}
      ]
    }},
    {{
      "Model": "computer chair",
      "is_static": true,
      "constraints": [
        {{"type": "beside", "target": "table_1", "side": "right", "offset": 0.0, "distance": 0.5, "hard": false, "weight": 3.0}},
        {{"type": "face_to", "target": "table_1", "hard": false, "weight": 2.0}}
      ]
    }},
    // Chairs for table_2 (chair_5 through chair_8)
    {{
      "Model": "computer chair",
      "is_static": true,
      "constraints": [
        {{"type": "beside", "target": "table_2", "side": "front", "offset": 0.0, "distance": 0.5, "hard": false, "weight": 3.0}},
        {{"type": "face_to", "target": "table_2", "hard": false, "weight": 2.0}}
      ]
    }},
    {{
      "Model": "computer chair",
      "is_static": true,
      "constraints": [
        {{"type": "beside", "target": "table_2", "side": "back", "offset": 0.0, "distance": 0.5, "hard": false, "weight": 3.0}},
        {{"type": "face_to", "target": "table_2", "hard": false, "weight": 2.0}}
      ]
    }},
    {{
      "Model": "computer chair",
      "is_static": true,
      "constraints": [
        {{"type": "beside", "target": "table_2", "side": "left", "offset": 0.0, "distance": 0.5, "hard": false, "weight": 3.0}},
        {{"type": "face_to", "target": "table_2", "hard": false, "weight": 2.0}}
      ]
    }},
    {{
      "Model": "computer chair",
      "is_static": true,
      "constraints": [
        {{"type": "beside", "target": "table_2", "side": "right", "offset": 0.0, "distance": 0.5, "hard": false, "weight": 3.0}},
        {{"type": "face_to", "target": "table_2", "hard": false, "weight": 2.0}}
      ]
    }}
  ]
}}
```
"""

