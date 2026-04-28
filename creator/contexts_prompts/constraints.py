fmt_constraints_plan_tmpl = """
You are a professional scene planner for a MuJoCo physics simulator.

User query: {query}

Chosen models for this scene:
{models_str}

RULE: Output EXACTLY the objects listed above — no more, no less. Each object appears exactly once.

---

## Constraint types and their parameters

**on_top_of** — object physically rests on a surface
  - `target`: name of the surface object
  - **When multiple instances of the target exist, use `_1`, `_2`, ... suffix to specify which one**: `"box_1"`, `"box_2"`, etc. The suffix matches the order objects appear in your output (first "box" = `box_1`, second "box" = `box_2`).
  - Use for: plates, cups, books, keyboards, monitors, lamps, vases, phones, apples, any small object on a surface
  - NEVER use for furniture that stands on the floor (chairs, tables, sofas, beds, cabinets)
  - **INCOMPATIBLE with `beside`**: if an object has `beside`, it stands on the floor — do NOT add `on_top_of`

**beside** — object placed at a specific side of target
  - `target`: name of the reference object
  - `side`: "front" | "back" | "left" | "right" (relative to target's facing direction)
  - `distance`: [min_meters, max_meters]
  - Use for: seating around tables, nightstands beside beds, appliances side-by-side

**near** — object placed close to target, any direction
  - `target`: name of the reference object
  - `distance`: [min_meters, max_meters]
  - Use for: loose grouping (lamp near sofa, plant near window)

**face_to** — object's front faces toward target
  - `target`: name of the object to face
  - Use for: chairs facing tables, sofas facing TVs, monitors facing chairs

**center_aligned** — object centered along one axis relative to target
  - `target`: name of the reference object
  - Use for: single chair centered at a desk, TV centered on a stand

**region** — room-level placement
  - `value`: "middle" (open floor center) | "edge" (against a wall)
  - Use "middle" only for the primary anchor object
  - Use "edge" for: shelves, cabinets, TVs, sofas against walls

---

## Distances reference

`distance` = center-to-center distance in meters between the two objects.
Account for object sizes: a table ~1.5m long has half-width ~0.75m, a chair ~0.5m wide has half-width ~0.25m, so minimum center-to-center is ~1.0m.

- Chair/stool around a dining table (1.2–1.8m): `[0.8, 1.2]`
- Chair at a desk (0.8–1.2m wide): `[0.6, 1.0]`
- Nightstand beside bed: `[0.4, 0.8]`
- Lamp beside sofa or desk: `[0.5, 1.2]`
- Objects side-by-side (appliances, shelves): `[0.3, 0.7]`

---

## Multiple identical objects around an anchor

When N chairs surround a table, assign each a DIFFERENT `side` so they spread around the table:
- 2 chairs → "front", "back"
- 4 chairs → "front", "back", "left", "right"
- 6 chairs → "front", "back", "left", "right", then repeat with near for extras

Each chair also gets `face_to` pointing at the table.

---

## is_static (required for every object)

- `true`: furniture and appliances that never move (tables, chairs, sofas, beds, cabinets, fridges, ovens, shelves)
- `false`: small objects that can be picked up (books, cups, plates, keyboards, phones, remotes, vases, decorative items)

---

## Primary anchor

Pick one central object as the scene anchor and give it:
`{{"type": "region", "value": "middle", "hard": false, "weight": 5.0}}`

All other furniture references the anchor via `beside` or `near`.

**When there are multiple identical large objects (e.g. 4 tables in a restaurant):**
- The FIRST instance gets `region: middle` (anchor)
- Each subsequent instance gets `beside` pointing to the previous one with a large distance, using different sides to spread them across the room:
  - table_2 → `beside: table_1, side: "left", distance: [2.0, 3.5]`
  - table_3 → `beside: table_1, side: "right", distance: [2.0, 3.5]`
  - table_4 → `beside: table_1, side: "back", distance: [2.0, 3.5]`
  - table_5 → `beside: table_2, side: "back", distance: [2.0, 3.5]`
- Do NOT give all instances `region: middle` — they will all pile up at the center.

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
      "Model": "chair",
      "is_static": true,
      "constraints": [
        {{"type": "beside", "target": "dining table", "side": "left", "distance": [0.8, 1.2], "hard": false, "weight": 3.0}},
        {{"type": "face_to", "target": "dining table", "hard": false, "weight": 2.0}}
      ]
    }},
    {{
      "Model": "chair",
      "is_static": true,
      "constraints": [
        {{"type": "beside", "target": "dining table", "side": "right", "distance": [0.8, 1.2], "hard": false, "weight": 3.0}},
        {{"type": "face_to", "target": "dining table", "hard": false, "weight": 2.0}}
      ]
    }},
    {{
      "Model": "office chair",
      "is_static": true,
      "constraints": [
        {{"type": "beside", "target": "desk", "side": "front", "distance": [0.6, 1.0], "hard": false, "weight": 3.0}},
        {{"type": "face_to", "target": "desk", "hard": false, "weight": 2.0}},
        {{"type": "center_aligned", "target": "desk", "hard": false, "weight": 2.0}}
      ]
    }},
    {{
      "Model": "plate",
      "is_static": false,
      "constraints": [
        {{"type": "on_top_of", "target": "dining table", "hard": false, "weight": 1.0}}
      ]
    }}
  ]
}}
```

### Example: two boxes on a table, apples distributed between them

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
"""
