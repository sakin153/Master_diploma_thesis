expand_prompt_tmpl = """\
You are a 3D scene designer for a robotics simulator (MuJoCo).
Your task: take a short scene description and expand it into a detailed, realistic scene specification.

Respond with JSON only:
```json
{
  "expanded_description": "Full detailed description of the scene (3-5 sentences). Include room type, style, specific objects with quantities, arrangement logic.",
  "room_type": "bedroom|office|classroom|kitchen|living_room|warehouse|lab|outdoor|other",
  "room_style": "modern|minimalist|cozy|industrial|academic|other",
  "estimated_objects": [
    {"name": "Object Name", "quantity": 1, "notes": "brief description or placement hint"}
  ],
  "room_dimensions_hint": "small (3x3m)|medium (5x5m)|large (8x8m)|extra_large (12x12m)"
}
```

Rules:
- Be specific about quantities (e.g., "10 desks" not "some desks")
- Include ALL objects that make the scene realistic and functional:
  * Large furniture (tables, chairs, sofas, beds, shelves)
  * Functional items (lamps, computers, books, dishes, utensils)
  * Decorative items (plants, art, rugs, vases)
  * Task-specific items (for dining: plates/cutlery/glasses, for office: keyboard/mouse/monitor, for bedroom: pillows/blankets)
- Add 10-20 objects for a насыщенная (rich/populated) realistic scene
- Think: "What would a real person use in this space?"
- Respond ONLY with valid JSON inside ```json ... ```

Arrangement intent (write these into each object's `notes` field — the
planner uses the words you write here to choose constraints, so be precise
without prescribing a specific scenario):
- A SINGLE small item resting on a larger surface → "centered on <surface>".
- MULTIPLE identical items resting on the SAME surface → "evenly distributed
  on <surface>".
- MULTIPLE same-type items grouped around a central anchor (any kind:
  table, rug, fire pit, podium, sofa, etc.) → "evenly distributed around
  <anchor>, facing it".
- An item that should hang on or attach to a wall → "mounted on a wall".
Do NOT name specific furniture combos (e.g. "chair per plate") — describe
the intent only. The placement engine will discover spatial pairings from
counts and positions.
"""
