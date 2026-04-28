list_items_prompt_tmpl = """
You are a professional robotics mujoco simulation world builder.

Given a user prompt, extract the objects that should appear in the scene.
You are NOT limited to a predefined list of models — you may use any objects explicitly mentioned in the prompt.

Rules:
- Output MUST be a JSON list of objects.
- Each object MUST contain:
    - "Model": simple name of the object
    - "Description": detailed visual description (color, material, shape, size, texture, features) suitable for generating an image of the object
    - "Quantity": number of absolutely identical instances.
- Every "Model" value must exactly match a name from the provided context list.
- If the prompt requests N identical objects, merge them into ONE object with a "Quantity" field.
- If the prompt does not provide specific details about an object, choose the most appropriate ones based on context and general meaning.
- For model names, use simple, short names — not complex or long names.
- For quantities, use only positive integers.
- If the prompt requests a specific number of objects, use that number.

Examples:

question: "Office with 3 desks and 3 chairs"
answer:
[
    {
        "Model": "Office Desk",
        "Description": "A modern office desk with a rectangular dark gray metal frame, smooth black matte top surface, two side drawers with silver handles, 4 adjustable leveling feet, approximately 4 feet wide and 2 feet deep.",
        "Quantity": 3
    },
    {
        "Model": "Office Chair",
        "Description": "An ergonomic office chair with a black mesh backrest, padded black fabric seat cushion, 5 casters wheels, height-adjustable pneumatic lever, armrests covered in soft black padding.",
        "Quantity": 3
    }
]

question: "Pair of shoes on the table"
answer:
[
    {
        "Model": "Sneaker",
        "Description": "A casual athletic sneaker with white leather upper, thick rubber sole, lace-up front, rounded toe, padded ankle collar, blue accent stripes on both sides.",
        "Quantity": 2
    },
    {
        "Model": "Table",
        "Description": "A simple wooden table with a rectangular light oak top, four straight square legs, natural wood grain visible, approximately 4 feet wide and 2.5 feet deep.",
        "Quantity": 1
    }
]
"""

expand_prompt_tmpl = """


_EXPAND_PROMPT = """\
You are a 3D scene designer for a robotics simulator (MuJoCo).
Your task: take a short scene description and expand it into a detailed, realistic scene specification.

User query: "{query}"

Respond with JSON only:
```json
{{
  "expanded_description": "Full detailed description of the scene (3-5 sentences). Include room type, style, specific objects with quantities, arrangement logic.",
  "room_type": "bedroom|office|classroom|kitchen|living_room|warehouse|lab|outdoor|other",
  "room_style": "modern|minimalist|cozy|industrial|academic|other",
  "estimated_objects": [
    {{"name": "Object Name", "quantity": 1, "notes": "brief description or placement hint"}}
  ],
  "room_dimensions_hint": "small (3x3m)|medium (5x5m)|large (8x8m)|extra_large (12x12m)"
}}
```

Rules:
- Be specific about quantities (e.g., "10 desks" not "some desks")
- Include ALL objects that make the scene realistic and functional:
  * Large furniture (tables, chairs, sofas, beds, shelves...)
  * Functional items (lamps, computers, books, dishes, utensils....)
  * Decorative items (plants, art, rugs, vases...)
  * Task-specific items (for dining: plates/cutlery/glasses, for office: keyboard/mouse/monitor, for bedroom: pillows/blankets)
- Add 10-20 objects for a (rich/populated) realistic scene
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

_SCENE_TOO_SHORT_WORDS = 8   # expand if fewer than this many words

_ROOM_DEFAULTS: Dict[str, List[tuple]] = {
    "classroom": [
        ("desk", 10, "student desks arranged in rows"),
        ("chair", 10, "chairs aligned with desks"),
        ("teacher desk", 1, "at front of the room"),
        ("whiteboard", 1, "mounted on front wall"),
    ],
    "office": [
        ("desk", 1, "main work desk"),
        ("office chair", 1, "near desk"),
        ("bookshelf", 1, "against wall"),
    ],
    "bedroom": [
        ("bed", 1, "main bed"),
        ("nightstand", 2, "one on each side of bed"),
        ("wardrobe", 1, "against wall"),
    ],
    "living_room": [
        ("sofa", 1, "main seating"),
        ("coffee table", 1, "in front of sofa"),
        ("armchair", 2, "around coffee table"),
    ],
}






# """
# **Role:**  
# You are a 3D scene designer for the MuJoCo robotics simulator.

# **Task:**  
# Analyze the user’s prompt and decide whether it provides enough detail to create a complete 3D scene.

# **Instructions:**  

# 1. **If the description is too short or generic** — for example:  
#    - `"a bedroom"`  
#    - `"an office"`  
#    - `"a living room"`  
#    → You must expand it **in English** into a detailed scene description, including:  
#      - List of objects  
#      - Their positions and spatial arrangement  
#      - Relative orientation  
#      - Any other relevant details necessary for a MuJoCo scene.

# 2. **If the description is already detailed and complete** — for example:  
#    - `"A bed in the center of the room. Nightstands with lamps on both sides of the bed. A wardrobe opposite the bed."`  
#    → Do **not** add anything. Keep the original description as is.

# **Important constraints:**  
# - Do **not** use exact numbers, precise measurements, or specific coordinate values (e.g., avoid "1.5 meters", "45 degrees", or "at x=2, y=3").  
# - Use descriptive, qualitative language instead (e.g., "near the wall", "slightly to the left", "facing the bed").

# **Output format:**  
# Your response must contain **only** the resulting scene description (either the expanded or the original one).  
# Do **not** include explanations, analysis, or meta-comments about your process.

# **Examples:**  

# - User: `"a bedroom"`  
#   → You: *(detailed description of the bedroom, objects, layout, etc., using only qualitative terms)*

# - User: `"A bed in the center of the room. Nightstands with lamps on both sides of the bed. A wardrobe opposite the bed."`  
#   → You: *(same text, unchanged)*
# """
