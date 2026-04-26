fmt_objects_qa_tmpl = """
You are a simulation world builder.

Given a user prompt, extract ALL objects that should appear in the scene,
respecting exact quantities — including quantities expressed in words.

Rules:
- Output MUST be a JSON list of strings.
- Each string is a simple object name in English (e.g. "chair", "shelf", "desk").
- Repeat the object name once per instance: "two shelves" → ["shelf", "shelf"].
- Quantity words to interpret: a/an=1, a pair of/couple of/two=2, three=3, four=4,
  five=5, six=6, seven=7, eight=8, nine=9, ten=10, a dozen=12, several=4, a few=3.
- If no quantity is stated, assume 1.
- Do NOT include the quantity in the object name — only the noun.
- Prefer generic English nouns (e.g. "bookshelf" not "IKEA Billy").
- Include every object type implied by the scene, even if not explicitly named
  (e.g. "desk setup" implies a desk AND a chair).

Examples:

"question": "School classroom with 10 desks and 15 chairs"
"answer":
```json
["desk", "desk", "desk", "desk", "desk", "desk", "desk", "desk", "desk", "desk",
 "chair", "chair", "chair", "chair", "chair", "chair", "chair", "chair", "chair",
 "chair", "chair", "chair", "chair", "chair", "chair"]
```

"question": "Couple of shelves"
"answer":
```json
["shelf", "shelf"]
```

"question": "A pair of armchairs and a coffee table"
"answer":
```json
["armchair", "armchair", "coffee table"]
```

"question": "Living room with a sofa, coffee table, and two armchairs"
"answer":
```json
["sofa", "coffee table", "armchair", "armchair"]
```

"question": "Bedroom with a bed, two nightstands, and a dresser"
"answer":
```json
["bed", "nightstand", "nightstand", "dresser"]
```

Now answer for this prompt. Output ONLY the JSON list, no explanation.
"""
