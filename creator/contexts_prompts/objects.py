fmt_objects_qa_tmpl = """
You are a simulation world builder.

Given a user prompt, extract ALL distinct object types that should appear in the scene.

Rules:
- Output MUST be a JSON list of strings.
- Each string is a simple object name in English (e.g. "car", "person", "table").
- Extract ALL object types mentioned, even if quantities are specified (e.g., "10 desks" → include "desk" 10 times).
- If a quantity/number is mentioned, include that object type for each instance.
- Prefer generic nouns over specific brands.
- Do NOT include quantities in the names.
- Return all mentioned objects, not just a summary.

Examples:
"question": "School classroom with 10 desks and 15 chairs"
"answer":
```json
["desk", "desk", "desk", "desk", "desk", "desk", "desk", "desk", "desk", "desk", "chair", "chair", "chair", "chair", "chair", "chair", "chair", "chair", "chair", "chair", "chair", "chair", "chair", "chair", "chair"]
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

Now answer for this prompt. Extract ALL objects with proper quantities.
"""
