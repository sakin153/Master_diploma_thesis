fmt_objects_qa_tmpl = """
You are a simulation world builder.

Given a user prompt, extract a concise list of scene objects that should appear in the scene.

Rules:
- Output MUST be a JSON list of strings.
- Each string is a simple object name in English (e.g. "car", "person", "table").
- Prefer generic nouns over specific brands.
- Keep it short: usually 2-10 objects.
- Do NOT include duplicates.

Example:
"question": "Pair of shoes on the table"
"answer":
```json
["shoes", "table"]
```

"question": "Two cars and a person next to them"
"answer":
```json
["car", "person"]
```

"question": "A dining room with a table and four chairs"
"answer":
```json
["table", "chair"]
```

Now answer for this prompt.
"""
