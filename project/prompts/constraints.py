fmt_constraints_plan_tmpl = """
You are a scene planner for a MuJoCo world generator.

User query: {query}

Chosen models for this scene:
{models_str}

Your task:
Return a semantic placement plan with detailed spatial relationships.
CRITICAL: Parse the user query carefully for object relationships:
- "X on Y" → add {{"type": "on_top_of", "target": "Y"}} to X
- "X with Y on it" → add {{"type": "on_top_of", "target": "X"}} to Y
- "X around Y" → add {{"type": "near", "target": "Y"}} + {{"type": "face_to", "target": "Y"}} to X

Allowed constraint types:
- region: value is "edge" or "middle"
- on_top_of: target (object should be on top of target)
- near: target + distance range [min,max]
- left_of / right_of / in_front_of / behind: target (directional placement)
- face_to: target (object faces toward target)

Rules:
- Use ONLY model names from chosen models above
- Always add on_top_of when user says "X on Y"
- For seating around tables: distribute using directional constraints + face_to
- Prefer soft constraints (hard=false)

Output JSON only:
```json
{{
  "objects": [
    {{
      "Model": "ModelName",
      "constraints": [
        {{"type": "on_top_of", "target": "OtherModel", "hard": false, "weight": 1.0}}
      ]
    }}
  ]
}}
```
"""
