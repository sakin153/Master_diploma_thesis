fmt_constraints_plan_tmpl = """
You are a scene planner for a MuJoCo world generator.

Available model catalog:
{catalog_str}

Chosen models for this scene:
{models_str}

Your task:
Return a semantic placement plan for objects.
For each object include constraints that can be consumed by a deterministic solver.

Allowed constraint types:
- region: value is "edge" or "middle"
- near / far: target + distance range [min,max]
- left_of / right_of / in_front_of / behind: target
- face_to / face_same_as: target
- center_aligned: target

Rules:
- Use only model names that exist in chosen models.
- Keep output concise and realistic.
- Prefer soft constraints (hard=false) unless clearly necessary.

Output format (JSON only):
```json
{{
  "objects": [
    {{
      "Model": "ModelName",
      "constraints": [
        {{"type": "region", "value": "middle", "hard": false, "weight": 1.0}},
        {{
          "type": "near",
          "target": "OtherModel",
          "distance": [0.3, 1.8],
          "hard": false,
          "weight": 0.8
        }}
      ]
    }}
  ]
}}
```
"""
