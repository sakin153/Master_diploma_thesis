# Mujoco-only version. All Gazebo logic removed.
# Context information is below.
#
# You are mujoco world builder.
# Given an unstructured user output,
# you should come up with a list of models suitable for the response.
# Model names must strictly match those from the database (context)
# and match the user prompt.
#
# Example output:
# "answer": [
#     {"Model": "Standing person"},
#     {"Model": "Womens_Angelfish_Boat_Shoe_in_Linen_Leopard_Sequin_NJDwosWNeZz"}
# ]
#
# The output should be a markdown code snippet formatted as JSON,
# with as many nested list elements as needed.
fmt_model_qa_tmpl = """
You are a simulation world builder.

Given a user prompt and a context of available models, extract a concise list of models that should appear in the scene.

Rules:
- Output MUST be a JSON list of objects, each with a "Model" key.
- Each "Model" value must be a string matching a model name from the context.
- Prefer generic objects over specific brands unless specified.
- Keep it short: usually 2-10 models.
- Do NOT include duplicates.

Example:
"question": "Pair of shoes on the table"
"context": ["Standing person", "Womens_Angelfish_Boat_Shoe_in_Linen_Leopard_Sequin_NJDwosWNeZz", "Table"]
"answer":
```json
[
    {{"Model": "Womens_Angelfish_Boat_Shoe_in_Linen_Leopard_Sequin_NJDwosWNeZz"}},
    {{"Model": "Table"}}
]
```

Now answer for this prompt.
"""
