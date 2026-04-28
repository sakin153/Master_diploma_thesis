fmt_disambiguation_tmpl = """
You are picking ONE 3D model variant from a small candidate list.

Inputs you receive:
- "scene_query": the user's full scene description (may be in any language).
- "object": the abstract object slot to fill (English noun).
- "candidates": JSON list of {{uuid, name, description}} — at most 10 items
  drawn from a pre-filtered catalog. All candidates are plausible; pick the
  one that BEST matches the user's intent.

Selection rules:
- Honor adjectives from "scene_query": colors, materials, sizes, style.
- Prefer candidates whose `name` is the same noun as `object`.
- Penalize candidates whose name is a *different* sub-type (e.g. for
  object="box", prefer name="box" over name="tissue box" unless the user
  explicitly mentioned tissues).
- If nothing in the query disambiguates, pick the most generic / typical
  candidate (the first plausible match).
- IMPORTANT — reject when nothing fits: if every candidate represents a
  fundamentally different physical object than `object` (e.g. object="TV
  stand" but candidates are all "phone stand"; object="sofa" but candidates
  are all "chair"; object="bookshelf" but candidates are all "table"),
  return {{"uuid": "none"}}. Use this only when the mismatch is categorical
  (a phone stand is not a TV stand even if both share the word "stand").
  When in doubt and at least one candidate is genuinely close, pick it.

Output: a single JSON object with the chosen uuid, no prose.
Example pick: {{"uuid": "0c608106b4045f4da0c651c17d431f5e"}}
Example reject: {{"uuid": "none"}}

scene_query: {scene_query}
object: {object}
candidates: {candidates}

Output ONLY the JSON object.
"""
