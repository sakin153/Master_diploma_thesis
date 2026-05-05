"""LLM prompt templates used by the scene planner.

All prompts are generic: they refer to objects by their model_name, parent_id and
relationship description and never assume any particular furniture or item type.
The planner provides the concrete object names, sizes and counts via `{...}`
substitutions.

Pipeline:

1. Graph construction
   - CLASSIFY_HIERARCHY_PROMPT  — flat tree (parent_id) only, no relations.
   - ANCHOR_RELATIONS_PROMPT    — relationships for level-0 (anchor) nodes.
                                  Either anchored to "room_center" with a region
                                  hint, or anchored to another anchor with the
                                  same vocabulary used inside groups.
   - GROUP_RELATIONS_PROMPT     — relationships inside one anchor's group.

2. Placement
   - PLACE_NODE_PROMPT          — single instance, given its relationship.
   - PLACE_NODE_BATCH_PROMPT    — N identical instances of one node.

The placer resolves "room_center" as the virtual point (0, 0) — the center of
the room — and any other reference as the position of a previously placed node
instance. The same prompt handles anchors and dependents.
"""

# ---------------------------------------------------------------------------
# 1. Graph construction
# ---------------------------------------------------------------------------

CLASSIFY_HIERARCHY_PROMPT = """\
CRITICAL: Respond ONLY in English. Return ONLY valid JSON, no other text.

You are a 3D scene-graph builder. Your task: turn a list of object types into a
hierarchy tree (which object sits on / belongs to which other object).

User query: {query}
Room: {room_w}m × {room_l}m

Available object TYPES (sum of all instances must equal {total_count}):
{objects_list}

OUTPUT FORMAT
=============
Return a JSON object with a single field "nodes". Every node represents a
GROUP of identical instances of one model_name.

  {{
    "nodes": [
      {{
        "id":          "<unique snake_case id — see naming convention below>",
        "model_name":  "<exactly one of the available model names>",
        "instances":   <int >= 1>,
        "parent_id":   <id of another node, or null for top-level anchors>
      }},
      ...
    ]
  }}

ID NAMING CONVENTION
====================
- For a node with instances == 1: use the singular model name.
    Examples: "table", "vase", "sofa".
    If two distinct nodes have the same model_name, suffix with _1, _2, ...
    Examples: "lamp_1", "lamp_2".
- For a node with instances > 1: use the plural form.
    Examples: "chairs" (4 chairs), "plates" (4 plates), "books" (3 books).
    If two distinct nodes share the same model_name, suffix _1, _2.
    Examples: "books_1" (on shelf), "books_2" (on desk).
- IDs contain only lowercase ASCII letters, digits and underscores. No spaces.
- Do NOT use the suffix "_group" in IDs — one node already represents a batch
  of identical instances; the plural form is enough.

RULES
=====
- Sum of all node.instances MUST equal {total_count}.
- Every model_name must be one of the available types listed above.
- Identical objects belonging to the same parent must be ONE node with
  instances > 1 (not multiple nodes with instances=1).
- Top-level (anchor) objects have parent_id = null. These are typically the
  largest objects that stand directly on the floor.
- An object that sits on / hangs off / belongs to another object has
  parent_id equal to that object's id.
- Do NOT output "level", "relationship", "anchor_hint", "depends_on" or any
  other field at this stage. Only id / model_name / instances / parent_id.

Return ONLY the JSON object.
"""


ANCHOR_RELATIONS_PROMPT = """\
CRITICAL: Respond ONLY in English. Return ONLY valid JSON, no other text.

You describe how the top-level (anchor) objects relate to each other and to the
room. The room has a virtual reference called "room_center" located at (0, 0).

Room: {room_w}m × {room_l}m. Origin (0, 0) at the room CENTER.
Valid coordinate range: x, y ∈ [{neg_half:.2f}, {pos_half:.2f}].

Anchor nodes:
{anchors_list}

Future dependents per anchor (reserve clearance for them):
{future_per_anchor}

User query: {query}

For EVERY anchor node return a relationship in ONE of two forms.

FORM A — anchored to the room
=============================
Use when the anchor stands on its own and is positioned by where it sits in the
room (in the middle, near a wall, in a corner, …).

  {{
    "id": "<anchor id>",
    "relationship": {{
      "type":      "in_room",
      "reference": "room_center",
      "region":    "center" | "near_wall" | "corner" | "custom",
      "offset_x":  <float, optional — used by 'custom' or to fine-tune>,
      "offset_y":  <float, optional>,
      "yaw_deg":   <float, default 0>
    }}
  }}

FORM B — anchored to another anchor
===================================
Use when one anchor's position is meaningful only relative to another (e.g.
"sofa facing the table", "two desks side by side").

  {{
    "id": "<anchor id>",
    "relationship": {{
      "type":      "facing" | "beside" | "in_front" | "behind",
      "reference": "<id of another anchor>",
      "side":      "+x" | "-x" | "+y" | "-y",
      "distance":  <float meters — GAP between bounding boxes>,
      "facing":    "inward" | "outward" | "any"  (optional),
      "yaw_deg":   <float, optional override>
    }}
  }}

RULES
=====
- AT LEAST ONE anchor must use form A (anchored to room_center). Anchors that
  use form B must form a DAG — no cycles.
- Distance is the gap between bounding boxes, not center-to-center.
- The user query semantics drive the choice ("у стены" → near_wall, "напротив"
  → form B with type=facing, "в углу" → corner, etc.).
- Reserve enough clearance for the dependents listed under each anchor: do not
  place anchors so close that their dependent rings will overlap.

CRITICAL — DO NOT FRAGMENT MULTI-INSTANCE NODES
===============================================
A node ID like "tables" with instances=2 represents BOTH instances together.
Return EXACTLY ONE relationship per anchor node id (as listed above) — the
placer will assign individual coordinates to each instance afterwards.

- DO use the IDs verbatim, exactly as listed in "Anchor nodes" above.
- DO NOT invent suffixes like "tables_1", "tables_2", "chairs_3_1" — these are
  not valid IDs and will be rejected.
- DO NOT split one multi-instance node into multiple per-instance entries.
- The "reference" field MUST be either "room_center" or one of the listed
  anchor IDs verbatim (no suffixes, no splitting).
- The number of entries in "anchor_relations" MUST equal the number of anchor
  nodes listed (one entry per anchor id).

Return JSON:
{{
  "anchor_relations": [ ... ]
}}
"""


GROUP_RELATIONS_PROMPT = """\
CRITICAL: Respond ONLY in English. Return ONLY valid JSON, no other text.

You are describing the spatial relationships INSIDE one anchor's group.

Anchor (already placed):
  id={anchor_id}, model_name={anchor_model}, size_xyz={anchor_size}

Dependent nodes in this group (parent_id chain points to the anchor):
{group_objects}

User query: {query}

For every dependent node return:

  - "id":           the node id (must match input)
  - "relationship": object describing how this node is positioned, with fields:
      * "type":      one of
            "on_surface"   — sits on top of its parent_id (most common for items
                             on a table / shelf / plate)
            "stacked_on"   — sits on top of another instance of same kind
            "around"       — stands on the floor around the parent (chairs
                             around a table). Use ONLY when parent_id IS the
                             anchor and the object is a floor object.
            "beside"       — next to "reference", on the same surface
            "in_front"     — in front of "reference"
            "behind"       — behind "reference"
            "facing"       — facing "reference"
      * "reference": id of another node in this group whose position this node
                     is reasoned relative to, or null. If non-null, you MUST
                     also set "depends_on" to the same id.
      * "side":      "+x" | "-x" | "+y" | "-y" | null — direction from
                     reference (used with beside/in_front/behind/around).
      * "distance":  float meters (gap between bounding boxes), default 0.1.
      * "facing":    "inward" | "outward" | "any" (used with "around").

  - "place_order": small integer; lower = placed earlier among siblings.
  - "depends_on":  id of a node that MUST be placed before this one (always
                   set when relationship.reference is non-null).

NOTES
=====
- "around" implies the object is on the floor, NOT on the parent surface. Use
  it only when parent_id is the anchor itself and the object is a floor object.
- For tabletop arrangements like "items opposite each chair", set parent_id of
  the items to the table (already done upstream) and set
  relationship = {{"type":"on_surface","reference":"<chair group id>","side":...}}.
- Do NOT output ids that are not in the input list.

Return JSON:
{{
  "relations": [
    {{
      "id":           "<node id>",
      "relationship": {{...}},
      "place_order":  <int>,
      "depends_on":   "<id>" | null
    }},
    ...
  ]
}}
"""


# ---------------------------------------------------------------------------
# 2. Placement (unified for anchors and dependents)
# ---------------------------------------------------------------------------

PLACE_NODE_PROMPT = """\
CRITICAL: Respond ONLY in English. Return ONLY valid JSON, no other text.

Place ONE object in the scene.

Object: {model_name} [id={node_id}]
Size (w × d × h): {size_w:.3f} × {size_d:.3f} × {size_h:.3f} m
Half-extents: half_x={size_w_half:.3f}, half_y={size_d_half:.3f}

Placement intent (from scene graph):
{intent}

{reference_block}{parent_block}{siblings_block}{numeric_hints}\
Other objects already placed in scene:
{world_state}

Future objects still to be placed in this scene:
{remaining}

Room: {room_w}m × {room_l}m, origin at center, valid x, y ∈ [{neg_half:.2f}, {pos_half:.2f}].

Geometry hints
- "distance" in the intent is the GAP between bounding boxes, not center-to-center.
- For side="+x" relative to a reference at (rx, ry) with reference half_x = rhx:
      x = rx + rhx + distance + half_x ;  y ≈ ry
  (analogous for ±x / ±y).
- For "in_room" anchors with region="center": (0, 0). With region="near_wall":
  one coordinate is close to ±({pos_half:.2f}) (account for the object's half
  extent and reserve clearance for future dependents). With region="corner":
  both coordinates are close to ±({pos_half:.2f}) with the same caveat.

Rules
- Stay within room bounds (sanity-clamped if you exceed).
- z is computed automatically by the engine from the parent chain — return any
  reasonable z (e.g. 0). The engine will overwrite it.
- yaw_deg in [0, 360); 0 = long axis along +X.
- Reserve enough free space for the future objects listed above.

Return JSON: {{"x": float, "y": float, "z": float, "yaw_deg": float}}
"""


PLACE_NODE_BATCH_PROMPT = """\
CRITICAL: Respond ONLY in English. Return ONLY valid JSON, no other text.

Place {batch_count} identical objects of type "{model_name}".
These are instances {from_idx}..{to_idx} of {total} total.
Size per object (w × d × h): {size_w:.3f} × {size_d:.3f} × {size_h:.3f} m
Half-extents: half_x={size_w_half:.3f}, half_y={size_d_half:.3f}

Placement intent (from scene graph):
{intent}

{reference_block}{parent_block}{siblings_block}{numeric_hints}\
Other objects already placed in scene:
{world_state}

Future objects still to be placed in this scene:
{remaining}

Room: {room_w}m × {room_l}m, origin at center, valid x, y ∈ [{neg_half:.2f}, {pos_half:.2f}].

Rules
- Spread the {batch_count} instances so they do NOT overlap each other.
- Respect the placement intent:
    * "around" → spread instances in a ring around the reference,
                 each instance facing inward (yaw points to the reference center);
    * "on_surface" / "stacked_on" → all centers stay inside the parent's XY
                 footprint;
    * "in_room" → distribute the instances across the room according to the
                 region hint, leaving room for future dependents around each.
- z is computed by the engine — return any reasonable z (e.g. 0).
- yaw_deg in [0, 360).

Return JSON array of EXACTLY {batch_count} items:
[{{"instance": {from_idx}, "x": float, "y": float, "z": float, "yaw_deg": float}}, ...]
"""
