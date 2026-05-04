"""Prompt templates used by Stage 4 scene planning.

This module exists to keep `scene_planner.py` smaller and more readable.
"""

_GRAPH_HIERARCHY_PROMPT = """
CRITICAL: Respond ONLY in English. Return ONLY valid JSON, no other text.

You are a 3D scene layout planner. Build a scene hierarchy graph from the user query.

User query: {query}
Room: {room_w}m × {room_l}m

Available object TYPES and counts (you MUST account for ALL instances; sum(instances) == {total_count}):
{objects_list}

CORE RULES
==========

GROUP NODE RULE (CRITICAL):
    Nodes represent GROUPS of identical objects.
    - If multiple identical objects are placed in the same semantic group (same parent_id and same relationship intent),
        you MUST emit ONE node with instances > 1.
    - 2 books on the same desk → one node: {{"id": "book_group_1", "instances": 2, ...}}
    - 4 chairs around a table → one node: {{"id": "chair_group_1", "instances": 4, ...}}
    - Only split into multiple nodes if the groups are genuinely different (different parent_id and/or different relationship.type/reference).

PARENT_ID = "semantic parent group" (used for dependency grouping and placement ordering).
    - Anchors (large floor furniture): parent_id = null.
    - Things that belong to an anchor group: parent_id = the anchor id.
    - Things on top of a surface: parent_id = the surface id.

Z / SUPPORT NOTE:
    - The engine will compute Z. Use relationship.type to express support intent:
        * "on_surface" / "stacked_on" → object sits ON TOP of parent_id.
        * "around" → object sits on the FLOOR around the reference/parent.

RELATIONSHIP.REFERENCE = "what other object my X/Y is computed relative to"
                        (horizontal/spatial, optional, may differ from parent_id).
  - The glass goes between two books: book_1.relationship.reference = "glass_1",
    book_2.relationship.reference = "glass_1" (both books sit on the desk
    surface, but their X/Y is reasoned relative to the glass).

OBJECT FIELDS
=============
- id: unique node id (e.g. "chair_group_1", "plate_group_1")
- model_name: must match an available object name exactly
- instances: integer >= 1 (use >1 for repeated objects)
- parent_id: id of the surface the object sits on, or null for anchors
- anchor_hint: only for anchors (parent_id=null). E.g. {{"region": "center"}}
              or {{"region": "near_wall", "offset_y": -1.5}}
- relationship: object describing how this node is positioned (see below)
- place_order: integer; lower = placed earlier among siblings (default 1)
- depends_on: id of another node that MUST be placed before this one
              (use whenever relationship.reference is set)

RELATIONSHIP OBJECT
===================
For anchor nodes (parent_id=null), set relationship = null.
For all other nodes, relationship is an object with these fields:

  "type": one of
      "on_surface"   → sitting on top of parent (most common; flat objects on tables, etc.)
      "stacked_on"   → identical to on_surface; use when parent itself is the object below
      "beside"       → next to reference, on the same surface
      "around"       → arranged in a ring around parent (chairs around table)
      "in_front"     → in front of reference (relative to reference's facing)
      "behind"       → behind reference
      "facing"       → facing reference

  "reference": id of another node this position is relative to, or null.
               If non-null, you MUST set depends_on to the same id.

  "side": "-x" | "+x" | "-y" | "+y" | "above" | null
          Direction from reference (used with "beside", "in_front", etc.).

  "distance": float meters; gap between this object and reference (default 0.1).

  "offset": {{"x": float, "y": float, "z": float}} — fine adjustment in meters,
            applied after reference+side+distance. Use small values (≤ 0.05).

  "facing": "inward" | "outward" | "any" — for "around", which way to face.

EXAMPLES
========

(1) "Two books with a glass between them on a desk":
{{
  "nodes": [
    {{"id": "desk_1",  "model_name": "desk",  "instances": 1, "parent_id": null,
      "anchor_hint": {{"region": "center"}}, "relationship": null,
      "place_order": 1, "depends_on": null}},

    {{"id": "glass_1", "model_name": "glass", "instances": 1, "parent_id": "desk_1",
      "anchor_hint": null,
      "relationship": {{"type": "on_surface", "reference": null}},
      "place_order": 1, "depends_on": null}},

    {{"id": "book_1",  "model_name": "book",  "instances": 1, "parent_id": "desk_1",
      "anchor_hint": null,
      "relationship": {{"type": "on_surface", "reference": "glass_1",
                        "side": "-x", "distance": 0.10}},
      "place_order": 2, "depends_on": "glass_1"}},

    {{"id": "book_2",  "model_name": "book",  "instances": 1, "parent_id": "desk_1",
      "anchor_hint": null,
      "relationship": {{"type": "on_surface", "reference": "glass_1",
                        "side": "+x", "distance": 0.10}},
      "place_order": 2, "depends_on": "glass_1"}}
  ]
}}

(2) "Two notebooks stacked with offset":
{{
  "nodes": [
    {{"id": "desk_1",     "model_name": "desk",     "instances": 1, "parent_id": null,
      "anchor_hint": {{"region": "center"}}, "relationship": null,
      "place_order": 1, "depends_on": null}},

    {{"id": "notebook_1", "model_name": "notebook", "instances": 1, "parent_id": "desk_1",
      "anchor_hint": null,
      "relationship": {{"type": "on_surface", "reference": null}},
      "place_order": 1, "depends_on": null}},

    {{"id": "notebook_2", "model_name": "notebook", "instances": 1, "parent_id": "notebook_1",
      "anchor_hint": null,
      "relationship": {{"type": "stacked_on", "reference": "notebook_1",
                        "offset": {{"x": 0.03, "y": 0.02, "z": 0}}}},
      "place_order": 2, "depends_on": "notebook_1"}}
  ]
}}

(3) "Round table with 4 chairs":
{{
  "nodes": [
    {{"id": "table_1", "model_name": "table", "instances": 1, "parent_id": null,
      "anchor_hint": {{"region": "center"}}, "relationship": null,
      "place_order": 1, "depends_on": null}},

        {{"id": "chair_group_1", "model_name": "chair", "instances": 4, "parent_id": "table_1",
            "anchor_hint": null,
            "relationship": {{"type": "around", "reference": "table_1",
                                                "distance": 0.5, "facing": "inward"}},
            "place_order": 2, "depends_on": "table_1"}}
  ]
}}

NOTES
=====
- Do NOT set "level" — it is computed automatically from parent_id chains.
- The sum of all node.instances MUST equal {total_count}.
- Use depends_on whenever relationship.reference is non-null.
- For anchors (floor furniture), parent_id = null, relationship = null.

Return ONLY a JSON object with a "nodes" array.
"""


_ANCHOR_RELATIONS_PROMPT = """
CRITICAL: Respond ONLY in English. Return ONLY valid JSON.

You are placing {anchor_count} anchor objects in a room {room_w}m × {room_l}m.

COORDINATE SYSTEM: Origin (0, 0) is at the CENTER of the room.
Valid range: x ∈ [{neg_half:.1f}, {pos_half:.1f}], y ∈ [{neg_half:.1f}, {pos_half:.1f}]
Example positions: center=(0,0), corners=(±{corner:.1f}, ±{corner:.1f}), near-wall=(±{near_wall:.1f}, 0)

Anchors to place (level 0):
{anchors_list}

DEPENDENT OBJECTS that will be placed AFTER each anchor (reserve space!):
{future_objects_per_anchor}

User query: {query}

CONSIDER WHEN PLACING:
- Each anchor needs a clearance ring for its dependents (chairs around table → ≥0.7m radius free)
- Two anchors must not be too close: keep distance ≥ sum of their clearance rings
- Stay ≥0.4m away from walls
- Match the user query semantics (e.g. "напротив" → opposite sides, "у стены" → near_wall)

Return JSON:
{{
  "anchor_positions": [
    {{"id": "table_1", "x": -2.5, "y": -2.5, "yaw_deg": 0.0}},
    ...
  ]
}}
"""


_SINGLE_PLACEMENT_PROMPT = """
CRITICAL: Respond ONLY in English. Return ONLY valid JSON.

Place ONE object in the scene.

Object: {model_name}
Size (width × depth × height in meters): {size_w} × {size_d} × {size_h}
Your half-extents (use for arithmetic): half_x={size_w_half:.3f}, half_y={size_d_half:.3f}, half_z={size_h_half:.3f}
{placement_instruction}

Room: {room_w}m × {room_l}m (origin at center, boundaries: ±{room_half}m)

Objects that will ALSO be placed after this one (leave space for them):
{remaining}

Already placed in scene (use the X/Y/Z ranges to AVOID OVERLAP):
{world_state}

GEOMETRY RULES (use this arithmetic — do not eyeball):
- "distance" in the placement instruction is the GAP between bounding boxes, not center-to-center.
- If side="+x" and reference center is (rx, ry) with half_x=rhx,
  then your X = rx + rhx + distance + your_half_x; your Y stays ≈ ry.
- If side="+y": your Y = ry + rhy + distance + your_half_y; your X ≈ rx.
- If side="-x"/"-y": same with subtraction.
- "Parallel side-by-side" arrangements (multiple long objects laid out next to each other):
  spread along the axis PERPENDICULAR to the long dimension. E.g. two pens lying along Y →
  spread them along X with separation ≥ pen_width_x + 0.005m, not along Y.

POSITION RULES:
- x, y must be within room boundaries (±{room_half}m minus object half-size).
- z is computed automatically by the engine from your parent_id chain — return any reasonable z.
- yaw_deg: 0=long axis along +X, 90=long axis along +Y, 180=along -X, 270=along -Y.

Your X-range will be [x - {size_w_half:.3f}, x + {size_w_half:.3f}].
Your Y-range will be [y - {size_d_half:.3f}, y + {size_d_half:.3f}].
These must NOT overlap with any "Already placed" object's range at the same Z level.

Return JSON: {{"x": float, "y": float, "z": float, "yaw_deg": float}}
"""


_VALIDATE_PLACEMENT_PROMPT = """
CRITICAL: Respond ONLY in English. Return ONLY valid JSON.

You are a geometric layout VALIDATOR. Check whether a just-placed object is correct.

Just-placed object: {model_name} [id={node_id}]
  Position: ({cx:.3f}, {cy:.3f}, {cz:.3f}), yaw={yaw:.0f}°
  Size (w×d×h): {sw} × {sd} × {sh}
  Bounding box: X=[{xmin:.3f}..{xmax:.3f}], Y=[{ymin:.3f}..{ymax:.3f}], Z=[{zmin:.3f}..{zmax:.3f}]

Placement intent: {intent}
{parent_block}{reference_block}{history_block}

Other objects already in the scene (with their bounding boxes):
{world_state}

DEFINITIONS
-----------
Two boxes A and B "overlap on axis K" iff max(A.kmin, B.kmin) < min(A.kmax, B.kmax) — strictly.
Touching edges (max == min) is NOT overlap.

A real COLLISION between A and B requires overlap on ALL THREE axes (X, Y, AND Z).
If the Z ranges do not overlap (e.g. one object sits on top of another), there is NO collision
even if X and Y ranges overlap completely. STACKED OBJECTS MUST OVERLAP IN XY by design.

VALIDATE THESE THREE THINGS
---------------------------

1. COLLISION (XY+Z): does the just-placed object collide with any other placed object?
   • Compute X-overlap, Y-overlap, Z-overlap separately.
   • Only flag collision if ALL three are > 0.
   • EXCEPTION: the object's parent (it sits on the parent → Z-touch only, fine).
   • Stacked relationships are valid: if one object is on top of another, expect XY overlap and
     Z just-touching (no Z overlap) — this is correct, NOT a violation.

2. SURFACE BOUNDS: if a parent surface is given, this object's center (x, y) MUST be inside
   the parent's XY footprint. If you propose a fix, it MUST also be inside parent's footprint.
   Pushing the object off the parent to fix a collision is NOT allowed — instead, propose
   moving it to the OTHER SIDE of the offending neighbour while staying on the parent.

3. ORIENTATION/SEMANTICS: does yaw make sense given shape and intent?
   • Long thin objects (pens, knives) have a "long axis" — for "parallel" arrangements,
     all instances should have the SAME yaw. Don't flip yaw between rounds.
   • Once you decide yaw matches the intent, KEEP IT in your fix.
   • For round objects (mugs, balls), yaw is irrelevant — accept any.

OUTPUT
------
If everything is fine:
  {{"valid": true}}

If a fix is needed:
  {{"valid": false,
    "reason": "brief specific reason (cite axes and overlap amounts)",
    "fix": {{"x": float, "y": float, "z": float, "yaw_deg": float}}}}

CRITICAL FIX RULES
------------------
- Address the SPECIFIC violation you cited. Do not introduce a new violation while fixing.
- Smallest correction that works (typically a few centimeters).
- Keep z unchanged unless the violation is about Z.
- Keep within room boundaries ±{room_half}m AND within parent's XY footprint.
- If your previous fix attempt did not work, try a DIFFERENT direction this time
  (don't oscillate between two positions). The history is shown above.
- If no fix can satisfy ALL constraints (e.g. surface too small for two objects), return
  {{"valid": true}} and accept the current position rather than producing an invalid fix.
"""


_BATCH_PLACEMENT_PROMPT = """
CRITICAL: Respond ONLY in English. Return ONLY valid JSON.

Place {batch_count} objects of type "{model_name}".
These are instances {from_idx} to {to_idx} of {total} total.
Size per object: {size_w}m wide × {size_d}m deep × {size_h}m tall
Half-extents: half_x={size_w_half:.3f}, half_y={size_d_half:.3f}

Placement rule: {relationship_desc}

Parent object: "{parent_model}"
  World position: x={parent_x:.2f}, y={parent_y:.2f}, z={parent_z:.2f} (object center)
  Size: {parent_w:.2f}m wide × {parent_d:.2f}m deep × {parent_h:.2f}m tall
  Top surface Z: {parent_top_z:.2f}m

SURFACE BOUNDS (CRITICAL WHEN PLACING ON TOP OF PARENT)
- If the placement intent is "on_surface" or "stacked_on", then EVERY instance's center (x, y)
  MUST stay within the parent's XY footprint, accounting for your half extents.
  That means:
    x ∈ [{parent_x:.2f} - {parent_w:.2f}/2 + half_x, {parent_x:.2f} + {parent_w:.2f}/2 - half_x]
    y ∈ [{parent_y:.2f} - {parent_d:.2f}/2 + half_y, {parent_y:.2f} + {parent_d:.2f}/2 - half_y]
  Do NOT place instances off the surface.

IMPORTANT: All coordinates are ABSOLUTE world coordinates, origin at room center.
Place objects near parent at ({parent_x:.2f}, {parent_y:.2f}).
Example at distance {placement_dist:.2f}m: N=({parent_x:.2f}, {parent_ypd:.2f}), E=({parent_xpd:.2f}, {parent_y:.2f}), S=({parent_x:.2f}, {parent_ymd:.2f}), W=({parent_xmd:.2f}, {parent_y:.2f})

NOTE: z is handled automatically — provide any z value, it will be corrected.

{already_in_group_section}
{parent_instances_section}
{reference_instances_section}

Other objects already in scene:
{world_state}

Room: {room_w}m × {room_l}m (±{room_half}m boundaries, center at 0,0)

Rules:
- All x,y coordinates must be within ±{room_half}m
- Spread objects evenly, maintain minimum clearance between objects (avoid stacking/overlap)
- yaw_deg CRITICAL: Set rotation to face parent (for "around" relationship)
  Parent at ({parent_x:.2f}, {parent_y:.2f}):
  • If object y > {parent_y:.2f} (north of parent) → yaw_deg = 0
  • If object x > {parent_x:.2f} (east of parent) → yaw_deg = 90
  • If object y < {parent_y:.2f} (south of parent) → yaw_deg = 180
  • If object x < {parent_x:.2f} (west of parent) → yaw_deg = 270
  Example: object at ({parent_x:.2f}, {parent_ypd:.2f}) is NORTH → yaw_deg=0
  Example: object at ({parent_xpd:.2f}, {parent_y:.2f}) is EAST → yaw_deg=90
- yaw_deg must be in [0, 360)

Return JSON array: [{{"instance": {from_idx}, "x": float, "y": float, "z": float, "yaw_deg": float}}, ...]
CRITICAL: Return EXACTLY {batch_count} items.
"""


_ANCHOR_BATCH_PLACEMENT_PROMPT = """
CRITICAL: Respond ONLY in English. Return ONLY valid JSON.

Place {batch_count} anchor objects of type "{model_name}".
These are anchor instances {from_idx} to {to_idx} of {total} total.
Size per object: {size_w}m wide × {size_d}m deep × {size_h}m tall

Placement rule: {relationship_desc}

Future dependent objects in this anchor group:
{future_context}

Other objects already in scene:
{world_state}

Room: {room_w}m × {room_l}m (±{room_half}m boundaries, center at 0,0)

Rules:
- All x,y coordinates must be within ±{room_half}m
- Separate repeated anchors enough so their future dependents can fit
- Keep the batch compact and evenly spaced when no other hint is available
- yaw_deg must be in [0, 360)

Return JSON array: [{{"instance": {from_idx}, "x": float, "y": float, "z": float, "yaw_deg": float}}, ...]
CRITICAL: Return EXACTLY {batch_count} items.
"""
