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

from pathlib import Path


def _load_prompt(filename):
    """Load a prompt from the prompts directory."""
    prompt_file = Path(__file__).parent / "prompts" / filename
    if not prompt_file.exists():
        raise FileNotFoundError(f"Prompt file not found: {prompt_file}")
    return prompt_file.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# 1. Graph construction
# ---------------------------------------------------------------------------

CLASSIFY_HIERARCHY_PROMPT = _load_prompt("classify_hierarchy.txt")
ANCHOR_RELATIONS_PROMPT = _load_prompt("anchor_relations.txt")
GROUP_RELATIONS_PROMPT = _load_prompt("group_relations.txt")


# ---------------------------------------------------------------------------
# 2. Placement (unified for anchors and dependents)
# ---------------------------------------------------------------------------

PLACE_NODE_PROMPT = _load_prompt("place_node.txt")
PLACE_NODE_BATCH_PROMPT = _load_prompt("place_node_batch.txt")
COPLACE_PROMPT = _load_prompt("coplace.txt")


# ---------------------------------------------------------------------------
# Pattern-specific batch prompts.
#
# Each one is a focused, single-pattern prompt used by _place_node_batch when
# the upstream surface plan has chosen that pattern. They are intentionally
# short so a small-context model only sees the parameters relevant to this
# one pattern (range along one axis, fixed value on the other; or grid bounds).
# ---------------------------------------------------------------------------

PLACE_BATCH_ROW_PROMPT = _load_prompt("place_batch_row.txt")
PLACE_BATCH_GRID_PROMPT = _load_prompt("place_batch_grid.txt")
PLACE_BATCH_FIX_PROMPT = _load_prompt("place_batch_fix.txt")


# ---------------------------------------------------------------------------
# Dynamic prompts (templates with runtime variable substitution)
# ---------------------------------------------------------------------------

SURFACE_ARRANGEMENT_PLAN_PROMPT = _load_prompt("surface_arrangement_plan.txt")
LAYERING_PLAN_PROMPT = _load_prompt("layering_plan.txt")

# Made with Bob
