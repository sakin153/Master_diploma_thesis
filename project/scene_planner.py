# Stage 4 — generate placement plan via LLM only.
#
# Pipeline
#   1. CLASSIFY_HIERARCHY (1 LLM call)         — flat tree (parent_id only).
#   2. ANCHOR_RELATIONS   (1 LLM call, opt.)   — relationships for level-0
#                                                 nodes. Either anchored to the
#                                                 virtual "room_center", or
#                                                 anchored to another anchor.
#                                                 Skipped when there is only
#                                                 one anchor instance — that
#                                                 single anchor goes to (0, 0).
#   3. GROUP_RELATIONS    (1 LLM call per anchor with descendants)
#                                              — relationships inside one group.
#   3.5 `_apply_physics_to_graph` (no LLM)     — stamps `is_static` onto every
#                                                 node from Stage-2.5 classifier
#                                                 output, then propagates the
#                                                 dynamic flag downward: if any
#                                                 ancestor is dynamic, the node
#                                                 is forced dynamic too (avoids
#                                                 items welded mid-air on a
#                                                 dynamic supporter).
#   4. PLACEMENT — anchors first (topological order on anchor relationships),
#                  then for each anchor walk its subtree level by level. At
#                  every level, siblings are grouped by
#                  `(parent_id, parent_instance_index)` — children targeting
#                  different instances of the same parent are NOT merged into
#                  one co-group.
#                  Per-group / per-node dispatch:
#                    * co-placement branch: when a sibling group qualifies
#                      (`_qualifies_for_coplace`), all its nodes are placed in
#                      one LLM call via `_place_co_group`.
#                    * single instance: `_place_node` → PLACE_NODE_PROMPT.
#                    * multiple instances: `_place_node_batch` → sub-batches
#                      of <= 6 via PLACE_NODE_BATCH_PROMPT. Before the batch
#                      LLM call, up to two extra planning calls run:
#                        - `_decide_surface_arrangement_plan` (surface layout
#                          pattern: row / grid / cluster / …),
#                        - `_decide_layering_plan` (vertical stacking inside
#                          containers).
#                      After the batch, a deterministic geometric-validation
#                      layer runs (`_violations_surface`,
#                      `_violations_overlap_pairs`,
#                      `_violations_overlap_with_siblings`,
#                      `_violations_overlap_mixed`, `_clamp_to_surface`).
#                      If violations remain, `_llm_fix_batch_violations`
#                      issues a corrective LLM call.
#   5. `_merge_to_output` (no LLM)             — final assembly: pulls back
#                                                 original model fields
#                                                 (`uuid`, `model_loc`,
#                                                 `_up_axis`), generates
#                                                 `save_fn`, and re-resolves
#                                                 `is_static` (the node flag
#                                                 from step 3.5 takes priority
#                                                 over the per-model
#                                                 classifier flag).
#
# Notes
#   * No heuristic / deterministic fallbacks for placement itself. Any
#     unrecoverable LLM failure raises. Sanity is limited to: keep (x, y) in
#     the room, keep z >= 0. The validation layer in step 4 is geometric,
#     not heuristic placement.
#   * The previous physics-settle stage (MuJoCo pre-roll) has been removed.

import json
import os
import re
from datetime import datetime

from project.llm_request import request as _llm_request, DEFAULT_MODEL
from project.scene_prompts import (
    ANCHOR_RELATIONS_PROMPT,
    CLASSIFY_HIERARCHY_PROMPT,
    COPLACE_PROMPT,
    GROUP_RELATIONS_PROMPT,
    LAYERING_PLAN_PROMPT,
    PLACE_BATCH_FIX_PROMPT,
    PLACE_BATCH_GRID_PROMPT,
    PLACE_BATCH_ROW_PROMPT,
    PLACE_NODE_BATCH_PROMPT,
    PLACE_NODE_PROMPT,
    SURFACE_ARRANGEMENT_PLAN_PROMPT,
)


# Relationship vocabulary used by the planner.
REL_AROUND = "around"
REL_ON_SURFACE = "on_surface"
REL_STACKED_ON = "stacked_on"
REL_IN = "in"
REL_BESIDE = "beside"
REL_IN_FRONT = "in_front"
REL_BEHIND = "behind"
REL_FACING = "facing"
REL_IN_ROOM = "in_room"

# Relationship types where the child's XY must lie inside the parent footprint
# (treated identically for footprint bounds, batch_size=1 and validation).
SURFACE_LIKE_RELS = (REL_ON_SURFACE, REL_STACKED_ON, REL_IN)

ROOM_CENTER = "room_center"

BATCH_SIZE = 6
LLM_RETRIES = 6
DOMAIN_RETRIES = 5  # extra retries when LLM violates surface bounds / overlap rules
PRIOR_BATCHES_CAP = 3  # cap on cross-batch context lines to keep prompt small
MAX_SIBLINGS_IN_CONTEXT = 10  # max siblings to show in _siblings_block to reduce context size

# --- Coplace (joint placement of mixed-type sibling groups) thresholds.
# When several sibling nodes share one parent, have no inter-references, and
# their combined instance count is small, place them all in ONE LLM call so
# the layout is coordinated (e.g. "2 bananas and 2 apples in a row at center"
# becomes 4 points on one line, not two competing groups).
COPLACE_MAX_TOTAL_INSTANCES = 8
COPLACE_MIN_NODES = 2

# Vertical spacing factor between layers when LLM uses multi-layer packing in
# `type="in"` containers. 1.0 = touching; > 1.0 leaves a small air gap so
# layered objects don't start in deep penetration.
LAYER_SPACING = 1.10

# Small vertical clearance (meters) to place movable objects slightly above
# supporting surfaces to avoid initial interpenetration and large solver
# impulses. Can be tuned via env var `WORLD_CREATOR_PLACEMENT_CLEARANCE`.
# Horizontal spacing still comes from `node["_surface_plan"]`.
# Increased default from 0.01 to 0.02 to reduce physics instability (performance optimization)
PLACEMENT_CLEARANCE = float(os.environ.get("WORLD_CREATOR_PLACEMENT_CLEARANCE", "0.02"))

# Conservative inflation applied ONLY to the per-child AABB used inside
# `_layer_capacity`. The LLM places objects on a soft grid and routinely
# under-spaces neighbours by a few percent (e.g. row Δy=0.15 m for bananas
# whose long axis is 0.2 m → adjacent overlap). Inflating the child's
# footprint by ε shrinks `cap_per_layer` so layering activates one batch
# earlier — instead of cramming 10 bananas into a 5×3 single layer where the
# LLM has zero slack, we ask for two layers of ~5. Doesn't change real
# placement coordinates, only the threshold at which LAYERING POLICY fires.
# Increased from 0.05 to 0.10 to reduce LLM retry failures (performance optimization)
LAYER_CAP_INFLATION = 0.10


def _layer_capacity(parent_obj, self_w, self_d, obj_h, margin=0.005):
    """Compute (capacity_per_layer, max_layers) for layered packing in a parent.

    capacity_per_layer = how many AABB-sized objects fit on a single layer
    inside the parent's footprint.
    max_layers         = how many such layers fit vertically inside the parent
                         (using LAYER_SPACING). Always ≥ 1.

    The child AABB (self_w, self_d, obj_h) is inflated by LAYER_CAP_INFLATION
    before fitting — see the constant's docstring for rationale.

    Returns (cap_per_layer, max_layers, range_x, range_y, parent_h).
    """
    psw, psd, _ = _size_wh(parent_obj["size"])
    eff_w = self_w * (1.0 + LAYER_CAP_INFLATION)
    eff_d = self_d * (1.0 + LAYER_CAP_INFLATION)
    eff_h = obj_h * (1.0 + LAYER_CAP_INFLATION)
    range_x = max(0.0, 2.0 * (psw / 2.0 - eff_w / 2.0 - margin))
    range_y = max(0.0, 2.0 * (psd / 2.0 - eff_d / 2.0 - margin))
    fit_x = max(1, int(range_x // eff_w) + 1) if eff_w > 0 else 1
    fit_y = max(1, int(range_y // eff_d) + 1) if eff_d > 0 else 1
    cap_per_layer = max(1, fit_x * fit_y)
    _, _, parent_h = _size_wh(parent_obj["size"])
    if eff_h > 0:
        max_layers = max(1, int(parent_h // (eff_h * LAYER_SPACING)))
    else:
        max_layers = 1
    return cap_per_layer, max_layers, range_x, range_y, parent_h

# Per-attempt sampling settings used in _place_node_batch domain retries.
# All attempts are intentionally non-deterministic: temperature is non-zero
# and seed is omitted. LLM still picks the coordinate; we only widen the
# sampler on later retries.
DOMAIN_RETRY_TEMPERATURES = (0.6, 0.8, 0.9, 1.0, 1.0, 1.0)
DOMAIN_RETRY_SEEDS = (None, None, None, None, None, None)


# ---------------------------------------------------------------------------
# Output saving
# ---------------------------------------------------------------------------

def _save_stage_output(data, stage_name, output_dir="output"):
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filepath = os.path.join(output_dir, f"{stage_name}_{timestamp}.json")
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False, default=str)
    print(f"[scene_planner] Saved {stage_name} → {filepath}")
    return filepath


# ---------------------------------------------------------------------------
# Model size helpers (size = [width_x, height_z, depth_y], engine convention)
# ---------------------------------------------------------------------------

def _model_size(models, model_name):
    for m in models:
        if m.get("Model") == model_name:
            return list(m.get("size", [1.0, 1.0, 1.0]))
    return [1.0, 1.0, 1.0]


def _size_wh(size):
    """Return (width_x, depth_y, height_z) for a size list.
    
    Input size format: [width_x, height_y, depth_z]
    Returns: (width_x, depth_z, height_y) to match the docstring promise of (width, depth, height)
    """
    sw = float(size[0]) if len(size) > 0 else 1.0
    sh = float(size[1]) if len(size) > 1 else sw
    sd = float(size[2]) if len(size) > 2 else sw
    # Return (width_x, depth_z, height_y) to match (width, depth, height) convention
    return sw, sd, sh


def _size_str(size):
    sw, sd, sh = _size_wh(size)
    return f"{sw:.2f}×{sd:.2f}×{sh:.2f}m"


# ---------------------------------------------------------------------------
# LLM JSON helper
# ---------------------------------------------------------------------------

def _normalise_batch_payload(raw):
    """Unwrap common LLM-emitted shapes into a flat list of placement dicts.

    Accepts (in order):
      • a list — returned as-is;
      • a dict with one of the wrapper keys "positions" / "items" /
        "answer" / "results" pointing at a list — unwrapped to that list;
      • a single placement dict with at least "x" and "y" — wrapped into
        a 1-element list (so a length-mismatch retry can talk in concrete
        cardinals instead of "got dict");
      • anything else — returned unchanged so the caller can complain.
    """
    if isinstance(raw, list):
        return raw
    if isinstance(raw, dict):
        for k in ("positions", "items", "answer", "results"):
            v = raw.get(k)
            if isinstance(v, list):
                return v
        if "x" in raw and "y" in raw:
            return [raw]
    return raw


def _llm_json(llm_fn, prompt, query, expect, retries=LLM_RETRIES, validation_context=None, **llm_kwargs):
    """Call llm_fn(prompt, query) until result satisfies expect(obj) -> (ok, err).

    Extra **llm_kwargs (e.g. temperature, seed) are forwarded to llm_fn so
    callers can vary sampling on retries. The closure llm_fn ignores
    unknown kwargs gracefully via **kwargs.

    validation_context: Optional dict with keys:
        - 'previous_attempts': list of dicts with 'coordinates', 'violations', 'attempt_num'
        - 'forbidden_points': list of (x, y) tuples to avoid
        - 'context_summary': string describing what went wrong in domain-specific terms

    On retry, the prompt is augmented with the previous failure AND a quick
    summary of what the model actually returned, so the LLM can self-
    correct on schema mismatches (e.g. emitted a single object instead of
    an array of N). On the LAST retry we bump the sampling temperature so
    the model gets unstuck if it keeps repeating the same wrong shape.
    
    ENHANCED: Now includes validation_context from domain retries to provide
    rich feedback about WHY previous attempts failed (not just JSON structure).
    """
    last_err = None
    last_raw_summary = ""
    cur_prompt = prompt
    
    # Build initial context from validation_context if provided
    if validation_context:
        context_lines = []
        if validation_context.get('context_summary'):
            context_lines.append(f"\n{'='*60}")
            context_lines.append("CONTEXT FROM PREVIOUS DOMAIN VALIDATION FAILURES:")
            context_lines.append(validation_context['context_summary'])
        
        if validation_context.get('previous_attempts'):
            context_lines.append("\nPREVIOUS FAILED ATTEMPTS:")
            for prev in validation_context['previous_attempts'][-3:]:  # Last 3 attempts
                context_lines.append(f"  Attempt #{prev['attempt_num']}:")
                if prev.get('coordinates'):
                    context_lines.append(f"    Coordinates: {prev['coordinates']}")
                if prev.get('violations'):
                    context_lines.append(f"    Violations: {'; '.join(prev['violations'][:3])}")
        
        if validation_context.get('forbidden_points'):
            context_lines.append("\nFORBIDDEN COORDINATES (avoid these):")
            for x, y in validation_context['forbidden_points'][:10]:  # Limit to 10
                context_lines.append(f"  - ({x:.3f}, {y:.3f})")
        
        if context_lines:
            context_lines.append(f"{'='*60}\n")
            cur_prompt = prompt + "\n".join(context_lines)
    
    for attempt in range(1, retries + 1):
        try:
            # ADDED: on the last retry, widen sampling to break out of stuck
            # wrong-shape answers. We only override temperature if the caller
            # passed one (so seed-only callers stay deterministic by default).
            kwargs = dict(llm_kwargs)
            if attempt == retries and "temperature" in kwargs:
                kwargs["temperature"] = max(float(kwargs.get("temperature", 0.0)), 0.9)
            raw = llm_fn(cur_prompt, query, **kwargs)
            if isinstance(raw, str):
                raw = json.loads(raw)
            ok, err = expect(raw)
            if ok:
                return raw
            last_err = err
            # Build a short summary of what the LLM actually returned, so
            # the next attempt receives a concrete description of the
            # mismatch (e.g. "you sent a single object {...}, but the
            # schema requires an array of 5").
            if isinstance(raw, dict):
                keys = ", ".join(sorted(raw.keys())[:8])
                last_raw_summary = f"a single JSON OBJECT with keys [{keys}]"
            elif isinstance(raw, list):
                last_raw_summary = f"a JSON ARRAY of length {len(raw)}"
            else:
                last_raw_summary = f"a {type(raw).__name__}"
        except Exception as exc:
            last_err = f"{type(exc).__name__}: {exc}"
            last_raw_summary = ""
        cur_prompt = (
            f"{prompt}\n\nPrevious attempt #{attempt} FAILED with: {last_err}.\n"
            f"You returned {last_raw_summary or 'a malformed response'}. "
            f"Re-read the schema description in the prompt above (look for "
            f"the 'Return JSON' line) and emit EXACTLY that structure — "
            f"if the schema asks for an ARRAY of N items, output an array "
            f"with N entries, NOT a single object. No prose, no markdown "
            f"fences, no extra fields outside the schema."
        )
    raise RuntimeError(f"LLM call failed after {retries} attempts: {last_err}")


# ---------------------------------------------------------------------------
# World state
# ---------------------------------------------------------------------------

def make_world_state(room_half_size):
    return {"placed": [], "remaining": [], "room_half_size": float(room_half_size)}


def _ws_add(world_state, instance_id, model_name, size, pose, yaw_deg):
    world_state["placed"].append({
        "id": instance_id,
        "Model": model_name,
        "size": list(size),
        "Pose": dict(pose),
        "yaw_deg": float(yaw_deg),
    })


def _ws_summary(world_state, current_node=None):
    """Generate summary of placed objects, optionally filtered by hierarchy.
    
    Args:
        world_state: Current world state with placed objects
        current_node: Optional node being placed. If provided, only shows:
                     - Objects at the same hierarchy level (siblings)
                     - Direct parent
                     - Direct children of parent (siblings)
                     This drastically reduces context size for large scenes.
    
    Returns:
        Formatted string of relevant placed objects
    """
    placed = world_state["placed"]
    if not placed:
        return "  (empty)"
    
    # If no node provided, return all objects (backward compatibility)
    if current_node is None:
        relevant_objects = placed
    else:
        # Filter by hierarchy: only show relevant objects
        relevant_objects = []
        current_parent_id = current_node.get("parent_id")
        
        for p in placed:
            obj_id = p["id"]
            
            # Always include direct parent
            if current_parent_id and obj_id.startswith(current_parent_id + "_inst_"):
                relevant_objects.append(p)
                continue
            
            # Include siblings (objects with same parent)
            # Extract parent from id like "table_1_inst_0" -> "table_1"
            if "_inst_" in obj_id:
                obj_parent_prefix = obj_id.rsplit("_inst_", 1)[0]
                # Check if this object shares the same parent
                if current_parent_id and obj_parent_prefix == current_parent_id:
                    relevant_objects.append(p)
                    continue
                
                # For objects at same level (e.g., other tables when placing a table)
                if current_parent_id is None:
                    # Current node is anchor (no parent), show other anchors
                    # Anchors have ids like "table_1_inst_0", "table_2_inst_0"
                    # Extract base: "table_1" from "table_1_inst_0"
                    current_base = current_node["id"]
                    if obj_parent_prefix.rsplit("_", 1)[0] == current_base.rsplit("_", 1)[0]:
                        relevant_objects.append(p)
    
    if not relevant_objects:
        return "  (empty - no relevant objects at this hierarchy level)"
    
    lines = []
    for p in relevant_objects:
        sw, sd, sh = _size_wh(p["size"])
        cx, cy, cz = p["Pose"]["x"], p["Pose"]["y"], p["Pose"]["z"]
        lines.append(
            f"  - {p['Model']} [{p['id']}]: center=({cx:.3f}, {cy:.3f}, {cz:.3f}), "
            f"yaw={p['yaw_deg']:.0f}°, "
            f"X=[{cx-sw/2:.3f}..{cx+sw/2:.3f}], Y=[{cy-sd/2:.3f}..{cy+sd/2:.3f}], "
            f"Z=[{cz-sh/2:.3f}..{cz+sh/2:.3f}]"
        )
    
    # Add summary line showing filtering
    if current_node is not None:
        total_placed = len(placed)
        shown = len(relevant_objects)
        if shown < total_placed:
            lines.insert(0, f"  (showing {shown} relevant objects out of {total_placed} total placed)")
    
    return "\n".join(lines)


def _ws_remaining_summary(world_state):
    remaining = world_state.get("remaining", [])
    if not remaining:
        return "  (none)"
    return "\n".join(f"  - {r['count']}× {r['model_name']} ({_size_str(r['size'])})" for r in remaining)


def _ws_find_all_by_node(world_state, node_id):
    if not node_id:
        return []
    return [p for p in world_state["placed"] if p["id"].startswith(node_id + "_inst_")]


# ---------------------------------------------------------------------------
# Graph helpers
# ---------------------------------------------------------------------------

def _compute_levels(nodes):
    by_id = {n["id"]: n for n in nodes}
    cache = {}

    def depth(node_id, seen):
        if node_id in cache:
            return cache[node_id]
        if node_id in seen:
            raise ValueError(f"Cycle in graph at node '{node_id}'")
        node = by_id.get(node_id)
        if node is None:
            return 0
        parent = node.get("parent_id")
        d = 0 if (parent is None or parent not in by_id) else depth(parent, seen | {node_id}) + 1
        cache[node_id] = d
        return d

    for n in nodes:
        n["level"] = depth(n["id"], set())


def _validate_basic_graph(nodes, available_models, expected_total):
    ids = {n["id"] for n in nodes}
    if len(ids) != len(nodes):
        return False, "duplicate node ids"
    available = set(available_models)
    for n in nodes:
        if n["model_name"] not in available:
            return False, f"unknown model '{n['model_name']}' in node '{n['id']}'"
        if int(n.get("instances", 0)) < 1:
            return False, f"node '{n['id']}' has instances < 1"
        pid = n.get("parent_id")
        if pid is not None and pid not in ids:
            return False, f"node '{n['id']}': parent_id '{pid}' is unknown"
    total = sum(int(n["instances"]) for n in nodes)
    if total != expected_total:
        return False, f"sum(instances)={total} but expected {expected_total}"
    return True, ""


def _collect_descendants(nodes, root_id):
    children = {}
    for n in nodes:
        children.setdefault(n.get("parent_id"), []).append(n)
    out = []
    stack = list(children.get(root_id, []))
    while stack:
        n = stack.pop()
        out.append(n)
        stack.extend(children.get(n["id"], []))
    return out


def _topological_sort(group_nodes):
    """Topological sort by 'depends_on' (within the given subset)."""
    by_id = {n["id"]: n for n in group_nodes}
    in_group = set(by_id.keys())
    visited, ordered = set(), []

    def visit(n):
        if n["id"] in visited:
            return
        visited.add(n["id"])
        dep = n.get("depends_on")
        if dep and dep in in_group:
            visit(by_id[dep])
        ordered.append(n)

    for n in sorted(group_nodes, key=lambda x: (int(x.get("place_order", 1)), x["id"])):
        visit(n)
    return ordered


# ---------------------------------------------------------------------------
# Phase A: CLASSIFY_HIERARCHY
# ---------------------------------------------------------------------------

def _phase_classify_hierarchy(models, room_half_size, query, llm_fn):
    type_counts = {}
    for m in models:
        type_counts[m["Model"]] = type_counts.get(m["Model"], 0) + 1
    total = sum(type_counts.values())
    objects_list = "\n".join(
        f"  - {name} ×{cnt} (size: {_size_str(_model_size(models, name))})"
        for name, cnt in type_counts.items()
    )

    room_w = room_half_size * 2
    prompt = CLASSIFY_HIERARCHY_PROMPT.format(
        query=query,
        room_w=f"{room_w:.1f}",
        room_l=f"{room_w:.1f}",
        objects_list=objects_list,
        total_count=total,
    )

    available = list(type_counts.keys())

    def expect(raw):
        if not isinstance(raw, dict) or "nodes" not in raw:
            return False, "missing 'nodes' field"
        nodes = raw["nodes"]
        if not isinstance(nodes, list) or not nodes:
            return False, "'nodes' must be a non-empty list"
        for n in nodes:
            for f in ("id", "model_name", "instances"):
                if f not in n:
                    return False, f"node missing field '{f}'"
            if "parent_id" not in n:
                n["parent_id"] = None
        # ADDED (path B): validate optional parent_instance_index field that
        # pins a child group to a specific instance of its parent. Lets a
        # multi-instance parent stay as ONE batched node while its children
        # attach to specific instances (e.g. "5 bananas in box[0], 5 apples
        # in box[1]" — boxes node remains instances=2, batched together).
        by_id = {n["id"]: n for n in nodes}
        for n in nodes:
            pii = n.get("parent_instance_index")
            if pii is None:
                continue
            try:
                pii_int = int(pii)
            except (ValueError, TypeError):
                return False, (
                    f"node '{n.get('id')}': parent_instance_index must be an int"
                )
            if pii_int < 0:
                return False, (
                    f"node '{n.get('id')}': parent_instance_index must be >= 0"
                )
            pid = n.get("parent_id")
            if not pid or pid not in by_id:
                return False, (
                    f"node '{n.get('id')}': parent_instance_index requires a "
                    f"valid parent_id (got '{pid}')"
                )
            parent_inst = int(by_id[pid].get("instances", 0) or 0)
            if pii_int >= parent_inst:
                return False, (
                    f"node '{n.get('id')}': parent_instance_index={pii_int} "
                    f">= parent '{pid}'.instances={parent_inst}"
                )
            n["parent_instance_index"] = pii_int  # normalize back to int
        return _validate_basic_graph(nodes, available, total)

    raw = _llm_json(llm_fn, prompt, query, expect)
    nodes = []
    for n in raw["nodes"]:
        # ADDED (path B): copy parent_instance_index when present.
        pii_raw = n.get("parent_instance_index")
        nodes.append({
            "id": str(n["id"]),
            "model_name": str(n["model_name"]),
            "instances": int(n["instances"]),
            "parent_id": n.get("parent_id"),
            "parent_instance_index": int(pii_raw) if pii_raw is not None else None,
            "relationship": None,
            "depends_on": None,
            "place_order": 1,
        })
    _compute_levels(nodes)
    print(f"[graph] phase A: {len(nodes)} nodes, levels 0..{max(n['level'] for n in nodes)}")
    return nodes


# ---------------------------------------------------------------------------
# Phase B: ANCHOR_RELATIONS — assign a relationship to every anchor.
# ---------------------------------------------------------------------------

def _future_per_anchor_text(nodes, models):
    lines = []
    anchors = [n for n in nodes if n.get("parent_id") is None]
    for an in anchors:
        descs = _collect_descendants(nodes, an["id"])
        if not descs:
            lines.append(f"  - {an['id']} ({an['model_name']}): (no dependents)")
            continue
        sub = []
        for d in descs:
            sub.append(
                f"      • {d['model_name']} ×{d['instances']} "
                f"({_size_str(_model_size(models, d['model_name']))})"
            )
        lines.append(f"  - {an['id']} ({an['model_name']}):\n" + "\n".join(sub))
    return "\n".join(lines)


def _phase_anchor_relations(anchor_nodes, future_text, models, room_half_size, query, llm_fn):
    """Assign 'relationship' (and derived 'depends_on') to every anchor node.

    No-op when there is only one anchor instance total — caller handles that.
    Otherwise mutates `anchor_nodes` in place.
    """
    total_instances = sum(an["instances"] for an in anchor_nodes)
    if total_instances <= 1:
        return

    anchors_list = "\n".join(
        f"  - {an['id']}: {an['model_name']} ×{an['instances']} "
        f"(size: {_size_str(_model_size(models, an['model_name']))})"
        for an in anchor_nodes
    )

    room_w = room_half_size * 2
    prompt = ANCHOR_RELATIONS_PROMPT.format(
        room_w=f"{room_w:.1f}",
        room_l=f"{room_w:.1f}",
        neg_half=-room_half_size,
        pos_half=room_half_size,
        anchors_list=anchors_list,
        future_per_anchor=future_text or "  (none)",
        query=query,
    )

    anchor_ids = {an["id"] for an in anchor_nodes}

    valid_ids_text = ", ".join(sorted(anchor_ids))

    def expect(raw):
        if not isinstance(raw, dict):
            return False, "expected JSON object"
        rels = raw.get("anchor_relations")
        if not isinstance(rels, list):
            return False, "missing 'anchor_relations' list"
        seen = set()
        has_room_center = False
        for r in rels:
            if not isinstance(r, dict) or "id" not in r:
                return False, "relation missing 'id'"
            rid = r["id"]
            if rid not in anchor_ids:
                return False, (
                    f"unknown anchor id '{rid}' in response. "
                    f"Each entry MUST use one of these EXACT ids verbatim: [{valid_ids_text}]. "
                    f"Do NOT invent instance suffixes like '_1', '_2'."
                )
            if rid in seen:
                return False, f"duplicate relation for anchor '{rid}'"
            seen.add(rid)
            rel = r.get("relationship") or {}
            if not isinstance(rel, dict):
                return False, f"anchor '{rid}': relationship must be object"
            ref = rel.get("reference")
            if ref == ROOM_CENTER:
                has_room_center = True
            elif ref not in anchor_ids:
                return False, (
                    f"anchor '{rid}': unknown reference '{ref}'. "
                    f"reference MUST be either 'room_center' or one of [{valid_ids_text}] verbatim."
                )
        missing = anchor_ids - seen
        if missing:
            return False, (
                f"missing relations for anchors {sorted(missing)}. "
                f"Provide EXACTLY one entry per anchor id from [{valid_ids_text}]."
            )
        if not has_room_center:
            return False, "at least one anchor must be anchored to 'room_center'"
        return True, ""

    raw = _llm_json(llm_fn, prompt, query, expect)

    by_id = {r["id"]: r for r in raw["anchor_relations"]}
    for an in anchor_nodes:
        rel = dict(by_id[an["id"]].get("relationship") or {})
        an["relationship"] = rel
        ref = rel.get("reference")
        an["depends_on"] = ref if ref and ref != ROOM_CENTER else None

    # Cycle check on anchor depends_on graph.
    visiting, visited = set(), set()
    by_anchor = {an["id"]: an for an in anchor_nodes}

    def dfs(aid):
        if aid in visited:
            return
        if aid in visiting:
            raise RuntimeError(f"Cycle in anchor relationships involving '{aid}'")
        visiting.add(aid)
        dep = by_anchor[aid].get("depends_on")
        if dep and dep in by_anchor:
            dfs(dep)
        visiting.discard(aid)
        visited.add(aid)

    for an in anchor_nodes:
        dfs(an["id"])


# ---------------------------------------------------------------------------
# Phase C: GROUP_RELATIONS — per anchor.
# ---------------------------------------------------------------------------

def _phase_group_relations(anchor_node, descendants, models, query, llm_fn):
    if not descendants:
        return

    desc_lines = []
    for d in descendants:
        # ADDED (path B): surface parent_instance_index in the descendant
        # listing so the GROUP_RELATIONS LLM understands which instance of a
        # multi-instance parent each child group is pinned to.
        pii = d.get("parent_instance_index")
        pii_str = f", parent_instance_index={pii}" if pii is not None else ""
        desc_lines.append(
            f"  - id={d['id']}, model_name={d['model_name']}, instances={d['instances']}, "
            f"parent_id={d.get('parent_id')}{pii_str}, "
            f"size={_size_str(_model_size(models, d['model_name']))}"
        )
    group_objects = "\n".join(desc_lines)

    prompt = GROUP_RELATIONS_PROMPT.format(
        anchor_id=anchor_node["id"],
        anchor_model=anchor_node["model_name"],
        anchor_size=_size_str(_model_size(models, anchor_node["model_name"])),
        group_objects=group_objects,
        query=query,
    )

    desc_ids = {d["id"] for d in descendants}

    def expect(raw):
        if not isinstance(raw, dict):
            return False, "expected JSON object"
        rels = raw.get("relations")
        if not isinstance(rels, list):
            return False, "missing 'relations' list"
        seen = set()
        for r in rels:
            if not isinstance(r, dict) or "id" not in r:
                return False, "relation entry missing 'id'"
            seen.add(r["id"])
        missing = desc_ids - seen
        if missing:
            return False, f"missing relations for ids {sorted(missing)}"
        return True, ""

    raw = _llm_json(llm_fn, prompt, query, expect)

    by_dep_id = {r["id"]: r for r in raw["relations"]}
    for d in descendants:
        r = by_dep_id[d["id"]]
        rel = r.get("relationship") or {}
        if not isinstance(rel, dict):
            rel = {}
        arrangement_raw = rel.get("arrangement")
        if isinstance(arrangement_raw, str):
            arr = arrangement_raw.strip().lower()
            if arr not in {"row", "grid", "scattered", "ring", "stack", "auto"}:
                arr = "auto"
        else:
            arr = "auto"
        d["relationship"] = {
            "type": str(rel.get("type", REL_ON_SURFACE)),
            "reference": rel.get("reference"),
            "side": rel.get("side"),
            "distance": float(rel["distance"]) if rel.get("distance") is not None else None,
            "facing": rel.get("facing"),
            "offset": rel.get("offset"),
            "arrangement": arr,
        }
        d["depends_on"] = r.get("depends_on")
        d["place_order"] = int(r.get("place_order", 1))


# ---------------------------------------------------------------------------
# Build whole graph
# ---------------------------------------------------------------------------

def _build_scene_graph(models, room_half_size, query, llm_fn):
    nodes = _phase_classify_hierarchy(models, room_half_size, query, llm_fn)
    anchors = [n for n in nodes if n.get("parent_id") is None]
    if not anchors:
        raise RuntimeError("Scene graph has no anchor nodes (parent_id=null)")

    future_text = _future_per_anchor_text(nodes, models)
    _phase_anchor_relations(anchors, future_text, models, room_half_size, query, llm_fn)
    print(f"[graph] phase B: anchor relationships set ({len(anchors)} anchors)")

    for an in anchors:
        descs = _collect_descendants(nodes, an["id"])
        if descs:
            _phase_group_relations(an, descs, models, query, llm_fn)
            print(f"[graph] phase C ({an['id']}): {len(descs)} relation(s) set")

    _apply_physics_to_graph(nodes, models)

    metadata = {
        "total_objects": sum(n["instances"] for n in nodes),
        "max_level": max(n["level"] for n in nodes),
    }
    return {"nodes": nodes, "metadata": metadata}


def _apply_physics_to_graph(nodes, models):
    """Stamp `is_static` onto every node and propagate dynamic downward.

    The classifier (Stage 2.5) sets `is_static` per model. Here we copy that
    onto each graph node, then enforce: if any ancestor of a node is dynamic
    (is_static is False), the node itself must be dynamic — otherwise items
    resting on a dynamic supporter would be welded mid-air relative to it.
    """
    by_name = {m.get("Model"): m for m in models if isinstance(m, dict)}
    by_id = {n["id"]: n for n in nodes}

    for n in nodes:
        m = by_name.get(n.get("model_name"))
        flag = m.get("is_static") if isinstance(m, dict) else None
        n["is_static"] = bool(flag) if flag is not None else None

    def has_dynamic_ancestor(node):
        seen = set()
        cur_pid = node.get("parent_id")
        while cur_pid and cur_pid not in seen:
            seen.add(cur_pid)
            parent = by_id.get(cur_pid)
            if parent is None:
                return False
            if parent.get("is_static") is False:
                return True
            cur_pid = parent.get("parent_id")
        return False

    flipped = 0
    for n in nodes:
        if n.get("is_static") is False:
            continue
        if has_dynamic_ancestor(n):
            n["is_static"] = False
            flipped += 1

    if flipped:
        print(f"[graph] physics propagation: {flipped} node(s) forced to dynamic by ancestor")


# ---------------------------------------------------------------------------
# Placement
# ---------------------------------------------------------------------------

def _clamp(v, lo, hi):
    return max(lo, min(hi, v))


def _sanity_xy(x, y, room_half_size):
    return (
        _clamp(float(x), -room_half_size, room_half_size),
        _clamp(float(y), -room_half_size, room_half_size),
    )


def _intent_text(node):
    rel = node.get("relationship") or {}
    if not rel:
        return "no relationship — use sensible defaults"
    parts = [f"type={rel.get('type', 'on_surface')}"]
    for k in ("reference", "region", "side", "distance", "facing", "yaw_deg", "offset_x", "offset_y"):
        v = rel.get(k)
        if v is not None and v != "":
            parts.append(f"{k}={v}")
    off = rel.get("offset")
    if off:
        parts.append(f"offset=(x={off.get('x', 0)}, y={off.get('y', 0)})")
    return ", ".join(parts)


def _reference_block(node, world_state):
    rel = node.get("relationship") or {}
    ref_id = rel.get("reference") if isinstance(rel, dict) else None
    if not ref_id:
        return ""
    if ref_id == ROOM_CENTER:
        room_half = world_state["room_half_size"]
        return (
            f"Reference: {ROOM_CENTER} — virtual point at the room center (0.0, 0.0). "
            f"Room is {room_half*2:.1f}m × {room_half*2:.1f}m.\n"
        )
    refs = _ws_find_all_by_node(world_state, ref_id)
    if not refs:
        return f"Reference '{ref_id}' is not yet placed; ignore it.\n"
    lines = [f"Reference instances for '{ref_id}':"]
    for r in refs:
        sw, sd, sh = _size_wh(r["size"])
        pose = r["Pose"]
        lines.append(
            f"  - {r['id']}: center=({pose['x']:.3f}, {pose['y']:.3f}, {pose['z']:.3f}), "
            f"size={_size_str(r['size'])}, yaw={r['yaw_deg']:.0f}°, "
            f"half_x={sw/2:.3f}, half_y={sd/2:.3f}"
        )
    return "\n".join(lines) + "\n"


    
    # ВСЕ УСЛОВИЯ ВЫПОЛНЕНЫ - добавляем инструкцию!
    return """
PROJECTION HINT (surface item ↔ "around" reference):
Your reference objects are positioned AROUND the parent (e.g. chairs around table).
To place yourself OPPOSITE the reference, project radially onto parent surface:
  1. Get reference center (rx, ry) from reference block above
  2. Compute direction from parent center: dx = rx - parent_x, dy = ry - parent_y
  3. Normalize: length = sqrt(dx² + dy²), unit_x = dx/length, unit_y = dy/length
  4. Project onto parent surface with inset from edge (use distance value, default 0.1m):
       target_x = parent_x + unit_x × (parent_half_x - inset)
       target_y = parent_y + unit_y × (parent_half_y - inset)
  5. Place at (target_x, target_y), ensuring you stay within parent bounds

"""


def _parent_block(node, world_state, relevant_parents=None):
    """Generate parent context block.
    
    Args:
        node: Current node being placed
        world_state: Global world state
        relevant_parents: Optional list of specific parent instances to describe.
                         If None, fetches all parents from world_state (old behavior).
                         Pass this to avoid describing irrelevant parent instances.
    """
    pid = node.get("parent_id")
    if not pid:
        return ""
    
    # Use provided relevant_parents if available, otherwise fall back to all parents
    if relevant_parents is not None:
        parents = relevant_parents
    else:
        single_parent = node.get("_single_parent_override")
        parents = [single_parent] if single_parent else _ws_find_all_by_node(world_state, pid)
    
    if not parents:
        return ""
    instances = int(node.get("instances", 1))

    # Child size (current node) — used for explicit footprint-area context
    # so the LLM can reason about how many fit per layer without re-deriving
    # geometry from half-extents in its head.
    csw, csd, csh = _size_wh(node.get("size_hint", [0.0, 0.0, 0.0]))
    child_footprint = csw * csd
    child_volume = csw * csd * csh

    def _area_block(p):
        sw, sd, sh = _size_wh(p["size"])
        floor_area = sw * sd
        # Heuristic interior height: full Z extent minus a small wall-thick
        # estimate (3 % of the smaller horizontal extent, capped at 2 cm).
        # The LLM sees this as a hint, not a hard constraint — it can ignore
        # it if the container is solid (in which case items go on top, not
        # in).
        wall_t = min(0.02, 0.03 * min(sw, sd))
        interior_h = max(0.0, sh - 2 * wall_t)
        # How many child footprints, with a 1.30× safety margin, fit on
        # the parent's floor in one layer (purely informational; LLM
        # decides actual count).
        per_layer_estimate = "?"
        if child_footprint > 0:
            per_layer_estimate = (
                f"≈ {floor_area / (child_footprint * 1.30):.1f}"
            )
        
        # Calculate max items per row along each axis (HARD CONSTRAINT)
        # This prevents LLM from placing more items in a row than physically fit
        max_items_x = "?"
        max_items_y = "?"
        if csw > 0 and csd > 0:
            # Use LAYER_CAP_INFLATION to match _layer_capacity logic
            eff_w = csw * (1.0 + LAYER_CAP_INFLATION)
            eff_d = csd * (1.0 + LAYER_CAP_INFLATION)
            margin = 0.005
            # Available range for placing items (accounting for child half-size and margin)
            range_x = max(0.0, 2.0 * (sw / 2.0 - eff_w / 2.0 - margin))
            range_y = max(0.0, 2.0 * (sd / 2.0 - eff_d / 2.0 - margin))
            # Max items that fit in a single row along each axis
            fit_x = max(1, int(range_x // eff_w) + 1) if eff_w > 0 else 1
            fit_y = max(1, int(range_y // eff_d) + 1) if eff_d > 0 else 1
            max_items_x = str(fit_x)
            max_items_y = str(fit_y)
        
        return (
            f"  floor_area={floor_area:.3f} m² ({sw:.2f}×{sd:.2f}), "
            f"interior_height≈{interior_h:.3f} m, "
            f"per_layer_est={per_layer_estimate} (child_footprint="
            f"{child_footprint:.4f} m² × 1.30 safety margin)\n"
            f"  MAX_ITEMS_PER_ROW: X-axis={max_items_x}, Y-axis={max_items_y} "
            f"(HARD LIMIT — do NOT place more items in a single row along either axis!)"
        )

    if len(parents) == 1:
        p = parents[0]
        sw, sd, sh = _size_wh(p["size"])
        pose = p["Pose"]
        parent_bottom_z = pose["z"] - sh / 2.0
        parent_top_z = pose["z"] + sh / 2.0
        return (
            f"Parent surface '{pid}' ({p['Model']}): "
            f"center=({pose['x']:.3f}, {pose['y']:.3f}, {pose['z']:.3f}), "
            f"size={_size_str(p['size'])}, top_z={parent_top_z:.3f}m. "
            f"surface_data: bottom_z={parent_bottom_z:.3f}m, top_z={parent_top_z:.3f}m, "
            f"surface_height={sh:.3f}m, place_child_at_z=top_z + child_half_h + clearance. "
            f"For 'on_surface'/'stacked_on' stay inside x∈[{pose['x']-sw/2:.3f}..{pose['x']+sw/2:.3f}], "
            f"y∈[{pose['y']-sd/2:.3f}..{pose['y']+sd/2:.3f}].\n"
            f"{_area_block(p)}\n"
            f"  child '{node['model_name']}' bbox={csw:.3f}×{csd:.3f}×{csh:.3f} m, "
            f"footprint={child_footprint:.4f} m², volume={child_volume:.5f} m³.\n"
        )

    # Multi-parent: list every parent instance.
    lines = [f"Parent group '{pid}' has {len(parents)} instances:"]
    for i, p in enumerate(parents, start=1):
        sw, sd, sh = _size_wh(p["size"])
        pose = p["Pose"]
        parent_bottom_z = pose["z"] - sh / 2.0
        parent_top_z = pose["z"] + sh / 2.0
        lines.append(
            f"  - parent_instance_{i} ({p['id']}): "
            f"center=({pose['x']:.3f}, {pose['y']:.3f}, {pose['z']:.3f}), "
            f"size={_size_str(p['size'])}, bottom_z={parent_bottom_z:.3f}m, top_z={parent_top_z:.3f}m, "
            f"surface_height={sh:.3f}m, place_child_at_z=top_z + child_half_h + clearance"
        )
    if instances == len(parents):
        lines.append(
            f"PARENT MAPPING: child instance i sits ON parent_instance_i "
            f"({instances} ↔ {instances}). Each child uses ITS OWN parent's footprint."
        )
    return "\n".join(lines) + "\n"


def _siblings_block(placed_in_group, self_w=None, self_d=None):
    """Format already-placed siblings within the current group.

    When self_w / self_d are provided, also pre-compute concrete forbidden
    coordinate ranges per sibling so the LLM does not have to do the math.
    
    OPTIMIZATION: Shows only the last MAX_SIBLINGS_IN_CONTEXT siblings to reduce
    context size. For large groups, this prevents exponential prompt growth.
    """
    if not placed_in_group:
        return ""
    
    total_count = len(placed_in_group)
    # Show only the most recent siblings to keep context manageable
    siblings_to_show = placed_in_group[-MAX_SIBLINGS_IN_CONTEXT:] if total_count > MAX_SIBLINGS_IN_CONTEXT else placed_in_group
    omitted_count = total_count - len(siblings_to_show)
    
    if omitted_count > 0:
        lines = [f"Already placed in this group ({total_count} total, showing last {len(siblings_to_show)}):"]
    else:
        lines = [f"Already placed in this group ({total_count}):"]
    
    for r in siblings_to_show:
        lyr = int(r.get("layer", 0))
        lyr_str = f", layer={lyr}" if lyr else ""
        lines.append(
            f"  - instance {r.get('instance', '?')}: pos=({r['x']:.3f}, {r['y']:.3f}, {r['z']:.3f}), "
            f"yaw={r.get('yaw_deg', 0):.0f}°{lyr_str}"
        )
    
    if self_w is not None and self_d is not None:
        lines.append(
            "Forbidden zones — apply per-layer (siblings on a DIFFERENT layer "
            "do NOT constrain your (x, y)). Your (x, y) MUST satisfy at least "
            "one of the following AGAINST EVERY sibling on YOUR layer:"
        )
        for r in siblings_to_show:
            xs, ys = r["x"], r["y"]
            lyr = int(r.get("layer", 0))
            lines.append(
                f"  - vs instance {r.get('instance', '?')} (layer={lyr}): "
                f"x ≤ {xs - self_w:.3f}  OR  x ≥ {xs + self_w:.3f}  "
                f"OR  y ≤ {ys - self_d:.3f}  OR  y ≥ {ys + self_d:.3f}"
            )
    
    if omitted_count > 0:
        lines.append(
            f"  Note: {omitted_count} earlier sibling(s) omitted for brevity. "
            f"Ensure your placement avoids ALL {total_count} siblings, not just the ones shown."
        )
    
    lines.append(
        "  >>> Pick a location DIFFERENT from every same-layer entry above. "
        "Do NOT reuse a same-layer sibling's (x, y). Spread across the "
        "available range."
    )
    return "\n".join(lines) + "\n"


def _parent_quota_block(node, batch_idx_in_parent, parent_obj, self_w, self_d):
    """Tell the LLM how many siblings go on the SAME parent and which axis
    is geometrically capable of fitting them side-by-side, so the layout
    does not crowd along an axis that physically cannot hold N items.

    CHANGED (task 2.1+2.2): the block is now emitted for EVERY single
    instance call (including the LAST one — previously skipped), and adds
    an IMPERATIVE directive when one axis cannot fit `instances`
    side-by-side. The LLM still picks the actual coordinates; this only
    surfaces a geometric fact LLM must respect.
    """
    if parent_obj is None:
        return ""
    instances = int(node.get("instances", 1))
    if instances <= 1:
        return ""
    remaining_after = instances - batch_idx_in_parent - 1
    cur_idx_1based = batch_idx_in_parent + 1

    psw, psd, _ = _size_wh(parent_obj["size"])
    phx, phy = psw / 2.0, psd / 2.0
    margin = 0.005
    self_hx, self_hy = self_w / 2.0, self_d / 2.0
    range_x = 2.0 * (phx - self_hx - margin)
    range_y = 2.0 * (phy - self_hy - margin)
    # N items fit side-by-side iff range >= (N-1) * self_size, hence
    # max N = floor(range / self_size) + 1, when the parent is wide enough for 1.
    fit_x = (int(range_x / self_w) + 1) if (self_w > 0 and range_x >= 0) else 0
    fit_y = (int(range_y / self_d) + 1) if (self_d > 0 and range_y >= 0) else 0
    longer_axis = "Y" if range_y >= range_x else "X"

    # Header — different wording for non-last vs last instance.
    if remaining_after > 0:
        header = (
            f"PARENT QUOTA — placing instance {cur_idx_1based} of {instances} "
            f"on parent {parent_obj['id']}; {remaining_after} more will be "
            f"placed AFTER this one (by future LLM calls).\n"
        )
        leave_room_line = (
            f"  >>> Do NOT take the parent center. Pick a position that "
            f"leaves room for {remaining_after} more instance(s) of size "
            f"{self_w:.3f} × {self_d:.3f}m.\n"
        )
    else:
        header = (
            f"PARENT QUOTA — placing instance {cur_idx_1based} of {instances} "
            f"on parent {parent_obj['id']} (LAST instance for this parent; "
            f"all earlier siblings are listed in the siblings block).\n"
        )
        leave_room_line = ""

    range_info = (
        f"  Available range on parent: X ≈ {range_x:.3f}m "
        f"(fits ~{fit_x} side-by-side), Y ≈ {range_y:.3f}m "
        f"(fits ~{fit_y} side-by-side). Longer axis: {longer_axis}.\n"
    )

    # IMPERATIVE axis-directive when one axis physically cannot hold N items.
    # This is a geometric fact, not a placement command — the LLM still
    # chooses the exact coordinates.
    if fit_x < instances and fit_y >= instances:
        directive = (
            f"  >>> AXIS X CANNOT FIT {instances} instances side-by-side "
            f"(fit_x={fit_x}). ANY layout that varies X across the {instances} "
            f"instances will FORCE OVERLAP. You MUST distribute the {instances} "
            f"instances along axis Y (fits {fit_y}).\n"
        )
    elif fit_y < instances and fit_x >= instances:
        directive = (
            f"  >>> AXIS Y CANNOT FIT {instances} instances side-by-side "
            f"(fit_y={fit_y}). ANY layout that varies Y across the {instances} "
            f"instances will FORCE OVERLAP. You MUST distribute the {instances} "
            f"instances along axis X (fits {fit_x}).\n"
        )
    elif fit_x < instances and fit_y < instances:
        directive = (
            f"  >>> WARNING: NEITHER axis fits {instances} side-by-side "
            f"(fit_x={fit_x}, fit_y={fit_y}). Some overlap is geometrically "
            f"unavoidable; use the LONGER axis ({longer_axis}) and minimise "
            f"the collision.\n"
        )
    else:
        directive = (
            f"  Hint: both axes can fit {instances} side-by-side. Prefer the "
            f"longer axis ({longer_axis}); place yourself near one edge so "
            f"the rest can line up next to you.\n"
        )

    return header + range_info + leave_room_line + directive + "\n"


def _prior_batches_block(prior_batches, query=""):
    """Cross-batch context for N:K split.

    DISABLED — superseded by `_decide_surface_arrangement_plan`.
    Identical parents naturally get identical plans → identical layouts,
    no need for explicit replicate-from-prior-batch instruction. The
    function is kept (and still called) so the prompt template's
    `{prior_batches_block}` placeholder still renders to an empty string
    instead of crashing. If you ever want this back, drop the early
    return below.

    Original docstring (for reference): "We DO NOT decide here whether
    the layout should match across parents — that decision is delegated
    to the LLM..."

    Capped at the last PRIOR_BATCHES_CAP entries to keep the prompt small.
    """
    # Disabled — see docstring. surface_plan + identical-parent semantics
    # provide cross-parent consistency naturally.
    return ""

    # Unreachable below — kept for easy revert.
    if not prior_batches:
        return ""
    tail = prior_batches[-PRIOR_BATCHES_CAP:]

    lines = [
        "Sibling sub-batches already placed on OTHER parents (this group "
        "is being distributed across multiple parents). For each entry "
        "below you see the parent's centre, the absolute coords of the "
        "items, and the SAME items expressed as offsets (dx, dy) FROM "
        "THAT PARENT'S CENTRE."
    ]
    for entry in tail:
        px = entry.get("parent_x", 0.0)
        py = entry.get("parent_y", 0.0)
        coords = ", ".join(
            f"({r['x']:.3f}, {r['y']:.3f})" for r in entry["results"]
        )
        rels = ", ".join(
            f"({r['dx']:+.3f}, {r['dy']:+.3f})"
            for r in entry.get("results_rel", [])
        ) or "n/a"
        lines.append(
            f"  - parent {entry['parent_id']} centre=({px:.3f}, {py:.3f}): "
            f"abs=[{coords}]  offsets_from_parent=[{rels}]"
        )
    lines.append(
        "DECIDE — read the user query above carefully:\n"
        "  (A) If the query implies the SAME arrangement on every parent "
        "(words like 'aligned', 'centrally', 'identical', 'each table the "
        "same', 'mirror', 'одинаково', 'симметрично', 'по центру на каждом', "
        "'выровнены', or any phrasing that treats the parents as a uniform "
        "set without per-parent variation) — REPLICATE the offsets "
        "verbatim: emit x = your_parent_centre_x + dx, y = your_parent_"
        "centre_y + dy for each item. Do NOT invent a new pattern.\n"
        "  (B) If the query explicitly asks for variety, randomness or "
        "describes a different layout per parent — pick a DIFFERENT "
        "pattern here that still satisfies the DEFAULT LAYOUT rules "
        "below (centroid balance, symmetry, edge margin).\n"
        "  When in doubt — choose (A). Identical layout across siblings is "
        "the safer default for any prompt that does not actively demand "
        "variety."
    )
    return "\n".join(lines) + "\n\n"


def _surface_bounds(parent_obj, self_w, self_d, margin=0.005):
    """(x_min, x_max, y_min, y_max) — center-of-child bounds inside parent."""
    psw, psd, _ = _size_wh(parent_obj["size"])
    phx, phy = psw / 2.0, psd / 2.0
    px, py = parent_obj["Pose"]["x"], parent_obj["Pose"]["y"]
    self_hx, self_hy = self_w / 2.0, self_d / 2.0
    lim_x = max(0.0, phx - self_hx - margin)
    lim_y = max(0.0, phy - self_hy - margin)
    return (px - lim_x, px + lim_x, py - lim_y, py + lim_y)


def _violations_surface(items, parents_per_item, self_w, self_d, rel_type=None):
    """For each item, check (x, y) lies inside its parent footprint.
    parents_per_item: list of parent objects matching items 1:1.
    rel_type: relationship type ("on_surface", "in", "stacked_on", ...).
        When equal to REL_IN, the bounds are TIGHTENED to require a full-
        child-size inset from each parent wall — items inside a container
        (box/drawer/bowl) must not sit on the rim, otherwise the physics
        solver will push them out at scene start. Universal: applies to
        any (child, parent) pair with type="in", no per-object table.
    Returns list of human-readable violation strings (empty = OK).
    """
    msgs = []
    is_in = (rel_type == REL_IN)
    for i, item in enumerate(items):
        p = parents_per_item[i]
        # Default footprint bounds (centre stays inside parent, with a tiny
        # 5 mm safety margin — see _surface_bounds).
        x_min, x_max, y_min, y_max = _surface_bounds(p, self_w, self_d)
        
        if not (x_min <= item["x"] <= x_max):
            msgs.append(
                f"item {i + 1}: x={item['x']:.3f} OUTSIDE parent {p['id']} bounds "
                f"[{x_min:.3f}, {x_max:.3f}]"
            )
        if not (y_min <= item["y"] <= y_max):
            msgs.append(
                f"item {i + 1}: y={item['y']:.3f} OUTSIDE parent {p['id']} bounds "
                f"[{y_min:.3f}, {y_max:.3f}]"
            )
        if is_in:
            # Tight inset: keep |offset_from_parent_centre| + child_half ≤
            # parent_half − child_size, i.e. a full-child-size gap from
            # each wall. Skip when the parent is too small to physically
            # allow such a gap (else we'd reject every position) — that
            # case is already a model-mismatch and the LLM cannot help.
            psw, psd, _ = _size_wh(p["size"])
            phx, phy = psw / 2.0, psd / 2.0
            self_hx, self_hy = self_w / 2.0, self_d / 2.0
            limit_x = phx - self_hx - self_w
            limit_y = phy - self_hy - self_d
            cx = p["Pose"]["x"]
            cy = p["Pose"]["y"]
            if limit_x > 0:
                ox = abs(item["x"] - cx)
                if ox > limit_x:
                    msgs.append(
                        f"item {i + 1}: too close to wall of '{p['id']}' on X — "
                        f"|Δx|={ox:.3f} but max allowed inside-container "
                        f"offset is {limit_x:.3f} (need full-child-size gap "
                        f"from each wall when type=in). Move toward parent "
                        f"centre x={cx:.3f}."
                    )
            if limit_y > 0:
                oy = abs(item["y"] - cy)
                if oy > limit_y:
                    msgs.append(
                        f"item {i + 1}: too close to wall of '{p['id']}' on Y — "
                        f"|Δy|={oy:.3f} but max allowed inside-container "
                        f"offset is {limit_y:.3f} (need full-child-size gap "
                        f"from each wall when type=in). Move toward parent "
                        f"centre y={cy:.3f}."
                    )
    return msgs


# Universal "comfortable gap" coefficient. Centres of two same-axis-aligned
# items must be separated by at least COMFORT_GAP_FACTOR × their full size
# along the constraint axis (i.e. a 15 % breathing room on top of bare
# touching). Universal for all object types — no per-object table.
COMFORT_GAP_FACTOR = 1.15


def _required_separations(self_w, self_d):
    """Minimum centre-to-centre separation along each axis (with comfort margin)."""
    return self_w * COMFORT_GAP_FACTOR, self_d * COMFORT_GAP_FACTOR


def _violations_surface_plan(items, parents_per_item, self_w, self_d, plan):
    """Check that placed items respect the upstream surface arrangement
    plan (decided by `_decide_surface_arrangement_plan`).

    Two checks (only the parts the plan specifies):
      • pairwise centre-to-centre distance ≥ plan.gap_m on at least one
        axis (we accept either-axis satisfaction so 2D patterns like
        dice/grid still pass);
      • each item's centre is at most `parent_half - child_half - plan.
        wall_margin_m` from the parent centre on each axis.

    Universal — gap_m and wall_margin_m are values the LLM decided per
    parent, no per-object table.
    """
    if not plan or not items or not parents_per_item:
        return []
    msgs = []
    gap = float(plan.get("gap_m", 0.0))
    wall = float(plan.get("wall_margin_m", 0.0))

    # Pairwise gap (looser than the strict universal rule: at least ONE
    # axis must clear gap_m, so 2D patterns survive).
    if gap > 0:
        for i in range(len(items)):
            for j in range(i + 1, len(items)):
                if int(items[i].get("layer", 0)) != int(items[j].get("layer", 0)):
                    continue
                dx = abs(items[i]["x"] - items[j]["x"])
                dy = abs(items[i]["y"] - items[j]["y"])
                if dx < gap and dy < gap:
                    msgs.append(
                        f"items {i + 1} and {j + 1} violate surface plan: "
                        f"plan.gap_m={gap:.3f} requires |Δx| ≥ {gap:.3f} OR "
                        f"|Δy| ≥ {gap:.3f}, got Δx={dx:.3f}, Δy={dy:.3f}."
                    )

    # Wall margin per parent.
    if wall > 0:
        self_hx, self_hy = self_w / 2.0, self_d / 2.0
        for i, item in enumerate(items):
            p = parents_per_item[i]
            if p is None:
                continue
            psw, psd, _ = _size_wh(p["size"])
            phx, phy = psw / 2.0, psd / 2.0
            cx = p["Pose"]["x"]
            cy = p["Pose"]["y"]
            # Max allowed |offset| from parent centre such that the child
            # bbox edge stays at least `wall` from the parent edge.
            limit_x = phx - self_hx - wall
            limit_y = phy - self_hy - wall
            if limit_x > 0:
                ox = abs(item["x"] - cx)
                if ox > limit_x:
                    msgs.append(
                        f"item {i + 1} violates surface plan wall margin "
                        f"({wall:.3f} m) on X: |Δx|={ox:.3f} > "
                        f"limit={limit_x:.3f} (parent='{p['id']}'). Move "
                        f"toward parent centre x={cx:.3f}."
                    )
            if limit_y > 0:
                oy = abs(item["y"] - cy)
                if oy > limit_y:
                    msgs.append(
                        f"item {i + 1} violates surface plan wall margin "
                        f"({wall:.3f} m) on Y: |Δy|={oy:.3f} > "
                        f"limit={limit_y:.3f} (parent='{p['id']}'). Move "
                        f"toward parent centre y={cy:.3f}."
                    )
    return msgs


def _violations_overlap_pairs(items, self_w, self_d):
    """Pairwise anti-overlap inside the same batch.

    Two items on DIFFERENT layers don't count as overlap (they're vertically
    separated by physics). Same-layer xy proximity must satisfy
    COMFORT_GAP_FACTOR — bounding boxes that merely touch produce broken
    contacts in the physics solver.
    """
    req_w, req_d = _required_separations(self_w, self_d)
    msgs = []
    for i in range(len(items)):
        for j in range(i + 1, len(items)):
            if int(items[i].get("layer", 0)) != int(items[j].get("layer", 0)):
                continue
            dx = abs(items[i]["x"] - items[j]["x"])
            dy = abs(items[i]["y"] - items[j]["y"])
            if dx < req_w and dy < req_d:
                # Surface gap (bbox edge to bbox edge) along the axis the LLM
                # was relying on for separation. Reported so retries see the
                # concrete deficit.
                gap_x = dx - self_w
                gap_y = dy - self_d
                need_x = req_w - dx
                need_y = req_d - dy
                msgs.append(
                    f"items {i + 1} and {j + 1} too close: "
                    f"|Δx|={dx:.3f} (need > {req_w:.3f}, missing "
                    f"{need_x:.3f}m on X) AND |Δy|={dy:.3f} (need > "
                    f"{req_d:.3f}, missing {need_y:.3f}m on Y); current "
                    f"surface gap = ({gap_x:+.3f}, {gap_y:+.3f}) m. Pick "
                    f"a clearly comfortable separation, do NOT sit at the "
                    f"strict overlap boundary."
                )
    return msgs


def _violations_overlap_with_siblings(items, siblings, self_w, self_d):
    """Anti-overlap between new items and already-placed siblings in the same group.

    Pairs with different layer ids are exempt (vertical separation).
    """
    req_w, req_d = _required_separations(self_w, self_d)
    msgs = []
    for i, item in enumerate(items):
        for s in siblings:
            if int(item.get("layer", 0)) != int(s.get("layer", 0)):
                continue
            dx = abs(item["x"] - s["x"])
            dy = abs(item["y"] - s["y"])
            if dx < req_w and dy < req_d:
                gap_x = dx - self_w
                gap_y = dy - self_d
                msgs.append(
                    f"item {i + 1} too close to sibling instance "
                    f"{s.get('instance', '?')} at ({s['x']:.3f}, {s['y']:.3f}): "
                    f"|Δx|={dx:.3f} (need > {req_w:.3f}), |Δy|={dy:.3f} "
                    f"(need > {req_d:.3f}); current surface gap = "
                    f"({gap_x:+.3f}, {gap_y:+.3f}) m. Comfortable separation "
                    f"required, not a flush touch."
                )
    return msgs


def _clamp_to_surface(item, parent_obj, self_w, self_d):
    """In-place clamp item's (x, y) into parent's center-bound rectangle."""
    x_min, x_max, y_min, y_max = _surface_bounds(parent_obj, self_w, self_d)
    item["x"] = _clamp(item["x"], x_min, x_max)
    item["y"] = _clamp(item["y"], y_min, y_max)
    return item


def _violations_overlap_mixed(items_with_sizes):
    """Pairwise overlap check for items with HETEROGENEOUS sizes.

    Each item is a dict with x, y, w, d (full bbox extents along respective
    axes). Pairs at different layers don't count as overlap.
    """
    msgs = []
    n = len(items_with_sizes)
    for i in range(n):
        a = items_with_sizes[i]
        for j in range(i + 1, n):
            b = items_with_sizes[j]
            if int(a.get("layer", 0)) != int(b.get("layer", 0)):
                continue
            min_dx = (a["w"] + b["w"]) / 2.0
            min_dy = (a["d"] + b["d"]) / 2.0
            dx = abs(a["x"] - b["x"])
            dy = abs(a["y"] - b["y"])
            if dx < min_dx and dy < min_dy:
                msgs.append(
                    f"items {i + 1} ({a.get('label', '?')}) and "
                    f"{j + 1} ({b.get('label', '?')}) overlap: "
                    f"|Δx|={dx:.3f} < {min_dx:.3f} AND "
                    f"|Δy|={dy:.3f} < {min_dy:.3f}"
                )
    return msgs


def _qualifies_for_coplace(group_nodes, world_state):
    """Decide if a sibling group should be placed in ONE coordinated LLM call.

    Rules (all must hold):
      • At least COPLACE_MIN_NODES nodes share the same parent_id.
      • No node in the group references another node in the group via
        relationship.reference (those have explicit pairing — keep separate).
      • Every node uses a surface-like relationship type (on_surface / in /
        stacked_on). Floor/around relations stay individual.
      • Combined instance count ≤ COPLACE_MAX_TOTAL_INSTANCES.
      • Parent has exactly one placed instance (multi-parent N:K split is
        already handled separately and does not benefit from coplace).
    """
    if len(group_nodes) < COPLACE_MIN_NODES:
        return False
    parent_ids = {n.get("parent_id") for n in group_nodes}
    if len(parent_ids) != 1:
        return False
    pid = next(iter(parent_ids))
    if pid is None:
        return False  # anchors don't coplace
    
    parent_instances = _ws_find_all_by_node(world_state, pid)
    if len(parent_instances) != 1:
        return False

    group_ids = {n["id"] for n in group_nodes}
    total_instances = 0
    for n in group_nodes:
        rel = n.get("relationship") or {}
        rtype = rel.get("type")
        if rtype not in SURFACE_LIKE_RELS:
            return False
        ref = rel.get("reference")
        if ref and ref in group_ids:
            return False  # explicit pairing inside the group
        total_instances += int(n.get("instances", 1))

    return total_instances <= COPLACE_MAX_TOTAL_INSTANCES


def _numeric_hints_block(node, world_state):
    """Pre-compute concrete numbers (footprint bounds, ring radius, target xy).

    The LLM is bad at edge-to-edge arithmetic; baking the numbers into the
    prompt makes "around" not collide with the parent and "on_surface" stay
    inside the parent footprint. Hints are advisory — LLM still chooses x, y.
    """
    rel = node.get("relationship") or {}
    if not isinstance(rel, dict):
        return ""

    rtype = rel.get("type")
    parent_id = node.get("parent_id")
    ref_id = rel.get("reference")
    side = rel.get("side")
    distance = rel.get("distance")
    distance = float(distance) if distance is not None else 0.1
    instances = int(node.get("instances", 1))

    sw, sd, sh = _size_wh(node["size_hint"])
    self_hx, self_hy = sw / 2.0, sd / 2.0

    single_parent = node.get("_single_parent_override")
    parents = [single_parent] if single_parent else (
        _ws_find_all_by_node(world_state, parent_id) if parent_id else []
    )
    parent_obj = parents[0] if parents else None

    # Detect 1:1 reference alignment (e.g. "plate opposite each chair", "apple
    # on each plate"). When active, REFERENCE-ALIGNMENT must dominate — any
    # spread / grid / layering hint conflicts with it and confuses the LLM.
    has_1to1_ref = False
    if ref_id and ref_id != ROOM_CENTER:
        ref_list_check = _ws_find_all_by_node(world_state, ref_id)
        if ref_list_check and len(ref_list_check) == instances and instances >= 1:
            has_1to1_ref = True

    out = []
    if has_1to1_ref:
        out.append(
            "PRIORITY ORDER (read first):\n"
            "  1. The REFERENCE-ALIGNMENT block below is BINDING — every "
            "instance you place MUST sit close to its assigned reference "
            "instance (instance i ↔ reference_i, no swapping).\n"
            "  2. Surface bounds are a HARD constraint — your (x, y) must "
            "stay inside them.\n"
            "  3. Any 'spread / grid / row / scattered' wording you may "
            "remember from earlier instructions is OVERRIDDEN here. Do NOT "
            "spread for the sake of spreading; follow the per-reference "
            "pairing instead."
        )

    # ----- on_surface / stacked_on / in: numeric center bounds inside parent.
    if rtype in SURFACE_LIKE_RELS and parents:
        margin = 0.005
        if len(parents) == 1:
            p = parents[0]
            psw, psd, _ = _size_wh(p["size"])
            phx, phy = psw / 2.0, psd / 2.0
            px, py = p["Pose"]["x"], p["Pose"]["y"]
            lim_x = max(0.0, phx - self_hx - margin)
            lim_y = max(0.0, phy - self_hy - margin)
            out.append(
                f"NUMERIC SURFACE BOUNDS — (x, y) is ANY point inside:\n"
                f"  x ∈ [{px - lim_x:.3f}, {px + lim_x:.3f}], "
                f"y ∈ [{py - lim_y:.3f}, {py + lim_y:.3f}].\n"
                f"  Use the FULL range — do NOT default to the parent center "
                f"({px:.3f}, {py:.3f}). When multiple instances share this surface, "
                f"spread them across the range so they do not overlap.\n"
                f"  (parent half = ({phx:.3f}, {phy:.3f}); self half = ({self_hx:.3f}, {self_hy:.3f}))"
            )
        else:
            lines = ["NUMERIC SURFACE BOUNDS per parent_instance — each child stays inside ITS parent:"]
            for i, p in enumerate(parents, start=1):
                psw, psd, _ = _size_wh(p["size"])
                phx, phy = psw / 2.0, psd / 2.0
                px, py = p["Pose"]["x"], p["Pose"]["y"]
                lim_x = max(0.0, phx - self_hx - margin)
                lim_y = max(0.0, phy - self_hy - margin)
                lines.append(
                    f"  parent_instance_{i} ({p['id']}) center=({px:.3f}, {py:.3f}): "
                    f"x ∈ [{px - lim_x:.3f}, {px + lim_x:.3f}], "
                    f"y ∈ [{py - lim_y:.3f}, {py + lim_y:.3f}]"
                )
            lines.append(
                "  Use the FULL range — do NOT default to parent centers. "
                "When multiple instances share a parent, spread them across the range."
            )
            if instances == len(parents):
                lines.append(
                    f"PARENT MAPPING: instance i must lie inside parent_instance_i bounds "
                    f"({instances} ↔ {instances})."
                )
            out.append("\n".join(lines))

        # --- ADDED: universal 1:1 reference-alignment GUIDANCE (information +
        # recommendation only — the LLM computes the final (x, y)). When
        # `relationship.reference` points to a sibling node whose placed-instance
        # count equals this node's instance count, expose:
        #   • the per-instance pairing (instance i ↔ reference_i),
        #   • each reference's raw (x, y, yaw) so the LLM can reason geometrically,
        #   • a textual rule for how to project the reference onto the surface
        #     (clamp into bounds when ref is outside; use ref position when ref
        #     is inside; inset by `distance` to clear the rim).
        # No coordinates are pre-computed for the LLM — it decides placement
        # itself. Works universally for ANY (object, surface, reference) triple
        # — e.g. "plates on table opposite chairs", "apple on each plate",
        # "book in each drawer".
        if ref_id and ref_id != ROOM_CENTER:
            ref_list = _ws_find_all_by_node(world_state, ref_id)
            if ref_list and len(ref_list) == instances and instances >= 1:
                lines = [
                    f"REFERENCE-ALIGNMENT (1:1 — counts match "
                    f"{instances}↔{len(ref_list)}). YOU choose every (x, y); "
                    f"this block only describes the pairing and gives geometric "
                    f"hints. The user query may override these defaults — read it.",
                    f"  Pairing: instance i ↔ reference_i (do NOT swap).",
                    f"  Reference instances of '{ref_id}':",
                ]
                for i, r in enumerate(ref_list, start=1):
                    lines.append(
                        f"    reference_{i} ({r['id']}): pos="
                        f"({r['Pose']['x']:.3f}, {r['Pose']['y']:.3f}), "
                        f"yaw={r['yaw_deg']:.0f}°"
                    )
                cur_idx = node.get("_current_instance_idx")
                if cur_idx is not None and 0 <= cur_idx < instances:
                    rcur = ref_list[cur_idx]
                    rcx, rcy = rcur["Pose"]["x"], rcur["Pose"]["y"]
                    # Compute the recommended projected target on the parent
                    # surface for this specific instance — this gives the LLM
                    # a concrete number to anchor on instead of guessing.
                    if parent_obj is not None:
                        pmargin = 0.005
                        psw_a, psd_a, _ = _size_wh(parent_obj["size"])
                        phx_a, phy_a = psw_a / 2.0, psd_a / 2.0
                        ppx, ppy = parent_obj["Pose"]["x"], parent_obj["Pose"]["y"]
                        x_lo, x_hi = ppx - phx_a + self_hx + pmargin, ppx + phx_a - self_hx - pmargin
                        y_lo, y_hi = ppy - phy_a + self_hy + pmargin, ppy + phy_a - self_hy - pmargin
                        # Clamp the reference's xy onto the surface bounds.
                        tx = max(x_lo, min(x_hi, rcx))
                        ty = max(y_lo, min(y_hi, rcy))
                        # Inset by `distance` toward the parent's center so the
                        # child clears the rim (only when ref was outside).
                        if rcx < x_lo or rcx > x_hi or rcy < y_lo or rcy > y_hi:
                            dx = ppx - tx
                            dy = ppy - ty
                            mag = (dx * dx + dy * dy) ** 0.5
                            if mag > 1e-6:
                                tx += dx / mag * distance
                                ty += dy / mag * distance
                        target_str = f"  RECOMMENDED TARGET (≈) for THIS instance: ({tx:.3f}, {ty:.3f}). Stay within ±5 cm of this point."
                    else:
                        target_str = ""
                    lines.append(
                        f"  THIS CALL places instance {cur_idx + 1} — its "
                        f"reference is reference_{cur_idx + 1} ({rcur['id']}) "
                        f"at ({rcx:.3f}, {rcy:.3f})."
                    )
                    if target_str:
                        lines.append(target_str)
                lines.append(
                    "  Geometric rule (default — adapt to the user query):"
                )
                lines.append(
                    "    • If reference_i lies INSIDE the surface bounds above, "
                    "place instance i at reference_i's (x, y)."
                )
                lines.append(
                    "    • If reference_i lies OUTSIDE the surface, find the "
                    "point on the surface NEAREST to reference_i (clamp "
                    "reference's xy into the surface bounds), then inset "
                    f"toward the parent's center by ≈ {distance:.3f}m so the "
                    "child does not sit flush against the rim."
                )
                lines.append(
                    "  Exception: if the user query specifies a different "
                    "arrangement (offset, rotation, custom side), follow the "
                    "query — the rule above is just the default for "
                    "'opposite/aligned with each reference'."
                )
                out.append("\n".join(lines))

    # ----- around: ring radius around parent + cardinal suggestions + yaw map.
    #       Two cases:
    #         a) single parent (classic "chairs around a table") — ring slots.
    #         b) multi-parent 1:1 mapping ("one chair next to each desk") —
    #            per-parent slot derived from `side`.
    if rtype == REL_AROUND and parent_obj is not None:
        if len(parents) > 1 and len(parents) == instances:
            lines = [
                "NUMERIC SLOT per parent_instance — instance i goes next to "
                f"parent_instance_i (1:1 mapping, {instances} ↔ {instances}):"
            ]
            for i, p in enumerate(parents, start=1):
                psw, psd, _ = _size_wh(p["size"])
                phx, phy = psw / 2.0, psd / 2.0
                px, py = p["Pose"]["x"], p["Pose"]["y"]
                if side == "+x":
                    tx = px + phx + distance + self_hx
                    ty = py
                    yaw = 180
                elif side == "-x":
                    tx = px - phx - distance - self_hx
                    ty = py
                    yaw = 0
                elif side == "+y":
                    tx = px
                    ty = py + phy + distance + self_hy
                    yaw = 270
                elif side == "-y":
                    tx = px
                    ty = py - phy - distance - self_hy
                    yaw = 90
                else:
                    # No side → cycle through cardinals, but warn.
                    cardinals = [(+1, 0, 180), (-1, 0, 0), (0, +1, 270), (0, -1, 90)]
                    dx, dy, yaw = cardinals[(i - 1) % 4]
                    tx = px + dx * (phx + distance + self_hx)
                    ty = py + dy * (phy + distance + self_hy)
                lines.append(
                    f"  parent_instance_{i} ({p['id']}) at ({px:.3f}, {py:.3f}) → "
                    f"instance {i} slot ≈ ({tx:.3f}, {ty:.3f}), yaw={yaw}"
                )
            lines.append(
                "  Use these target slots — each instance MUST stay near its OWN "
                "parent. Do NOT cluster multiple instances near the same parent, "
                "and do NOT overlap the parent footprint."
            )
            out.append("\n".join(lines))
        else:
            psw, psd, _ = _size_wh(parent_obj["size"])
            phx, phy = psw / 2.0, psd / 2.0
            px, py = parent_obj["Pose"]["x"], parent_obj["Pose"]["y"]
            rx = phx + distance + self_hx
            ry = phy + distance + self_hy
            out.append(
                f"NUMERIC RING AROUND parent at ({px:.3f}, {py:.3f}):\n"
                f"  ring_offset_x = parent_half_x + distance + self_half_x "
                f"= {phx:.3f} + {distance:.3f} + {self_hx:.3f} = {rx:.3f}\n"
                f"  ring_offset_y = parent_half_y + distance + self_half_y "
                f"= {phy:.3f} + {distance:.3f} + {self_hy:.3f} = {ry:.3f}\n"
                f"  Cardinal slots (use these or interpolate angles for >4 instances):\n"
                f"    E=({px + rx:.3f}, {py:.3f})    yaw=180  (faces -X toward parent)\n"
                f"    W=({px - rx:.3f}, {py:.3f})    yaw=0    (faces +X toward parent)\n"
                f"    N=({px:.3f}, {py + ry:.3f})    yaw=270  (faces -Y toward parent)\n"
                f"    S=({px:.3f}, {py - ry:.3f})    yaw=90   (faces +Y toward parent)\n"
                f"  All centers MUST be at least ring_offset away from parent — never inside it."
            )

    # ----- beside / in_front / behind / facing: per-reference target xy.
    if rtype in (REL_BESIDE, REL_IN_FRONT, REL_BEHIND, REL_FACING) and ref_id and side:
        if ref_id == ROOM_CENTER:
            ref_list = [{"id": ROOM_CENTER, "Pose": {"x": 0.0, "y": 0.0}, "size": [0.0, 0.0, 0.0]}]
        else:
            ref_list = _ws_find_all_by_node(world_state, ref_id)
        if ref_list:
            lines = ["NUMERIC TARGETS for each reference (approximate; fine-tune if needed):"]
            for idx, r in enumerate(ref_list, start=1):
                rsw, rsd, _ = _size_wh(r.get("size", [0, 0, 0]))
                rhx, rhy = rsw / 2.0, rsd / 2.0
                rx, ry = r["Pose"]["x"], r["Pose"]["y"]
                if side == "+x":
                    tx, ty = rx + rhx + distance + self_hx, ry
                elif side == "-x":
                    tx, ty = rx - rhx - distance - self_hx, ry
                elif side == "+y":
                    tx, ty = rx, ry + rhy + distance + self_hy
                elif side == "-y":
                    tx, ty = rx, ry - rhy - distance - self_hy
                else:
                    tx, ty = rx, ry
                lines.append(f"  ref_instance_{idx} ({r['id']}) → target ≈ ({tx:.3f}, {ty:.3f})")
            out.append("\n".join(lines))

    # ----- in_room: target from region hint.
    if rtype == REL_IN_ROOM and ref_id == ROOM_CENTER:
        room_half = world_state["room_half_size"]
        region = (rel.get("region") or "center").lower()
        ox = float(rel.get("offset_x") or 0.0)
        oy = float(rel.get("offset_y") or 0.0)
        margin = 0.4
        if region == "center":
            tx, ty = ox, oy
        elif region == "near_wall":
            wall_y = -(room_half - self_hy - margin)
            tx = ox
            ty = oy if oy != 0.0 else wall_y
        elif region == "corner":
            tx = ox if ox != 0.0 else -(room_half - self_hx - margin)
            ty = oy if oy != 0.0 else -(room_half - self_hy - margin)
        else:
            tx, ty = ox, oy
        out.append(
            f"NUMERIC TARGET (region={region}) ≈ ({tx:.3f}, {ty:.3f}). "
            f"Adjust within room bounds if it conflicts with future dependents."
        )

    # ----- batch instance ↔ ref_instance mapping when counts match.
    if instances > 1 and ref_id and ref_id != ROOM_CENTER:
        ref_list = _ws_find_all_by_node(world_state, ref_id)
        if ref_list and len(ref_list) == instances:
            out.append(
                f"REFERENCE MAPPING — counts match ({instances} ↔ {instances}). "
                f"Place instance i NEXT TO ref_instance_i exactly. "
                f"DO NOT cluster multiple instances near the same ref_instance."
            )

    # ----- batch instance ↔ parent_instance mapping when counts match
    #       and parent_id has multiple instances (e.g. one banana per plate,
    #       or one chair beside each desk). Wording depends on relationship
    #       type: "ON" only for surface-like, "next to" for around/beside-like.
    if instances > 1 and parents and len(parents) == instances and len(parents) > 1:
        if rtype in SURFACE_LIKE_RELS:
            verb = "ON"
        elif rtype in (REL_AROUND, REL_BESIDE, REL_IN_FRONT, REL_BEHIND, REL_FACING):
            verb = "next to"
        else:
            verb = "associated with"
        out.append(
            f"PARENT MAPPING — counts match ({instances} ↔ {instances}). "
            f"Place instance i {verb} parent_instance_i. Each instance has its OWN parent. "
            f"DO NOT cluster multiple instances at the same parent."
        )

    # ----- LAYERING POLICY: vertical packing inside containers.
    # Active ONLY for type="in" when the per-layer footprint cannot hold all
    # instances on one floor. Suppressed when a 1:1 reference is in effect —
    # in that mode every instance has its own dedicated target so layering
    # becomes nonsense.
    if rtype == REL_IN and parents and instances > 1 and not has_1to1_ref:
        ref_p = parents[0]
        cap_per_layer, max_layers, _, _, parent_h = _layer_capacity(
            ref_p, sw, sd, sh
        )
        needed_layers = (instances + cap_per_layer - 1) // cap_per_layer
        if needed_layers >= 2 and max_layers >= 2:
            usable_layers = min(needed_layers, max_layers)
            layer_step = sh * LAYER_SPACING
            out.append(
                f"LAYERING POLICY (type=\"in\" + overflow).\n"
                f"  N={instances} > capacity_per_layer={cap_per_layer} for "
                f"this container; container fits up to {max_layers} layers "
                f"vertically (parent_h={parent_h:.3f} m, "
                f"layer_step={layer_step:.3f} m).\n"
                f"  RULE: each instance must include integer field \"layer\" "
                f"in its JSON output.\n"
                f"  Assignment: instance idx i (0-based, GLOBAL across the "
                f"whole {instances}-set) → layer = i // {cap_per_layer}, "
                f"capped at {usable_layers - 1}.\n"
                f"  Effect: instances 0..{cap_per_layer - 1} are layer 0 "
                f"(bottom), {cap_per_layer}..{2 * cap_per_layer - 1} are "
                f"layer 1, and so on. Within each layer spread (x, y) over "
                f"the FULL footprint — overlap inside one layer is forbidden, "
                f"but two instances on DIFFERENT layers may share (x, y).\n"
                f"  Physics will settle layered instances into the container "
                f"by gravity — your job is just to assign correct layer ids."
            )

    # ----- ARRANGEMENT POLICY: distribution hint for surface-like multi-instance.
    # Suppressed when 1:1 reference alignment is active — that pairing
    # already dictates per-instance placement and any spread/grid hint
    # contradicts it (LLM gets pulled in two directions and ignores one).
    if rtype in SURFACE_LIKE_RELS and parents and instances > 1 and not has_1to1_ref:
        arrangement = (rel.get("arrangement") or "auto").lower()
        if arrangement not in {"row", "grid", "scattered", "ring", "stack", "auto"}:
            arrangement = "auto"

        ref_p = parents[0]
        psw, psd, _ = _size_wh(ref_p["size"])
        margin = 0.005
        range_x = max(0.0, 2.0 * (psw / 2.0 - self_hx - margin))
        range_y = max(0.0, 2.0 * (psd / 2.0 - self_hy - margin))
        long_axis = "X" if range_x >= range_y else "Y"
        short_axis = "Y" if long_axis == "X" else "X"
        long_range = max(range_x, range_y)
        short_range = min(range_x, range_y)

        # Capacity along the longer axis if everything were on a single line.
        cap_long = max(1, int(long_range // max(self_hx, self_hy, 1e-3) // 2))

        # Universal MIN-SEPARATION block — concrete numbers for the LLM.
        # The LLM still chooses the actual gap; this only states the lower
        # bound that any pair MUST satisfy. Using ≥ here (not ≤) emphasises
        # that the size value is the MINIMUM separation, not a step.
        gap_floor = 0.01  # 1 cm safety gap baseline
        min_sep = (
            f"PAIRWISE MIN-SEPARATION (HARD — applies to ALL pairs):\n"
            f"  For every two placements (i, j) at least one must hold:\n"
            f"      |xi − xj| ≥ {sw:.3f} m   (≥ obj width along X)\n"
            f"      |yi − yj| ≥ {sd:.3f} m   (≥ obj depth along Y)\n"
            f"  Add a comfortable gap of your choice (≥ {gap_floor:.3f} m, "
            f"typically 0.01–0.03 m) so neighbours don't touch."
        )

        if arrangement == "row":
            self_along_long = self_hx if long_axis == "X" else self_hy
            min_step_long = 2.0 * self_along_long
            need_long = instances * min_step_long
            policy = (
                f"ARRANGEMENT POLICY = row.\n"
                f"  Place {instances} instances PREDOMINANTLY along the "
                f"{long_axis} axis (parent's longer side). It is a band, "
                f"not a math line — small variation along {short_axis} is "
                f"allowed.\n"
                f"  Available along {long_axis}: {long_range:.3f} m. Each "
                f"object spans {min_step_long:.3f} m along {long_axis}, so "
                f"a single straight line needs ≥ {need_long:.3f} m.\n"
                f"  • If {need_long:.3f} ≤ {long_range:.3f}: one straight "
                f"line, gap of YOUR choice (≥ {gap_floor:.3f} m).\n"
                f"  • Otherwise (does NOT fit): use TWO rows along "
                f"{long_axis}, offset along {short_axis}."
            )
        elif arrangement == "scattered":
            policy = (
                f"ARRANGEMENT POLICY = scattered.\n"
                f"  Distribute {instances} instances IRREGULARLY across the "
                f"FULL footprint. Use BOTH X and Y axes; vary every "
                f"instance's (x, y) noticeably.\n"
                f"  Available: X≈{range_x:.3f} m, Y≈{range_y:.3f} m."
            )
        elif arrangement in ("grid", "auto"):
            import math
            aspect = (long_range + 1e-6) / (short_range + 1e-6)
            elongated = aspect >= 1.5  # parent is clearly longer along one axis
            prefer_line = (instances <= 3) or (instances == 2)

            if arrangement == "auto" and prefer_line:
                # For 2-3 items the natural default IS a single line along the
                # parent's longer axis. LLM picks the exact spacing.
                policy = (
                    f"ARRANGEMENT POLICY = auto (default).\n"
                    f"  Parent's longer axis is {long_axis} "
                    f"(range {long_range:.3f} m vs {short_axis} "
                    f"{short_range:.3f} m"
                    f"{', clearly elongated' if elongated else ''}).\n"
                    f"  RECOMMENDED for {instances} items WHEN the user did "
                    f"not specify a layout: place them in a single line "
                    f"along {long_axis}, with {short_axis} ≈ parent center "
                    f"(small variation OK).\n"
                    f"  YOU pick the gap between centres along {long_axis} — "
                    f"any value satisfying the MIN-SEPARATION rule above. "
                    f"Equal spacing ≈ {long_range / max(1, instances):.3f} m "
                    f"is a reasonable default but not required.\n"
                    f"  If a single line clearly does not fit, fall back to "
                    f"a 2D layout (~{max(1, int(math.ceil(math.sqrt(instances * aspect))))}"
                    f" along {long_axis})."
                )
            else:
                cols = max(1, int(math.ceil(math.sqrt(instances * aspect))))
                rows = max(1, int(math.ceil(instances / cols)))
                label = "grid" if arrangement == "grid" else "auto (default — 2D spread)"
                policy = (
                    f"ARRANGEMENT POLICY = {label}.\n"
                    f"  Spread {instances} instances over BOTH axes — do not "
                    f"pile them on a single line. Suggested: ~{cols} along "
                    f"{long_axis} × ~{rows} along {short_axis}.\n"
                    f"  Available: X≈{range_x:.3f} m, Y≈{range_y:.3f} m. "
                    f"YOU pick the exact spacing satisfying the "
                    f"MIN-SEPARATION rule above."
                )
        else:
            policy = ""

        out.append(min_sep)
        if policy:
            out.append(policy)

    if not out:
        return ""
    return "\n".join(out) + "\n\n"


def _decide_surface_arrangement_plan(node, parent_obj, query, llm_fn):
    """Ask the LLM for a high-level horizontal-arrangement plan for one
    placement batch on a single parent.

    Mirrors `_decide_layering_plan` but for the XY plane: pattern, axis,
    minimum sibling gap, minimum wall margin. Placement then has to
    respect these as HARD MINIMUMS — a placement may pick a larger gap
    or wider margin, but never less.

    Returns a dict (or None on LLM failure → callers fall back to the
    existing universal rules).
    """
    instances = int(node["instances"])
    sw, sd, sh = _size_wh(node["size_hint"])
    psw, psd, _ = _size_wh(parent_obj["size"])
    cx = parent_obj["Pose"]["x"]
    cy = parent_obj["Pose"]["y"]
    floor_area = psw * psd
    child_footprint = sw * sd
    rel = node.get("relationship") or {}
    rel_type = rel.get("type") if isinstance(rel, dict) else None

    # Hard floors enforced by existing validators — surface plan must
    # not undercut these (we clamp on the way out).
    min_gap_floor = max(sw, sd) * 1.15
    min_wall_margin_floor = max(sw, sd) if rel_type == REL_IN else 0.005

    free_x = max(0.0, psw / 2.0 - sw / 2.0 - 0.005)
    free_y = max(0.0, psd / 2.0 - sd / 2.0 - 0.005)
    if free_y >= free_x:
        long_axis_hint = "y"
        long_free_hint = free_y
    else:
        long_axis_hint = "x"
        long_free_hint = free_x

    prompt = SURFACE_ARRANGEMENT_PLAN_PROMPT.format(
        instances=instances,
        model_name=node['model_name'],
        parent_id=parent_obj['id'],
        parent_model=parent_obj['Model'],
        rel_type=rel_type,
        psw=psw,
        psd=psd,
        floor_area=floor_area,
        cx=cx,
        cy=cy,
        long_axis_hint=long_axis_hint.upper(),
        long_free_hint=long_free_hint,
        sw=sw,
        sd=sd,
        sh=sh,
        child_footprint=child_footprint,
        query=query,
        min_gap_floor=min_gap_floor,
        min_wall_margin_floor=min_wall_margin_floor,
    )

    valid_patterns = {"single", "row", "grid", "dice", "scatter", "ring"}
    valid_axes = {"x", "y", "centered"}

    def expect(raw):
        if not isinstance(raw, dict):
            return False, f"expected JSON object, got {type(raw).__name__}"
        for k in ("pattern", "axis", "gap_m", "wall_margin_m"):
            if k not in raw:
                return False, f"missing field '{k}'"
        if raw["pattern"] not in valid_patterns:
            return False, (
                f"unknown pattern '{raw['pattern']}'; allowed: "
                + ", ".join(sorted(valid_patterns))
            )
        if raw["axis"] not in valid_axes:
            return False, (
                f"unknown axis '{raw['axis']}'; allowed: "
                + ", ".join(sorted(valid_axes))
            )
        try:
            float(raw["gap_m"])
            float(raw["wall_margin_m"])
        except (TypeError, ValueError):
            return False, "gap_m and wall_margin_m must be numbers"
        return True, ""

    try:
        raw = _llm_json(llm_fn, prompt, query, expect)
        plan = {
            "pattern": str(raw["pattern"]),
            "axis": str(raw["axis"]),
            "gap_m": max(min_gap_floor, float(raw["gap_m"])),
            "wall_margin_m": max(
                min_wall_margin_floor, float(raw["wall_margin_m"])
            ),
            "rationale": str(raw.get("rationale", "")).strip(),
        }
        print(
            f"[surface_plan] '{node['id']}' on '{parent_obj['id']}': "
            f"pattern={plan['pattern']}, axis={plan['axis']}, "
            f"gap={plan['gap_m']:.3f}m, "
            f"wall_margin={plan['wall_margin_m']:.3f}m"
            + (f"  ({plan['rationale']})" if plan["rationale"] else "")
        )
        return plan
    except Exception as exc:
        print(
            f"[surface_plan] WARN: failed for "
            f"'{node['id']}'/'{parent_obj['id']}' ({exc}); "
            f"falling back to universal-rules-only placement."
        )
        return None


def _decide_layering_plan(node, parent_obj, query, llm_fn):
    """Ask the LLM how to split N items inside a single parent container
    into one or more horizontal layers.

    Only meaningful for `type="in"` (items physically inside a container).
    Returns a list of dicts:
        [{"items_count": int, "z_world": float}, ...]
    where z_world is the absolute Z of the centre of items in that layer.
    Sum of items_count equals node.instances.

    The function makes an EXTRA LLM call per parent — by design, per the
    user instruction "не экономить на запросах". Falls back to a single-
    layer plan on any LLM failure (so the existing pipeline still runs).
    """
    instances = int(node["instances"])
    sw, sd, sh = _size_wh(node["size_hint"])
    psw, psd, psh = _size_wh(parent_obj["size"])
    cx = parent_obj["Pose"]["x"]
    cy = parent_obj["Pose"]["y"]
    cz = parent_obj["Pose"]["z"]
    parent_floor_z = cz - psh / 2.0  # bottom of parent container in world Z
    parent_top_z = cz + psh / 2.0
    floor_area = psw * psd
    child_footprint = sw * sd
    child_volume = sw * sd * sh
    wall_t = min(0.02, 0.03 * min(psw, psd))
    interior_h = max(0.0, psh - 2 * wall_t)
    per_layer_estimate = (
        floor_area / (child_footprint * 1.30) if child_footprint > 0 else 1.0
    )

    layers_needed = max(1, int((instances + per_layer_estimate - 1) // max(1, per_layer_estimate)))
    prompt = LAYERING_PLAN_PROMPT.format(
        instances=instances,
        model_name=node['model_name'],
        parent_id=parent_obj['id'],
        parent_model=parent_obj['Model'],
        floor_area=floor_area,
        psw=psw,
        psd=psd,
        interior_h=interior_h,
        parent_floor_z=parent_floor_z,
        parent_top_z=parent_top_z,
        cx=cx,
        cy=cy,
        sw=sw,
        sd=sd,
        sh=sh,
        child_footprint=child_footprint,
        child_volume=child_volume,
        per_layer_estimate=per_layer_estimate,
        layers_needed=layers_needed,
    )

    def expect(raw):
        if not isinstance(raw, dict):
            return False, f"expected JSON object, got {type(raw).__name__}"
        layers = raw.get("layers")
        if not isinstance(layers, list) or not layers:
            return False, "field 'layers' must be a non-empty array"
        total = 0
        for i, L in enumerate(layers):
            if not isinstance(L, dict):
                return False, f"layer {i}: must be an object"
            try:
                cnt = int(L.get("items_count"))
                z = float(L.get("z_world"))
            except (TypeError, ValueError):
                return False, f"layer {i}: items_count int + z_world float required"
            if cnt <= 0:
                return False, f"layer {i}: items_count must be > 0"
            if not (parent_floor_z - 0.05 <= z <= parent_top_z + 0.05):
                return False, (
                    f"layer {i}: z_world={z:.3f} outside container "
                    f"[{parent_floor_z:.3f}, {parent_top_z:.3f}]"
                )
            total += cnt
        if total != instances:
            return False, (
                f"sum of items_count = {total}, expected {instances}"
            )
        return True, ""

    try:
        raw = _llm_json(llm_fn, prompt, query, expect)
        layers = [
            {"items_count": int(L["items_count"]), "z_world": float(L["z_world"])}
            for L in raw["layers"]
        ]
        print(f"[layering] '{node['id']}' in '{parent_obj['id']}': "
              f"{len(layers)} layer(s) — "
              + ", ".join(f"{L['items_count']}@z={L['z_world']:.3f}" for L in layers))
        return layers
    except Exception as exc:
        # Fallback: single layer at child-half above parent floor.
        print(f"[layering] WARN: LLM layering plan failed for "
              f"'{node['id']}'/'{parent_obj['id']}' ({exc}); "
              f"defaulting to a single layer.")
        return [{
            "items_count": instances,
            "z_world": parent_floor_z + sh / 2.0 + 0.005,
        }]


def _place_node(node, world_state, query, llm_fn):
    """Unified placement for any node (anchor or dependent)."""
    sw, sd, sh = _size_wh(node["size_hint"])
    room_half = world_state["room_half_size"]
    instances = int(node["instances"])

    # --- N:K split: when N children distribute evenly across K parents (N>K),
    #     place children_per_parent for each parent in a separate LLM call.
    parent_id = node.get("parent_id")
    parents = _ws_find_all_by_node(world_state, parent_id) if parent_id else []

    # ADDED (path B): honor parent_instance_index — pin this child group to
    # one specific instance of the parent. We reuse the existing
    # `_single_parent_override` plumbing (already wired into _parent_block,
    # _numeric_hints_block, _place_node_batch validation, _register_results).
    # Effect: parent list collapses to a single instance, N:K split is
    # skipped, surface bounds collapse to that one parent — and the multi-
    # instance parent itself stays as ONE batched node placed earlier with
    # full validation.
    pii = node.get("parent_instance_index")
    if pii is not None and parents and 0 <= pii < len(parents):
        node["_single_parent_override"] = parents[pii]
        parents = [parents[pii]]

    num_parents = len(parents)
    # ADDED: optional toggle. When WORLD_CREATOR_NK_SPLIT=0, skip the per-
    # parent N:K split and fall through to the "Normal path" below — that
    # places ALL N items in ONE LLM call, with the prompt seeing every
    # parent at once. Slower / larger prompt but lets the LLM coordinate
    # the layout across siblings without relying on prior_batches
    # replication. Default = 1 (keep existing per-parent split).
    nk_split_enabled = os.environ.get(
        "WORLD_CREATOR_NK_SPLIT", "1"
    ).strip().lower() not in ("0", "false", "no", "off")

    if (nk_split_enabled
            and num_parents > 1
            and instances > num_parents
            and instances % num_parents == 0):
        children_per_parent = instances // num_parents
        prior_batches = []

        # ADDED (layering): for type="in" (items inside a container) ask
        # the LLM upfront how many vertical layers to use per parent and
        # how many items per layer. Each layer then becomes its own
        # placement batch with a fixed z_world. For other relationship
        # types we keep the legacy single-batch-per-parent behaviour.
        rel = node.get("relationship") or {}
        rtype = rel.get("type") if isinstance(rel, dict) else None
        use_layering = (rtype == REL_IN)

        for pidx, parent_obj in enumerate(parents):
            ppx = parent_obj["Pose"]["x"]
            ppy = parent_obj["Pose"]["y"]

            if use_layering:
                sub_node_for_layering = dict(node)
                sub_node_for_layering["instances"] = children_per_parent
                sub_node_for_layering["_single_parent_override"] = parent_obj
                layer_plan = _decide_layering_plan(
                    sub_node_for_layering, parent_obj, query, llm_fn,
                )
            else:
                # Single-layer plan with engine-computed z (preserves the
                # pre-layering behaviour for on_surface / stacked_on).
                layer_plan = [{
                    "items_count": children_per_parent,
                    "z_world": None,
                }]

            parent_total_results = []
            cum_offset = 0
            for layer_idx, L in enumerate(layer_plan):
                lcount = int(L["items_count"])
                lz = L.get("z_world")

                sub_node = dict(node)
                sub_node["instances"] = lcount
                sub_node["_single_parent_override"] = parent_obj
                if lz is not None:
                    # Pin z for this layer; _place_node_batch will see this
                    # as a hint and pass it through to the consumer side.
                    sub_node["_z_override"] = float(lz)
                    sub_node["_layer_index"] = layer_idx
                    sub_node["_layer_count"] = len(layer_plan)

                # ADDED (surface plan): the LLM decides the layout strategy
                # (pattern, axis, gap, wall margin) for this batch BEFORE
                # placement. The placement call then has to respect the plan
                # as a hard minimum. None on LLM failure → falls back to
                # universal anti-overlap / inside-container rules already in
                # the validators.
                sub_node["_surface_plan"] = _decide_surface_arrangement_plan(
                    sub_node, parent_obj, query, llm_fn,
                )

                sub_results = _place_node_batch(
                    sub_node, world_state, query, llm_fn,
                    instance_offset=pidx * children_per_parent + cum_offset,
                    total_label=instances,
                    prior_batches=prior_batches,
                )
                parent_total_results.extend(sub_results)
                cum_offset += lcount

            prior_batches.append({
                "parent_id": parent_obj["id"],
                "parent_x": ppx,
                "parent_y": ppy,
                "results": [{"x": r["x"], "y": r["y"]} for r in parent_total_results],
                "results_rel": [
                    {"dx": r["x"] - ppx, "dy": r["y"] - ppy}
                    for r in parent_total_results
                ],
            })
        return

    # --- Normal path (single parent, 1:1 mapping, or indivisible) ---
    # OPTIMIZATION: Pass only relevant parents to reduce context size
    relevant_parents_for_normal = parents if parents else None
    
    base_kwargs = dict(
        model_name=node["model_name"],
        node_id=node["id"],
        size_w=sw, size_d=sd, size_h=sh,
        size_w_half=sw / 2.0, size_d_half=sd / 2.0,
        intent=_intent_text(node),
        reference_block=_reference_block(node, world_state),
        parent_block=_parent_block(node, world_state, relevant_parents=relevant_parents_for_normal),
        numeric_hints=_numeric_hints_block(node, world_state),
        world_state=_ws_summary(world_state, current_node=node),
        remaining=_ws_remaining_summary(world_state),
        room_w=f"{room_half*2:.1f}", room_l=f"{room_half*2:.1f}",
        neg_half=-room_half, pos_half=room_half,
        placement_clearance=PLACEMENT_CLEARANCE,
    )

    if instances == 1:
        prompt = PLACE_NODE_PROMPT.format(
            siblings_block="", prior_batches_block="",
            parent_quota_block="", **base_kwargs)

        def expect(raw):
            if not isinstance(raw, dict):
                return False, "expected JSON object"
            
            # Handle case where LLM wraps response in node_id key
            node_id = node["id"]
            
            # Try to unwrap if response is nested under node_id
            if node_id in raw:
                nested = raw[node_id]
                if isinstance(nested, list):
                    if len(nested) > 0:
                        raw = nested[0]
                    else:
                        return False, f"empty array in '{node_id}'"
                elif isinstance(nested, dict):
                    # LLM wrapped single object under node_id key
                    raw = nested
            
            # Check if we have the required keys at root level
            for k in ("x", "y"):
                if k not in raw:
                    return False, f"missing '{k}'"
            return True, ""

        raw = _llm_json(llm_fn, prompt, query, expect)
        
        # Unwrap if needed (in case expect() didn't catch it)
        node_id = node["id"]
        if node_id in raw:
            nested = raw[node_id]
            if isinstance(nested, list) and len(nested) > 0:
                raw = nested[0]
            elif isinstance(nested, dict):
                raw = nested
        x, y = _sanity_xy(raw["x"], raw["y"], room_half)
        z = max(0.0, float(raw.get("z", 0.0)))
        result = {"x": x, "y": y, "z": z, "yaw_deg": float(raw.get("yaw_deg", 0.0))}
        _register_results(node, [result], world_state)
        return

    # Multi-instance: split into batches of <= BATCH_SIZE.
    _place_node_batch(node, world_state, query, llm_fn,
                      instance_offset=0, total_label=instances)


def _build_pattern_specific_prompt(node, plan, batch_count, sw, sd, sh,
                                   parent_obj, query, intent_text):
    """Build a focused per-pattern prompt for batch placement.

    Returns (prompt_str, expect_fn) on success, or (None, None) if the plan
    is missing / unrecognised / unsuitable for this batch — the caller then
    falls back to the universal PLACE_NODE_BATCH_PROMPT.

    The pattern-specific prompts are intentionally short: a small-context
    model only sees the parameters relevant to one pattern, instead of the
    full kitchen-sink prompt that mixes all patterns and rules.
    """
    if not plan or not parent_obj or batch_count < 2:
        return None, None
    pattern = str(plan.get("pattern", "")).lower()
    axis = str(plan.get("axis", "centered")).lower()
    gap = float(plan.get("gap_m", max(sw, sd) * 1.15))
    wall = float(plan.get("wall_margin_m", 0.005))

    psw, psd, _ = _size_wh(parent_obj["size"])
    cx = parent_obj["Pose"]["x"]
    cy = parent_obj["Pose"]["y"]

    # Per-axis allowed range for child centre, accounting for own half-extent
    # and the plan's wall margin (additional safety beyond bbox).
    x_min = cx - psw / 2.0 + sw / 2.0 + wall
    x_max = cx + psw / 2.0 - sw / 2.0 - wall
    y_min = cy - psd / 2.0 + sd / 2.0 + wall
    y_max = cy + psd / 2.0 - sd / 2.0 - wall
    if x_min > x_max or y_min > y_max:
        # Footprint too small after margins — give up on specialisation,
        # let the universal prompt try with looser interpretation.
        return None, None

    query_excerpt = (query or "").strip()
    if len(query_excerpt) > 240:
        query_excerpt = query_excerpt[:240] + "..."

    if pattern == "row":
        # Pick the moving axis: explicit plan.axis if x/y, else the longer
        # free range. The other axis stays fixed at the parent centre.
        if axis == "x":
            move_axis, fixed_axis = "x", "y"
            axis_min, axis_max = x_min, x_max
            fixed_value = cy
        elif axis == "y":
            move_axis, fixed_axis = "y", "x"
            axis_min, axis_max = y_min, y_max
            fixed_value = cx
        else:
            # 'centered' or unknown — pick longer axis.
            if (x_max - x_min) >= (y_max - y_min):
                move_axis, fixed_axis = "x", "y"
                axis_min, axis_max = x_min, x_max
                fixed_value = cy
            else:
                move_axis, fixed_axis = "y", "x"
                axis_min, axis_max = y_min, y_max
                fixed_value = cx

        prompt = PLACE_BATCH_ROW_PROMPT.format(
            batch_count=batch_count,
            model_name=node["model_name"],
            parent_id=parent_obj["id"],
            size_w=sw, size_d=sd,
            parent_x=cx, parent_y=cy,
            axis=move_axis, fixed_axis=fixed_axis,
            axis_min=axis_min, axis_max=axis_max,
            fixed_axis_value=fixed_value,
            gap_m=gap,
            query_excerpt=query_excerpt,
        )
    elif pattern in ("grid", "dice"):
        prompt = PLACE_BATCH_GRID_PROMPT.format(
            batch_count=batch_count,
            model_name=node["model_name"],
            parent_id=parent_obj["id"],
            size_w=sw, size_d=sd,
            parent_x=cx, parent_y=cy,
            x_min=x_min, x_max=x_max,
            y_min=y_min, y_max=y_max,
            gap_m=gap,
            query_excerpt=query_excerpt,
        )
    else:
        # ring / scatter / single / unknown → no specialisation
        return None, None

    def expect(raw, _bc=batch_count):
        payload = _normalise_batch_payload(raw)
        if not isinstance(payload, list):
            return False, (
                f"schema mismatch: expected JSON array of {_bc} items, "
                f"got {type(raw).__name__}"
            )
        if len(payload) != _bc:
            return False, (
                f"schema mismatch: expected EXACTLY {_bc} items, got "
                f"{len(payload)}"
            )
        for it in payload:
            if not isinstance(it, dict) or "x" not in it or "y" not in it:
                return False, (
                    "each array item must be an object with numeric 'x' "
                    "and 'y' fields"
                )
        return True, ""

    return prompt, expect


def _compute_row_suggestions(parent_obj, sw, sd, batch_count, plan):
    """Geometrically valid target points for `row` pattern, used as a hint
    for the FIX prompt. Code computes them; LLM still emits the final answer.
    """
    if not parent_obj or batch_count < 1:
        return []
    psw, psd, _ = _size_wh(parent_obj["size"])
    cx = parent_obj["Pose"]["x"]
    cy = parent_obj["Pose"]["y"]
    plan = plan or {}
    axis = str(plan.get("axis", "centered")).lower()
    wall = float(plan.get("wall_margin_m", 0.005))

    x_min = cx - psw / 2.0 + sw / 2.0 + wall
    x_max = cx + psw / 2.0 - sw / 2.0 - wall
    y_min = cy - psd / 2.0 + sd / 2.0 + wall
    y_max = cy + psd / 2.0 - sd / 2.0 - wall

    if axis == "x" or (axis not in ("x", "y") and (x_max - x_min) >= (y_max - y_min)):
        lo, hi, fixed = x_min, x_max, ("y", cy)
        moving_axis = "x"
    else:
        lo, hi, fixed = y_min, y_max, ("x", cx)
        moving_axis = "y"

    if batch_count == 1:
        mid = 0.5 * (lo + hi)
        return [(mid, fixed[1]) if moving_axis == "x" else (fixed[1], mid)]

    span = hi - lo
    if span <= 0:
        center = 0.5 * (lo + hi)
        return [(center, fixed[1]) if moving_axis == "x" else (fixed[1], center)
                for _ in range(batch_count)]

    step = span / (batch_count - 1)
    points = []
    for i in range(batch_count):
        v = lo + i * step
        if moving_axis == "x":
            points.append((v, fixed[1]))
        else:
            points.append((fixed[0] if False else cx, v))  # x = parent center
    return points


def _llm_fix_batch_violations(node, items, violations, parent_obj, sw, sd,
                              gap_m, query, llm_fn, batch_count, previous_attempts=None):
    """Last-resort focused LLM call after DOMAIN_RETRIES failures.

    Builds geometrically valid suggested targets (deterministically) and
    asks the LLM to emit a corrected batch — the LLM still owns the final
    coordinates, but it now has a concrete numerical anchor instead of a
    pile of forbidden points.

    ENHANCED: Now accepts previous_attempts history to provide rich context
    about what has been tried and why it failed.

    Returns a list of corrected items (same shape as the retry consumer
    expects) or None if even this call fails.
    """
    if not parent_obj:
        return None

    cx = parent_obj["Pose"]["x"]
    cy = parent_obj["Pose"]["y"]

    plan = node.get("_surface_plan") or {}
    suggestions = _compute_row_suggestions(parent_obj, sw, sd, batch_count, plan)

    last_items_lines = []
    for i, it in enumerate(items, 1):
        last_items_lines.append(
            f"  item {i}: x={it.get('x', 0.0):.3f}, y={it.get('y', 0.0):.3f}, "
            f"yaw={it.get('yaw_deg', 0.0):.1f}"
        )
    last_items_block = "\n".join(last_items_lines) if last_items_lines else "  (none)"

    suggestions_lines = []
    for i, (sx, sy) in enumerate(suggestions[:batch_count], 1):
        suggestions_lines.append(f"  item {i}: x={sx:.3f}, y={sy:.3f}")
    suggestions_block = (
        "\n".join(suggestions_lines)
        if suggestions_lines
        else "  (no suggestions — choose any valid layout)"
    )

    violations_lines = []
    for v in violations[:6]:
        violations_lines.append(f"  - {v}")
    violations_block = "\n".join(violations_lines) if violations_lines else "  (unknown)"

    # ENHANCED: Add history of previous attempts if available
    history_block = ""
    if previous_attempts:
        history_lines = ["\n\nHISTORY OF PREVIOUS FAILED ATTEMPTS:"]
        for prev in previous_attempts[-5:]:  # Last 5 attempts
            history_lines.append(f"  Attempt #{prev['attempt_num']}:")
            history_lines.append(f"    Tried: {prev.get('coordinates', 'N/A')}")
            if prev.get('violations'):
                top_violations = prev['violations'][:2]  # Top 2 violations per attempt
                history_lines.append(f"    Failed because: {'; '.join(top_violations)}")
        history_lines.append("\nLEARN FROM THESE MISTAKES - do not repeat the same errors!\n")
        history_block = "\n".join(history_lines)
    
    prompt = PLACE_BATCH_FIX_PROMPT.format(
        batch_count=batch_count,
        model_name=node["model_name"],
        parent_id=parent_obj["id"],
        size_w=sw, size_d=sd,
        parent_x=cx, parent_y=cy,
        last_items_block=last_items_block,
        violations_block=violations_block,
        suggestions_block=suggestions_block,
        gap_m=gap_m,
    ) + history_block

    def expect(raw, _bc=batch_count):
        payload = _normalise_batch_payload(raw)
        if not isinstance(payload, list) or len(payload) != _bc:
            return False, f"expected JSON array of {_bc} items"
        for it in payload:
            if not isinstance(it, dict) or "x" not in it or "y" not in it:
                return False, "each item needs numeric x, y"
        return True, ""

    # ENHANCED: Build validation context for _llm_json
    validation_ctx = None
    if previous_attempts:
        forbidden_pts = [(it['x'], it['y']) for it in items] if items else []
        validation_ctx = {
            'previous_attempts': previous_attempts,
            'forbidden_points': forbidden_pts,
            'context_summary': (
                f"FIX-BATCH call after {len(previous_attempts)} failed attempts. "
                f"This is the LAST CHANCE to get it right. Use the suggested coordinates "
                f"as a starting point and avoid all previous mistakes."
            ),
        }
    
    try:
        raw = _llm_json(llm_fn, prompt, query, expect, validation_context=validation_ctx)
        payload = _normalise_batch_payload(raw)
        out = []
        for it in payload:
            out.append({
                "x": float(it["x"]),
                "y": float(it["y"]),
                "z": float(it.get("z", 0.0)),
                "yaw_deg": float(it.get("yaw_deg", 0.0)),
                "layer": int(it.get("layer", 0)) if isinstance(it.get("layer"), (int, float)) else 0,
            })
        print(
            f"[scene_planner] FIX-BATCH succeeded for node {node['id']} "
            f"after DOMAIN_RETRIES exhausted."
        )
        return out
    except Exception as exc:
        print(
            f"[scene_planner] FIX-BATCH also failed for node {node['id']}: {exc}"
        )
        return None


def _place_node_batch(node, world_state, query, llm_fn, instance_offset=0,
                      total_label=None, prior_batches=None):
    """Place multiple instances of a node in batches of <= BATCH_SIZE.

    instance_offset: global instance index offset (for N:K split sub-batches).
    total_label: total instances shown to LLM (for context).
    prior_batches: list of previously placed sub-batch results (from N:K split)
                   used for cross-batch variety; capped in the prompt.

    For surface-like relationships (on_surface / stacked_on / in) all
    instances of a node are placed in ONE LLM call (batch_size = min(N,
    BATCH_SIZE)) so the model sees the whole layout at once and can plan
    globally — instead of placing one instance at a time and clustering.
    Anti-overlap and footprint validation work on the joint answer; on
    violation the whole batch is retried with feedback up to DOMAIN_RETRIES.

    Returns: list of all placed results (also registered into world_state).
    """
    sw, sd, sh = _size_wh(node["size_hint"])
    room_half = world_state["room_half_size"]
    instances = int(node["instances"])
    if total_label is None:
        total_label = instances
    prior_batches = prior_batches or []

    rel = node.get("relationship") or {}
    rtype = rel.get("type") if isinstance(rel, dict) else None
    is_surface = rtype in SURFACE_LIKE_RELS
    # Choose batch size:
    #   • 1:1 reference (e.g. "plate opposite each chair") → ONE LLM call per
    #     instance so the per-instance RECOMMENDED TARGET hint kicks in. With
    #     a target like "≈ (+0.43, 0) within ±5 cm" the LLM has a single
    #     well-bounded decision per call and pairings can't be swapped.
    #   • No reference → joint batch (LLM sees all instances together and
    #     plans the global layout in one go).
    ref_id_for_batch = rel.get("reference") if isinstance(rel, dict) else None
    has_1to1_ref = False
    if is_surface and ref_id_for_batch and ref_id_for_batch != ROOM_CENTER:
        ref_list_for_batch = _ws_find_all_by_node(world_state, ref_id_for_batch)
        if ref_list_for_batch and len(ref_list_for_batch) == instances:
            has_1to1_ref = True
    effective_batch_size = (
        1 if (is_surface and has_1to1_ref)
        else (min(instances, BATCH_SIZE) if is_surface else BATCH_SIZE)
    )

    # Resolve parent(s) for validation. Mirrors _register_results logic.
    single_parent = node.get("_single_parent_override")
    if single_parent:
        validation_parents = [single_parent]
    else:
        parent_id = node.get("parent_id")
        validation_parents = (
            _ws_find_all_by_node(world_state, parent_id) if parent_id else []
        )
    use_per_instance_parent = (
        len(validation_parents) > 1 and len(validation_parents) == instances
    )

    def parent_for_local(local_idx):
        if not validation_parents:
            return None
        if use_per_instance_parent:
            return validation_parents[local_idx]
        if single_parent:
            return single_parent
        return validation_parents[0]

    # Render the surface arrangement plan (if any) as a prompt block.
    # This is the semantic-decision layer: pattern / axis / gap /
    # wall-margin chosen by a separate LLM call upstream. Placement
    # below has to honour these as HARD MINIMUMS (validator backstop).
    sp = node.get("_surface_plan")
    if sp:
        surface_plan_block = (
            "SURFACE ARRANGEMENT PLAN (chosen by a previous LLM call — "
            "treat these as HARD MINIMUMS, not suggestions):\n"
            f"  • Pattern        : {sp['pattern']}\n"
            f"  • Primary axis   : {sp['axis']}\n"
            f"  • Min sibling gap: {sp['gap_m']:.3f} m (centre-to-centre)\n"
            f"  • Min wall margin: {sp['wall_margin_m']:.3f} m (parent edge "
            f"to child edge)\n"
        )
        if sp.get("rationale"):
            surface_plan_block += f"  • Rationale      : {sp['rationale']}\n"
        surface_plan_block += (
            "Items must satisfy: every pairwise distance ≥ "
            f"{sp['gap_m']:.3f} m on at least one axis, AND every item's "
            "centre lies at least (parent_half − child_half − "
            f"{sp['wall_margin_m']:.3f}) from the parent centre on both "
            "axes. Use the plan's pattern/axis as the layout shape.\n\n"
        )
    else:
        surface_plan_block = ""

    # OPTIMIZATION: Get unique parents for this batch to avoid describing all parents
    # when we only need specific ones. This significantly reduces context size.
    # Use dict IDs to deduplicate since parent objects are unhashable dicts
    seen_parent_ids = set()
    unique_batch_parents = []
    for p in validation_parents:
        if p is not None:
            pid = p.get("id")
            if pid not in seen_parent_ids:
                seen_parent_ids.add(pid)
                unique_batch_parents.append(p)
    
    base_kwargs = dict(
        model_name=node["model_name"],
        node_id=node["id"],
        size_w=sw, size_d=sd, size_h=sh,
        size_w_half=sw / 2.0, size_d_half=sd / 2.0,
        intent=_intent_text(node),
        reference_block=_reference_block(node, world_state),
        parent_block=_parent_block(node, world_state, relevant_parents=unique_batch_parents),
        numeric_hints=_numeric_hints_block(node, world_state),
        world_state=_ws_summary(world_state, current_node=node),
        remaining=_ws_remaining_summary(world_state),
        room_w=f"{room_half*2:.1f}", room_l=f"{room_half*2:.1f}",
        neg_half=-room_half, pos_half=room_half,
        placement_clearance=PLACEMENT_CLEARANCE,
        prior_batches_block=_prior_batches_block(prior_batches, query),
        surface_plan_block=surface_plan_block,
    )

    placed_in_group = []
    all_results = []
    for batch_start in range(0, instances, effective_batch_size):
        batch_end = min(batch_start + effective_batch_size, instances)
        batch_count = batch_end - batch_start
        from_idx = instance_offset + batch_start + 1
        to_idx = instance_offset + batch_end

        # Resolve which parents the items in this batch belong to.
        batch_parents = [parent_for_local(batch_start + k) for k in range(batch_count)]

        # ADDED: when surface placement runs one instance per LLM call,
        # tell _numeric_hints_block which instance is being placed so it can
        # surface the per-instance REFERENCE-ALIGNED TARGET for this call.
        # Universal: any node with reference of matching count benefits.
        if batch_count == 1 and is_surface:
            node["_current_instance_idx"] = batch_start
            cur_kwargs = dict(
                base_kwargs,
                numeric_hints=_numeric_hints_block(node, world_state),
            )
        else:
            cur_kwargs = base_kwargs

        # Quota hint: only meaningful when placing one-by-one on a single parent.
        if batch_count == 1 and is_surface and batch_parents and batch_parents[0] is not None:
            quota = _parent_quota_block(node, batch_start, batch_parents[0], sw, sd)
        else:
            quota = ""

        # --- Build prompt + extract items + validate, with domain retries.
        feedback = ""
        items = []
        # ENHANCED: Track previous attempts for rich feedback to _llm_json
        previous_attempts = []
        
        for attempt in range(DOMAIN_RETRIES + 1):
            sb = _siblings_block(placed_in_group, self_w=sw, self_d=sd)
            if batch_count == 1:
                # CHANGED: was **base_kwargs — now uses cur_kwargs so the
                # per-instance REFERENCE-ALIGNED TARGET (added above) lands in
                # this single-instance call.
                prompt = PLACE_NODE_PROMPT.format(
                    siblings_block=sb, parent_quota_block=quota, **cur_kwargs)

                def expect(raw):
                    if not isinstance(raw, dict):
                        return False, "expected JSON object"
                    for k in ("x", "y"):
                        if k not in raw:
                            return False, f"missing '{k}'"
                    return True, ""
            else:
                # ADDED: try a pattern-specific compact prompt first (row /
                # grid / dice). Falls through to the universal prompt when
                # the surface plan is missing or the pattern is one we
                # haven't specialised yet (ring / scatter / single).
                pattern_parent = (
                    batch_parents[0] if batch_parents and batch_parents[0]
                    else None
                )
                pat_prompt, pat_expect = _build_pattern_specific_prompt(
                    node, node.get("_surface_plan"), batch_count,
                    sw, sd, sh, pattern_parent, query, _intent_text(node),
                )
                if pat_prompt is not None:
                    prompt = pat_prompt
                    expect = pat_expect
                    # Skip the universal prompt build below for this attempt.
                    if feedback:
                        prompt = prompt + feedback
                    
                    # ENHANCED: Build validation context from previous attempts
                    validation_ctx = None
                    if previous_attempts:
                        validation_ctx = {
                            'previous_attempts': previous_attempts,
                            'context_summary': f"Pattern '{node.get('_surface_plan',{}).get('pattern', 'unknown')}' placement has failed {len(previous_attempts)} time(s). Learn from these mistakes.",
                        }
                    
                    t_idx = min(attempt, len(DOMAIN_RETRY_TEMPERATURES) - 1)
                    raw = _llm_json(
                        llm_fn, prompt, query, expect,
                        validation_context=validation_ctx,
                        temperature=DOMAIN_RETRY_TEMPERATURES[t_idx],
                        seed=DOMAIN_RETRY_SEEDS[t_idx],
                    )
                    cap_layers = 1
                    cap_per_layer = None
                    if rtype == REL_IN and validation_parents:
                        cap_per_layer, max_layers, _, _, _ = _layer_capacity(
                            validation_parents[0], sw, sd, sh
                        )
                        needed = (instances + cap_per_layer - 1) // cap_per_layer
                        cap_layers = max(1, min(max_layers, needed))
                    payload = _normalise_batch_payload(raw)
                    items = [{
                        "x": float(it["x"]),
                        "y": float(it["y"]),
                        "z": float(it.get("z", 0.0)),
                        "yaw_deg": float(it.get("yaw_deg", 0.0)),
                        "layer": max(0, min(
                            cap_layers - 1,
                            int(it.get("layer", 0)) if isinstance(it.get("layer"), (int, float)) else 0,
                        )),
                    } for it in payload]
                    # Run the same validators as the universal path; on
                    # failure, retry the same pattern prompt with feedback.
                    violations = []
                    if is_surface and all(p is not None for p in batch_parents):
                        violations.extend(
                            _violations_surface(items, batch_parents, sw, sd, rel_type=rtype)
                        )
                    if is_surface:
                        violations.extend(_violations_overlap_pairs(items, sw, sd))
                        violations.extend(
                            _violations_overlap_with_siblings(items, placed_in_group, sw, sd)
                        )
                        _surface_plan_for_check = node.get("_surface_plan")
                        violations.extend(
                            _violations_surface_plan(
                                items, batch_parents, sw, sd, _surface_plan_for_check
                            )
                        )
                    if not violations:
                        break
                    
                    # ENHANCED: Record this failed attempt for future retries
                    coords_summary = ", ".join([f"({it['x']:.2f},{it['y']:.2f})" for it in items[:3]])
                    if len(items) > 3:
                        coords_summary += f" ... ({len(items)} total)"
                    previous_attempts.append({
                        'attempt_num': attempt + 1,
                        'coordinates': coords_summary,
                        'violations': violations[:5],  # Top 5 violations
                    })
                    
                    if attempt < DOMAIN_RETRIES:
                        feedback = (
                            "\n\nYour previous answer was REJECTED:\n"
                            + "\n".join(f"  - {v}" for v in violations)
                            + "\nFix the listed issues and re-emit the JSON array.\n"
                        )
                        continue
                    # Exhausted retries on the pattern path — same FIX-BATCH
                    # last-resort as the universal path uses.
                    gap_for_fix = (
                        float(node.get("_surface_plan", {}).get("gap_m"))
                        if node.get("_surface_plan")
                        else max(sw, sd) * 1.15
                    )
                    fixed_items = _llm_fix_batch_violations(
                        node, items, violations, pattern_parent, sw, sd,
                        gap_for_fix, query, llm_fn, batch_count,
                    )
                    if fixed_items is not None:
                        items = fixed_items
                    else:
                        raise RuntimeError(
                            f"[scene_planner] pattern '{node.get('_surface_plan',{}).get('pattern')}' "
                            f"exhausted DOMAIN_RETRIES and FIX-BATCH for "
                            f"node {node['id']}. Last violations: {violations}"
                        )
                    break
                prompt = PLACE_NODE_BATCH_PROMPT.format(
                    batch_count=batch_count,
                    from_idx=from_idx, to_idx=to_idx, total=total_label,
                    siblings_block=sb,
                    parent_quota_block="",
                    **base_kwargs,
                )

                def expect(raw, _bc=batch_count):
                    # Use the same normaliser as the consumer below — keeps
                    # validation and consumption in lockstep, so any wrapper
                    # we accept here is also unwrappable downstream.
                    payload = _normalise_batch_payload(raw)
                    if not isinstance(payload, list):
                        return False, (
                            f"schema mismatch: expected a JSON array of "
                            f"{_bc} placement objects, got "
                            f"{type(raw).__name__}"
                        )
                    if len(payload) != _bc:
                        return False, (
                            f"schema mismatch: expected EXACTLY {_bc} items "
                            f"in the array, got {len(payload)}. Re-emit ALL "
                            f"{_bc} placements as one JSON array."
                        )
                    for it in payload:
                        if not isinstance(it, dict) or "x" not in it or "y" not in it:
                            return False, (
                                "each array item must be an object with "
                                "numeric 'x' and 'y' fields"
                            )
                    return True, ""

            if feedback:
                prompt = prompt + feedback

            # ENHANCED: Build validation context from previous attempts
            validation_ctx = None
            if previous_attempts:
                forbidden_pts = [(it['x'], it['y']) for it in items] if items else []
                validation_ctx = {
                    'previous_attempts': previous_attempts,
                    'forbidden_points': forbidden_pts,
                    'context_summary': f"Batch placement for {node['model_name']} has failed {len(previous_attempts)} time(s). Avoid repeating the same coordinates.",
                }

            # ADDED (task: temperature gradient): on retries widen sampling
            # so a stuck LLM has license to pick a different point. Falls
            # back to the last entry if attempt exceeds the table.
            t_idx = min(attempt, len(DOMAIN_RETRY_TEMPERATURES) - 1)
            raw = _llm_json(
                llm_fn, prompt, query, expect,
                validation_context=validation_ctx,
                temperature=DOMAIN_RETRY_TEMPERATURES[t_idx],
                seed=DOMAIN_RETRY_SEEDS[t_idx],
            )
            # Compute layer cap so we can clamp LLM-supplied layer ids.
            # Only meaningful for "in" with a known parent and overflow.
            cap_layers = 1
            cap_per_layer = None
            if rtype == REL_IN and validation_parents:
                cap_per_layer, max_layers, _, _, _ = _layer_capacity(
                    validation_parents[0], sw, sd, sh
                )
                needed = (instances + cap_per_layer - 1) // cap_per_layer
                cap_layers = max(1, min(max_layers, needed))

            def _layer_of(raw_item):
                try:
                    L = int(raw_item.get("layer", 0))
                except (TypeError, ValueError):
                    L = 0
                if L < 0:
                    L = 0
                if L > cap_layers - 1:
                    L = cap_layers - 1
                return L

            if batch_count == 1:
                items = [{
                    "x": float(raw["x"]),
                    "y": float(raw["y"]),
                    "z": float(raw.get("z", 0.0)),
                    "yaw_deg": float(raw.get("yaw_deg", 0.0)),
                    "layer": _layer_of(raw),
                }]
            else:
                # Same normaliser as in expect() — handles
                # positions/items/answer/results wrappers and the single-
                # dict-instead-of-array case in one place.
                payload = _normalise_batch_payload(raw)
                items = [{
                    "x": float(it["x"]),
                    "y": float(it["y"]),
                    "z": float(it.get("z", 0.0)),
                    "yaw_deg": float(it.get("yaw_deg", 0.0)),
                    "layer": _layer_of(it),
                } for it in payload]

            # Validate (only for surface placements).
            violations = []
            if is_surface and all(p is not None for p in batch_parents):
                violations.extend(
                    _violations_surface(items, batch_parents, sw, sd, rel_type=rtype)
                )
            if is_surface:
                violations.extend(_violations_overlap_pairs(items, sw, sd))
                violations.extend(
                    _violations_overlap_with_siblings(items, placed_in_group, sw, sd)
                )
                # ADDED: surface-plan compliance check. The plan was
                # decided by a separate LLM call upstream; placement
                # must respect its gap_m and wall_margin_m as HARD
                # MINIMUMS (over and above the universal 1.15× / full-
                # child-size validators above). Without this check the
                # plan would be advisory-only.
                _surface_plan_for_check = node.get("_surface_plan")
                violations.extend(
                    _violations_surface_plan(
                        items, batch_parents, sw, sd, _surface_plan_for_check
                    )
                )

            if not violations:
                break

            # ENHANCED: Record this failed attempt for future retries
            coords_summary = ", ".join([f"({it['x']:.2f},{it['y']:.2f})" for it in items[:3]])
            if len(items) > 3:
                coords_summary += f" ... ({len(items)} total)"
            previous_attempts.append({
                'attempt_num': attempt + 1,
                'coordinates': coords_summary,
                'violations': violations[:5],  # Top 5 violations
            })

            if attempt < DOMAIN_RETRIES:
                # ADDED (task 1.1): explicit FORBIDDEN POINTS list — every
                # already-placed sibling in this group plus the just-rejected
                # attempt items. Gives the LLM a concrete anti-target list so
                # it doesn't repeat the same coordinate that just failed.
                forbidden = []
                for s in placed_in_group:
                    forbidden.append(
                        f"({s['x']:.3f}, {s['y']:.3f}) "
                        f"[sibling instance {s.get('instance', '?')}]"
                    )
                for k, it in enumerate(items, start=1):
                    item_label = (
                        f"item {k} of your rejected attempt"
                        if len(items) > 1
                        else "your rejected attempt"
                    )
                    forbidden.append(
                        f"({it['x']:.3f}, {it['y']:.3f}) [{item_label}]"
                    )
                forbidden_block = ""
                if forbidden:
                    forbidden_block = (
                        "\nFORBIDDEN POINTS — overlapping any of these makes "
                        "the answer invalid; required separation from EACH "
                        f"is |Δx| > {sw:.3f} OR |Δy| > {sd:.3f}:\n"
                        + "\n".join(f"  - {p}" for p in forbidden)
                        + "\n"
                    )

                # Minimal feedback — just the violations + forbidden
                # points. The semantic plan upstream already tells the
                # LLM what to do; the verify-pair / axis-flip / zone-rule
                # nudges that used to live here were compensation for
                # not having a plan, and now muddy the signal.
                feedback = (
                    "\n\nYour previous answer was REJECTED for the following reasons:\n"
                    + "\n".join(f"  - {v}" for v in violations)
                    + forbidden_block
                    + "\nFix the listed issues, follow the SURFACE "
                    "ARRANGEMENT PLAN (if present), stay inside the "
                    "parent footprint, return STRICT valid JSON.\n"
                )
            else:
                # FIX-BATCH: one final focused LLM call with deterministic
                # suggested targets (computed by code, returned by LLM).
                # The targets satisfy the footprint and the gap; LLM may
                # accept them as-is or nudge slightly. This replaces the
                # previous WARN-passthrough that silently shipped broken
                # placements.
                fix_parent = (
                    batch_parents[0] if batch_parents and batch_parents[0]
                    else None
                )
                gap_for_fix = (
                    float(node.get("_surface_plan", {}).get("gap_m"))
                    if node.get("_surface_plan")
                    else max(sw, sd) * 1.15
                )
                fixed_items = _llm_fix_batch_violations(
                    node, items, violations, fix_parent, sw, sd,
                    gap_for_fix, query, llm_fn, batch_count,
                    previous_attempts=previous_attempts,
                )
                if fixed_items is not None:
                    items = fixed_items
                else:
                    raise RuntimeError(
                        f"[scene_planner] domain violations after "
                        f"{DOMAIN_RETRIES + 1} attempts AND FIX-BATCH for "
                        f"node {node['id']} batch {from_idx}..{to_idx}. "
                        f"Last violations: {violations}"
                    )

        # Register sanitized items.
        # ADDED (layering): if the upstream layering planner pinned an
        # explicit z_world for this batch (one layer of a stack), use it
        # verbatim instead of trusting the LLM-supplied / engine-derived z.
        # This is what makes the two-layer apple stack actually sit at the
        # decided heights rather than collapsing onto a single plane.
        z_override = node.get("_z_override")
        for i, item in enumerate(items):
            x, y = _sanity_xy(item["x"], item["y"], room_half)
            if z_override is not None:
                z = float(z_override)
            else:
                z = max(0.0, float(item["z"]))
            r = {
                "instance": from_idx + i,
                "x": x, "y": y, "z": z,
                "yaw_deg": float(item["yaw_deg"]),
                "layer": int(item.get("layer", 0)),
            }
            placed_in_group.append(r)
            all_results.append(r)

    # ADDED: clean up the per-instance index marker we set for surface placements
    # so the same node dict can be reused safely (e.g. through later phases or
    # debugging dumps) without a stale instance pointer.
    node.pop("_current_instance_idx", None)

    _register_results(node, all_results, world_state, instance_offset=instance_offset)
    return all_results


def _register_results(node, results, world_state, instance_offset=0):
    sz = node["size_hint"]
    _, _, sh = _size_wh(sz)
    obj_h = sh

    rel = node.get("relationship") or {}
    rel_type = rel.get("type") if isinstance(rel, dict) else None

    parent_id = node.get("parent_id")
    # N:K split passes a single-parent override on the sub-node.
    single_parent = node.get("_single_parent_override")
    if single_parent:
        parents = [single_parent]
    else:
        parents = _ws_find_all_by_node(world_state, parent_id) if parent_id else []
    # When child instances == parent instances, each child sits on its own parent.
    use_per_instance_parent = len(parents) > 1 and len(parents) == len(results)

    def z_for_instance(idx):
        layer = int(results[idx].get("layer", 0)) if idx < len(results) else 0
        layer_offset = layer * obj_h * LAYER_SPACING
        clearance = PLACEMENT_CLEARANCE
        if rel_type == REL_AROUND:
            return obj_h / 2.0 + layer_offset + clearance
        if use_per_instance_parent:
            p = parents[idx]
        elif parents:
            p = parents[0]
        else:
            return obj_h / 2.0 + layer_offset + clearance
        _, _, ph = _size_wh(p["size"])
        return p["Pose"]["z"] + ph / 2.0 + obj_h / 2.0 + layer_offset + clearance

    for i, r in enumerate(results):
        instance_id = f"{node['id']}_inst_{instance_offset + i}"
        pose = {"x": float(r["x"]), "y": float(r["y"]), "z": float(z_for_instance(i))}
        _ws_add(world_state, instance_id, node["model_name"], sz, pose, float(r.get("yaw_deg", 0.0)))

    placed_count = len(results)
    new_remaining = []
    for rem in world_state.get("remaining", []):
        if rem.get("model_name") != node["model_name"]:
            new_remaining.append(rem)
            continue
        new_count = int(rem.get("count", 0)) - placed_count
        if new_count > 0:
            new_remaining.append({**rem, "count": new_count})
    world_state["remaining"] = new_remaining


def _place_co_group(group_nodes, world_state, query, llm_fn):
    """Coordinated placement of independent sibling nodes in ONE LLM call.

    Builds a flat list of every instance across all member nodes (in node-list
    order, then instance order within the node), prompts the LLM with the
    parent footprint and a per-item bounds line, validates against pairwise
    overlap (mixed sizes) and bounds, retries with feedback up to
    DOMAIN_RETRIES, then registers results back into world_state under each
    original node.
    """
    from project.scene_prompts import COPLACE_PROMPT

    # Order: by place_order then by id to keep output stable.
    group_nodes = sorted(group_nodes, key=lambda n: (int(n.get("place_order", 1)), n["id"]))
    parent_id = group_nodes[0]["parent_id"]
    parent_obj = _ws_find_all_by_node(world_state, parent_id)[0]

    psw, psd, psh = _size_wh(parent_obj["size"])
    parent_hx, parent_hy = psw / 2.0, psd / 2.0
    parent_x, parent_y = parent_obj["Pose"]["x"], parent_obj["Pose"]["y"]
    parent_top_z = parent_obj["Pose"]["z"] + psh / 2.0
    margin = 0.005

    # Build the flat per-instance list (with sizes for validation and prompt).
    flat = []  # list of dicts { node, node_idx, inst_idx_in_node, w, d, h, label }
    for n in group_nodes:
        sw, sd, sh = _size_wh(n["size_hint"])
        for k in range(int(n.get("instances", 1))):
            flat.append({
                "node": n,
                "node_idx": group_nodes.index(n),
                "inst_idx_in_node": k,
                "w": sw, "d": sd, "h": sh,
                "label": f"{n['model_name']}#{k + 1}",
            })

    n_total = len(flat)

    # Prompt parts.
    items_lines = []
    bounds_lines = []
    for i, it in enumerate(flat, start=1):
        items_lines.append(
            f"  {i}. type={it['node']['model_name']}, "
            f"size_w={it['w']:.3f}, size_d={it['d']:.3f}, size_h={it['h']:.3f}"
        )
        lim_x = max(0.0, parent_hx - it["w"] / 2.0 - margin)
        lim_y = max(0.0, parent_hy - it["d"] / 2.0 - margin)
        bounds_lines.append(
            f"  item {i}: x ∈ [{parent_x - lim_x:.3f}, {parent_x + lim_x:.3f}], "
            f"y ∈ [{parent_y - lim_y:.3f}, {parent_y + lim_y:.3f}]"
        )

    items_block = "\n".join(items_lines)
    bounds_block = "\n".join(bounds_lines)

    print(
        f"[place] coplace group on '{parent_id}' "
        f"({len(group_nodes)} nodes, {n_total} instances): "
        f"{[n['model_name'] + 'x' + str(n.get('instances', 1)) for n in group_nodes]}"
    )

    feedback = ""
    items = []
    for attempt in range(DOMAIN_RETRIES + 1):
        prompt = COPLACE_PROMPT.format(
            n_total=n_total,
            parent_id=parent_obj["id"],
            parent_model=parent_obj.get("Model", "?"),
            parent_x=parent_x, parent_y=parent_y, parent_top_z=parent_top_z,
            parent_hx=parent_hx, parent_hy=parent_hy,
            query=query,
            items_block=items_block,
            bounds_block=bounds_block,
        )
        if feedback:
            prompt = prompt + feedback

        def expect(raw, _n=n_total):
            if isinstance(raw, dict) and "positions" in raw:
                raw = raw["positions"]
            if not isinstance(raw, list) or len(raw) != _n:
                return False, (
                    f"expected list of {_n} positions, got "
                    f"{type(raw).__name__} len="
                    f"{len(raw) if isinstance(raw, list) else '?'}"
                )
            for it in raw:
                if not isinstance(it, dict) or "x" not in it or "y" not in it:
                    return False, "each item must have 'x' and 'y'"
            return True, ""

        t_idx = min(attempt, len(DOMAIN_RETRY_TEMPERATURES) - 1)
        raw = _llm_json(
            llm_fn, prompt, query, expect,
            temperature=DOMAIN_RETRY_TEMPERATURES[t_idx],
            seed=DOMAIN_RETRY_SEEDS[t_idx],
        )
        if isinstance(raw, dict) and "positions" in raw:
            raw = raw["positions"]
        items = [{
            "x": float(it["x"]),
            "y": float(it["y"]),
            "yaw_deg": float(it.get("yaw_deg", 0.0)),
            "layer": int(it.get("layer", 0)) if isinstance(it.get("layer", 0), (int, float)) else 0,
            "w": flat[i]["w"], "d": flat[i]["d"],
            "label": flat[i]["label"],
        } for i, it in enumerate(raw)]

        # Validate: bounds + mixed pairwise overlap.
        violations = []
        for i, item in enumerate(items, start=1):
            lim_x = max(0.0, parent_hx - item["w"] / 2.0 - margin)
            lim_y = max(0.0, parent_hy - item["d"] / 2.0 - margin)
            x_lo, x_hi = parent_x - lim_x, parent_x + lim_x
            y_lo, y_hi = parent_y - lim_y, parent_y + lim_y
            if not (x_lo <= item["x"] <= x_hi):
                violations.append(
                    f"item {i} ({item['label']}): x={item['x']:.3f} OUTSIDE "
                    f"[{x_lo:.3f}, {x_hi:.3f}]"
                )
            if not (y_lo <= item["y"] <= y_hi):
                violations.append(
                    f"item {i} ({item['label']}): y={item['y']:.3f} OUTSIDE "
                    f"[{y_lo:.3f}, {y_hi:.3f}]"
                )
        violations.extend(_violations_overlap_mixed(items))

        # Soft centroid-balance check: enforce only if the query has no
        # explicit asymmetric trigger (corner, wall, side, edge, etc.).
        q_lower = (query or "").lower()
        asymmetric_trigger = any(
            kw in q_lower for kw in (
                "corner", "угол", "near wall", "у стены", "side", "edge",
                "край", "по краю", "одной стороне", "сбоку",
            )
        )
        if not asymmetric_trigger and items:
            mean_x = sum(it["x"] for it in items) / len(items)
            mean_y = sum(it["y"] for it in items) / len(items)
            if abs(mean_x - parent_x) > 0.05:
                violations.append(
                    f"centroid x={mean_x:.3f} too far from parent center "
                    f"{parent_x:.3f} (>0.05 m). Rebalance: shift the whole "
                    f"composition along X so mean(x_i) ≈ {parent_x:.3f}."
                )
            if abs(mean_y - parent_y) > 0.05:
                violations.append(
                    f"centroid y={mean_y:.3f} too far from parent center "
                    f"{parent_y:.3f} (>0.05 m). Rebalance: shift the whole "
                    f"composition along Y so mean(y_i) ≈ {parent_y:.3f}."
                )

        if not violations:
            break

        if attempt < DOMAIN_RETRIES:
            feedback = (
                "\n\nYour previous answer was REJECTED for the following reasons:\n"
                + "\n".join(f"  - {v}" for v in violations)
                + "\nPick coordinates that satisfy ALL constraints. "
                "Stay strictly inside the per-item surface bounds and ensure "
                "pairwise gaps. Return STRICT valid JSON.\n"
            )
        else:
            raise RuntimeError(
                "[place] coplace failed after retries; last violations: "
                + "; ".join(violations)
            )

    # Distribute items back to their nodes and call _register_results per node.
    by_node = {n["id"]: [] for n in group_nodes}
    for k, it in enumerate(items):
        info = flat[k]
        by_node[info["node"]["id"]].append({
            "x": it["x"], "y": it["y"], "z": 0.0,
            "yaw_deg": it["yaw_deg"], "layer": it["layer"],
        })

    for n in group_nodes:
        results = by_node[n["id"]]
        _register_results(n, results, world_state)
        for r in results:
            print(
                f"[place]   {n['model_name']}: pos=({r['x']:.2f}, "
                f"{r['y']:.2f}), layer={r['layer']}"
            )


def _place_all(graph, world_state, query, llm_fn):
    nodes = graph["nodes"]
    anchors = [n for n in nodes if n.get("parent_id") is None]

    total_anchor_instances = sum(an["instances"] for an in anchors)

    # Topologically place anchors in dependency order.
    ordered_anchors = _topological_sort(anchors)
    for an in ordered_anchors:
        _place_node(an, world_state, query, llm_fn)

    # For every anchor, walk its subtree level by level. Within each level,
    # group siblings under one parent and run coordinated placement when the
    # group qualifies; the rest fall through to per-node placement.
    for an in anchors:
        descendants = _collect_descendants(nodes, an["id"])
        if not descendants:
            continue
        max_level = max(d["level"] for d in descendants)
        for level in range(an["level"] + 1, max_level + 1):
            in_level = [d for d in descendants if d["level"] == level]
            placed_ids = set()

            # Group by (parent_id, parent_instance_index); eligibility is per-group.
            # This ensures nodes targeting different parent instances are placed separately.
            by_parent = {}
            for n in in_level:
                key = (n.get("parent_id"), n.get("parent_instance_index"))
                by_parent.setdefault(key, []).append(n)
            for (pid, p_inst_idx), kids in by_parent.items():
                if _qualifies_for_coplace(kids, world_state):
                    _place_co_group(kids, world_state, query, llm_fn)
                    placed_ids.update(n["id"] for n in kids)

            for node in _topological_sort(in_level):
                if node["id"] in placed_ids:
                    continue
                _place_node(node, world_state, query, llm_fn)


# ---------------------------------------------------------------------------
# Output merge
# ---------------------------------------------------------------------------

def _merge_to_output(world_state, models, graph):
    queues = {}
    for m in models:
        queues.setdefault(m.get("Model", ""), []).append(m)

    nodes_by_id = {n["id"]: n for n in (graph or {}).get("nodes", [])}

    result = []
    for i, placed in enumerate(world_state["placed"]):
        model_name = placed["Model"]
        original = queues.get(model_name, []).pop(0) if queues.get(model_name) else {}

        obj_uuid = original.get("uuid", placed["id"])
        safe_uuid = re.sub(r"[^a-zA-Z0-9_]+", "_", str(obj_uuid))

        # Resolve is_static: prefer per-node decision (after dynamic-propagation
        # in _apply_physics_to_graph) over the per-model classifier output.
        node_id = str(placed["id"]).rsplit("_inst_", 1)[0]
        node = nodes_by_id.get(node_id)
        node_flag = node.get("is_static") if isinstance(node, dict) else None
        if node_flag is None:
            node_flag = original.get("is_static")

        merged = dict(original)
        merged.update({
            "Model": model_name,
            "uuid": obj_uuid,
            "model_loc": original.get("model_loc", ""),
            "save_fn": f"{safe_uuid}_{i}",
            "_up_axis": original.get("_up_axis", "y"),
            "size": placed["size"],
            "is_static": node_flag,
            "Pose": dict(placed["Pose"]),
            "yaw_deg": placed["yaw_deg"],
        })
        result.append(merged)
        print(
            f"[scene_planner]   {model_name}: pos=({placed['Pose']['x']:.2f}, "
            f"{placed['Pose']['y']:.2f}, {placed['Pose']['z']:.2f}), yaw={placed['yaw_deg']:.0f}°"
        )
    return result


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def generate_placement_plan(models, room_half_size, query, prompt_model_fn=None, save_outputs=True):
    """Generate placement plan and return list of models with Pose / yaw_deg / size."""
    if not models:
        print("[scene_planner] WARNING: empty models list")
        return []

    model_id = DEFAULT_MODEL

    def llm_fn(prompt, query_text, **kwargs):
        # ADDED (task: temperature gradient): forward temperature/seed (and
        # any other llm options) to the underlying request. The external
        # `prompt_model_fn` override accepts kwargs too; if it doesn't
        # recognise them, it can ignore via **kwargs in its own signature.
        if prompt_model_fn is not None:
            try:
                raw = prompt_model_fn(prompt, query_text, model_id, **kwargs)
            except TypeError:
                # Older callbacks without **kwargs — fall back gracefully.
                raw = prompt_model_fn(prompt, query_text, model_id)
        else:
            raw = _llm_request(system=prompt, user=query_text, model=model_id, **kwargs)
        if isinstance(raw, str):
            try:
                return json.loads(raw)
            except json.JSONDecodeError:
                return raw
        return raw

    print(f"[scene_planner] Stage 4: {len(models)} objects, room={room_half_size*2:.1f}m")

    # Phase 1: scene graph.
    graph = _build_scene_graph(models, room_half_size, query, llm_fn)
    if save_outputs:
        _save_stage_output({
            "stage": "1_scene_graph",
            "query": query,
            "room_half_size": room_half_size,
            "graph": graph,
        }, "stage1_scene_graph")

    for node in graph["nodes"]:
        node["size_hint"] = _model_size(models, node["model_name"])

    world_state = make_world_state(room_half_size)
    type_counts = {}
    for m in models:
        type_counts[m["Model"]] = type_counts.get(m["Model"], 0) + 1
    world_state["remaining"] = [
        {"model_name": name, "count": cnt, "size": _model_size(models, name)}
        for name, cnt in type_counts.items()
    ]

    _place_all(graph, world_state, query, llm_fn)
    print(f"[scene_planner] Placed {len(world_state['placed'])} objects")

    if save_outputs:
        _save_stage_output({
            "stage": "2_hierarchical_placement",
            "world_state": world_state,
            "graph": graph,
        }, "stage2_world_state")

    result = _merge_to_output(world_state, models, graph)
    print(f"[scene_planner] Stage 4 complete: {len(result)} models with placement")

    if save_outputs:
        _save_stage_output({
            "stage": "3_final_layout",
            "models": result,
            "room_half_size": room_half_size,
            "query": query,
        }, "stage3_final_layout")

    return result
