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
#   4. PLACEMENT — anchors first (topological order on anchor relationships),
#                  then for each anchor walk its subtree level by level. Every
#                  node — anchor or dependent — is placed by the same LLM-only
#                  function: PLACE_NODE for instances=1, PLACE_NODE_BATCH for
#                  instances>1 (sub-batches of <= 6).
#
# No heuristic / deterministic fallbacks. Any unrecoverable LLM failure raises.
# Sanity is limited to: keep (x, y) in the room, keep z >= 0.

import json
import os
import re
from datetime import datetime

from project.llm_request import request as _llm_request, DEFAULT_MODEL
from project.scene_prompts import (
    ANCHOR_RELATIONS_PROMPT,
    CLASSIFY_HIERARCHY_PROMPT,
    GROUP_RELATIONS_PROMPT,
    PLACE_NODE_BATCH_PROMPT,
    PLACE_NODE_PROMPT,
)


# Relationship vocabulary used by the planner.
REL_AROUND = "around"
REL_ON_SURFACE = "on_surface"
REL_STACKED_ON = "stacked_on"
REL_BESIDE = "beside"
REL_IN_FRONT = "in_front"
REL_BEHIND = "behind"
REL_FACING = "facing"
REL_IN_ROOM = "in_room"

ROOM_CENTER = "room_center"

BATCH_SIZE = 6
LLM_RETRIES = 3


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
    """Return (width_x, depth_y, height_z) for a size list."""
    sw = float(size[0]) if len(size) > 0 else 1.0
    sh = float(size[1]) if len(size) > 1 else sw
    sd = float(size[2]) if len(size) > 2 else sw
    return sw, sd, sh


def _size_str(size):
    sw, sd, sh = _size_wh(size)
    return f"{sw:.2f}×{sd:.2f}×{sh:.2f}m"


# ---------------------------------------------------------------------------
# LLM JSON helper
# ---------------------------------------------------------------------------

def _llm_json(llm_fn, prompt, query, expect, retries=LLM_RETRIES):
    """Call llm_fn(prompt, query) until result satisfies expect(obj) -> (ok, err)."""
    last_err = None
    cur_prompt = prompt
    for attempt in range(1, retries + 1):
        try:
            raw = llm_fn(cur_prompt, query)
            if isinstance(raw, str):
                raw = json.loads(raw)
            ok, err = expect(raw)
            if ok:
                return raw
            last_err = err
        except Exception as exc:
            last_err = f"{type(exc).__name__}: {exc}"
        cur_prompt = (
            f"{prompt}\n\nPrevious attempt #{attempt} failed: {last_err}\n"
            "Return STRICT valid JSON matching the requested schema."
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


def _ws_summary(world_state):
    placed = world_state["placed"]
    if not placed:
        return "  (empty)"
    lines = []
    for p in placed:
        sw, sd, sh = _size_wh(p["size"])
        cx, cy, cz = p["Pose"]["x"], p["Pose"]["y"], p["Pose"]["z"]
        lines.append(
            f"  - {p['Model']} [{p['id']}]: center=({cx:.3f}, {cy:.3f}, {cz:.3f}), "
            f"yaw={p['yaw_deg']:.0f}°, "
            f"X=[{cx-sw/2:.3f}..{cx+sw/2:.3f}], Y=[{cy-sd/2:.3f}..{cy+sd/2:.3f}], "
            f"Z=[{cz-sh/2:.3f}..{cz+sh/2:.3f}]"
        )
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
        return _validate_basic_graph(nodes, available, total)

    raw = _llm_json(llm_fn, prompt, query, expect)
    nodes = []
    for n in raw["nodes"]:
        nodes.append({
            "id": str(n["id"]),
            "model_name": str(n["model_name"]),
            "instances": int(n["instances"]),
            "parent_id": n.get("parent_id"),
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
        desc_lines.append(
            f"  - id={d['id']}, model_name={d['model_name']}, instances={d['instances']}, "
            f"parent_id={d.get('parent_id')}, size={_size_str(_model_size(models, d['model_name']))}"
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
        d["relationship"] = {
            "type": str(rel.get("type", REL_ON_SURFACE)),
            "reference": rel.get("reference"),
            "side": rel.get("side"),
            "distance": float(rel["distance"]) if rel.get("distance") is not None else None,
            "facing": rel.get("facing"),
            "offset": rel.get("offset"),
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

    metadata = {
        "total_objects": sum(n["instances"] for n in nodes),
        "max_level": max(n["level"] for n in nodes),
    }
    return {"nodes": nodes, "metadata": metadata}


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


def _parent_block(node, world_state):
    pid = node.get("parent_id")
    if not pid:
        return ""
    parents = _ws_find_all_by_node(world_state, pid)
    if not parents:
        return ""
    instances = int(node.get("instances", 1))

    if len(parents) == 1:
        p = parents[0]
        sw, sd, sh = _size_wh(p["size"])
        pose = p["Pose"]
        parent_top_z = pose["z"] + sh / 2.0
        return (
            f"Parent surface '{pid}' ({p['Model']}): "
            f"center=({pose['x']:.3f}, {pose['y']:.3f}, {pose['z']:.3f}), "
            f"size={_size_str(p['size'])}, top_z={parent_top_z:.3f}m. "
            f"For 'on_surface'/'stacked_on' stay inside x∈[{pose['x']-sw/2:.3f}..{pose['x']+sw/2:.3f}], "
            f"y∈[{pose['y']-sd/2:.3f}..{pose['y']+sd/2:.3f}].\n"
        )

    # Multi-parent: list every parent instance.
    lines = [f"Parent group '{pid}' has {len(parents)} instances:"]
    for i, p in enumerate(parents, start=1):
        sw, sd, sh = _size_wh(p["size"])
        pose = p["Pose"]
        parent_top_z = pose["z"] + sh / 2.0
        lines.append(
            f"  - parent_instance_{i} ({p['id']}): "
            f"center=({pose['x']:.3f}, {pose['y']:.3f}, {pose['z']:.3f}), "
            f"size={_size_str(p['size'])}, top_z={parent_top_z:.3f}m"
        )
    if instances == len(parents):
        lines.append(
            f"PARENT MAPPING: child instance i sits ON parent_instance_i "
            f"({instances} ↔ {instances}). Each child uses ITS OWN parent's footprint."
        )
    return "\n".join(lines) + "\n"


def _siblings_block(placed_in_group):
    if not placed_in_group:
        return ""
    lines = [f"Already placed in this group ({len(placed_in_group)}):"]
    for r in placed_in_group:
        lines.append(
            f"  - instance {r.get('instance', '?')}: pos=({r['x']:.3f}, {r['y']:.3f}, {r['z']:.3f}), "
            f"yaw={r.get('yaw_deg', 0):.0f}°"
        )
    return "\n".join(lines) + "\n"


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

    sw, sd, _ = _size_wh(node["size_hint"])
    self_hx, self_hy = sw / 2.0, sd / 2.0

    parents = _ws_find_all_by_node(world_state, parent_id) if parent_id else []
    parent_obj = parents[0] if parents else None

    out = []

    # ----- on_surface / stacked_on: numeric center bounds inside parent.
    if rtype in (REL_ON_SURFACE, REL_STACKED_ON) and parents:
        margin = 0.005
        if len(parents) == 1:
            p = parents[0]
            psw, psd, _ = _size_wh(p["size"])
            phx, phy = psw / 2.0, psd / 2.0
            px, py = p["Pose"]["x"], p["Pose"]["y"]
            lim_x = max(0.0, phx - self_hx - margin)
            lim_y = max(0.0, phy - self_hy - margin)
            out.append(
                f"NUMERIC SURFACE BOUNDS — center MUST stay inside:\n"
                f"  x ∈ [{px - lim_x:.3f}, {px + lim_x:.3f}], "
                f"y ∈ [{py - lim_y:.3f}, {py + lim_y:.3f}].\n"
                f"  (parent half = ({phx:.3f}, {phy:.3f}); self half = ({self_hx:.3f}, {self_hy:.3f}))"
            )
        else:
            lines = ["NUMERIC SURFACE BOUNDS per parent_instance — each child must stay inside ITS parent:"]
            for i, p in enumerate(parents, start=1):
                psw, psd, _ = _size_wh(p["size"])
                phx, phy = psw / 2.0, psd / 2.0
                px, py = p["Pose"]["x"], p["Pose"]["y"]
                lim_x = max(0.0, phx - self_hx - margin)
                lim_y = max(0.0, phy - self_hy - margin)
                lines.append(
                    f"  parent_instance_{i} ({p['id']}): "
                    f"x ∈ [{px - lim_x:.3f}, {px + lim_x:.3f}], "
                    f"y ∈ [{py - lim_y:.3f}, {py + lim_y:.3f}]"
                )
            if instances == len(parents):
                lines.append(
                    f"PARENT MAPPING: instance i must lie inside parent_instance_i bounds "
                    f"({instances} ↔ {instances})."
                )
            out.append("\n".join(lines))

    # ----- around: ring radius around parent + cardinal suggestions + yaw map.
    #       (uses parents[0] — "around" is a multi-instance ring around ONE parent
    #       by definition; multi-parent "around" is not a meaningful relationship.)
    if rtype == REL_AROUND and parent_obj is not None:
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
    #       and parent_id has multiple instances (e.g. one banana per plate).
    if instances > 1 and parents and len(parents) == instances and len(parents) > 1:
        out.append(
            f"PARENT MAPPING — counts match ({instances} ↔ {instances}). "
            f"Place instance i ON parent_instance_i. Each instance has its OWN parent."
        )

    if not out:
        return ""
    return "\n".join(out) + "\n\n"


def _place_node(node, world_state, query, llm_fn):
    """Unified placement for any node (anchor or dependent)."""
    sw, sd, sh = _size_wh(node["size_hint"])
    room_half = world_state["room_half_size"]
    instances = int(node["instances"])

    base_kwargs = dict(
        model_name=node["model_name"],
        node_id=node["id"],
        size_w=sw, size_d=sd, size_h=sh,
        size_w_half=sw / 2.0, size_d_half=sd / 2.0,
        intent=_intent_text(node),
        reference_block=_reference_block(node, world_state),
        parent_block=_parent_block(node, world_state),
        numeric_hints=_numeric_hints_block(node, world_state),
        world_state=_ws_summary(world_state),
        remaining=_ws_remaining_summary(world_state),
        room_w=f"{room_half*2:.1f}", room_l=f"{room_half*2:.1f}",
        neg_half=-room_half, pos_half=room_half,
    )

    if instances == 1:
        prompt = PLACE_NODE_PROMPT.format(siblings_block="", **base_kwargs)

        def expect(raw):
            if not isinstance(raw, dict):
                return False, "expected JSON object"
            for k in ("x", "y"):
                if k not in raw:
                    return False, f"missing '{k}'"
            return True, ""

        raw = _llm_json(llm_fn, prompt, query, expect)
        x, y = _sanity_xy(raw["x"], raw["y"], room_half)
        z = max(0.0, float(raw.get("z", 0.0)))
        result = {"x": x, "y": y, "z": z, "yaw_deg": float(raw.get("yaw_deg", 0.0))}
        _register_results(node, [result], world_state)
        return

    # Multi-instance: split into batches of <= BATCH_SIZE.
    placed_in_group = []
    all_results = []
    for batch_start in range(0, instances, BATCH_SIZE):
        batch_end = min(batch_start + BATCH_SIZE, instances)
        batch_count = batch_end - batch_start
        from_idx, to_idx = batch_start + 1, batch_end

        prompt = PLACE_NODE_BATCH_PROMPT.format(
            batch_count=batch_count,
            from_idx=from_idx, to_idx=to_idx, total=instances,
            siblings_block=_siblings_block(placed_in_group),
            **base_kwargs,
        )

        def expect(raw, _bc=batch_count):
            if isinstance(raw, dict) and "positions" in raw:
                raw = raw["positions"]
            if not isinstance(raw, list) or len(raw) != _bc:
                return False, (
                    f"expected list of {_bc}, got "
                    f"{type(raw).__name__} len={len(raw) if isinstance(raw, list) else '?'}"
                )
            for item in raw:
                if not isinstance(item, dict) or "x" not in item or "y" not in item:
                    return False, "each item must have 'x' and 'y'"
            return True, ""

        raw = _llm_json(llm_fn, prompt, query, expect)
        if isinstance(raw, dict) and "positions" in raw:
            raw = raw["positions"]

        for i, item in enumerate(raw):
            x, y = _sanity_xy(item["x"], item["y"], room_half)
            z = max(0.0, float(item.get("z", 0.0)))
            r = {
                "instance": from_idx + i,
                "x": x, "y": y, "z": z,
                "yaw_deg": float(item.get("yaw_deg", 0.0)),
            }
            placed_in_group.append(r)
            all_results.append(r)

    _register_results(node, all_results, world_state)


def _register_results(node, results, world_state):
    sz = node["size_hint"]
    _, _, sh = _size_wh(sz)
    obj_h = sh

    rel = node.get("relationship") or {}
    rel_type = rel.get("type") if isinstance(rel, dict) else None

    parent_id = node.get("parent_id")
    parents = _ws_find_all_by_node(world_state, parent_id) if parent_id else []
    # When child instances == parent instances, each child sits on its own parent.
    use_per_instance_parent = len(parents) > 1 and len(parents) == len(results)

    def z_for_instance(idx):
        # Vertical placement:
        #   - "around"   → on the floor (chairs around a table do not climb the table);
        #   - has parent → sit on top of parent (per-instance when counts match);
        #   - else       → on the floor (anchors).
        if rel_type == REL_AROUND:
            return obj_h / 2.0
        if use_per_instance_parent:
            p = parents[idx]
        elif parents:
            p = parents[0]
        else:
            return obj_h / 2.0
        _, _, ph = _size_wh(p["size"])
        return p["Pose"]["z"] + ph / 2.0 + obj_h / 2.0

    for i, r in enumerate(results):
        instance_id = f"{node['id']}_inst_{i}"
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


def _place_all(graph, world_state, query, llm_fn):
    nodes = graph["nodes"]
    anchors = [n for n in nodes if n.get("parent_id") is None]

    total_anchor_instances = sum(an["instances"] for an in anchors)

    # Special case: a single anchor in the entire scene goes to (0, 0).
    if total_anchor_instances == 1 and len(anchors) == 1:
        an = anchors[0]
        _, _, sh = _size_wh(an["size_hint"])
        _register_results(an, [{"x": 0.0, "y": 0.0, "z": sh / 2.0, "yaw_deg": 0.0}], world_state)
        print(f"[place] {an['id']} ({an['model_name']}): (0.00, 0.00) [single anchor → center]")
    else:
        # Topologically place anchors in dependency order.
        ordered_anchors = _topological_sort(anchors)
        for an in ordered_anchors:
            _place_node(an, world_state, query, llm_fn)

    # For every anchor, walk its subtree level by level.
    for an in anchors:
        descendants = _collect_descendants(nodes, an["id"])
        if not descendants:
            continue
        max_level = max(d["level"] for d in descendants)
        for level in range(an["level"] + 1, max_level + 1):
            in_level = [d for d in descendants if d["level"] == level]
            for node in _topological_sort(in_level):
                _place_node(node, world_state, query, llm_fn)


# ---------------------------------------------------------------------------
# Output merge
# ---------------------------------------------------------------------------

def _merge_to_output(world_state, models, graph):
    queues = {}
    for m in models:
        queues.setdefault(m.get("Model", ""), []).append(m)

    result = []
    for i, placed in enumerate(world_state["placed"]):
        model_name = placed["Model"]
        original = queues.get(model_name, []).pop(0) if queues.get(model_name) else {}

        obj_uuid = original.get("uuid", placed["id"])
        safe_uuid = re.sub(r"[^a-zA-Z0-9_]+", "_", str(obj_uuid))

        merged = dict(original)
        merged.update({
            "Model": model_name,
            "uuid": obj_uuid,
            "model_loc": original.get("model_loc", ""),
            "save_fn": f"{safe_uuid}_{i}",
            "_up_axis": original.get("_up_axis", "y"),
            "size": placed["size"],
            "is_static": original.get("is_static", True),
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

def generate_placement_plan(models, room_half_size, room_type, query, prompt_model_fn=None, save_outputs=True):
    """Generate placement plan and return list of models with Pose / yaw_deg / size."""
    if not models:
        print("[scene_planner] WARNING: empty models list")
        return []

    model_id = DEFAULT_MODEL

    def llm_fn(prompt, query_text):
        if prompt_model_fn is not None:
            raw = prompt_model_fn(prompt, query_text, model_id)
        else:
            raw = _llm_request(system=prompt, user=query_text, model=model_id)
        if isinstance(raw, str):
            try:
                return json.loads(raw)
            except json.JSONDecodeError:
                return raw
        return raw

    print(f"[scene_planner] Stage 4: {len(models)} objects, room={room_half_size*2:.1f}m, type={room_type}")

    # Phase 1: scene graph.
    graph = _build_scene_graph(models, room_half_size, query, llm_fn)
    if save_outputs:
        _save_stage_output({
            "stage": "1_scene_graph",
            "query": query,
            "room_half_size": room_half_size,
            "room_type": room_type,
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
            "room_type": room_type,
        }, "stage3_final_layout")

    return result
