# Stage 4 - generates placement plan via LLM + enforcement pipeline.
# Reads models with size/scale, returns models with Pose, yaw_deg, is_static.

import json
import math
import os
import re
from dataclasses import dataclass
from datetime import datetime

from project.llm_request import request as _llm_request, DEFAULT_MODEL
from project.scene_prompts import (
    _ANCHOR_BATCH_PLACEMENT_PROMPT,
    _ANCHOR_RELATIONS_PROMPT,
    _BATCH_PLACEMENT_PROMPT,
    _GRAPH_HIERARCHY_PROMPT,
    _SINGLE_PLACEMENT_PROMPT,
    _VALIDATE_PLACEMENT_PROMPT,
)


# ---------------------------------------------------------------------------
# Output saving utilities
# ---------------------------------------------------------------------------

def _save_stage_output(data, stage_name, output_dir="output"):
    """Сохранить результаты стадии в JSON файл."""
    try:
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{stage_name}_{timestamp}.json"
        filepath = os.path.join(output_dir, filename)
        
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        print(f"[scene_planner] Saved {stage_name} → {filepath}")
        return filepath
    except Exception as e:
        print(f"[scene_planner] WARNING: Failed to save {stage_name}: {e}")
        return None


# ---------------------------------------------------------------------------
# OBB Geometry (from creator.placement.geometry)
# ---------------------------------------------------------------------------

@dataclass
class Vec2:
    x: float = 0.0
    y: float = 0.0

    def __add__(self, other):
        return Vec2(self.x + other.x, self.y + other.y)

    def __sub__(self, other):
        return Vec2(self.x - other.x, self.y - other.y)

    def __mul__(self, s):
        return Vec2(self.x * s, self.y * s)

    def dot(self, other):
        return self.x * other.x + self.y * other.y

    def length(self):
        return math.sqrt(self.x * self.x + self.y * self.y)

    def normalized(self):
        ln = self.length()
        if ln < 1e-12:
            return Vec2(1.0, 0.0)
        return Vec2(self.x / ln, self.y / ln)


@dataclass
class OBB:
    """Oriented Bounding Box in 2D (XY plane)."""
    center: Vec2
    half_extents: Vec2
    yaw_rad: float = 0.0

    def corners(self):
        c = math.cos(self.yaw_rad)
        s = math.sin(self.yaw_rad)
        ax = Vec2(c, s)
        ay = Vec2(-s, c)
        hx = self.half_extents.x
        hy = self.half_extents.y
        return [
            self.center + ax * hx + ay * hy,
            self.center - ax * hx + ay * hy,
            self.center - ax * hx - ay * hy,
            self.center + ax * hx - ay * hy,
        ]

    def axes(self):
        c = math.cos(self.yaw_rad)
        s = math.sin(self.yaw_rad)
        return [Vec2(c, s), Vec2(-s, c)]

    def aabb(self):
        pts = self.corners()
        xs = [p.x for p in pts]
        ys = [p.y for p in pts]
        return AABB(min(xs), max(xs), min(ys), max(ys))


@dataclass
class AABB:
    min_x: float = 0.0
    max_x: float = 0.0
    min_y: float = 0.0
    max_y: float = 0.0

    def overlaps(self, other, margin=0.0):
        sep_x = (self.max_x + margin < other.min_x) or (other.max_x + margin < self.min_x)
        sep_y = (self.max_y + margin < other.min_y) or (other.max_y + margin < self.min_y)
        return not (sep_x or sep_y)


def _project_corners(corners, axis):
    dots = [c.dot(axis) for c in corners]
    return min(dots), max(dots)


def obb_overlap(a, b, margin=0.0):
    """SAT overlap test for two OBBs."""
    corners_a = a.corners()
    corners_b = b.corners()
    for ax in a.axes() + b.axes():
        min_a, max_a = _project_corners(corners_a, ax)
        min_b, max_b = _project_corners(corners_b, ax)
        if max_a + margin < min_b or max_b + margin < min_a:
            return False
    return True


def obb_overlap_depth(a, b):
    """Return penetration depth (>0 means overlap)."""
    corners_a = a.corners()
    corners_b = b.corners()
    min_overlap = float("inf")
    for ax in a.axes() + b.axes():
        min_a, max_a = _project_corners(corners_a, ax)
        min_b, max_b = _project_corners(corners_b, ax)
        overlap = min(max_a, max_b) - max(min_a, min_b)
        if overlap < 0:
            return overlap
        min_overlap = min(min_overlap, overlap)
    return min_overlap


def obb_separation_vector(a, b):
    """Return (direction, depth) for minimum translation to separate a from b."""
    corners_a = a.corners()
    corners_b = b.corners()
    best_depth = float("inf")
    best_axis = Vec2(1.0, 0.0)
    for ax in a.axes() + b.axes():
        min_a, max_a = _project_corners(corners_a, ax)
        min_b, max_b = _project_corners(corners_b, ax)
        overlap = min(max_a, max_b) - max(min_a, min_b)
        if overlap < 0:
            return ax, overlap
        if overlap < best_depth:
            best_depth = overlap
            ca = a.center.dot(ax)
            cb = b.center.dot(ax)
            best_axis = ax if ca >= cb else ax * (-1.0)
    return best_axis.normalized(), best_depth


def model_to_obb(model, inflation=0.0, pos_override=None, yaw_override=None):
    """Convert a placed model dict to an OBB."""
    size = model.get("size", [1.0, 1.0, 1.0])
    raw_sx = float(size[0]) if len(size) > 0 else 1.0
    raw_sy = float(size[2]) if len(size) > 2 else 1.0
    sx = max(0.02, raw_sx)
    sy = max(0.02, raw_sy)
    pose = model.get("Pose") or {}
    if pos_override is not None:
        cx, cy = pos_override
    else:
        cx = float(pose.get("x", 0.0))
        cy = float(pose.get("y", 0.0))
    if yaw_override is not None:
        yaw_rad = math.radians(yaw_override)
    else:
        yaw_rad = math.radians(float(model.get("yaw_deg", 0.0)))
    return OBB(
        center=Vec2(cx, cy),
        half_extents=Vec2(sx / 2.0 + inflation, sy / 2.0 + inflation),
        yaw_rad=yaw_rad,
    )


def model_half_height(model):
    """Return half-height of model in scene (Z direction)."""
    size = model.get("size", [1.0, 1.0, 1.0])
    if len(size) < 3:
        return 0.5
    sx, sy, sz = float(size[0]), float(size[1]), float(size[2])
    height = sy if sy >= 0.05 * max(sx, sy, sz) else max(sx, sy, sz)
    return max(0.01, height / 2.0)


def _z_intervals_overlap(a, b):
    """Check if two models overlap in Z (same height level)."""
    size_a = a.get("size", [1.0, 1.0, 1.0])
    size_b = b.get("size", [1.0, 1.0, 1.0])
    pose_a = a.get("Pose") or {}
    pose_b = b.get("Pose") or {}
    za = float(pose_a.get("z", 0.0))
    zb = float(pose_b.get("z", 0.0))
    ha = max(0.01, float(size_a[1]) if len(size_a) > 1 else 1.0) / 2.0
    hb = max(0.01, float(size_b[1]) if len(size_b) > 1 else 1.0) / 2.0
    
    # УЛУЧШЕНИЕ: Если объекты на явно разных уровнях (разница > 15см), не считать коллизией
    # Это предотвращает ложные коллизии между объектами на столе и на полу
    z_diff = abs(za - zb)
    if z_diff > 0.15:  # 15см разница = разные уровни
        return False
    
    return not ((za + ha < zb - hb) or (zb + hb < za - ha))


def gradient_resolve_overlaps(models, room_half_size=5.0, iterations=200, step_size=0.05, collision_margin=0.01, semantic_plan=None, scene_graph=None):
    """Move objects apart using gradient-based overlap resolution.
    
    Args:
        scene_graph: Optional scene graph with hierarchy info. If provided,
                     objects on surfaces (level > 0) won't be moved independently.
    """
    resolved = [dict(m) for m in models]
    n = len(resolved)
    overlap_history = []
    stagnation_threshold = 20
    
    # Построить карту: model_name → level из графа
    # Защищаем только объекты НА ПОВЕРХНОСТИ (on_surface), не "around"
    surface_objects = set()
    if scene_graph:
        for node in scene_graph.get("nodes", []):
            level = node.get("level", 0)
            rel = node.get("relationship") or {}
            model_name = node.get("model_name", "")
            if level > 0 and rel.get("type") in ("on_surface", "stacked_on"):
                surface_objects.add(model_name)
        if surface_objects:
            print(f"[gradient_resolve] Surface objects (won't move in XY): {surface_objects}")

    for iteration_idx in range(iterations):
        grads = [Vec2(0.0, 0.0) for _ in range(n)]
        any_overlap = False
        overlap_count = 0

        for i in range(n):
            for j in range(i + 1, n):
                if not _z_intervals_overlap(resolved[i], resolved[j]):
                    continue
                pose_i = resolved[i].get("Pose") or {"z": 0.0}
                pose_j = resolved[j].get("Pose") or {"z": 0.0}
                z_i = float(pose_i.get("z", 0.0))
                z_j = float(pose_j.get("z", 0.0))
                if z_i > 0.5 and z_j > 0.5:
                    continue
                obb_i = model_to_obb(resolved[i], inflation=collision_margin)
                obb_j = model_to_obb(resolved[j], inflation=collision_margin)
                sep_dir, depth = obb_separation_vector(obb_i, obb_j)
                if depth <= 0:
                    continue
                any_overlap = True
                overlap_count += 1
                dist = (obb_i.center - obb_j.center).length()
                if depth > 0.1:
                    step_multiplier = 6.0
                elif depth < 0.02:
                    step_multiplier = 2.0
                else:
                    step_multiplier = 4.0
                scale = step_multiplier * depth / max(0.01, dist)
                grad = sep_dir * (scale * 0.625)
                grads[i] = grads[i] + grad
                grads[j] = grads[j] - grad

        overlap_history.append(overlap_count)
        if len(overlap_history) >= stagnation_threshold:
            recent_counts = overlap_history[-stagnation_threshold:]
            min_recent = min(recent_counts)
            if overlap_count > 0 and overlap_count >= min_recent:
                if all(c >= min_recent for c in recent_counts):
                    print(f"[gradient_resolve] Convergence failure at iteration {iteration_idx}: stuck at {overlap_count}")
                    return resolved, False

        all_in_bounds = True
        for i in range(n):
            obb_i = model_to_obb(resolved[i])
            aabb_i = obb_i.aabb()
            lo = -room_half_size
            hi = room_half_size
            if aabb_i.min_x < lo:
                grads[i] = grads[i] + Vec2(lo - aabb_i.min_x, 0.0)
                all_in_bounds = False
            if aabb_i.max_x > hi:
                grads[i] = grads[i] + Vec2(hi - aabb_i.max_x, 0.0)
                all_in_bounds = False
            if aabb_i.min_y < lo:
                grads[i] = grads[i] + Vec2(0.0, lo - aabb_i.min_y)
                all_in_bounds = False
            if aabb_i.max_y > hi:
                grads[i] = grads[i] + Vec2(0.0, hi - aabb_i.max_y)
                all_in_bounds = False

        if not any_overlap and all_in_bounds:
            print(f"[gradient_resolve] Converged successfully at iteration {iteration_idx}")
            break

        for i in range(n):
            g = grads[i]
            if g.length() < 1e-8:
                continue
            pose = dict(resolved[i].get("Pose") or {"x": 0.0, "y": 0.0, "z": 0.0})
            pose["x"] = float(pose.get("x", 0.0)) + step_size * g.x
            pose["y"] = float(pose.get("y", 0.0)) + step_size * g.y
            resolved[i]["Pose"] = pose

    for i in range(n):
        obb_i = model_to_obb(resolved[i])
        pose = dict(resolved[i].get("Pose") or {"x": 0.0, "y": 0.0, "z": 0.0})
        cx = float(pose.get("x", 0.0))
        cy = float(pose.get("y", 0.0))
        aabb_i = obb_i.aabb()
        lo = -room_half_size
        hi = room_half_size
        if aabb_i.min_x < lo:
            cx += lo - aabb_i.min_x
        elif aabb_i.max_x > hi:
            cx += hi - aabb_i.max_x
        if aabb_i.min_y < lo:
            cy += lo - aabb_i.min_y
        elif aabb_i.max_y > hi:
            cy += hi - aabb_i.max_y
        pose["x"] = cx
        pose["y"] = cy
        hh = model_half_height(resolved[i])
        cz = float(pose.get("z", hh))
        floor_z = hh
        if cz < floor_z - 1e-3:
            pose["z"] = floor_z
        elif abs(cz - floor_z) < max(0.05, 0.1 * hh):
            pose["z"] = floor_z
        resolved[i]["Pose"] = pose

    return resolved, True


# ---------------------------------------------------------------------------
# Relationship type constants
# ---------------------------------------------------------------------------

REL_AROUND = "around"
REL_ON_SURFACE = "on_surface"
REL_BESIDE = "beside"
REL_FACING = "facing"
REL_OPPOSITE = "opposite"
REL_IN_FRONT = "in_front"
REL_BEHIND = "behind"
REL_ABSOLUTE = "absolute"


# ---------------------------------------------------------------------------
# Scene graph data structures
# ---------------------------------------------------------------------------

def make_scene_node(id, model_name, level, instances, parent_id, relationship, anchor_hint, place_order=1, depends_on=None):
    """Create a scene graph node dict."""
    return {
        "id": id,
        "model_name": model_name,
        "level": level,
        "instances": instances,
        "parent_id": parent_id,
        "relationship": relationship,
        "anchor_hint": anchor_hint,
        "place_order": place_order,
        "depends_on": depends_on,
    }


def make_scene_graph(nodes, anchor_relations=None):
    """Create a scene graph dict from a list of nodes."""
    max_level = max((n.get("level", 0) for n in nodes), default=0)
    total = sum(n.get("instances", 1) for n in nodes)
    return {
        "nodes": nodes,
        "anchor_relations": anchor_relations or [],
        "metadata": {
            "total_objects": total,
            "max_level": max_level,
        }
    }


def make_world_state(room_half_size):
    """Create a fresh world state dict."""
    return {
        "placed": [],
        "remaining": [],
        "room_half_size": room_half_size,
    }


def _collect_groups(nodes):
    """Группирует узлы по parent_id."""
    groups = {}
    for node in nodes:
        pid = node["parent_id"]
        groups.setdefault(pid, []).append(node)
    return groups


# ---------------------------------------------------------------------------
# WorldState utilities
# ---------------------------------------------------------------------------

def _ws_add(world_state, placed_id, model_name, size_list, pose, yaw_deg):
    """Добавить размещённый объект в world_state."""
    world_state["placed"].append({
        "id": placed_id,
        "Model": model_name,
        "size": size_list,
        "Pose": pose,
        "yaw_deg": yaw_deg,
    })


def _ws_summary(world_state, max_items=None):
    """Описание уже размещённых объектов с AABB и Z-границами для промпта."""
    placed = world_state["placed"]
    if not placed:
        return "  (empty)"
    items = placed if max_items is None else placed[-max_items:]
    lines = []
    for p in items:
        pose = p["Pose"]
        sz = p["size"]
        sx = float(sz[0]) if len(sz) > 0 else 1.0
        sz_world_y = float(sz[2]) if len(sz) > 2 else sx
        sz_world_z = float(sz[1]) if len(sz) > 1 else sx
        cx, cy, cz = float(pose["x"]), float(pose["y"]), float(pose["z"])
        x_min, x_max = cx - sx / 2.0, cx + sx / 2.0
        y_min, y_max = cy - sz_world_y / 2.0, cy + sz_world_y / 2.0
        z_min, z_max = cz - sz_world_z / 2.0, cz + sz_world_z / 2.0
        lines.append(
            f"  - {p['Model']} [{p['id']}]: "
            f"center=({cx:.3f}, {cy:.3f}, {cz:.3f}), "
            f"yaw={p['yaw_deg']:.0f}°, "
            f"X=[{x_min:.3f}..{x_max:.3f}], Y=[{y_min:.3f}..{y_max:.3f}], "
            f"Z=[{z_min:.3f}..{z_max:.3f}]"
        )
    if max_items is not None and len(placed) > max_items:
        lines.append(f"  ... and {len(placed) - max_items} more objects")
    return "\n".join(lines)


def _ws_remaining_summary(world_state):
    """Краткое описание оставшихся объектов для промпта."""
    remaining = world_state.get("remaining", [])
    if not remaining:
        return "  (нет)"
    return "\n".join(
        f"  - {r['count']}× {r['model_name']} ({r['size'][0]:.2f}×{r['size'][2]:.2f}×{r['size'][1]:.2f}м)"
        for r in remaining
    )


def _ws_find(world_state, object_id):
    """Найти объект в world_state по id."""
    for p in world_state["placed"]:
        if p["id"] == object_id:
            return p
    return None


def _ws_find_all_by_node(world_state, node_id_prefix):
    """Найти все объекты размещённые для данного узла (по префиксу id)."""
    return [p for p in world_state["placed"] if p["id"].startswith(node_id_prefix + "_inst_")]


# ---------------------------------------------------------------------------
# Scene graph validator
# ---------------------------------------------------------------------------

def _compute_levels(graph):
    """Set node['level'] from parent_id chain depth (0 for anchors)."""
    nodes = graph.get("nodes", [])
    by_id = {n["id"]: n for n in nodes}
    cache = {}

    def depth(node_id, seen):
        if node_id in cache:
            return cache[node_id]
        if node_id in seen:
            return 0  # cycle guard; validator will flag separately
        node = by_id.get(node_id)
        if node is None:
            return 0
        parent = node.get("parent_id")
        if parent is None or parent not in by_id:
            d = 0
        else:
            d = depth(parent, seen | {node_id}) + 1
        cache[node_id] = d
        return d

    for n in nodes:
        n["level"] = depth(n["id"], set())
    return graph


def _validate_graph(graph, available_model_names):
    """Validate scene graph structure and references."""
    errors = []
    nodes = graph.get("nodes", [])
    ids = {n["id"] for n in nodes}
    available = set(available_model_names)

    for i, node in enumerate(nodes):
        parent_id = node.get("parent_id")
        if parent_id is not None and parent_id not in ids:
            errors.append(f"Node '{node['id']}': parent_id '{parent_id}' not found")
        if node["model_name"] not in available:
            errors.append(f"Node '{node['id']}': unknown model '{node['model_name']}'")
        if node.get("instances", 0) < 1:
            errors.append(f"Node '{node['id']}': instances must be >= 1")
        if node.get("depends_on") and node["depends_on"] not in ids:
            errors.append(f"Node '{node['id']}': depends_on '{node['depends_on']}' not found")
        rel = node.get("relationship") or {}
        ref = rel.get("reference") if isinstance(rel, dict) else None
        if ref and ref not in ids:
            errors.append(f"Node '{node['id']}': relationship.reference '{ref}' not found")
        if ref and ref == node["id"]:
            errors.append(f"Node '{node['id']}': relationship.reference cannot equal node id")

    # Проверка циклов (DFS)
    def has_cycle(node_id, visiting, visited):
        visiting.add(node_id)
        node = next((n for n in nodes if n["id"] == node_id), None)
        if node and node["parent_id"] and node["parent_id"] in ids:
            pid = node["parent_id"]
            if pid in visiting:
                return True
            if pid not in visited:
                if has_cycle(pid, visiting, visited):
                    return True
        visiting.remove(node_id)
        visited.add(node_id)
        return False

    visited, visiting = set(), set()
    for node in nodes:
        if node["id"] not in visited:
            if has_cycle(node["id"], visiting, visited):
                errors.append("Cycle detected in scene graph")
                break

    return len(errors) == 0, errors


# ---------------------------------------------------------------------------
# Heuristic graph fallback (no LLM)
# ---------------------------------------------------------------------------

_ANCHOR_KEYWORDS = ("table", "sofa", "bed", "desk", "shelf", "cabinet", "bookcase", "стол", "диван", "кровать")
_LEVEL1_KEYWORDS = ("chair", "stool", "lamp", "plate", "bowl", "cup", "vase", "стул", "лампа", "тарелка")
_LEVEL2_KEYWORDS = ("apple", "book", "pen", "bottle", "glass", "candle", "fruit", "яблоко", "книга")


def _heuristic_graph(models):
    """Fallback: построить граф эвристически без LLM."""
    # Собрать уникальные типы и количества
    type_counts = {}
    for m in models:
        name = m.get("Model", "")
        type_counts[name] = type_counts.get(name, 0) + 1

    nodes = []
    anchors = []

    for model_name, count in type_counts.items():
        lname = model_name.lower()
        if any(kw in lname for kw in _ANCHOR_KEYWORDS):
            level = 0
            parent_id = None
            relationship = None
            anchor_hint = {"region": "center", "offset_x": 0.0, "offset_y": 0.0}
        elif any(kw in lname for kw in _LEVEL2_KEYWORDS):
            level = 2
            parent_id = None  # заполним ниже
            relationship = {"type": REL_ON_SURFACE, "direction": "center", "distance": 0.0, "facing": "any"}
            anchor_hint = None
        else:
            level = 1
            parent_id = None
            relationship = {"type": REL_AROUND, "direction": "all_sides", "distance": 0.6, "facing": "inward"}
            anchor_hint = None

        node_id = re.sub(r"[^a-zA-Z0-9]+", "_", model_name.lower()) + "_group"
        node = make_scene_node(
            id=node_id,
            model_name=model_name,
            level=level,
            instances=count,
            parent_id=parent_id,
            relationship=relationship,
            anchor_hint=anchor_hint,
            place_order=1,
            depends_on=None,
        )
        nodes.append(node)
        if level == 0:
            anchors.append(node_id)

    # Привязать level 1 и 2 к первому anchor
    default_anchor = anchors[0] if anchors else None
    level1_ids = []
    for node in nodes:
        if node["level"] == 1:
            node["parent_id"] = default_anchor
            level1_ids.append(node["id"])
        elif node["level"] == 2:
            node["parent_id"] = level1_ids[0] if level1_ids else default_anchor

    print(f"[graph_builder] Heuristic graph: {len(nodes)} nodes, anchors={anchors}")
    return make_scene_graph(nodes)


# ---------------------------------------------------------------------------
# LLM prompt templates for scene graph and placement
# ---------------------------------------------------------------------------

# Prompt templates are defined in project/scene_prompts.py and imported above.


def _validate_graph_counts(graph, type_counts):
    """Validate that sum(instances) per model_name matches the input model counts."""
    errors = []
    nodes = graph.get("nodes", [])

    for name, expected in type_counts.items():
        got = 0
        for n in nodes:
            if n.get("model_name") == name:
                got += int(n.get("instances", 1))
        if got != expected:
            errors.append(f"Count mismatch for model '{name}': expected {expected}, got {got}")

    extra = {n.get("model_name") for n in nodes} - set(type_counts.keys())
    for name in sorted(x for x in extra if x):
        errors.append(f"Graph contains model '{name}' which is not in available objects")

    return len(errors) == 0, errors


def _normalize_grouped_graph(graph):
    """Normalize LLM-produced graphs to the grouped-node format.

    If the LLM outputs per-instance nodes (e.g. chair_1..chair_4), we merge them into a
    single group node with instances>1 and rewrite relationship/dependency references.

    The normalization is intentionally conservative and focuses on the common patterns
    used by this project (notably 'around' groups like chairs around a table).
    """

    def _sanitize_id(text):
        return re.sub(r"[^a-zA-Z0-9]+", "_", str(text).strip().lower()).strip("_")

    nodes = list(graph.get("nodes", []))
    if not nodes:
        return graph

    # Ensure instances are ints
    for n in nodes:
        try:
            n["instances"] = int(n.get("instances", 1))
        except Exception:
            n["instances"] = 1
        if n["instances"] < 1:
            n["instances"] = 1

        # Semantic grouping: if something is "around" a reference, it belongs to that group.
        rel = n.get("relationship")
        if isinstance(rel, dict) and rel.get("type") == "around" and rel.get("reference"):
            if n.get("parent_id") is None:
                n["parent_id"] = rel.get("reference")
            if n.get("depends_on") is None:
                n["depends_on"] = rel.get("reference")

    existing_ids = {n.get("id") for n in nodes}

    def _fresh_group_id(model_name, existing):
        base = _sanitize_id(model_name) or "obj"
        i = 1
        while True:
            cand = f"{base}_group_{i}"
            if cand not in existing:
                existing.add(cand)
                return cand
            i += 1

    def _clean_group_relationship(rel):
        if not isinstance(rel, dict):
            return None
        rel = dict(rel)
        rel.pop("side", None)
        rel.pop("offset", None)
        return rel

    def _rewrite_refs(n, id_map):
        if not id_map:
            return
        if n.get("parent_id") in id_map:
            n["parent_id"] = id_map[n["parent_id"]]
        if n.get("depends_on") in id_map:
            n["depends_on"] = id_map[n["depends_on"]]
        rel = n.get("relationship")
        if isinstance(rel, dict) and rel.get("reference") in id_map:
            rel = dict(rel)
            rel["reference"] = id_map[rel["reference"]]
            n["relationship"] = rel

    # Pass 1: merge common "around" patterns (chairs around table, etc.)
    around_groups = {}
    for n in nodes:
        rel = n.get("relationship")
        if not isinstance(rel, dict):
            continue
        if rel.get("type") != "around":
            continue
        ref = rel.get("reference")
        if not ref:
            continue
        key = (n.get("model_name"), ref)
        around_groups.setdefault(key, []).append(n)

    id_map = {}
    merged_nodes = []
    consumed = set()
    for (model_name, ref), group in around_groups.items():
        if len(group) < 2:
            continue
        new_id = _fresh_group_id(model_name, existing_ids)
        template = dict(group[0])
        template["id"] = new_id
        template["instances"] = sum(int(x.get("instances", 1)) for x in group)
        template["relationship"] = _clean_group_relationship(template.get("relationship"))
        # Semantic grouping: objects "around" belong to the reference anchor group.
        template["parent_id"] = ref
        template["depends_on"] = ref
        template["anchor_hint"] = None
        template["place_order"] = min(int(x.get("place_order", 1)) for x in group)

        merged_nodes.append(template)
        for old in group:
            consumed.add(old.get("id"))
            id_map[old.get("id")] = new_id

    # Keep all nodes not consumed, plus merged ones
    nodes2 = [n for n in nodes if n.get("id") not in consumed] + merged_nodes

    # Rewrite references due to the merge
    for n in nodes2:
        _rewrite_refs(n, id_map)

    # Pass 2: merge any remaining per-instance nodes that now share the same signature
    def _sig(n):
        rel = n.get("relationship")
        rel_type = rel.get("type") if isinstance(rel, dict) else None
        rel_ref = rel.get("reference") if isinstance(rel, dict) else None
        rel_facing = rel.get("facing") if isinstance(rel, dict) else None
        dist = rel.get("distance") if isinstance(rel, dict) else None
        dist_key = None if dist is None else round(float(dist), 3)
        return (
            n.get("model_name"),
            n.get("parent_id"),
            rel_type,
            rel_ref,
            rel_facing,
            dist_key,
        )

    buckets = {}
    for n in nodes2:
        buckets.setdefault(_sig(n), []).append(n)

    out = []
    id_map2 = {}
    for sig, group in buckets.items():
        if len(group) == 1:
            out.append(group[0])
            continue
        # Merge only if they look like per-instance duplicates.
        if any(int(x.get("instances", 1)) != 1 for x in group):
            out.extend(group)
            continue

        model_name = group[0].get("model_name")
        new_id = _fresh_group_id(model_name, existing_ids)
        template = dict(group[0])
        template["id"] = new_id
        template["instances"] = len(group)
        template["relationship"] = _clean_group_relationship(template.get("relationship"))
        template["anchor_hint"] = None if template.get("parent_id") is not None else template.get("anchor_hint")
        template["place_order"] = min(int(x.get("place_order", 1)) for x in group)
        rel = template.get("relationship")
        if isinstance(rel, dict) and rel.get("reference"):
            template["depends_on"] = rel.get("reference")

        out.append(template)
        for old in group:
            id_map2[old.get("id")] = new_id

    if id_map2:
        for n in out:
            _rewrite_refs(n, id_map2)

    graph["nodes"] = out
    return graph


# ---------------------------------------------------------------------------
# SceneGraphBuilder
# ---------------------------------------------------------------------------

def _expand_anchor_groups(graph):
    """Если LLM создала anchor-узел с instances>1, раскрыть его в отдельные узлы.

    Для каждого level-0 узла с instances=N создаёт N узлов с instances=1.
    Узлы-потомки, которые ссылались на исходный anchor, распределяются равномерно
    между новыми anchor-узлами.
    """
    nodes = graph["nodes"]
    anchor_multi = [n for n in nodes if n["level"] == 0 and n.get("instances", 1) > 1]
    if not anchor_multi:
        return graph

    new_nodes = []
    replaced_ids = {}  # old_id → [new_id_0, new_id_1, ...]

    for n in nodes:
        if n["level"] == 0 and n.get("instances", 1) > 1:
            count = n["instances"]
            generated = []
            for i in range(count):
                new_id = f"{n['id']}_{i+1}"
                new_node = dict(n)
                new_node["id"] = new_id
                new_node["instances"] = 1
                new_nodes.append(new_node)
                generated.append(new_id)
            replaced_ids[n["id"]] = generated
            print(f"[graph_builder] Expanded anchor '{n['id']}' (×{count}) → {generated}")
        else:
            new_nodes.append(n)

    if not replaced_ids:
        return graph

    # Перераспределить потомков по новым anchors.
    # Если узел имеет instances=N и N делится на кол-во anchors — разбить на N/K подузлов.
    # Иначе — назначить каждый узел одному anchor (round-robin).
    final_nodes = []
    child_counters = {old_id: 0 for old_id in replaced_ids}
    for n in new_nodes:
        parent = n.get("parent_id")
        if parent in replaced_ids:
            new_anchors = replaced_ids[parent]
            k = len(new_anchors)
            instances = n.get("instances", 1)
            if instances >= k and instances % k == 0:
                # Равномерно делить instances между anchors
                per_anchor = instances // k
                for i, anchor_id in enumerate(new_anchors):
                    new_n = dict(n)
                    new_n["id"] = f"{n['id']}_{i+1}"
                    new_n["parent_id"] = anchor_id
                    new_n["instances"] = per_anchor
                    final_nodes.append(new_n)
            else:
                # Round-robin: каждый узел целиком одному anchor
                idx = child_counters[parent] % k
                child_counters[parent] += 1
                new_n = dict(n)
                new_n["parent_id"] = new_anchors[idx]
                new_n["id"] = f"{n['id']}_{idx+1}"
                final_nodes.append(new_n)
        else:
            final_nodes.append(n)

    return make_scene_graph(final_nodes, graph.get("anchor_relations", []))


def _classify_objects_by_role(type_counts, models):
    """Классифицировать объекты на якоря и зависимые объекты.
    
    Возвращает (anchors, dependents):
    - anchors: dict {model_name: count} для крупной мебели (столы, диваны и т.д.)
    - dependents: dict {model_name: count} для мелких объектов (стулья, тарелки и т.д.)
    """
    anchors = {}
    dependents = {}
    
    for model_name, count in type_counts.items():
        lname = model_name.lower()
        # Эвристика: якоря - это крупная мебель
        if any(kw in lname for kw in _ANCHOR_KEYWORDS):
            anchors[model_name] = count
        else:
            dependents[model_name] = count
    
    # Если нет якорей - первый тип становится якорем
    if not anchors and type_counts:
        first_type = list(type_counts.keys())[0]
        anchors[first_type] = type_counts[first_type]
        dependents = {k: v for k, v in type_counts.items() if k != first_type}
    
    return anchors, dependents


def _build_anchor_nodes_batch(anchor_types, models, room_half_size, query, llm_fn, batch_size=10):
    """Генерировать узлы якорей батчами через LLM.
    
    Args:
        anchor_types: dict {model_name: count}
        batch_size: максимум якорей в одном запросе к LLM
    
    Returns:
        list of anchor nodes
    """
    all_anchor_nodes = []
    anchor_items = list(anchor_types.items())
    
    # Разбить на батчи
    total_anchors = sum(anchor_types.values())
    print(f"[graph_builder] Generating {total_anchors} anchor nodes in batches of {batch_size}...")
    
    current_batch = []
    current_count = 0
    batch_num = 1
    
    for model_name, count in anchor_items:
        for instance_idx in range(count):
            current_batch.append((model_name, instance_idx + 1))
            current_count += 1
            
            if current_count >= batch_size:
                # Генерировать батч
                nodes = _generate_anchor_batch_llm(
                    current_batch, models, room_half_size, query, llm_fn, batch_num
                )
                all_anchor_nodes.extend(nodes)
                current_batch = []
                current_count = 0
                batch_num += 1
    
    # Последний батч
    if current_batch:
        nodes = _generate_anchor_batch_llm(
            current_batch, models, room_half_size, query, llm_fn, batch_num
        )
        all_anchor_nodes.extend(nodes)
    
    return all_anchor_nodes


def _generate_anchor_batch_llm(batch, models, room_half_size, query, llm_fn, batch_num):
    """Генерировать один батч якорных узлов через LLM."""
    room_w = room_half_size * 2
    
    objects_list = "\n".join(
        f"  - {model_name.replace(' ', '_')}_inst{idx}: {model_name} (size: {_model_avg_size(models, model_name)})"
        for model_name, idx in batch
    )
    
    prompt = f"""CRITICAL: Respond ONLY in English. Return ONLY valid JSON.

Generate anchor nodes (level 0, floor furniture) for a scene.

User query: {query}
Room: {room_w}m × {room_w}m

Anchor objects to generate (batch {batch_num}):
{objects_list}

CRITICAL RULES FOR IDs:
- IDs MUST NOT contain spaces
- Use underscores instead of spaces: "table_1" NOT "table 1"
- Use lowercase with underscores: "computer_chair_1" NOT "computer chair_1"

For each object, create a node with:
- id: unique identifier WITHOUT SPACES (e.g., "table_1", "computer_chair_2")
- model_name: exact model name from the list (can have spaces)
- instances: always 1
- parent_id: always null (anchors have no parent)
- anchor_hint: placement hint, e.g. {{"region": "center", "offset_x": 0.0, "offset_y": 0.0}}
- relationship: always null (anchors have no relationship)
- place_order: integer (lower = placed first)
- depends_on: always null

Return JSON: {{"nodes": [...]}}

Example for 2 tables:
{{
  "nodes": [
    {{
      "id": "table_1",
      "model_name": "table",
      "instances": 1,
      "parent_id": null,
      "anchor_hint": {{"region": "center", "offset_x": -2.0, "offset_y": 0.0}},
      "relationship": null,
      "place_order": 1,
      "depends_on": null
    }},
    {{
      "id": "table_2",
      "model_name": "table",
      "instances": 1,
      "parent_id": null,
      "anchor_hint": {{"region": "center", "offset_x": 2.0, "offset_y": 0.0}},
      "relationship": null,
      "place_order": 2,
      "depends_on": null
    }}
  ]
}}

Example for computer chairs (note: id has underscores, model_name has spaces):
{{
  "nodes": [
    {{
      "id": "computer_chair_1",
      "model_name": "computer chair",
      "instances": 1,
      "parent_id": null,
      "anchor_hint": {{"region": "center", "offset_x": 0.0, "offset_y": 0.0}},
      "relationship": null,
      "place_order": 1,
      "depends_on": null
    }}
  ]
}}
"""
    
    for attempt in range(3):
        try:
            raw = llm_fn(prompt, query)
            if isinstance(raw, str):
                raw = json.loads(raw)
            nodes = raw.get("nodes", [])
            
            if len(nodes) != len(batch):
                print(f"[graph_builder] WARNING: Expected {len(batch)} nodes, got {len(nodes)}")
            
            print(f"[graph_builder] Batch {batch_num}: generated {len(nodes)} anchor nodes")
            return nodes
        except Exception as e:
            print(f"[graph_builder] Batch {batch_num} attempt {attempt+1} failed: {e}")
            if attempt == 2:
                # Fallback: создать узлы эвристически
                print(f"[graph_builder] Using heuristic fallback for batch {batch_num}")
                return _create_anchor_nodes_heuristic(batch, room_half_size)
    
    return []


def _create_anchor_nodes_heuristic(batch, room_half_size):
    """Создать якорные узлы эвристически (fallback)."""
    nodes = []
    for i, (model_name, idx) in enumerate(batch):
        node_id = f"{re.sub(r'[^a-zA-Z0-9]+', '_', model_name.lower())}_{idx}"
        # Простая сетка
        offset_x = (i % 3 - 1) * 2.0
        offset_y = (i // 3 - 1) * 2.0
        nodes.append({
            "id": node_id,
            "model_name": model_name,
            "instances": 1,
            "parent_id": None,
            "anchor_hint": {"region": "center", "offset_x": offset_x, "offset_y": offset_y},
            "relationship": None,
            "place_order": i + 1,
            "depends_on": None,
        })
    return nodes


def _build_dependent_nodes_batch(dependent_types, anchor_nodes, models, query, llm_fn, batch_size=15):
    """Генерировать зависимые узлы (стулья, тарелки и т.д.) батчами.
    
    Args:
        dependent_types: dict {model_name: count}
        anchor_nodes: list of anchor nodes to attach dependents to
        batch_size: максимум зависимых объектов в одном запросе
    
    Returns:
        list of dependent nodes
    """
    if not dependent_types or not anchor_nodes:
        return []
    
    all_dependent_nodes = []
    total_dependents = sum(dependent_types.values())
    print(f"[graph_builder] Generating {total_dependents} dependent nodes in batches of {batch_size}...")
    
    # Создать список всех зависимых объектов
    dependent_items = []
    for model_name, count in dependent_types.items():
        for instance_idx in range(count):
            dependent_items.append((model_name, instance_idx + 1))
    
    # Разбить на батчи
    batch_num = 1
    for i in range(0, len(dependent_items), batch_size):
        batch = dependent_items[i:i + batch_size]
        nodes = _generate_dependent_batch_llm(
            batch, anchor_nodes, models, query, llm_fn, batch_num
        )
        all_dependent_nodes.extend(nodes)
        batch_num += 1
    
    return all_dependent_nodes


def _generate_dependent_batch_llm(batch, anchor_nodes, models, query, llm_fn, batch_num):
    """Генерировать один батч зависимых узлов через LLM."""
    objects_list = "\n".join(
        f"  - {model_name.replace(' ', '_')}_inst{idx}: {model_name} (size: {_model_avg_size(models, model_name)})"
        for model_name, idx in batch
    )
    
    anchor_list = "\n".join(
        f"  - {node['id']}: {node['model_name']}"
        for node in anchor_nodes
    )
    
    prompt = f"""CRITICAL: Respond ONLY in English. Return ONLY valid JSON.

Generate dependent nodes (objects that relate to anchors) for a scene.

User query: {query}

Available anchor objects (already placed):
{anchor_list}

Dependent objects to generate (batch {batch_num}):
{objects_list}

CRITICAL RULES FOR IDs:
- IDs MUST NOT contain spaces
- Use underscores instead of spaces: "chair_1" NOT "chair 1"
- Use lowercase with underscores: "computer_chair_1" NOT "computer chair_1"
- reference and depends_on MUST match anchor IDs exactly

For each dependent object, create a node with:
- id: unique identifier WITHOUT SPACES (e.g., "chair_1", "computer_chair_2")
- model_name: exact model name from the list (can have spaces)
- instances: always 1
- parent_id: null for floor objects, or anchor id for objects on surfaces
- anchor_hint: always null (only anchors have this)
- relationship: object describing spatial relationship
  * type: "around" (chairs around table), "on_surface" (plate on table), "beside", etc.
  * reference: id of anchor object this relates to (WITHOUT SPACES)
  * side: "+y", "-y", "+x", "-x" (direction from reference)
  * distance: float (meters from reference)
  * facing: "inward", "outward", "any"
- place_order: integer
- depends_on: same as relationship.reference (WITHOUT SPACES)

Return JSON: {{"nodes": [...]}}

Example for 4 chairs around table_1:
{{
  "nodes": [
    {{
      "id": "chair_1",
      "model_name": "computer chair",
      "instances": 1,
      "parent_id": null,
      "anchor_hint": null,
      "relationship": {{
        "type": "around",
        "reference": "table_1",
        "side": "+y",
        "distance": 0.5,
        "facing": "inward"
      }},
      "place_order": 1,
      "depends_on": "table_1"
    }},
    {{
      "id": "chair_2",
      "model_name": "computer chair",
      "instances": 1,
      "parent_id": null,
      "anchor_hint": null,
      "relationship": {{
        "type": "around",
        "reference": "table_1",
        "side": "+x",
        "distance": 0.5,
        "facing": "inward"
      }},
      "place_order": 2,
      "depends_on": "table_1"
    }},
    {{
      "id": "chair_3",
      "model_name": "computer chair",
      "instances": 1,
      "parent_id": null,
      "anchor_hint": null,
      "relationship": {{
        "type": "around",
        "reference": "table_1",
        "side": "-y",
        "distance": 0.5,
        "facing": "inward"
      }},
      "place_order": 3,
      "depends_on": "table_1"
    }},
    {{
      "id": "chair_4",
      "model_name": "computer chair",
      "instances": 1,
      "parent_id": null,
      "anchor_hint": null,
      "relationship": {{
        "type": "around",
        "reference": "table_1",
        "side": "-x",
        "distance": 0.5,
        "facing": "inward"
      }},
      "place_order": 4,
      "depends_on": "table_1"
    }}
  ]
}}
"""
    
    for attempt in range(3):
        try:
            raw = llm_fn(prompt, query)
            if isinstance(raw, str):
                raw = json.loads(raw)
            nodes = raw.get("nodes", [])
            
            if len(nodes) != len(batch):
                print(f"[graph_builder] WARNING: Expected {len(batch)} nodes, got {len(nodes)}")
            
            print(f"[graph_builder] Batch {batch_num}: generated {len(nodes)} dependent nodes")
            return nodes
        except Exception as e:
            print(f"[graph_builder] Batch {batch_num} attempt {attempt+1} failed: {e}")
            if attempt == 2:
                # Fallback: создать узлы эвристически
                print(f"[graph_builder] Using heuristic fallback for batch {batch_num}")
                return _create_dependent_nodes_heuristic(batch, anchor_nodes)
    
    return []


def _create_dependent_nodes_heuristic(batch, anchor_nodes):
    """Создать зависимые узлы эвристически (fallback)."""
    nodes = []
    anchor_idx = 0
    
    for i, (model_name, idx) in enumerate(batch):
        node_id = f"{re.sub(r'[^a-zA-Z0-9]+', '_', model_name.lower())}_{idx}"
        # Привязать к якорю round-robin
        anchor = anchor_nodes[anchor_idx % len(anchor_nodes)]
        anchor_idx += 1
        
        # Определить сторону (циклически)
        sides = ["+y", "+x", "-y", "-x"]
        side = sides[i % 4]
        
        nodes.append({
            "id": node_id,
            "model_name": model_name,
            "instances": 1,
            "parent_id": None,
            "anchor_hint": None,
            "relationship": {
                "type": "around",
                "reference": anchor["id"],
                "side": side,
                "distance": 0.5,
                "facing": "inward"
            },
            "place_order": i + 1,
            "depends_on": anchor["id"],
        })
    return nodes


def _build_scene_graph(models, room_half_size, query, llm_fn):
    """Построить граф сцены через LLM (с fallback)."""
    # Собрать уникальные типы и их количества
    type_counts = {}
    for m in models:
        name = m.get("Model", "")
        type_counts[name] = type_counts.get(name, 0) + 1

    total_count = sum(type_counts.values())
    objects_list = "\n".join(
        f"  - {name} ×{cnt} (size: {_model_avg_size(models, name)})"
        for name, cnt in type_counts.items()
    )

    room_w = room_half_size * 2
    prompt = _GRAPH_HIERARCHY_PROMPT.format(
        query=query,
        room_w=room_w,
        room_l=room_w,
        objects_list=objects_list,
        total_count=total_count,
    )

    available_names = list(type_counts.keys())

    graph = None
    for attempt in range(3):
        print(f"[graph_builder] Building hierarchy (attempt {attempt+1}/3)...")
        try:
            raw = llm_fn(prompt, query)
            if isinstance(raw, str):
                raw = json.loads(raw)
            nodes = raw.get("nodes", [])
            # LLM no longer sets level — compute it from parent_id chain.
            for n in nodes:
                n.setdefault("level", 0)
            graph = make_scene_graph(nodes)
            graph = _normalize_grouped_graph(graph)
            graph = _compute_levels(graph)
            graph = make_scene_graph(graph["nodes"])  # refresh metadata.max_level
            valid, errors = _validate_graph(graph, available_names)
            counts_ok, count_errors = _validate_graph_counts(graph, type_counts)
            if valid and counts_ok:
                print(f"[graph_builder] Graph OK: {len(nodes)} nodes, max_level={graph['metadata']['max_level']}")
                _log_graph(graph)
                break
            err_str = "\n".join(errors + count_errors)
            print(f"[graph_builder] Validation failed: {err_str}")
            prompt = f"Previous attempt had errors:\n{err_str}\n\nFix and regenerate.\n\n{prompt}"
        except Exception as e:
            print(f"[graph_builder] Error: {e}")
            if attempt == 2:
                print("[graph_builder] Falling back to heuristic graph")
                graph = _heuristic_graph(models)
                break
    else:
        print("[graph_builder] All attempts failed, using heuristic graph")
        graph = _heuristic_graph(models)

    return graph


def _resolve_single_anchor_pos(node, room_half_size):
    """Определить позицию для единственного anchor в комнате на основе anchor_hint.region."""
    hint = node.get("anchor_hint") or {}
    region = hint.get("region", "center")
    margin = 0.6
    limit = max(0.0, room_half_size - margin)
    if region == "center":
        return {"x": 0.0, "y": 0.0, "yaw_deg": 0.0}
    if region == "near_wall":
        ox = float(hint.get("offset_x", 0.0))
        oy = float(hint.get("offset_y", -limit))  # по умолчанию у стены -Y, лицом в комнату
        return {"x": max(-limit, min(limit, ox)), "y": max(-limit, min(limit, oy)), "yaw_deg": float(hint.get("yaw_deg", 90.0))}
    if region == "corner":
        ox = float(hint.get("offset_x", -limit * 0.7))
        oy = float(hint.get("offset_y", -limit * 0.7))
        return {"x": max(-limit, min(limit, ox)), "y": max(-limit, min(limit, oy)), "yaw_deg": float(hint.get("yaw_deg", 45.0))}
    # Custom — explicit offsets
    ox = float(hint.get("offset_x", 0.0))
    oy = float(hint.get("offset_y", 0.0))
    return {"x": max(-limit, min(limit, ox)), "y": max(-limit, min(limit, oy)), "yaw_deg": float(hint.get("yaw_deg", 0.0))}


def _collect_descendants(graph, root_id):
    """Вернуть все узлы графа, чей parent_id (прямой или транзитивный) = root_id."""
    children_by_parent = {}
    for n in graph["nodes"]:
        children_by_parent.setdefault(n.get("parent_id"), []).append(n)
    out = []
    stack = list(children_by_parent.get(root_id, []))
    while stack:
        n = stack.pop()
        out.append(n)
        stack.extend(children_by_parent.get(n["id"], []))
    return out


def _model_avg_size(models, model_name):
    """Return size string for a model type."""
    for m in models:
        if m.get("Model") == model_name:
            sz = m.get("size", [1.0, 1.0, 1.0])
            return f"{sz[0]:.2f}×{sz[2]:.2f}×{sz[1]:.2f}м"
    return "?"


def _log_graph(graph):
    """Print graph summary per level."""
    max_level = graph["metadata"]["max_level"]
    for lvl in range(max_level + 1):
        nodes = [n for n in graph["nodes"] if n["level"] == lvl]
        parts = [f"{n['model_name']}×{n['instances']}({n['id']})" for n in nodes]
        print(f"[graph_builder]   Level {lvl}: {', '.join(parts)}")


def _place_anchors_by_llm(graph, anchor_nodes, type_counts, models, room_half_size, query, llm_fn):
    """Разместить несколько anchor объектов через LLM."""
    room_w = room_half_size * 2

    anchors_list = "\n".join(
        f"  - {n['id']}: {n['model_name']} ×{n['instances']} ({_model_avg_size(models, n['model_name'])})"
        for n in anchor_nodes
    )

    # Полное описание будущих объектов per-anchor с размерами и типом связи
    future_lines = []
    for an in anchor_nodes:
        # Все потомки (рекурсивно) этого anchor
        descendants = _collect_descendants(graph, an["id"])
        if not descendants:
            future_lines.append(f"  - {an['id']}: (no dependents)")
            continue
        descs = []
        for d in descendants:
            sz_str = _model_avg_size(models, d["model_name"])
            rel = d.get("relationship") or {}
            rel_type = rel.get("type", "?")
            rel_dist = rel.get("distance")
            rel_desc = rel_type if rel_dist is None else f"{rel_type} @ {rel_dist}m"
            descs.append(f"      • {d['model_name']}×{d['instances']} ({sz_str}) — {rel_desc}")
        future_lines.append(f"  - around {an['id']}:\n" + "\n".join(descs))
    future_objects_per_anchor = "\n".join(future_lines) if future_lines else "  (none)"

    prompt = _ANCHOR_RELATIONS_PROMPT.format(
        anchor_count=len(anchor_nodes),
        room_w=room_w,
        room_l=room_w,
        anchors_list=anchors_list,
        future_objects_per_anchor=future_objects_per_anchor,
        query=query,
        neg_half=-room_half_size,
        pos_half=room_half_size,
        corner=round(room_half_size * 0.7, 1),
        near_wall=round(room_half_size * 0.8, 1),
    )

    try:
        raw = llm_fn(prompt, query)
        if isinstance(raw, str):
            raw = json.loads(raw)
        positions = {p["id"]: p for p in raw.get("anchor_positions", [])}
        margin = 0.6
        limit = room_half_size - margin
        for an in anchor_nodes:
            pos = positions.get(an["id"])
            if pos:
                x = max(-limit, min(limit, float(pos["x"])))
                y = max(-limit, min(limit, float(pos["y"])))
                an["_anchor_pos"] = {"x": x, "y": y, "yaw_deg": float(pos.get("yaw_deg", 0.0))}
            else:
                an["_anchor_pos"] = {"x": 0.0, "y": 0.0, "yaw_deg": 0.0}
        print(f"[graph_builder] Anchor positions set via LLM")
    except Exception as e:
        print(f"[graph_builder] Anchor placement LLM failed ({e}), using grid fallback")
        n = len(anchor_nodes)
        for i, an in enumerate(anchor_nodes):
            offset = (i - (n - 1) / 2.0) * (room_half_size / max(n, 1))
            an["_anchor_pos"] = {"x": offset, "y": 0.0, "yaw_deg": 0.0}

    return graph


# ---------------------------------------------------------------------------
# HierarchicalPlacer
# ---------------------------------------------------------------------------

def _rel_description(node):
    """Описание связи объекта с родителем/reference в виде строки для промпта."""
    rel = node.get("relationship")
    if not rel:
        hint = node.get("anchor_hint") or {}
        return f"Anchor object. Place in room region: {hint.get('region', 'center')}"
    rtype = rel.get("type", "on_surface")
    parts = [f"Placement type: {rtype}"]
    ref = rel.get("reference")
    if ref:
        parts.append(f"Position relative to: {ref}")
    side = rel.get("side")
    if side:
        parts.append(f"Side from reference: {side}")
    dist = rel.get("distance")
    if dist is not None:
        parts.append(f"Distance from reference: {dist}m")
    offset = rel.get("offset")
    if offset:
        parts.append(f"Fine offset (m): x={offset.get('x', 0)}, y={offset.get('y', 0)}, z={offset.get('z', 0)}")
    facing = rel.get("facing")
    if facing:
        parts.append(f"Facing: {facing}")
    return ". ".join(parts)


def _world_xy_half_sizes(obj_or_node):
    """Return (half_x, half_y) in world units for either a placed object or graph node."""
    sz = obj_or_node.get("size") or obj_or_node.get("size_hint") or [1.0, 1.0, 1.0]
    sx = float(sz[0]) if len(sz) > 0 else 1.0
    sy = float(sz[2]) if len(sz) > 2 else sx
    return sx / 2.0, sy / 2.0


def _validate_xy_within_parent_footprint(x, y, node, parent_obj, margin=0.02):
    """Validate (x,y) is inside parent's footprint, accounting for child half-extents.

    Unlike clamping, this does NOT modify the coordinates. It is used to enforce
    LLM output correctness without deterministic post-processing.
    """
    if parent_obj is None:
        return True, ""

    parent_pose = parent_obj.get("Pose") or {}
    px = float(parent_pose.get("x", 0.0))
    py = float(parent_pose.get("y", 0.0))

    psz = parent_obj.get("size") or [1.0, 1.0, 1.0]
    pw = float(psz[0]) if len(psz) > 0 else 1.0
    pd = float(psz[2]) if len(psz) > 2 else pw

    self_hx, self_hy = _world_xy_half_sizes(node)
    limit_x = max(0.0, pw / 2.0 - self_hx - margin)
    limit_y = max(0.0, pd / 2.0 - self_hy - margin)

    # Numerical tolerance: prevents false negatives from rounding and float noise.
    # NOTE: Prompts often show bounds rounded to 3 decimals, so the LLM may return
    # a value that's within the printed bounds but slightly outside the true
    # computed limit by up to ~5e-4. Keep epsilon comfortably above that.
    eps = 1e-3

    fx = float(x)
    fy = float(y)
    if not (px - limit_x - eps <= fx <= px + limit_x + eps):
        return False, f"x={fx:.6f} outside parent footprint [{px - limit_x:.6f}..{px + limit_x:.6f}]"
    if not (py - limit_y - eps <= fy <= py + limit_y + eps):
        return False, f"y={fy:.6f} outside parent footprint [{py - limit_y:.6f}..{py + limit_y:.6f}]"
    return True, ""


def _min_clearance_for_group(node):
    sz = node.get("size_hint") or [0.1, 0.1, 0.1]
    sw = float(sz[0]) if len(sz) > 0 else 0.1
    sd = float(sz[2]) if len(sz) > 2 else sw
    # Conservative: keep at least ~one footprint apart.
    return max(0.05, 0.6 * max(sw, sd))


def _validate_min_clearance_xy(positions, min_clearance, already=None):
    """Check that XY distance between any pair is >= min_clearance."""
    pts = []
    for p in (already or []):
        pts.append((float(p.get("x", 0.0)), float(p.get("y", 0.0)), str(p.get("instance", "?"))))
    base_n = len(pts)
    for p in positions:
        pts.append((float(p.get("x", 0.0)), float(p.get("y", 0.0)), str(p.get("instance", "?"))))

    for i in range(len(pts)):
        xi, yi, li = pts[i]
        for j in range(i + 1, len(pts)):
            xj, yj, lj = pts[j]
            dx = xi - xj
            dy = yi - yj
            if (dx * dx + dy * dy) ** 0.5 < float(min_clearance):
                # Report the first collision found.
                left = "existing" if i < base_n else "new"
                right = "existing" if j < base_n else "new"
                return False, f"too close in XY: {left} instance {li} vs {right} instance {lj} (< {min_clearance:.3f}m)"
    return True, ""


def _compute_suggested_xy(node, ref_obj, parent_obj):
    """Detminisitic XY suggestion from relationship.{side,distance,offset}.

    Edge-to-edge math: target = base_center + side * (base_half + distance + self_half) + offset.
    Returns (x, y) or None if neither reference nor parent is available.
    """
    rel = node.get("relationship") or {}
    if not isinstance(rel, dict):
        return None

    side = rel.get("side")
    raw_distance = rel.get("distance")
    distance = float(raw_distance) if raw_distance is not None else 0.05
    offset = rel.get("offset") or {}
    off_x = float(offset.get("x", 0.0) or 0.0)
    off_y = float(offset.get("y", 0.0) or 0.0)

    base = ref_obj if ref_obj else parent_obj
    if base is None:
        return None

    bx = float(base["Pose"]["x"])
    by = float(base["Pose"]["y"])
    base_hx, base_hy = _world_xy_half_sizes(base)
    self_hx, self_hy = _world_xy_half_sizes(node)

    if side == "+x":
        x = bx + (base_hx + distance + self_hx)
        y = by
    elif side == "-x":
        x = bx - (base_hx + distance + self_hx)
        y = by
    elif side == "+y":
        x = bx
        y = by + (base_hy + distance + self_hy)
    elif side == "-y":
        x = bx
        y = by - (base_hy + distance + self_hy)
    else:
        # "above" or no side → directly above/at base center; offset will adjust
        x = bx
        y = by

    return (x + off_x, y + off_y)


def _ws_describe_object(obj):
    """Подробное описание одного размещённого объекта для промпта LLM."""
    pose = obj.get("Pose", {})
    sz = obj.get("size", [1.0, 1.0, 1.0])
    return (
        f"id={obj['id']}, model={obj['Model']}, "
        f"pos=({pose.get('x', 0):.2f}, {pose.get('y', 0):.2f}, {pose.get('z', 0):.2f}), "
        f"size_xyz=({sz[0]:.2f}, {sz[2] if len(sz) > 2 else sz[0]:.2f}, {sz[1] if len(sz) > 1 else sz[0]:.2f})m, "
        f"yaw={obj.get('yaw_deg', 0):.0f}°"
    )


def _validate_pos(result, room_half_size):
    """Проверить что позиция в границах комнаты."""
    x = result.get("x", 0.0)
    y = result.get("y", 0.0)
    z = result.get("z", 0.0)
    if abs(x) > room_half_size * 1.1 or abs(y) > room_half_size * 1.1:
        return False, f"Position ({x:.2f}, {y:.2f}) outside room ±{room_half_size}m"
    if z < -0.1:
        return False, f"z={z:.2f} is below floor"
    return True, ""


def _fallback_pos(node, parent_placed, room_half_size):
    """Запасная позиция если LLM не справилась."""
    if parent_placed:
        pose = parent_placed["Pose"]
        return {"x": pose["x"] + 0.8, "y": pose["y"], "z": 0.0, "yaw_deg": 0.0}
    sz = node.get("size_hint", [1.0, 1.0, 1.0])
    hz = sz[1] / 2.0 if len(sz) > 1 else 0.5
    return {"x": 0.0, "y": 0.0, "z": hz, "yaw_deg": 0.0}


def _place_single_node(node, world_state, room_half_size, query, llm_fn):
    """Разместить один объект (instances=1) через LLM."""
    room_w = room_half_size * 2
    sz = node.get("size_hint", [1.0, 1.0, 1.0])

    # Anchor — используем _anchor_pos если есть
    if node["level"] == 0 and "_anchor_pos" in node:
        ap = node["_anchor_pos"]
        hz = sz[1] / 2.0 if len(sz) > 1 else 0.5
        result = {"x": ap["x"], "y": ap["y"], "z": hz, "yaw_deg": ap.get("yaw_deg", 0.0)}
        print(f"[placer] Anchor {node['id']} ({node['model_name']}): pos=({result['x']:.2f}, {result['y']:.2f}, {result['z']:.2f}), yaw={result['yaw_deg']:.0f}°")
        return [result]

    parent_placed = (_ws_find_all_by_node(world_state, node["parent_id"]) or [None])[0] if node["parent_id"] else None

    placement_instruction = _rel_description(node)
    if parent_placed:
        pose = parent_placed["Pose"]
        psz = parent_placed["size"]
        parent_top_z = pose["z"] + (psz[1] / 2.0 if len(psz) > 1 else psz[0] / 2.0)
        placement_instruction += (
            f"\nSurface parent '{node['parent_id']}' ({parent_placed['Model']}): "
            f"pos=({pose['x']:.2f}, {pose['y']:.2f}, {pose['z']:.2f}), "
            f"size_xyz=({psz[0]:.2f}, {psz[2] if len(psz) > 2 else psz[0]:.2f}, {psz[1] if len(psz) > 1 else psz[0]:.2f})m, "
            f"top_surface_z={parent_top_z:.2f}m. "
            f"Your X must be in [{pose['x'] - psz[0]/2:.2f}, {pose['x'] + psz[0]/2:.2f}], "
            f"Y in [{pose['y'] - (psz[2] if len(psz) > 2 else psz[0])/2:.2f}, "
            f"{pose['y'] + (psz[2] if len(psz) > 2 else psz[0])/2:.2f}]."
        )

    rel = node.get("relationship") or {}
    ref_id = rel.get("reference") if isinstance(rel, dict) else None
    if ref_id:
        ref_objs = _ws_find_all_by_node(world_state, ref_id)
        if ref_objs:
            placement_instruction += "\nReference object(s) for spatial positioning:"
            for r in ref_objs:
                placement_instruction += "\n  " + _ws_describe_object(r)
            # Common case: placing something on a surface relative to objects that are NOT on that surface.
            # E.g. apples/plates on a table "in front of" chairs around the table.
            if parent_placed and rel.get("type") in (REL_ON_SURFACE, "stacked_on") and ref_id != node.get("parent_id"):
                placement_instruction += (
                    "\nIMPORTANT: If the reference objects are outside the parent surface footprint (e.g. chairs around a table), "
                    "DO NOT copy the reference x/y directly. Instead, use the DIRECTION from the parent center to the reference, "
                    "and choose a point ON THE PARENT SURFACE near the corresponding edge, staying within the numeric surface bounds."
                )
        else:
            placement_instruction += f"\nReference '{ref_id}' is not yet placed (skip reference, use parent surface only)."

    sw = float(sz[0]) if len(sz) > 0 else 1.0
    sd = float(sz[2]) if len(sz) > 2 else sw
    sh = float(sz[1]) if len(sz) > 1 else sw
    prompt = _SINGLE_PLACEMENT_PROMPT.format(
        model_name=node["model_name"],
        size_w=sw, size_d=sd, size_h=sh,
        size_w_half=sw / 2.0, size_d_half=sd / 2.0, size_h_half=sh / 2.0,
        placement_instruction=placement_instruction,
        room_w=room_w, room_l=room_w, room_half=room_half_size,
        remaining=_ws_remaining_summary(world_state),
        world_state=_ws_summary(world_state),
    )

    for attempt in range(6):
        try:
            raw = llm_fn(prompt, query)
            if isinstance(raw, str):
                raw = json.loads(raw)
            valid, err = _validate_pos(raw, room_half_size)
            if valid:
                rel = node.get("relationship") or {}
                rel_type = rel.get("type") if isinstance(rel, dict) else None
                if parent_placed and rel_type in (REL_ON_SURFACE, "stacked_on"):
                    ok_fp, err_fp = _validate_xy_within_parent_footprint(raw.get("x", 0.0), raw.get("y", 0.0), node, parent_placed)
                    if not ok_fp:
                        valid = False
                        err = f"Surface bounds violation: {err_fp}"
                if valid:
                    print(f"[placer] {node['id']} ({node['model_name']}): pos=({raw['x']:.2f}, {raw['y']:.2f}, {raw['z']:.2f}), yaw={raw.get('yaw_deg', 0):.0f}°")
                    return [raw]
            print(f"[placer] Invalid position: {err}")
            prompt += f"\nPrevious attempt invalid: {err}. Try again."
            if attempt == 2:
                prompt += (
                    "\nRELAXED FALLBACK (still LLM-only): If you cannot satisfy the spatial relationship, "
                    "prioritize producing ANY valid non-overlapping position that satisfies ALL numeric constraints "
                    "(room bounds and surface bounds if applicable)."
                )
        except Exception as e:
            print(f"[placer] Error on attempt {attempt+1}: {e}")
            prompt += f"\nPrevious attempt error: {e}. Return valid JSON."

    raise RuntimeError(
        f"LLM placement failed for single node {node.get('id')} ({node.get('model_name')}) after retries"
    )


def _place_batch_node(node, world_state, room_half_size, query, llm_fn):
    """Разместить несколько экземпляров одного типа (instances>1) батчами по ≤6."""
    instances = node["instances"]
    batch_size = 6
    room_w = room_half_size * 2
    sz = node.get("size_hint", [1.0, 1.0, 1.0])

    parent_objects = _ws_find_all_by_node(world_state, node["parent_id"]) if node["parent_id"] else []
    parent_placed = parent_objects[0] if parent_objects else None
    parent_model = parent_placed["Model"] if parent_placed else "?"
    parent_pose = parent_placed["Pose"] if parent_placed else {"x": 0.0, "y": 0.0, "z": 0.0}
    parent_sz = parent_placed["size"] if parent_placed else [1.0, 1.0, 1.0]

    rel_desc = _rel_description(node)

    rel = node.get("relationship") or {}
    rel_type = rel.get("type") if isinstance(rel, dict) else None
    ref_id = rel.get("reference") if isinstance(rel, dict) else None

    # Precompute explicit parent surface bounds for on-surface placements.
    margin = 0.02
    self_hx, self_hy = _world_xy_half_sizes(node)
    surface_xmin = float(parent_pose.get("x", 0.0)) - max(0.0, parent_sz[0] / 2.0 - self_hx - margin)
    surface_xmax = float(parent_pose.get("x", 0.0)) + max(0.0, parent_sz[0] / 2.0 - self_hx - margin)
    pd = parent_sz[2] if len(parent_sz) > 2 else parent_sz[0]
    surface_ymin = float(parent_pose.get("y", 0.0)) - max(0.0, pd / 2.0 - self_hy - margin)
    surface_ymax = float(parent_pose.get("y", 0.0)) + max(0.0, pd / 2.0 - self_hy - margin)

    # Если зависит от другого узла — собрать их позиции для контекста
    depends_context = ""
    if node.get("depends_on"):
        dep_objs = _ws_find_all_by_node(world_state, node["depends_on"])
        if dep_objs:
            lines = [f"  Positions of '{node['depends_on']}' objects (use for alignment):"]
            for d in dep_objs:
                dp = d["Pose"]
                lines.append(f"    - {d['id']}: pos=({dp['x']:.2f}, {dp['y']:.2f})")
            depends_context = "\n".join(lines) + "\n"

    parent_instances_section = ""
    if parent_objects:
        lines = [f"Parent instances for '{node['parent_id']}':"]
        for idx, parent in enumerate(parent_objects, start=1):
            pose = parent["Pose"]
            size = parent["size"]
            lines.append(
                f"  - parent_instance_{idx}: id={parent['id']}, model={parent['Model']}, "
                f"pos=({pose['x']:.2f}, {pose['y']:.2f}, {pose['z']:.2f}), "
                f"size=({size[0]:.2f}, {size[2] if len(size) > 2 else size[0]:.2f}, {size[1] if len(size) > 1 else size[0]:.2f})"
            )
        if len(parent_objects) > 1:
            lines.append("  - Match child instance i to parent_instance_i when possible.")
        parent_instances_section = "\n".join(lines)

    all_results = []
    placed_in_batch = []

    reference_instances_section = ""
    if ref_id and isinstance(ref_id, str):
        ref_objs = _ws_find_all_by_node(world_state, ref_id)
        if ref_objs:
            lines = [f"Reference instances for '{ref_id}' (use these for alignment):"]
            for idx, robj in enumerate(ref_objs, start=1):
                rp = robj.get("Pose") or {}
                lines.append(f"  - ref_instance_{idx}: id={robj['id']}, pos=({rp.get('x', 0.0):.2f}, {rp.get('y', 0.0):.2f}, {rp.get('z', 0.0):.2f}), yaw={robj.get('yaw_deg', 0.0):.0f}°")
            if len(ref_objs) == instances:
                lines.append("  - IMPORTANT: Match your placed instance i to ref_instance_i.")
            if parent_placed and rel_type in (REL_ON_SURFACE, "stacked_on") and ref_id != node.get("parent_id"):
                lines.append(
                    "  - CRITICAL: If the reference instances are OFF the parent surface (e.g. chairs around the table), "
                    "do NOT reuse ref x/y for the on-surface objects. Use the direction from parent center to ref_instance_i and "
                    "place the on-surface object near the corresponding edge ON the parent surface, within the numeric bounds."
                )
            reference_instances_section = "\n".join(lines)

    batches = [list(range(i, min(i + batch_size, instances))) for i in range(0, instances, batch_size)]
    for batch_idxs in batches:
        from_idx = batch_idxs[0] + 1
        to_idx = batch_idxs[-1] + 1
        batch_count = len(batch_idxs)

        already_section = ""
        if placed_in_batch:
            lines = [f"Already placed in this group ({len(placed_in_batch)} objects):"]
            for r in placed_in_batch:
                lines.append(f"  - instance {r['instance']}: pos=({r['x']:.2f}, {r['y']:.2f}, {r['z']:.2f}), yaw={r.get('yaw_deg', 0):.0f}°")
            already_section = "\n".join(lines)

        px = parent_pose["x"]
        py = parent_pose["y"]
        pz = parent_pose["z"]
        pw = parent_sz[0]
        pd = parent_sz[2] if len(parent_sz) > 2 else parent_sz[0]
        ph = parent_sz[1] if len(parent_sz) > 1 else parent_sz[0]
        obj_h = sz[1] if len(sz) > 1 else sz[0]
        parent_top_z = pz + ph / 2.0

        rel = node.get("relationship") or {}
        placement_dist = float(rel.get("distance") or max(pw, pd) / 2.0 + max(sz[0], sz[2] if len(sz) > 2 else sz[0]) / 2.0 + 0.1)

        prompt = _BATCH_PLACEMENT_PROMPT.format(
            batch_count=batch_count,
            model_name=node["model_name"],
            from_idx=from_idx, to_idx=to_idx, total=instances,
            size_w=sz[0], size_d=sz[2] if len(sz) > 2 else sz[0], size_h=obj_h,
            size_w_half=(float(sz[0]) / 2.0 if len(sz) > 0 else 0.05),
            size_d_half=(float(sz[2]) / 2.0 if len(sz) > 2 else float(sz[0]) / 2.0 if len(sz) > 0 else 0.05),
            relationship_desc=rel_desc + ("\n" + depends_context if depends_context else ""),
            parent_model=parent_model,
            parent_x=px, parent_y=py, parent_z=pz,
            parent_w=pw, parent_d=pd, parent_h=ph,
            parent_top_z=parent_top_z,
            surface_xmin=surface_xmin,
            surface_xmax=surface_xmax,
            surface_ymin=surface_ymin,
            surface_ymax=surface_ymax,
            placement_dist=placement_dist,
            parent_xpd=px + placement_dist, parent_xmd=px - placement_dist,
            parent_ypd=py + placement_dist, parent_ymd=py - placement_dist,
            already_in_group_section=already_section,
            parent_instances_section=parent_instances_section,
            reference_instances_section=reference_instances_section,
            world_state=_ws_summary(world_state),
            room_w=room_w, room_l=room_w, room_half=room_half_size,
        )

        for attempt in range(3):
            try:
                raw = llm_fn(prompt, query)
                if isinstance(raw, str):
                    raw = json.loads(raw)
                if isinstance(raw, dict) and "positions" in raw:
                    raw = raw["positions"]
                if not isinstance(raw, list) or len(raw) != batch_count:
                    raise ValueError(f"Expected list of {batch_count}, got {type(raw).__name__} len={len(raw) if isinstance(raw, list) else '?'}")

                valid_results = []
                for item in raw:
                    ok, err = _validate_pos(item, room_half_size)
                    if not ok:
                        raise ValueError(f"Invalid position: {err}")
                    valid_results.append(item)

                if parent_placed and rel_type in (REL_ON_SURFACE, "stacked_on"):
                    for item in valid_results:
                        ok_fp, err_fp = _validate_xy_within_parent_footprint(item.get("x", 0.0), item.get("y", 0.0), node, parent_placed)
                        if not ok_fp:
                            raise ValueError(f"Surface bounds violation: {err_fp}")

                min_clearance = _min_clearance_for_group(node)
                ok_clear, err_clear = _validate_min_clearance_xy(valid_results, min_clearance=min_clearance, already=placed_in_batch)
                if not ok_clear:
                    raise ValueError(err_clear)

                placed_in_batch.extend(valid_results)
                all_results.extend(valid_results)
                break
            except Exception as e:
                print(f"[placer] Batch attempt {attempt+1} failed: {e}")
                prompt += f"\nPrevious attempt error: {e}. Return exactly {batch_count} items."

        if len(all_results) < to_idx:
            # Hybrid fallback: if the batch couldn't be placed reliably via batch calls, do sequential LLM.
            print(f"[placer] Batch {from_idx}-{to_idx} failed after retries; falling back to sequential LLM placement")
            seq = _place_group_sequential_via_llm(
                node,
                world_state=world_state,
                room_half_size=room_half_size,
                query=query,
                llm_fn=llm_fn,
                from_instance=from_idx,
                to_instance=to_idx,
                already_placed_in_group=placed_in_batch,
            )
            placed_in_batch.extend(seq)
            all_results.extend(seq)

    print(f"[placer] {node['id']} ({node['model_name']}×{instances}): placed {len(all_results)} instances")
    return all_results


def _place_group_sequential_via_llm(
    node,
    world_state,
    room_half_size,
    query,
    llm_fn,
    from_instance,
    to_instance,
    already_placed_in_group=None,
):
    """LLM-only sequential fallback for a subset of instances in a grouped node."""
    parent_placed = (_ws_find_all_by_node(world_state, node.get("parent_id")) or [None])[0] if node.get("parent_id") else None
    sz = node.get("size_hint", [1.0, 1.0, 1.0])
    room_w = room_half_size * 2
    results = []
    placed = list(already_placed_in_group or [])

    rel = node.get("relationship") or {}
    rel_type = rel.get("type") if isinstance(rel, dict) else None

    # Numeric parent footprint ranges (center coordinates), for explicit constraint prompting.
    surface_range = None
    if parent_placed and rel_type in (REL_ON_SURFACE, "stacked_on"):
        margin = 0.02
        ppose = parent_placed["Pose"]
        psz = parent_placed["size"]
        pw = float(psz[0]) if len(psz) > 0 else 1.0
        pd = float(psz[2]) if len(psz) > 2 else pw
        self_hx, self_hy = _world_xy_half_sizes(node)
        lim_x = max(0.0, pw / 2.0 - self_hx - margin)
        lim_y = max(0.0, pd / 2.0 - self_hy - margin)
        surface_range = (
            float(ppose.get("x", 0.0)) - lim_x,
            float(ppose.get("x", 0.0)) + lim_x,
            float(ppose.get("y", 0.0)) - lim_y,
            float(ppose.get("y", 0.0)) + lim_y,
        )

    for inst in range(int(from_instance), int(to_instance) + 1):
        # Build an instruction that makes instance numbering explicit and avoids overlap.
        placement_instruction = _rel_description(node)
        placement_instruction += f"\nYou are placing instance {inst} of {int(node.get('instances', 1))} for node '{node.get('id')}'."

        if placed:
            lines = ["Already placed in this group (avoid overlap and stacking):"]
            for p in placed:
                lines.append(f"  - instance {p.get('instance')}: pos=({float(p.get('x', 0.0)):.3f}, {float(p.get('y', 0.0)):.3f}, {float(p.get('z', 0.0)):.3f}), yaw={float(p.get('yaw_deg', 0.0)):.0f}°")
            placement_instruction += "\n" + "\n".join(lines)

        if parent_placed and rel_type in (REL_ON_SURFACE, "stacked_on"):
            pose = parent_placed["Pose"]
            psz = parent_placed["size"]
            parent_top_z = pose["z"] + (psz[1] / 2.0 if len(psz) > 1 else psz[0] / 2.0)
            placement_instruction += (
                f"\nSurface parent '{node.get('parent_id')}' ({parent_placed['Model']}): "
                f"pos=({pose['x']:.2f}, {pose['y']:.2f}, {pose['z']:.2f}), "
                f"size_xyz=({psz[0]:.2f}, {psz[2] if len(psz) > 2 else psz[0]:.2f}, {psz[1] if len(psz) > 1 else psz[0]:.2f})m, "
                f"top_surface_z={parent_top_z:.2f}m. "
                f"Your (x,y) MUST be inside parent's XY footprint."
            )
            if surface_range is not None:
                xmin, xmax, ymin, ymax = surface_range
                placement_instruction += (
                    f"\nCRITICAL numeric bounds: x in [{xmin:.6f}, {xmax:.6f}], y in [{ymin:.6f}, {ymax:.6f}]."
                    "\nTo avoid rounding issues, stay at least 0.005m INSIDE these bounds (not exactly on the edge)."
                )

        sw = float(sz[0]) if len(sz) > 0 else 1.0
        sd = float(sz[2]) if len(sz) > 2 else sw
        sh = float(sz[1]) if len(sz) > 1 else sw
        prompt = _SINGLE_PLACEMENT_PROMPT.format(
            model_name=node["model_name"],
            size_w=sw, size_d=sd, size_h=sh,
            size_w_half=sw / 2.0, size_d_half=sd / 2.0, size_h_half=sh / 2.0,
            placement_instruction=placement_instruction,
            room_w=room_w, room_l=room_w, room_half=room_half_size,
            remaining=_ws_remaining_summary(world_state),
            world_state=_ws_summary(world_state),
        )

        min_clearance = _min_clearance_for_group(node)
        last_error = None
        for attempt in range(8):
            try:
                raw = llm_fn(prompt, query)
                if isinstance(raw, str):
                    raw = json.loads(raw)
                ok, err = _validate_pos(raw, room_half_size)
                if not ok:
                    last_error = f"Invalid position: {err}"
                    raise ValueError(last_error)
                if parent_placed and rel_type in (REL_ON_SURFACE, "stacked_on"):
                    ok_fp, err_fp = _validate_xy_within_parent_footprint(raw.get("x", 0.0), raw.get("y", 0.0), node, parent_placed)
                    if not ok_fp:
                        last_error = f"Surface bounds violation: {err_fp}"
                        raise ValueError(last_error)

                candidate = dict(raw)
                candidate["instance"] = inst
                ok_clear, err_clear = _validate_min_clearance_xy([candidate], min_clearance=min_clearance, already=placed)
                if not ok_clear:
                    last_error = err_clear
                    raise ValueError(last_error)

                results.append(candidate)
                placed.append(candidate)
                break
            except Exception as e:
                msg = str(e)
                prompt += f"\nPrevious attempt invalid: {msg}. Try again."
                if attempt == 2:
                    prompt += (
                        "\nRELAXED FALLBACK (still LLM-only): If the reference alignment keeps failing, "
                        "ignore the reference and choose ANY valid (x,y) within the numeric surface bounds "
                        "that avoids overlap with already placed instances."
                    )
        else:
            raise RuntimeError(
                f"LLM sequential placement failed for node {node.get('id')} instance {inst}: {last_error}"
            )

    return results


def _place_anchor_batch_node(node, world_state, room_half_size, query, llm_fn):
    """Разместить групповой anchor узел с instances>1 батчами по ≤6."""
    instances = node["instances"]
    batch_size = 6
    room_w = room_half_size * 2
    sz = node.get("size_hint", [1.0, 1.0, 1.0])
    rel_desc = _rel_description(node)
    future_context_text = _ws_remaining_summary(world_state)

    all_results = []
    batches = [list(range(i, min(i + batch_size, instances))) for i in range(0, instances, batch_size)]
    for batch_idxs in batches:
        from_idx = batch_idxs[0] + 1
        to_idx = batch_idxs[-1] + 1
        batch_count = len(batch_idxs)
        prompt = _ANCHOR_BATCH_PLACEMENT_PROMPT.format(
            batch_count=batch_count,
            model_name=node["model_name"],
            from_idx=from_idx, to_idx=to_idx, total=instances,
            size_w=sz[0], size_d=sz[2] if len(sz) > 2 else sz[0], size_h=sz[1] if len(sz) > 1 else sz[0],
            relationship_desc=rel_desc,
            future_context=future_context_text,
            world_state=_ws_summary(world_state),
            room_w=room_w, room_l=room_w, room_half=room_half_size,
        )

        for attempt in range(3):
            try:
                raw = llm_fn(prompt, query)
                if isinstance(raw, str):
                    raw = json.loads(raw)
                if isinstance(raw, dict) and "positions" in raw:
                    raw = raw["positions"]
                if not isinstance(raw, list) or len(raw) != batch_count:
                    raise ValueError(f"Expected list of {batch_count}, got {type(raw).__name__} len={len(raw) if isinstance(raw, list) else '?'}")
                valid_results = []
                for item in raw:
                    ok, err = _validate_pos(item, room_half_size)
                    if not ok:
                        raise ValueError(f"Invalid position: {err}")
                    valid_results.append(item)
                all_results.extend(valid_results)
                break
            except Exception as e:
                print(f"[placer] Anchor batch attempt {attempt+1} failed: {e}")
                if attempt == 2:
                    print(f"[placer] Using fallback positions for anchor batch {from_idx}-{to_idx}")
                    for idx in batch_idxs:
                        fb = _fallback_pos(node, None, room_half_size)
                        fb["instance"] = idx + 1
                        fb["x"] += idx * 0.8
                        all_results.append(fb)
                prompt += f"\nPrevious attempt error: {e}. Return exactly {batch_count} items."

    print(f"[placer] {node['id']} ({node['model_name']}×{instances}): placed {len(all_results)} anchor instances")
    return all_results


_VALIDATOR_MAX_ROUNDS = 3


def _validate_placement_via_llm(node, candidate, world_state, room_half_size, query, llm_fn, history=None):
    """Один раунд LLM-валидации позиции candidate для node.

    Возвращает (is_valid: bool, fix_dict: dict|None, reason: str).
    """
    sz = node.get("size_hint", [1.0, 1.0, 1.0])
    sw = float(sz[0]) if len(sz) > 0 else 1.0
    sd = float(sz[2]) if len(sz) > 2 else sw
    sh = float(sz[1]) if len(sz) > 1 else sw

    cx = float(candidate.get("x", 0.0))
    cy = float(candidate.get("y", 0.0))
    cz = float(candidate.get("z", 0.0))
    yaw = float(candidate.get("yaw_deg", 0.0))

    history_block = ""
    if history:
        history_lines = ["\nPREVIOUS ROUNDS for this object (do NOT repeat the same fix; pick a different direction):"]
        for i, h in enumerate(history, start=1):
            history_lines.append(
                f"  round {i}: tried ({h['x']:.3f}, {h['y']:.3f}, {h['z']:.3f}, yaw={h['yaw_deg']:.0f}°) "
                f"— rejected because: {h['reason']}"
            )
        history_block = "\n".join(history_lines)

    parent_id = node.get("parent_id")
    parent_placed = (_ws_find_all_by_node(world_state, parent_id) or [None])[0] if parent_id else None
    parent_block = ""
    if parent_placed:
        ppose = parent_placed["Pose"]
        psz = parent_placed["size"]
        parent_block = (
            f"\nParent surface '{parent_id}' ({parent_placed['Model']}): "
            f"center=({ppose['x']:.3f}, {ppose['y']:.3f}, {ppose['z']:.3f}), "
            f"size_xyz=({psz[0]:.3f}, {psz[2] if len(psz) > 2 else psz[0]:.3f}, "
            f"{psz[1] if len(psz) > 1 else psz[0]:.3f})m. "
            f"Object MUST stay within parent's XY footprint "
            f"(X within ±{psz[0]/2:.3f} of parent.x, Y within ±{(psz[2] if len(psz) > 2 else psz[0])/2:.3f} of parent.y)."
        )

    rel = node.get("relationship") or {}
    ref_id = rel.get("reference") if isinstance(rel, dict) else None
    reference_block = ""
    if ref_id:
        ref_objs = _ws_find_all_by_node(world_state, ref_id)
        if ref_objs:
            reference_block = "\nReference object(s) for spatial intent:"
            for r in ref_objs:
                reference_block += "\n  " + _ws_describe_object(r)

    intent = _rel_description(node)

    prompt = _VALIDATE_PLACEMENT_PROMPT.format(
        model_name=node["model_name"], node_id=node["id"],
        cx=cx, cy=cy, cz=cz, yaw=yaw,
        sw=sw, sd=sd, sh=sh,
        xmin=cx - sw / 2.0, xmax=cx + sw / 2.0,
        ymin=cy - sd / 2.0, ymax=cy + sd / 2.0,
        zmin=cz - sh / 2.0, zmax=cz + sh / 2.0,
        intent=intent,
        parent_block=parent_block,
        reference_block=reference_block,
        history_block=history_block,
        world_state=_ws_summary(world_state),
        room_half=room_half_size,
    )

    try:
        raw = llm_fn(prompt, query)
        if isinstance(raw, str):
            raw = json.loads(raw)
        if not isinstance(raw, dict):
            return True, None, "validator returned non-dict; skipping"
        if raw.get("valid") is True:
            return True, None, ""
        fix = raw.get("fix")
        reason = str(raw.get("reason", "no reason given"))
        if not isinstance(fix, dict):
            return True, None, f"validator said invalid but no fix; ignoring ({reason})"
        # carry over missing fields from candidate
        merged = dict(candidate)
        for k in ("x", "y", "z", "yaw_deg"):
            if k in fix:
                merged[k] = float(fix[k])
        ok, err = _validate_pos(merged, room_half_size)
        if not ok:
            return True, None, f"fix out of bounds ({err}); keeping original"
        return False, merged, reason
    except Exception as e:
        return True, None, f"validator error: {e}"


def _place_all(graph, world_state, query, llm_fn):
    """Разместить все объекты согласно иерархии графа."""
    room_half_size = world_state["room_half_size"]
    max_level = graph["metadata"]["max_level"]

    def _place_then_validate(node):
        if node["instances"] == 1:
            results = _place_single_node(node, world_state, room_half_size, query, llm_fn)
        else:
            results = _place_batch_node(node, world_state, room_half_size, query, llm_fn)
        # Валидатор работает только для одиночных узлов (instances=1), что и есть наш случай.
        if len(results) == 1 and node.get("level", 0) > 0:
            history = []
            for round_idx in range(_VALIDATOR_MAX_ROUNDS):
                ok, fix, reason = _validate_placement_via_llm(
                    node, results[0], world_state, room_half_size, query, llm_fn,
                    history=history,
                )
                if ok:
                    if round_idx == 0:
                        print(f"[validator] {node['id']}: OK")
                    else:
                        print(f"[validator] {node['id']}: OK after {round_idx} fix(es)")
                    break
                if fix is None:
                    print(f"[validator] {node['id']} round {round_idx + 1}: {reason} (no fix applied)")
                    break
                print(f"[validator] {node['id']} round {round_idx + 1}: {reason} → "
                      f"({fix['x']:.3f}, {fix['y']:.3f}, {fix['z']:.3f}, yaw={fix['yaw_deg']:.0f}°)")
                history.append({**results[0], "reason": reason})
                results = [fix]
            else:
                print(f"[validator] {node['id']}: gave up after {_VALIDATOR_MAX_ROUNDS} rounds, accepting last fix")
        _register_results(node, results, world_state)

    for level in range(max_level + 1):
        nodes_at_level = sorted(
            [n for n in graph["nodes"] if n["level"] == level],
            key=lambda n: (n.get("place_order", 1), n["id"]),
        )

        if level == 0:
            for node in nodes_at_level:
                if node["instances"] > 1:
                    results = _place_anchor_batch_node(node, world_state, room_half_size, query, llm_fn)
                    _register_results(node, results, world_state)
                else:
                    _place_then_validate(node)
        else:
            groups = _collect_groups(nodes_at_level)
            for parent_id in sorted(groups.keys()):
                group_nodes = groups[parent_id]
                ordered = _topological_sort_group(group_nodes)
                for node in ordered:
                    _place_then_validate(node)


def _topological_sort_group(group_nodes):
    """Сортировка узлов одной группы с учётом depends_on.

    depends_on указывает, что узел зависит от позиций другого узла.
    → зависимый размещается позже.
    Tiebreaker: place_order, затем id.
    """
    by_id = {n["id"]: n for n in group_nodes}
    in_group_ids = set(by_id.keys())

    # Граф зависимостей: dep → [зависящие]
    visited = set()
    result = []

    def visit(n):
        if n["id"] in visited:
            return
        visited.add(n["id"])
        dep_id = n.get("depends_on")
        if dep_id and dep_id in in_group_ids:
            visit(by_id[dep_id])
        result.append(n)

    # Стартовать с узлов в порядке (place_order, id) для детерминизма
    for n in sorted(group_nodes, key=lambda x: (x.get("place_order", 1), x["id"])):
        visit(n)
    return result


def _register_results(node, results, world_state):
    """Записать результаты размещения в world_state."""
    sz = node.get("size_hint", [1.0, 1.0, 1.0])
    obj_h = sz[1] if len(sz) > 1 else sz[0]

    # Определить корректный z детерминистически
    rel = node.get("relationship") or {}
    rel_type = rel.get("type", "")
    parent_placed = None
    if node.get("parent_id"):
        candidates = _ws_find_all_by_node(world_state, node["parent_id"])
        parent_placed = candidates[0] if candidates else None

    # parent_id определяет вертикаль: если есть parent — сидим у него на верху;
    # если parent_id=None — это пол. relationship.type на Z больше не влияет
    # (он отвечает только за X/Y reasoning через reference/side/distance).
    # Исключение: "around" — стулья вокруг стола стоят на полу, не на столе.
    if rel_type == "around":
        forced_z = obj_h / 2.0
    elif parent_placed:
        psz = parent_placed["size"]
        ph = psz[1] if len(psz) > 1 else psz[0]
        pz = parent_placed["Pose"]["z"]
        forced_z = pz + ph / 2.0 + obj_h / 2.0
    else:
        forced_z = obj_h / 2.0  # Floor object (anchors)

    for i, r in enumerate(results):
        instance_id = f"{node['id']}_inst_{i}"
        z = forced_z if forced_z is not None else float(r.get("z", obj_h / 2.0))
        pose = {"x": float(r.get("x", 0.0)), "y": float(r.get("y", 0.0)), "z": z}
        yaw = float(r.get("yaw_deg", 0.0))
        _ws_add(world_state, instance_id, node["model_name"], sz, pose, yaw)
    # Обновить remaining (decrement count for this model_name)
    placed_count = len(results)
    new_remaining = []
    for rem in world_state.get("remaining", []):
        if rem.get("model_name") != node.get("model_name"):
            new_remaining.append(rem)
            continue
        new_cnt = int(rem.get("count", 0)) - placed_count
        if new_cnt > 0:
            rem2 = dict(rem)
            rem2["count"] = new_cnt
            new_remaining.append(rem2)
    world_state["remaining"] = new_remaining


# ---------------------------------------------------------------------------
# Merge placed objects back to output format
# ---------------------------------------------------------------------------

def _merge_to_output(world_state, models, graph):
    """Сопоставить размещённые объекты с оригинальными моделями."""
    result = []
    # Для каждого типа: очередь оригинальных моделей
    model_queues = {}
    for m in models:
        name = m.get("Model", "")
        model_queues.setdefault(name, []).append(m)

    for i, placed in enumerate(world_state["placed"]):
        model_name = placed["Model"]
        queue = model_queues.get(model_name, [])
        original = queue.pop(0) if queue else {}

        obj_uuid = original.get("uuid", placed["id"])
        safe_uuid = re.sub(r"[^a-zA-Z0-9_]+", "_", str(obj_uuid))
        save_fn = f"{safe_uuid}_{i}"

        sz = placed["size"]
        pose = placed["Pose"]
        yaw = placed["yaw_deg"]

        merged = dict(original)
        merged.update({
            "Model": model_name,
            "uuid": obj_uuid,
            "model_loc": original.get("model_loc", ""),
            "save_fn": save_fn,
            "_up_axis": original.get("_up_axis", "y"),
            "size": sz if isinstance(sz, list) else [sz.get("width", 1.0), sz.get("height", 1.0), sz.get("length", 1.0)],
            "is_static": original.get("is_static", True),
            "Pose": {"x": pose["x"], "y": pose["y"], "z": pose["z"]},
            "yaw_deg": yaw,
        })
        result.append(merged)

        print(
            f"[scene_planner]   {model_name}: "
            f"pos=({pose['x']:.2f}, {pose['y']:.2f}, {pose['z']:.2f}), "
            f"yaw={yaw:.0f}°"
        )

    return result


# ---------------------------------------------------------------------------
# Collision checking and repair (from creator.placement.physics)
# ---------------------------------------------------------------------------

def check_collisions(placed_models, collision_margin=0.01):
    """Check for collisions between objects using OBB + SAT."""
    collisions = []
    n = len(placed_models)
    for i in range(n):
        for j in range(i + 1, n):
            if not _z_intervals_overlap(placed_models[i], placed_models[j]):
                continue
            obb_i = model_to_obb(placed_models[i], inflation=collision_margin)
            obb_j = model_to_obb(placed_models[j], inflation=collision_margin)
            if obb_overlap(obb_i, obb_j, margin=0.0):
                depth = obb_overlap_depth(obb_i, obb_j)
                name_i = str(placed_models[i].get("Model") or placed_models[i].get("name") or f"object_{i}")
                name_j = str(placed_models[j].get("Model") or placed_models[j].get("name") or f"object_{j}")
                pose_i = placed_models[i].get("Pose") or {}
                pose_j = placed_models[j].get("Pose") or {}
                collisions.append({
                    "object_a": name_i,
                    "object_b": name_j,
                    "overlap_depth": depth,
                    "position_a": {
                        "x": float(pose_i.get("x", 0.0)),
                        "y": float(pose_i.get("y", 0.0)),
                        "z": float(pose_i.get("z", 0.0)),
                    },
                    "position_b": {
                        "x": float(pose_j.get("x", 0.0)),
                        "y": float(pose_j.get("y", 0.0)),
                        "z": float(pose_j.get("z", 0.0)),
                    },
                })
    return collisions


def _llm_replan_layout(placed_models, semantic_plan, room_half_size, collision_details, max_retries=3, scene_graph=None):
    """LLM fallback for replanning object layout when gradient resolution fails.
    
    Args:
        scene_graph: Optional scene graph with hierarchy info. Objects on surfaces
                     (level > 0) won't be repositioned.
    """
    print(f"[llm_replan_layout] Invoking LLM fallback for {len(placed_models)} objects")
    print(f"[llm_replan_layout] Reason: {len(collision_details)} collision(s) after gradient resolution")
    
    # Построить set объектов на поверхности (не должны двигаться)
    surface_objects = set()
    if scene_graph:
        for node in scene_graph.get("nodes", []):
            if node.get("level", 0) > 0:
                rel = node.get("relationship", {})
                if rel.get("type") in ("on_surface", "stacked_on"):
                    surface_objects.add(node["model_name"])
        if surface_objects:
            print(f"[llm_replan_layout] Surface objects (won't be repositioned): {surface_objects}")

    objects_info = []
    for i, obj in enumerate(placed_models):
        model_name = obj.get("Model", f"object_{i}")
        size = obj.get("size", [1.0, 1.0, 1.0])
        pose = obj.get("Pose", {})
        current_pos = (float(pose.get("x", 0.0)), float(pose.get("y", 0.0)), float(pose.get("z", 0.0)))
        objects_info.append({"index": i, "name": model_name, "size": size, "current_position": current_pos})

    collision_summary = []
    for collision in collision_details:
        collision_summary.append({
            "object_a": collision["object_a"],
            "object_b": collision["object_b"],
            "overlap_depth": collision["overlap_depth"],
        })

    system_prompt = """You are a 3D scene layout optimizer. Reposition objects to eliminate overlaps while preserving room boundaries.

Return JSON:
{
  "positions": [
    {"index": 0, "x": 1.5, "y": 0.0, "z": 0.4},
    ...
  ],
  "reasoning": "Brief explanation"
}

Rules:
- Eliminate ALL overlaps
- Keep all objects within room boundaries
- Maintain minimum 0.05m clearance
- Keep Z positions unchanged"""

    user_prompt = f"""Room boundaries: [-{room_half_size}, {room_half_size}] in both X and Y

Objects:
{json.dumps(objects_info, indent=2)}

Collisions to eliminate:
{json.dumps(collision_summary, indent=2)}

Provide new X,Y positions that eliminate all overlaps."""

    for attempt in range(max_retries):
        try:
            print(f"[llm_replan_layout] LLM attempt {attempt + 1}/{max_retries}...")
            raw = _llm_request(system=system_prompt, user=user_prompt, model=DEFAULT_MODEL)

            if isinstance(raw, str):
                response = json.loads(raw)
            elif isinstance(raw, dict):
                response = raw
            else:
                raise ValueError(f"Unexpected response type: {type(raw)}")

            if not isinstance(response, dict) or "positions" not in response:
                print(f"[llm_replan_layout] Invalid response format (attempt {attempt + 1})")
                if attempt < max_retries - 1:
                    user_prompt += "\n\nPrevious attempt failed. Return valid JSON with 'positions' array."
                    continue
                else:
                    return placed_models, False

            new_positions = response["positions"]
            reasoning = response.get("reasoning", "No reasoning provided")
            print(f"[llm_replan_layout] LLM reasoning: {reasoning}")

            replanned_models = []
            for obj in placed_models:
                replanned_models.append(dict(obj))

            for pos_update in new_positions:
                idx = pos_update.get("index")
                if idx is None or idx < 0 or idx >= len(replanned_models):
                    continue
                
                # НОВОЕ: Не изменять позиции объектов на поверхности
                model_name = replanned_models[idx].get("Model", "")
                if model_name in surface_objects:
                    print(f"[llm_replan_layout] DEBUG: Skipping reposition for {model_name} (surface object)")
                    continue
                
                pose = dict(replanned_models[idx].get("Pose", {}))
                pose["x"] = float(pos_update.get("x", pose.get("x", 0.0)))
                pose["y"] = float(pos_update.get("y", pose.get("y", 0.0)))
                replanned_models[idx]["Pose"] = pose

            final_collisions = check_collisions(replanned_models, collision_margin=0.01)
            if final_collisions:
                print(f"[llm_replan_layout] LLM solution still has {len(final_collisions)} collision(s)")
                if attempt < max_retries - 1:
                    remaining = [f"{c['object_a']} <-> {c['object_b']}" for c in final_collisions]
                    user_prompt += f"\n\nPrevious solution had collisions: {remaining}. Increase spacing."
                    continue
                else:
                    print("[llm_replan_layout] WARNING: Using LLM solution with collisions (best effort)")
                    return replanned_models, True

            print("[llm_replan_layout] SUCCESS: LLM replanning eliminated all collisions")
            return replanned_models, True

        except Exception as e:
            print(f"[llm_replan_layout] Error during attempt {attempt + 1}: {e}")
            if attempt < max_retries - 1:
                continue
            else:
                return placed_models, False

    return placed_models, False


def validate_and_repair_layout(placed_models, min_z=0.01, room_half_size=5.0, repair_iters=8, push_step=0.06, semantic_plan=None, scene_graph=None):
    """Validate layout and resolve overlaps using gradient-based resolution.
    
    Args:
        scene_graph: Optional scene graph with hierarchy info. Objects on surfaces
                     (level > 0) won't be moved independently during collision resolution.
    """
    print(f"[validate_and_repair] Starting with {len(placed_models)} models...")
    repaired = []
    for m in placed_models:
        item = dict(m)
        pose = dict(item.get("Pose") or {"x": 0.0, "y": 0.0, "z": min_z})
        z = float(pose.get("z", min_z))
        if z < min_z:
            z = min_z
        pose["z"] = z
        item["Pose"] = pose
        repaired.append(item)

    num_objects = len(repaired)
    base_iterations = repair_iters * 20
    if num_objects > 5:
        bonus_iterations = (num_objects - 5) * 50
        adaptive_iterations = base_iterations + bonus_iterations
    else:
        adaptive_iterations = base_iterations
    adaptive_iterations = min(adaptive_iterations, 500)

    print(f"[validate_and_repair] Using adaptive iteration limit: {adaptive_iterations} iterations (base={base_iterations}, objects={num_objects})")

    total_footprint_area = 0.0
    for m in repaired:
        size = m.get("size", [1.0, 1.0, 1.0])
        width = max(0.01, float(size[0]) if len(size) > 0 else 1.0)
        depth = max(0.01, float(size[2]) if len(size) > 2 else 1.0)
        total_footprint_area += width * depth

    available_area = (2.0 * room_half_size) ** 2
    density_ratio = total_footprint_area / max(0.01, available_area)
    print(f"[validate_and_repair] Density check: footprint={total_footprint_area:.2f}m², available={available_area:.2f}m², ratio={density_ratio:.2f}")

    # Ранний выход: если плотность > 0.8 — сразу LLM (градиент не сойдётся)
    if density_ratio > 0.8:
        print("[validate_and_repair] INFEASIBILITY DETECTED: density > 0.8, invoking LLM fallback immediately...")
        replanned_models, llm_success = _llm_replan_layout(
            placed_models=repaired,
            semantic_plan=semantic_plan or {},
            room_half_size=room_half_size,
            collision_details=[],
            max_retries=3,
            scene_graph=scene_graph,  # ← Передаём граф!
        )
        if llm_success:
            print("[validate_and_repair] LLM fallback succeeded")
            repaired = replanned_models
            final_collisions = check_collisions(repaired, collision_margin=0.02)
            if final_collisions:
                print(f"[validate_and_repair] WARNING: {len(final_collisions)} collision(s) remain after LLM fallback")
            else:
                print("[validate_and_repair] SUCCESS: All collisions eliminated by LLM fallback")
            return repaired
        else:
            print("[validate_and_repair] LLM fallback failed, falling through to gradient resolution")

    repaired, converged = gradient_resolve_overlaps(
        repaired,
        room_half_size=room_half_size,
        iterations=adaptive_iterations,
        step_size=push_step,
        collision_margin=0.02,
        semantic_plan=semantic_plan,
        scene_graph=scene_graph,  # ← Передаём граф!
    )

    if not converged:
        print("[validate_and_repair] Gradient resolution did not converge")

    collisions = check_collisions(repaired, collision_margin=0.02)
    if collisions:
        print(f"[validate_and_repair] CONVERGENCE FAILURE: {len(collisions)} collision(s) remain")
        print("[validate_and_repair] Invoking LLM fallback...")
        replanned_models, llm_success = _llm_replan_layout(
            placed_models=repaired,
            semantic_plan=semantic_plan or {},
            room_half_size=room_half_size,
            collision_details=collisions,
            max_retries=3,
            scene_graph=scene_graph,  # ← Передаём граф!
        )
        if llm_success:
            print("[validate_and_repair] LLM fallback succeeded")
            repaired = replanned_models
            final_collisions = check_collisions(repaired, collision_margin=0.02)
            if final_collisions:
                print(f"[validate_and_repair] WARNING: {len(final_collisions)} collision(s) remain after LLM fallback")
            else:
                print("[validate_and_repair] SUCCESS: All collisions eliminated by LLM fallback")
        else:
            print("[validate_and_repair] LLM fallback failed")
    else:
        print("[validate_and_repair] SUCCESS: All collisions resolved after gradient resolution")

    return repaired


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def generate_placement_plan(models, room_half_size, room_type, query, prompt_model_fn=None, save_outputs=True):
    """Stage 4: generate placement plan for objects.

    Accepts: list of models with size/scale, room half-size, room type, query.
    Returns: list of models with added fields Pose, yaw_deg, is_static, size (final).
    
    Args:
        save_outputs: If True, saves intermediate results to output/ directory
    """
    if not models:
        print("[scene_planner] WARNING: empty models list")
        return []

    model_id = DEFAULT_MODEL

    def llm_fn(prompt, query_text):
        if prompt_model_fn is not None:
            raw = prompt_model_fn(prompt, query_text, model_id)
            if isinstance(raw, (dict, list)):
                return raw
            if isinstance(raw, str):
                return json.loads(raw)
            raise ValueError(f"Unexpected type: {type(raw)}")
        raw = _llm_request(system=prompt, user=query_text, model=model_id)
        if isinstance(raw, str):
            try:
                return json.loads(raw)
            except json.JSONDecodeError:
                return raw
        return raw

    print(f"[scene_planner] Stage 4: {len(models)} objects, room={room_half_size*2:.1f}m, type={room_type}")

    # Фаза 1: Построение графа
    graph = _build_scene_graph(models, room_half_size, query, llm_fn)
    
    # Сохранить граф сцены
    if save_outputs:
        _save_stage_output({
            "stage": "1_scene_graph",
            "query": query,
            "room_half_size": room_half_size,
            "room_type": room_type,
            "graph": graph,
        }, "stage1_scene_graph")

    # Инъекция size_hint в узлы графа
    for node in graph["nodes"]:
        for m in models:
            if m.get("Model") == node["model_name"]:
                node["size_hint"] = m.get("size", [1.0, 1.0, 1.0])
                break

    # Инициализировать world_state с remaining
    world_state = make_world_state(room_half_size)
    type_counts = {}
    for m in models:
        name = m.get("Model", "")
        type_counts[name] = type_counts.get(name, 0) + 1
    world_state["remaining"] = [
        {"model_name": name, "count": cnt, "size": next(
            (m.get("size", [1.0, 1.0, 1.0]) for m in models if m.get("Model") == name), [1.0, 1.0, 1.0]
        )}
        for name, cnt in type_counts.items()
    ]

    # Фаза 2: Иерархическое размещение
    _place_all(graph, world_state, query, llm_fn)
    print(f"[scene_planner] Placed {len(world_state['placed'])} objects")
    
    # Сохранить world_state после размещения
    if save_outputs:
        _save_stage_output({
            "stage": "2_hierarchical_placement",
            "world_state": world_state,
            "graph": graph,
        }, "stage2_world_state")

    # Merge
    result = _merge_to_output(world_state, models, graph)
    print(f"[scene_planner] Stage 4 complete: {len(result)} models with placement")
    
    # Сохранить результат перед коррекцией коллизий
    if save_outputs:
        _save_stage_output({
            "stage": "3_merged_output",
            "models": result,
            "room_half_size": room_half_size,
        }, "stage3_merged_output")

    # ВРЕМЕННО ЗАКОММЕНТИРОВАНО ДЛЯ ТЕСТИРОВАНИЯ
    # result = validate_and_repair_layout(
    #     result,
    #     room_half_size=room_half_size,
    #     scene_graph=graph
    # )
    
    # Сохранить финальный результат после коррекции
    if save_outputs:
        _save_stage_output({
            "stage": "4_final_layout",
            "models": result,
            "room_half_size": room_half_size,
            "query": query,
            "room_type": room_type,
        }, "stage4_final_layout")

    return result
