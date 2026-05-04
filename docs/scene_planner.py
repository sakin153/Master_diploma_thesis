# Stage 4 - generates placement plan via LLM + enforcement pipeline.
# Reads models with size/scale, returns models with Pose, yaw_deg, is_static.

import json
import math
import re
from dataclasses import dataclass

from project.llm_request import request as _llm_request, DEFAULT_MODEL


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
    return not ((za + ha < zb - hb) or (zb + hb < za - ha))


def gradient_resolve_overlaps(models, room_half_size=5.0, iterations=200, step_size=0.05, collision_margin=0.01, semantic_plan=None):
    """Move objects apart using gradient-based overlap resolution."""
    resolved = [dict(m) for m in models]
    n = len(resolved)
    overlap_history = []
    stagnation_threshold = 20

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
# Prompt templates
# ---------------------------------------------------------------------------

_SEMANTIC_PLAN_PROMPT = """
CRITICAL: You MUST respond ONLY in English. Do NOT use any other language (Chinese, Russian, etc.) in your response.
CRITICAL: Your response MUST be valid JSON only. Do NOT include any text, comments, or explanations outside the JSON structure.

You are a professional scene composer and architect for a MuJoCo physics simulator.
Your role is to create detailed placement instructions for every object in the scene.

User query: {query}

Room dimensions:
- Width: {room_width} meters
- Length: {room_length} meters
- Height: {room_height} meters

Available models for this scene:
{models_str}

---

## Your Role as Scene Composer

You have COMPLETE CONTROL over the scene composition:
- Decide exact positions for every object (absolute coordinates or relative to other objects)
- Decide exact orientations for every object (absolute angles or facing directions)
- Specify distances between objects
- Define spatial relationships ("chair faces table", "lamp on table surface")

The system will execute your instructions EXACTLY as specified without modifications.

---

## Output Format

Generate a JSON semantic plan with this structure:

```json
{{
  "schema_version": "1.0",
  "room_size": {{
    "width": {room_width},
    "length": {room_length},
    "height": {room_height}
  }},
  "objects": [
    // Array of objects with placement instructions
  ]
}}
```

---

## Object Specification

Each object MUST have these fields:

**Required fields:**
- `id`: Unique identifier (e.g., "table_1", "chair_1", "chair_2")
- `Model`: Exact model name from the available models list above
- `type`: "furniture" (tables, chairs, sofas, beds) or "small_object" (plates, cups, books, lamps)
- `size`: Object dimensions in meters: {{"width": float, "length": float, "height": float}}
- `is_static`: true for furniture, false for small movable objects
- `position`: Position specification (see below)
- `orientation`: Orientation specification (see below)

**Note**: Do NOT include `model_loc` field - the system will add it automatically based on the Model name.

---

## Position Specification

Choose ONE of these two options for each object:

### Option 1: Absolute Position
Use when you want to specify exact coordinates in the room:

```json
"position": {{
  "absolute": {{
    "x": 5.0,    // meters from room origin (center of room)
    "y": 5.0,    // meters from room origin
    "z": 0.0     // height (0.0 for floor objects)
  }}
}}
```

### Option 2: Relative Position
Use when you want to place object relative to another object:

```json
"position": {{
  "relative": {{
    "relative_to": "table_1",        // ID of target object
    "direction": "front",             // Direction from target
    "distance": 0.5,                  // Distance in meters
    "reference_point": "center"       // Point on target to measure from
  }}
}}
```

**Available directions:**
- `"front"`: In front of target's front face
- `"back"`: Behind target's back face
- `"left"`: To the left of target
- `"right"`: To the right of target
- `"front_left"`: 45 degrees between front and left
- `"front_right"`: 45 degrees between front and right
- `"back_left"`: 45 degrees between back and left
- `"back_right"`: 45 degrees between back and right
- `"above"`: Directly above target
- `"below"`: Directly below target

**Available reference points:**
- `"center"`: Geometric center of target object
- `"front_edge"`: Center of front face
- `"back_edge"`: Center of back face
- `"left_edge"`: Center of left face
- `"right_edge"`: Center of right face
- `"top_surface"`: Center of top face (for placing objects on surfaces)

**Distance guidelines:**
- Chair around dining table: 0.5-0.8 meters
- Small object on table surface: use "above" direction with distance 0.0
- Side table next to sofa: 0.3-0.6 meters
- Objects that should touch: 0.0 meters

---

## Orientation Specification

Choose ONE of these two options for each object:

### Option 1: Absolute Orientation
Use when you want to specify exact angles:

```json
"orientation": {{
  "absolute": {{
    "yaw_deg": 0.0,      // Rotation around vertical axis (0-360)
    "pitch_deg": 0.0,    // Usually 0 for furniture
    "roll_deg": 0.0      // Usually 0 for furniture
  }}
}}
```

**Yaw angle reference:**
- 0 degrees: Facing positive X direction
- 90 degrees: Facing positive Y direction
- 180 degrees: Facing negative X direction
- 270 degrees: Facing negative Y direction

### Option 2: Relative Orientation
Use when you want object to face another object:

```json
"orientation": {{
  "relative": {{
    "facing": "table_1",              // ID of target object to face
    "facing_direction": "front",      // Which side of target to face
    "facing_away": false              // Optional: face away instead of toward
  }}
}}
```

**Available facing directions:**
- `"front"`: Face the front of target object
- `"back"`: Face the back of target object
- `"left_side"`: Face the left side of target object
- `"right_side"`: Face the right side of target object

---

## Important Rules

1. **Object Count**: You MUST generate EXACTLY one object specification for EACH entry in the "Available models for this scene" list above.
2. **Unique IDs**: Each object must have a unique ID. Use pattern: `model_type_number` (e.g., "table_1", "chair_1", "chair_2").
3. **Relationships**: When user specifies relationships like "4 chairs per table", create explicit relative positioning for each chair to its specific table.
4. **Z-coordinate**: For floor objects (tables, chairs), use z=0.0 in absolute positioning. For objects on surfaces, use "above" direction with distance 0.0.
5. **Model Names**: Use EXACT model names from the available models list provided above.
6. **Distances**: Be realistic with distances. Chairs around tables: 0.5-0.8m. Objects on surfaces: use "above" with distance 0.0.
7. **Orientation Logic**: Chairs should face their tables. Use relative orientation when objects should face each other.

---

## Your Task

Based on the user query and available models above, generate a complete semantic plan in JSON format.
Generate the semantic plan now:
"""

_PLACEMENT_PRIORITY_PROMPT = """
CRITICAL: You MUST respond ONLY in English. Do NOT use any other language.
CRITICAL: Your response MUST be valid JSON only.

You are a scene planning expert. Your task is to analyze a scene description
and determine the optimal order for placing objects.

User query: {query}

Available objects to place:
{objects_str}

---

## Your Task

Analyze the scene and divide objects into priority groups:

1. **Anchor objects** - Objects that should be placed first with absolute positions
   - Usually: tables, sofas, beds, large furniture
   - These define the main structure of the scene

2. **Dependent objects** - Objects that should be placed relative to anchors
   - Usually: chairs (around tables), lamps (on tables), cushions (on sofas)
   - These are positioned based on anchor objects

---

## Output Format

Generate a JSON response with this structure:

```json
{{
  "anchor_objects": [
    {{"index": 0, "reason": "Main table - defines dining area"}}
  ],
  "dependent_objects": [
    {{"index": 1, "reason": "Chair - should be placed around table"}}
  ]
}}
```

Rules:
1. Use index numbers from the "Available objects" list above (0-indexed).
2. Anchor objects should be large, structural items that define the scene.
3. Dependent objects should be items that naturally relate to anchors.

Now analyze the scene and generate the placement priority plan:
"""


# ---------------------------------------------------------------------------
# SchemaValidator — inline, без импорта из creator
# ---------------------------------------------------------------------------

class _ValidationResult:
    def __init__(self, success, errors):
        self.success = success
        self.errors = errors


class _SchemaValidator:
    SUPPORTED_VERSIONS = ["1.0"]

    def validate_plan(self, plan):
        errors = []
        if not isinstance(plan, dict):
            return _ValidationResult(False, ["Semantic plan must be a dictionary"])

        if "schema_version" not in plan:
            errors.append("Missing required field: schema_version")
        elif plan["schema_version"] not in self.SUPPORTED_VERSIONS:
            errors.append(f"Unsupported schema version: {plan['schema_version']}")

        if "objects" not in plan:
            errors.append("Missing required field: objects")
        elif not isinstance(plan["objects"], list):
            errors.append("Field 'objects' must be a list")
        elif len(plan["objects"]) == 0:
            errors.append("Field 'objects' must contain at least one object")

        if "room_size" not in plan:
            errors.append("Missing required field: room_size")
        elif not isinstance(plan["room_size"], dict):
            errors.append("Field 'room_size' must be a dictionary")
        else:
            for dim in ("width", "length", "height"):
                if dim not in plan["room_size"]:
                    errors.append(f"room_size missing required dimension: {dim}")

        if errors:
            return _ValidationResult(False, errors)

        objects = plan["objects"]
        seen_ids = set()

        for i, obj in enumerate(objects):
            if not isinstance(obj, dict):
                errors.append(f"Object at index {i} must be a dictionary")
                continue

            for field in ("id", "Model", "type", "size", "is_static", "position", "orientation"):
                if field not in obj:
                    errors.append(f"Object at index {i} missing required field: {field}")

            if "id" in obj:
                if obj["id"] in seen_ids:
                    errors.append(f"Duplicate object ID: {obj['id']}")
                seen_ids.add(obj["id"])

            if "size" in obj and isinstance(obj["size"], dict):
                for dim in ("width", "length", "height"):
                    if dim not in obj["size"]:
                        errors.append(f"Object at index {i}: size missing dimension '{dim}'")

            # position: exactly one of absolute/relative
            if "position" in obj and isinstance(obj["position"], dict):
                pos = obj["position"]
                has_abs = "absolute" in pos and pos["absolute"] is not None
                has_rel = "relative" in pos and pos["relative"] is not None
                if not has_abs and not has_rel:
                    errors.append(f"Object at index {i}: position must specify 'absolute' or 'relative'")
                elif has_abs and has_rel:
                    errors.append(f"Object at index {i}: position cannot specify both 'absolute' and 'relative'")

            # orientation: exactly one of absolute/relative
            if "orientation" in obj and isinstance(obj["orientation"], dict):
                ori = obj["orientation"]
                has_abs = "absolute" in ori and ori["absolute"] is not None
                has_rel = "relative" in ori and ori["relative"] is not None
                if not has_abs and not has_rel:
                    errors.append(f"Object at index {i}: orientation must specify 'absolute' or 'relative'")
                elif has_abs and has_rel:
                    errors.append(f"Object at index {i}: orientation cannot specify both 'absolute' and 'relative'")

        # validate cross-references
        for i, obj in enumerate(objects):
            if not isinstance(obj, dict):
                continue
            pos = obj.get("position", {})
            if isinstance(pos.get("relative"), dict):
                target = pos["relative"].get("relative_to")
                if target and target not in seen_ids:
                    errors.append(
                        f"Object at index {i}: relative position references non-existent ID '{target}'"
                    )
            ori = obj.get("orientation", {})
            if isinstance(ori.get("relative"), dict):
                target = ori["relative"].get("facing")
                if target and target not in seen_ids:
                    errors.append(
                        f"Object at index {i}: relative orientation references non-existent ID '{target}'"
                    )

        if errors:
            return _ValidationResult(False, errors)
        return _ValidationResult(True, [])


# ---------------------------------------------------------------------------
# DistanceResolver — inline
# ---------------------------------------------------------------------------

class _DistanceResolver:
    # Base direction vectors in XY plane (unit vectors)
    DIRECTION_VECTORS = {
        "front":       (0.0,   1.0),
        "back":        (0.0,  -1.0),
        "left":        (-1.0,  0.0),
        "right":       (1.0,   0.0),
        "front_left":  (-0.707, 0.707),
        "front_right": (0.707,  0.707),
        "back_left":   (-0.707, -0.707),
        "back_right":  (0.707,  -0.707),
        "above":       (0.0,   0.0),
        "below":       (0.0,   0.0),
    }

    def resolve_positions(self, plan):
        resolved = {
            "schema_version": plan["schema_version"],
            "room_size": plan["room_size"],
            "objects": [dict(o) for o in plan["objects"]],
            "metadata": plan.get("metadata", {}),
        }
        objects = resolved["objects"]

        deps = self._build_deps(objects)
        self._check_cycles(deps)
        order = self._topo_sort(objects, deps)

        for obj_id in order:
            obj = self._find(objects, obj_id)
            if obj is None:
                continue
            pos = obj["position"]
            if pos.get("absolute") is not None:
                continue
            rel = pos.get("relative")
            if rel is None:
                continue
            target = self._find(objects, rel["relative_to"])
            if target is None:
                raise ValueError(f"Object '{obj_id}' references non-existent '{rel['relative_to']}'")
            abs_pos = self._calc_abs(obj, target, rel)
            obj["position"] = {"absolute": abs_pos, "relative": None}

        return resolved

    def _build_deps(self, objects):
        deps = {}
        for obj in objects:
            oid = obj["id"]
            deps[oid] = set()
            rel = obj.get("position", {}).get("relative")
            if rel:
                target = rel.get("relative_to")
                if target:
                    deps[oid].add(target)
        return deps

    def _check_cycles(self, deps):
        visited = set()
        rec = set()

        def dfs(node, path):
            visited.add(node)
            rec.add(node)
            path.append(node)
            for nb in deps.get(node, set()):
                if nb not in visited:
                    dfs(nb, path)
                elif nb in rec:
                    cycle = path[path.index(nb):] + [nb]
                    raise RuntimeError(f"Circular dependency: {' -> '.join(cycle)}")
            rec.remove(node)
            path.pop()

        for node in deps:
            if node not in visited:
                dfs(node, [])

    def _topo_sort(self, objects, deps):
        abs_ids = []
        rel_ids = []
        for obj in objects:
            if obj.get("position", {}).get("absolute") is not None:
                abs_ids.append(obj["id"])
            else:
                rel_ids.append(obj["id"])

        in_deg = {oid: 0 for oid in rel_ids}
        for oid in rel_ids:
            for dep in deps.get(oid, set()):
                if dep in in_deg:
                    in_deg[oid] += 1

        queue = [oid for oid in rel_ids if in_deg[oid] == 0]
        sorted_rel = []
        while queue:
            cur = queue.pop(0)
            sorted_rel.append(cur)
            for oid in rel_ids:
                if cur in deps.get(oid, set()):
                    in_deg[oid] -= 1
                    if in_deg[oid] == 0:
                        queue.append(oid)

        return abs_ids + sorted_rel

    def _calc_abs(self, obj, target, rel):
        target_pos = target["position"].get("absolute")
        if target_pos is None:
            raise ValueError(f"Target '{target['id']}' has no absolute position yet")

        ref = self._get_ref_point(target, rel.get("reference_point", "center"))
        direction = rel["direction"]
        distance = rel["distance"]
        dx, dy = self._dir_vec(direction, target)

        x = ref[0] + dx * distance
        y = ref[1] + dy * distance
        z = self._calc_z(obj, target, direction, ref[2])
        return {"x": x, "y": y, "z": z}

    def _dir_vec(self, direction, target):
        if direction not in self.DIRECTION_VECTORS:
            raise ValueError(f"Unknown direction: {direction}")
        if direction in ("above", "below"):
            return (0.0, 0.0)

        base_dx, base_dy = self.DIRECTION_VECTORS[direction]
        ori = target.get("orientation", {}).get("absolute")
        if ori is None:
            raise ValueError(f"Target '{target['id']}' has no absolute orientation yet")
        yaw = math.radians(ori.get("yaw_deg", 0.0))
        cos_y = math.cos(yaw)
        sin_y = math.sin(yaw)
        return (base_dx * cos_y - base_dy * sin_y, base_dx * sin_y + base_dy * cos_y)

    def _get_ref_point(self, target, ref_name):
        pos = target["position"]["absolute"]
        cx, cy, cz = pos["x"], pos["y"], pos["z"]
        size = target["size"]
        w, l, h = size["width"], size["length"], size["height"]
        ori = target.get("orientation", {}).get("absolute", {})
        yaw = math.radians(ori.get("yaw_deg", 0.0))
        cos_y = math.cos(yaw)
        sin_y = math.sin(yaw)

        if ref_name == "center":
            return (cx, cy, cz)
        elif ref_name == "front_edge":
            ox, oy = 0.0, l / 2.0
        elif ref_name == "back_edge":
            ox, oy = 0.0, -l / 2.0
        elif ref_name == "left_edge":
            ox, oy = -w / 2.0, 0.0
        elif ref_name == "right_edge":
            ox, oy = w / 2.0, 0.0
        elif ref_name == "top_surface":
            return (cx, cy, cz + h / 2.0)
        else:
            raise ValueError(f"Unknown reference point: {ref_name}")

        rx = ox * cos_y - oy * sin_y
        ry = ox * sin_y + oy * cos_y
        return (cx + rx, cy + ry, cz)

    def _calc_z(self, obj, target, direction, ref_z):
        obj_h = obj["size"]["height"]
        obj_type = obj.get("type", "furniture")

        if direction == "above":
            target_h = target["size"]["height"]
            return ref_z + target_h / 2.0 + obj_h / 2.0
        if direction == "below":
            target_h = target["size"]["height"]
            return ref_z - target_h / 2.0 - obj_h / 2.0

        if obj_type == "furniture":
            return obj_h / 2.0
        else:
            target_type = target.get("type", "furniture")
            if target_type == "furniture":
                t_pos = target["position"]["absolute"]
                t_h = target["size"]["height"]
                surface_z = t_pos["z"] + t_h / 2.0
                return surface_z + obj_h / 2.0
            return obj_h / 2.0

    def _find(self, objects, oid):
        for o in objects:
            if o["id"] == oid:
                return o
        return None


# ---------------------------------------------------------------------------
# OrientationResolver — inline
# ---------------------------------------------------------------------------

class _OrientationResolver:
    def resolve_orientations(self, plan):
        resolved = {
            "schema_version": plan["schema_version"],
            "room_size": plan["room_size"],
            "objects": [dict(o) for o in plan["objects"]],
            "metadata": plan.get("metadata", {}),
        }
        objects = resolved["objects"]

        for obj in objects:
            ori = obj.get("orientation", {})
            if ori.get("absolute") is not None:
                continue
            rel = ori.get("relative")
            if rel is None:
                continue

            target_id = rel["facing"]
            target = self._find(objects, target_id)
            if target is None:
                raise ValueError(f"Object '{obj['id']}' references non-existent orientation target '{target_id}'")

            src_pos = obj["position"].get("absolute")
            if src_pos is None:
                raise ValueError(f"Object '{obj['id']}' has no absolute position (resolve distances first)")
            tgt_pos = target["position"].get("absolute")
            if tgt_pos is None:
                raise ValueError(f"Target '{target_id}' has no absolute position")
            tgt_ori = target["orientation"].get("absolute")
            if tgt_ori is None:
                raise ValueError(f"Target '{target_id}' has no absolute orientation")

            facing_dir = rel.get("facing_direction", "front")
            facing_away = rel.get("facing_away", False)
            yaw = self._calc_facing(
                (src_pos["x"], src_pos["y"], src_pos["z"]),
                (tgt_pos["x"], tgt_pos["y"], tgt_pos["z"]),
                tgt_ori["yaw_deg"],
                facing_dir,
                facing_away,
            )
            obj["orientation"] = {
                "absolute": {"yaw_deg": yaw, "pitch_deg": 0.0, "roll_deg": 0.0},
                "relative": None,
            }

        return resolved

    def _calc_facing(self, src, tgt, tgt_yaw, facing_dir, facing_away):
        offsets = {
            "front": 0.0, "back": 180.0,
            "left_side": 90.0, "right_side": -90.0,
            "front_left": 45.0, "front_right": -45.0,
            "back_left": 135.0, "back_right": -135.0,
        }
        if facing_dir not in offsets:
            print(f"[scene_planner] WARNING: unknown facing_direction '{facing_dir}', using 'front'")
            facing_dir = "front"

        side_angle = math.radians(tgt_yaw + offsets[facing_dir])
        side_x = tgt[0] + math.sin(side_angle)
        side_y = tgt[1] + math.cos(side_angle)
        dx = side_x - src[0]
        dy = side_y - src[1]
        yaw = math.degrees(math.atan2(dx, dy))
        if facing_away:
            yaw += 180.0
        return yaw % 360.0

    def _find(self, objects, oid):
        for o in objects:
            if o["id"] == oid:
                return o
        return None


# ---------------------------------------------------------------------------
# PlacementExecutor — inline (dumb executor, places objects as-is)
# ---------------------------------------------------------------------------

class _PlacementExecutor:
    def execute(self, plan):
        """Place all objects from resolved plan, return list of placed dicts."""
        placed = []
        for obj in plan["objects"]:
            pos = obj["position"].get("absolute")
            ori = obj["orientation"].get("absolute")
            if pos is None:
                raise ValueError(f"Object {obj['id']} has no absolute position")
            if ori is None:
                raise ValueError(f"Object {obj['id']} has no absolute orientation")
            placed.append({
                "id": obj["id"],
                "Model": obj["Model"],
                "type": obj.get("type", "furniture"),
                "size": dict(obj["size"]),
                "is_static": obj.get("is_static", True),
                "model_loc": obj.get("model_loc", ""),
                "uuid": obj.get("uuid", obj["id"]),
                "save_fn": obj.get("save_fn", ""),
                "_up_axis": obj.get("_up_axis", "y"),
                "position": dict(pos),
                "orientation": dict(ori),
            })
        return placed


# ---------------------------------------------------------------------------
# LLM helpers
# ---------------------------------------------------------------------------

def _llm(prompt, query, model):
    """Thin wrapper: call llm_request, accept dict or raw string response."""
    raw = _llm_request(system=prompt, user=query, model=model)
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        return json.loads(raw)
    raise ValueError(f"Unexpected LLM response type: {type(raw)}")


def _build_models_str(models):
    """Format models list for the prompt."""
    lines = []
    for i, m in enumerate(models):
        name = m.get("Model", f"object_{i}")
        size = m.get("size", [1.0, 1.0, 1.0])
        uuid = m.get("uuid", "")
        lines.append(
            f"  - Model: {name}\n"
            f"    Size: {{'width': {size[0]:.2f}, 'length': {size[1]:.2f}, 'height': {size[2]:.2f}}}\n"
            f"    uuid: {uuid}"
        )
    text = "\n".join(lines)
    count = len(models)
    text += (
        f"\n\nCRITICAL: You MUST generate EXACTLY {count} objects in your semantic plan. "
        f"The list above contains {count} entries - create one object specification for each entry, "
        f"even if some have the same Model name."
    )
    return text


# ---------------------------------------------------------------------------
# _generate_plan — single call with retry + validation feedback
# ---------------------------------------------------------------------------

def _generate_plan(models, room_half_size, query, llm_fn, max_retries=3, skip_validation=False):
    """Generate semantic plan from LLM with validation retry loop."""
    models_str = _build_models_str(models)
    room_w = room_half_size * 2
    room_l = room_half_size * 2
    room_h = 3.0

    prompt = _SEMANTIC_PLAN_PROMPT.format(
        query=query,
        room_width=room_w,
        room_length=room_l,
        room_height=room_h,
        models_str=models_str,
    )

    validator = _SchemaValidator()

    for attempt in range(max_retries):
        print(f"[scene_planner] Generating semantic plan (attempt {attempt + 1}/{max_retries})")
        try:
            plan = llm_fn(prompt, query)

            if skip_validation:
                print("[scene_planner] Skipping validation (intermediate batch)")
                return plan

            result = validator.validate_plan(plan)
            if result.success:
                print(f"[scene_planner] Schema validation passed on attempt {attempt + 1}")
                return plan

            err_msg = "\n".join(result.errors)
            print(f"[scene_planner] Schema validation failed: {err_msg}")
            if attempt < max_retries - 1:
                prompt = (
                    f"The previous semantic plan had validation errors:\n\n{err_msg}\n\n"
                    f"Please fix these errors and generate a corrected semantic plan.\n\n"
                    f"Original request:\n{prompt}"
                )
            else:
                raise RuntimeError(f"Schema validation failed after {max_retries} attempts:\n{err_msg}")

        except (json.JSONDecodeError, RuntimeError) as exc:
            print(f"[scene_planner] Error on attempt {attempt + 1}: {exc}")
            if attempt < max_retries - 1:
                prompt = (
                    f"An error occurred: {exc}\n\nPlease try again.\n\nOriginal request:\n{prompt}"
                )
            else:
                raise RuntimeError(f"Failed to generate semantic plan after {max_retries} attempts: {exc}") from exc

    raise RuntimeError(f"Failed to generate valid semantic plan after {max_retries} attempts")


# ---------------------------------------------------------------------------
# _determine_priorities — anchor vs dependent split
# ---------------------------------------------------------------------------

def _determine_priorities(models, query, llm_fn):
    """Ask LLM to split models into anchors and dependent objects."""
    lines = []
    for i, m in enumerate(models):
        name = m.get("Model", f"object_{i}")
        size = m.get("size", [1.0, 1.0, 1.0])
        lines.append(f"  {i}. {name} (size: {size[0]:.2f}m x {size[1]:.2f}m x {size[2]:.2f}m)")
    objects_str = "\n".join(lines)

    prompt = _PLACEMENT_PRIORITY_PROMPT.format(query=query, objects_str=objects_str)
    try:
        resp = llm_fn(prompt, query)
        anchor_indices = [item["index"] for item in resp.get("anchor_objects", [])]
        dep_indices = [item["index"] for item in resp.get("dependent_objects", [])]
        anchors = [models[i] for i in anchor_indices if i < len(models)]
        deps = [models[i] for i in dep_indices if i < len(models)]
        print(f"[scene_planner] Priorities: {len(anchors)} anchors, {len(deps)} dependent")
        return anchors, deps
    except Exception as exc:
        print(f"[scene_planner] Priority detection failed ({exc}), using heuristic split")
        anchors, deps = [], []
        for m in models:
            name = m.get("Model", "").lower()
            if any(kw in name for kw in ("table", "sofa", "bed", "desk")):
                anchors.append(m)
            else:
                deps.append(m)
        return anchors, deps


# ---------------------------------------------------------------------------
# _group_by_anchor — distribute dependent objects evenly among anchors
# ---------------------------------------------------------------------------

def _group_by_anchor(anchor_objects, dependent_models):
    """Distribute dependent models evenly among placed anchor objects."""
    if not anchor_objects or not dependent_models:
        return {}

    n_anchors = len(anchor_objects)
    n_deps = len(dependent_models)
    per_anchor = n_deps // n_anchors
    remainder = n_deps % n_anchors

    groups = {}
    idx = 0
    for i, anchor_obj in enumerate(anchor_objects):
        anchor_id = anchor_obj.get("id", f"anchor_{i}")
        count = per_anchor + (1 if i < remainder else 0)
        groups[anchor_id] = dependent_models[idx:idx + count]
        idx += count
        print(f"[scene_planner]   {anchor_id}: {len(groups[anchor_id])} dependent objects")
    return groups


# ---------------------------------------------------------------------------
# _generate_plan_in_batches — batch generation for large scenes
# ---------------------------------------------------------------------------

def _generate_plan_in_batches(models, room_half_size, query, llm_fn, batch_size=8, max_retries=3):
    """Generate semantic plan in batches for scenes with more than batch_size objects."""
    if len(models) <= batch_size:
        print(f"[scene_planner] {len(models)} objects, using single-pass generation")
        return _generate_plan(models, room_half_size, query, llm_fn, max_retries=max_retries)

    print(f"[scene_planner] {len(models)} objects, using batch generation (batch_size={batch_size})")

    anchors, deps = _determine_priorities(models, query, llm_fn)
    print(f"[scene_planner] Batch 1: generating {len(anchors)} anchor objects")

    anchor_plan = _generate_plan(
        anchors, room_half_size, query, llm_fn,
        max_retries=max_retries, skip_validation=True
    )
    all_objects = anchor_plan.get("objects", [])
    print(f"[scene_planner] Batch 1 done: {len(all_objects)} objects")

    dep_groups = _group_by_anchor(all_objects, deps)

    batch_num = 2
    for anchor_id, group_models in dep_groups.items():
        if not group_models:
            continue
        print(f"[scene_planner] Batch {batch_num}: {len(group_models)} dependent objects for {anchor_id}")

        context_lines = [
            "\n\nALREADY PLACED OBJECTS (use these IDs for relative positioning):",
            f"\nFOCUS: Place objects around '{anchor_id}'",
        ]
        for obj in all_objects:
            oid = obj.get("id", "unknown")
            omodel = obj.get("Model", "unknown")
            osize = obj.get("size", {})
            pos_info = ""
            if "position" in obj:
                if isinstance(obj["position"].get("absolute"), dict):
                    ap = obj["position"]["absolute"]
                    pos_info = f"at x={ap.get('x', 0):.1f}m, y={ap.get('y', 0):.1f}m"
            marker = " <- TARGET ANCHOR" if oid == anchor_id else ""
            context_lines.append(
                f"  - {oid}: {omodel} "
                f"(size: {osize.get('width', 0):.2f}m x {osize.get('length', 0):.2f}m x {osize.get('height', 0):.2f}m) "
                f"{pos_info}{marker}"
            )

        context_str = "\n".join(context_lines)
        batch_query = (
            f"{query}. {context_str}\n\n"
            f"Now place the following new objects using relative positioning to '{anchor_id}'."
        )

        batch_plan = _generate_plan(
            group_models, room_half_size, batch_query, llm_fn,
            max_retries=max_retries, skip_validation=True
        )
        batch_objects = batch_plan.get("objects", [])
        all_objects.extend(batch_objects)
        print(f"[scene_planner] Batch {batch_num} done: {len(batch_objects)} objects, total: {len(all_objects)}")
        batch_num += 1

    final_plan = {
        "schema_version": "1.0",
        "room_size": anchor_plan.get("room_size", {
            "width": room_half_size * 2,
            "length": room_half_size * 2,
            "height": 3.0,
        }),
        "objects": all_objects,
    }

    # Validate final merged plan
    print("[scene_planner] Validating final merged plan...")
    validator = _SchemaValidator()
    result = validator.validate_plan(final_plan)
    if not result.success:
        err_msg = "\n".join(result.errors)
        raise RuntimeError(f"Final merged plan validation failed:\n{err_msg}")

    print("[scene_planner] Final plan validation passed")
    return final_plan


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


def _llm_replan_layout(placed_models, semantic_plan, room_half_size, collision_details, max_retries=3):
    """LLM fallback for replanning object layout when gradient resolution fails."""
    print(f"[llm_replan_layout] Invoking LLM fallback for {len(placed_models)} objects")
    print(f"[llm_replan_layout] Reason: {len(collision_details)} collision(s) after gradient resolution")
    
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


def validate_and_repair_layout(placed_models, min_z=0.01, room_half_size=5.0, repair_iters=8, push_step=0.06, semantic_plan=None):
    """Validate layout and resolve overlaps using gradient-based resolution."""
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

def generate_placement_plan(models, room_half_size, room_type, query, prompt_model_fn=None):
    """Stage 4: generate placement plan for objects.

    Accepts: list of models with size/scale, room half-size, room type, query.
    Returns: list of models with added fields Pose, yaw_deg, is_static, size (final).
    """
    if not models:
        print("[scene_planner] WARNING: empty models list, returning empty result")
        return []

    model = DEFAULT_MODEL

    # Wrap the LLM function so internal helpers have a consistent signature (prompt, query)
    def llm_fn(prompt, query_text):
        if prompt_model_fn is not None:
            raw = prompt_model_fn(prompt, query_text, model)
            if isinstance(raw, dict):
                return raw
            if isinstance(raw, str):
                return json.loads(raw)
            raise ValueError(f"Unexpected response type from prompt_model_fn: {type(raw)}")
        return _llm_request(system=prompt, user=query_text, model=model)

    print(f"[scene_planner] Stage 4: generating placement plan for {len(models)} objects")
    print(f"[scene_planner] Room: {room_half_size*2:.1f}m x {room_half_size*2:.1f}m, type: {room_type}")

    # Generate raw semantic plan from LLM
    semantic_plan = _generate_plan_in_batches(
        models, room_half_size, query, llm_fn, batch_size=8, max_retries=3
    )
    print(f"[scene_planner] LLM produced {len(semantic_plan.get('objects', []))} objects")

    # Attach model_loc and extra fields to each object in the plan
    # (LLM doesn't know about model_loc, we inject it from the input list)
    for obj in semantic_plan.get("objects", []):
        model_name = obj.get("Model", "")
        match = next(
            (m for m in models if m.get("Model") == model_name),
            None
        )
        if match:
            obj["model_loc"] = match.get("model_loc", "")
            obj["uuid"] = match.get("uuid", obj.get("id", ""))
            obj["save_fn"] = match.get("save_fn", "")
            obj["_up_axis"] = match.get("_up_axis", "y")
            loc_preview = obj["model_loc"][:50] + "..." if len(obj["model_loc"]) > 50 else obj["model_loc"]
            print(f"[scene_planner]   {model_name}: model_loc={loc_preview}")
        else:
            print(f"[scene_planner] WARNING: no model_loc found for '{model_name}', using empty string")
            obj["model_loc"] = ""
            obj["uuid"] = obj.get("id", "")
            obj["save_fn"] = ""
            obj["_up_axis"] = "y"

    # Distance resolution: convert relative positions to absolute
    dist_resolver = _DistanceResolver()
    semantic_plan = dist_resolver.resolve_positions(semantic_plan)
    print("[scene_planner] Distance resolution done")

    # Orientation resolution: convert relative orientations to absolute
    ori_resolver = _OrientationResolver()
    semantic_plan = ori_resolver.resolve_orientations(semantic_plan)
    print("[scene_planner] Orientation resolution done")

    # Placement execution: extract final positions
    executor = _PlacementExecutor()
    placed = executor.execute(semantic_plan)
    print(f"[scene_planner] Placed {len(placed)} objects")

    # Build the output list: merge original model fields with resolved placement
    result = []
    for i, placed_obj in enumerate(placed):
        obj_uuid = placed_obj.get("uuid") or placed_obj.get("id", "")
        original = next(
            (m for m in models if m.get("uuid") == obj_uuid),
            next((m for m in models if m.get("Model") == placed_obj["Model"]), {})
        )

        # Generate unique save_fn per instance to avoid name collisions
        safe_uuid = re.sub(r"[^a-zA-Z0-9_]+", "_", obj_uuid)
        unique_save_fn = f"{safe_uuid}_{i}"

        sz = placed_obj["size"]
        pos = placed_obj["position"]
        ori = placed_obj["orientation"]

        merged = dict(original)  # carry all original fields (uuid, model_loc, scale, etc.)
        merged.update({
            "Model": placed_obj["Model"],
            "uuid": obj_uuid,
            "model_loc": placed_obj.get("model_loc", original.get("model_loc", "")),
            "save_fn": unique_save_fn,
            "_up_axis": original.get("_up_axis", "y"),
            "size": [sz["width"], sz["length"], sz["height"]],
            "is_static": placed_obj["is_static"],
            "Pose": {
                "x": pos["x"],
                "y": pos["y"],
                "z": pos["z"],
            },
            "yaw_deg": ori["yaw_deg"],
        })
        result.append(merged)

        print(
            f"[scene_planner]   {merged['Model']}: "
            f"pos=({pos['x']:.2f}, {pos['y']:.2f}, {pos['z']:.2f}), "
            f"yaw={ori['yaw_deg']:.0f}deg"
        )

    print(f"[scene_planner] Stage 4 complete: {len(result)} models with placement")
    
    # Collision resolution
    result = validate_and_repair_layout(result, room_half_size=room_half_size)
    
    return result
