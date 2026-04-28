"""Oriented Bounding Box (OBB) geometry and SAT collision detection.

Best practices from:
- ImperativeScene: gradient-based overlap resolution, multi-loss scoring
- SceneSmith: clearance zones, support surface validation
- SceneWeaver: boundary-constrained placement
"""

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple


@dataclass
class Vec2:
    x: float = 0.0
    y: float = 0.0

    def __add__(self, other: "Vec2") -> "Vec2":
        return Vec2(self.x + other.x, self.y + other.y)

    def __sub__(self, other: "Vec2") -> "Vec2":
        return Vec2(self.x - other.x, self.y - other.y)

    def __mul__(self, s: float) -> "Vec2":
        return Vec2(self.x * s, self.y * s)

    def dot(self, other: "Vec2") -> float:
        return self.x * other.x + self.y * other.y

    def length(self) -> float:
        return math.sqrt(self.x * self.x + self.y * self.y)

    def normalized(self) -> "Vec2":
        ln = self.length()
        if ln < 1e-12:
            return Vec2(1.0, 0.0)
        return Vec2(self.x / ln, self.y / ln)


@dataclass
class OBB:
    """Oriented Bounding Box in 2D (XY plane)."""

    center: Vec2
    half_extents: Vec2  # half-width along local X, half-depth along local Y
    yaw_rad: float = 0.0

    def corners(self) -> List[Vec2]:
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

    def axes(self) -> List[Vec2]:
        c = math.cos(self.yaw_rad)
        s = math.sin(self.yaw_rad)
        return [Vec2(c, s), Vec2(-s, c)]

    def aabb(self) -> "AABB":
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

    def overlaps(self, other: "AABB", margin: float = 0.0) -> bool:
        sep_x = (self.max_x + margin < other.min_x) or (
            other.max_x + margin < self.min_x
        )
        sep_y = (self.max_y + margin < other.min_y) or (
            other.max_y + margin < self.min_y
        )
        return not (sep_x or sep_y)

    def center(self) -> Vec2:
        return Vec2(
            (self.min_x + self.max_x) * 0.5,
            (self.min_y + self.max_y) * 0.5,
        )

    def contains_point(self, p: Vec2) -> bool:
        return self.min_x <= p.x <= self.max_x and self.min_y <= p.y <= self.max_y


# ---------------------------------------------------------------------------
# SAT-based OBB overlap test
# ---------------------------------------------------------------------------

def _project_corners(corners: List[Vec2], axis: Vec2) -> Tuple[float, float]:
    dots = [c.dot(axis) for c in corners]
    return min(dots), max(dots)


def obb_overlap(a: OBB, b: OBB, margin: float = 0.0) -> bool:
    """SAT overlap test for two OBBs with optional inflation margin."""
    corners_a = a.corners()
    corners_b = b.corners()

    for ax in a.axes() + b.axes():
        min_a, max_a = _project_corners(corners_a, ax)
        min_b, max_b = _project_corners(corners_b, ax)
        if max_a + margin < min_b or max_b + margin < min_a:
            return False
    return True


def obb_overlap_depth(a: OBB, b: OBB) -> float:
    """Return penetration depth (>0 means overlap). Uses SAT."""
    corners_a = a.corners()
    corners_b = b.corners()
    min_overlap = float("inf")

    for ax in a.axes() + b.axes():
        min_a, max_a = _project_corners(corners_a, ax)
        min_b, max_b = _project_corners(corners_b, ax)
        overlap = min(max_a, max_b) - max(min_a, min_b)
        if overlap < 0:
            return overlap  # separated
        min_overlap = min(min_overlap, overlap)

    return min_overlap


def obb_separation_vector(a: OBB, b: OBB) -> Tuple[Vec2, float]:
    """Return (direction, depth) for the minimum translation to separate a from b.

    Direction points from b toward a. depth > 0 means they overlap.
    Inspired by ImperativeScene's gradient-based overlap resolution.
    """
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
            # Ensure direction points from b center to a center
            ca = a.center.dot(ax)
            cb = b.center.dot(ax)
            best_axis = ax if ca >= cb else ax * (-1.0)

    return best_axis.normalized(), best_depth


# ---------------------------------------------------------------------------
# Polygon utilities
# ---------------------------------------------------------------------------

def point_in_polygon(p: Vec2, polygon: Sequence[Sequence[float]]) -> bool:
    """Ray-casting point-in-polygon test."""
    inside = False
    n = len(polygon)
    if n < 3:
        return False
    j = n - 1
    for i in range(n):
        xi, yi = float(polygon[i][0]), float(polygon[i][1])
        xj, yj = float(polygon[j][0]), float(polygon[j][1])
        if ((yi > p.y) != (yj > p.y)) and (
            p.x < (xj - xi) * (p.y - yi) / max(1e-9, (yj - yi)) + xi
        ):
            inside = not inside
        j = i
    return inside


def obb_inside_polygon(obb: OBB, polygon: Sequence[Sequence[float]]) -> bool:
    """Check all 4 corners of an OBB are inside a polygon."""
    return all(point_in_polygon(c, polygon) for c in obb.corners())


def obb_inside_room(obb: OBB, room_half_size: float) -> bool:
    """Check all 4 corners are within a square room centered at origin."""
    for c in obb.corners():
        if abs(c.x) > room_half_size or abs(c.y) > room_half_size:
            return False
    return True


# ---------------------------------------------------------------------------
# Clearance zones (inspired by SceneSmith)
# ---------------------------------------------------------------------------

@dataclass
class ClearanceZone:
    """Rectangular clearance zone (e.g. in front of a door)."""

    aabb: AABB
    zone_type: str = "door"  # door | window | passage

    def violates(self, obb: OBB) -> bool:
        """Check if an OBB intrudes into this clearance zone."""
        return self.aabb.overlaps(obb.aabb())


def build_door_clearance(
    door_polygon: Sequence[Sequence[float]],
    clearance_depth: float = 0.8,
) -> ClearanceZone:
    """Build a clearance zone extending from a door polygon."""
    xs = [float(p[0]) for p in door_polygon]
    ys = [float(p[1]) for p in door_polygon]
    return ClearanceZone(
        aabb=AABB(
            min(xs) - clearance_depth,
            max(xs) + clearance_depth,
            min(ys) - clearance_depth,
            max(ys) + clearance_depth,
        ),
        zone_type="door",
    )


# ---------------------------------------------------------------------------
# Helpers for converting model dicts to geometry
# ---------------------------------------------------------------------------

def model_to_obb(
    model: Dict[str, Any],
    *,
    inflation: float = 0.0,
    pos_override: Optional[Tuple[float, float]] = None,
    yaw_override: Optional[float] = None,
) -> OBB:
    """Convert a placed model dict to an OBB."""
    size = model.get("size", [1.0, 1.0, 1.0])
    # After euler="90 0 yaw" rotation (Y-up → Z-up):
    #   mesh X → scene X (width)
    #   mesh Z → scene Y (depth)   ← floor footprint uses this
    #   mesh Y → scene Z (height)
    raw_sx = float(size[0]) if len(size) > 0 else 1.0
    raw_sy = float(size[2]) if len(size) > 2 else 1.0  # mesh Z = scene depth
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


def model_half_height(model: Dict[str, Any]) -> float:
    """Return half-height of model in the scene (Z direction).

    Objaverse GLB models use Y-up convention. MuJoCo assembly applies
    euler="90 0 yaw" which maps mesh-Y → scene-Z.
    Therefore the scene height = size[1] (mesh Y extent).
    Falls back to max(size) if size[1] is suspiciously small.
    """
    size = model.get("size", [1.0, 1.0, 1.0])
    if len(size) < 3:
        return 0.5
    sx, sy, sz = float(size[0]), float(size[1]), float(size[2])
    # Use mesh Y as height; fallback to max if Y is implausibly tiny
    height = sy if sy >= 0.05 * max(sx, sy, sz) else max(sx, sy, sz)
    return max(0.01, height / 2.0)


# ---------------------------------------------------------------------------
# Gradient-based overlap resolution (inspired by ImperativeScene)
# ---------------------------------------------------------------------------

def _z_intervals_overlap(a: Dict[str, Any], b: Dict[str, Any]) -> bool:
    """Check if two models overlap in Z (they're at the same height level).

    Objects at different Z levels (one stacked on another) should not be
    separated in XY — only objects at the same floor level should be pushed apart.

    Height in scene = size[1] (mesh Y, after euler="90 0 yaw").
    """
    size_a = a.get("size", [1.0, 1.0, 1.0])
    size_b = b.get("size", [1.0, 1.0, 1.0])
    pose_a = a.get("Pose") or {}
    pose_b = b.get("Pose") or {}
    za = float(pose_a.get("z", 0.0))
    zb = float(pose_b.get("z", 0.0))
    # size[1] = mesh Y = scene Z height (not size[2] which is scene Y depth)
    ha = max(0.01, float(size_a[1]) if len(size_a) > 1 else 1.0) / 2.0
    hb = max(0.01, float(size_b[1]) if len(size_b) > 1 else 1.0) / 2.0
    return not ((za + ha < zb - hb) or (zb + hb < za - ha))


def gradient_resolve_overlaps(
    models: List[Dict[str, Any]],
    *,
    room_half_size: float = 5.0,
    iterations: int = 200,
    step_size: float = 0.05,
    collision_margin: float = 0.01,
    semantic_plan: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """Move objects apart using gradient-based overlap resolution.

    For each pair of overlapping objects at the same Z level, compute a
    separation gradient proportional to overlap depth (from ImperativeScene:
    4.0 * depth / dist * dir).

    Objects at different Z levels (stacked) are excluded from XY separation
    to preserve "on" placements.
    
    Objects with on_top_of constraints are marked as anchored and excluded
    from gradient updates to preserve LLM-based surface positioning.
    """
    resolved = [dict(m) for m in models]
    n = len(resolved)
    
    # Identify anchored objects (on surfaces) from semantic plan
    anchored = [False] * n
    if semantic_plan and isinstance(semantic_plan, dict):
        objects = semantic_plan.get("objects", [])
        if isinstance(objects, list):
            # Build name->constraints map
            constraints_by_name: Dict[str, List[Dict]] = {}
            for obj in objects:
                if not isinstance(obj, dict):
                    continue
                name = str(obj.get("Model") or obj.get("name") or "")
                if not name:
                    continue
                cs = obj.get("constraints", [])
                if isinstance(cs, list):
                    constraints_by_name.setdefault(name, []).extend(cs)
            
            # Mark objects with on_top_of as anchored
            name_counts: Dict[str, int] = {}
            for i, m in enumerate(resolved):
                name = str(m.get("Model") or m.get("name") or "")
                if not name:
                    continue
                idx = name_counts.get(name, 0)
                name_counts[name] = idx + 1
                
                cs = constraints_by_name.get(name, [])
                for c in cs:
                    if not isinstance(c, dict):
                        continue
                    ctype = str(c.get("type", "")).lower()
                    if ctype in {"on", "on_top_of", "on-top-of", "on top of", "on_top"}:
                        anchored[i] = True
                        break
                    # Also anchor the scene anchor (region:middle + is_static)
                    if (ctype == "region"
                            and str(c.get("value", "")).lower() == "middle"
                            and c.get("weight", 1.0) >= 4.0):
                        m = resolved[i]
                        if m.get("is_static", True):
                            anchored[i] = True
                            print(f"[anchor] {m.get('Model')} anchored (region:middle)")
                        break

    for _ in range(iterations):
        grads = [Vec2(0.0, 0.0) for _ in range(n)]
        any_overlap = False

        # Pairwise overlap gradients (only for objects at same Z level)
        for i in range(n):
            for j in range(i + 1, n):
                # Skip if both objects are anchored (on surfaces)
                if anchored[i] and anchored[j]:
                    continue
                
                # Skip objects at different Z levels (stacked objects)
                if not _z_intervals_overlap(resolved[i], resolved[j]):
                    continue
                
                # Skip small objects that are likely on surfaces (z > 0.5m)
                # These were carefully positioned by LLM and shouldn't be moved
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
                dist = (obb_i.center - obb_j.center).length()
                scale = 4.0 * depth / max(0.01, dist)
                grad = sep_dir * (scale * 0.625)
                grads[i] = grads[i] + grad
                grads[j] = grads[j] - grad

        # Out-of-bounds gradients
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
            break

        # Apply gradients (skip anchored objects on surfaces)
        for i in range(n):
            if anchored[i]:
                continue
            
            g = grads[i]
            if g.length() < 1e-8:
                continue
            
            pose = dict(resolved[i].get("Pose") or {"x": 0.0, "y": 0.0, "z": 0.0})
            pose["x"] = float(pose.get("x", 0.0)) + step_size * g.x
            pose["y"] = float(pose.get("y", 0.0)) + step_size * g.y
            resolved[i]["Pose"] = pose

    # Final hard clamp: ensure all objects stay within room bounds AND
    # rest correctly on the floor (or on top of their support if stacked).
    # Floor objects: bottom must be at z=0, so z = half-height.
    # Stacked objects: keep their existing z (set by small-objects/repair pass).
    
    # Restore surface-relative positions for anchored objects
    for i in range(n):
        if not anchored[i] or "_surface_offset" not in resolved[i]:
            continue
            
        offset = resolved[i]["_surface_offset"]
        receptacle_name = offset.get("receptacle", "")
        if not receptacle_name:
            continue
        
        # Find receptacle by name
        for j in range(n):
            if i == j:
                continue
            model_name = str(resolved[j].get("Model") or resolved[j].get("name") or "")
            if model_name == receptacle_name:
                pose_j = resolved[j].get("Pose") or {}
                rx = float(pose_j.get("x", 0.0))
                ry = float(pose_j.get("y", 0.0))
                
                # Restore relative position
                pose = dict(resolved[i].get("Pose") or {})
                pose["x"] = rx + offset["x"]
                pose["y"] = ry + offset["y"]
                resolved[i]["Pose"] = pose
                break
    
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

        # Z-clamp: an object whose bottom would be below floor (z < half-height)
        # must be lifted up. Objects stacked on others (z noticeably above
        # half-height) keep their stacked z so we don't break "on top of".
        hh = model_half_height(resolved[i])
        cz = float(pose.get("z", hh))
        floor_z = hh
        # Tolerance: anything within 5% of floor_z is treated as a floor object.
        if cz < floor_z - 1e-3:
            pose["z"] = floor_z
        elif abs(cz - floor_z) < max(0.05, 0.1 * hh):
            # Snap exactly to floor (kills small drift from gradient steps).
            pose["z"] = floor_z

        resolved[i]["Pose"] = pose

    return resolved


# ---------------------------------------------------------------------------
# Multi-loss scoring (inspired by ImperativeScene)
# ---------------------------------------------------------------------------

@dataclass
class PlacementLoss:
    """Aggregated loss for a placement candidate."""

    inbound: float = 0.0
    overlap: float = 0.0
    standing: float = 0.0
    clearance: float = 0.0
    constraint: float = 0.0

    def total(self) -> float:
        return self.inbound + self.overlap + self.standing + self.clearance + self.constraint

    def is_feasible(self) -> bool:
        return self.inbound == 0.0 and self.overlap == 0.0 and self.standing == 0.0


def compute_inbound_loss(obb: OBB, room_half_size: float) -> float:
    """Penalty for OBB exceeding room boundaries."""
    loss = 0.0
    for corner in obb.corners():
        dx = max(0.0, abs(corner.x) - room_half_size)
        dy = max(0.0, abs(corner.y) - room_half_size)
        loss += dx + dy
    return loss


def compute_overlap_loss(
    obb: OBB,
    placed_obbs: Sequence[OBB],
    margin: float = 0.01,
) -> float:
    """Sum of penetration depths against all placed objects."""
    loss = 0.0
    for other in placed_obbs:
        depth = obb_overlap_depth(obb, other)
        if depth > -margin:
            loss += max(0.0, depth + margin)
    return loss


def compute_clearance_loss(
    obb: OBB,
    clearance_zones: Sequence[ClearanceZone],
) -> float:
    """Penalty for intruding into clearance zones."""
    loss = 0.0
    for zone in clearance_zones:
        if zone.violates(obb):
            obj_aabb = obb.aabb()
            # Approximate overlap area
            dx = min(obj_aabb.max_x, zone.aabb.max_x) - max(
                obj_aabb.min_x, zone.aabb.min_x
            )
            dy = min(obj_aabb.max_y, zone.aabb.max_y) - max(
                obj_aabb.min_y, zone.aabb.min_y
            )
            if dx > 0 and dy > 0:
                loss += dx * dy
    return loss
