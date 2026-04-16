from __future__ import annotations

import math
import random
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from typing import Iterable

from mujoco_scene_editor.pipeline.scene_graph import RelationType, SceneGraph, SceneObject, SceneRelation, index_objects


@dataclass(frozen=True)
class PlacedObject:
    obj: SceneObject
    pos_xyz: tuple[float, float, float]


def _density_kg_m3(material: str) -> float:
    m = (material or "generic").lower()
    if m == "plastic":
        return 950.0
    if m == "wood":
        return 700.0
    if m == "metal":
        return 7800.0
    if m == "glass":
        return 2500.0
    if m == "rubber":
        return 1100.0
    if m == "paper":
        return 300.0
    return 1000.0


def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def _default_size_m(class_name: str) -> tuple[float, float, float]:
    c = (class_name or "").lower().strip()

    # Furniture / anchors.
    if any(k in c for k in ("desk", "table")):
        return 1.2, 0.8, 0.75
    if "shelf" in c:
        return 1.0, 0.3, 1.6
    if any(k in c for k in ("cabinet", "cupboard")):
        return 0.9, 0.45, 1.8

    # Common tabletop objects.
    if any(k in c for k in ("mug", "cup")):
        return 0.09, 0.09, 0.11
    if any(k in c for k in ("bottle", "can")):
        return 0.08, 0.08, 0.26
    if any(k in c for k in ("plate", "dish")):
        return 0.25, 0.25, 0.03
    if any(k in c for k in ("laptop", "notebook")):
        return 0.34, 0.24, 0.025
    if any(k in c for k in ("keyboard",)):
        return 0.45, 0.16, 0.03
    if any(k in c for k in ("monitor", "screen")):
        return 0.55, 0.20, 0.40
    if any(k in c for k in ("book",)):
        return 0.22, 0.16, 0.035

    # Generic prop.
    return 0.18, 0.18, 0.18


def _primitive_for_class(class_name: str) -> str:
    c = (class_name or "").lower()
    if any(k in c for k in ("mug", "cup", "bottle", "can")):
        return "cylinder"
    if any(k in c for k in ("plate", "dish")):
        return "cylinder"
    if any(k in c for k in ("ball", "sphere")):
        return "sphere"
    return "box"


def _rgba_for_class(class_name: str) -> tuple[float, float, float, float]:
    c = (class_name or "").lower()
    if any(k in c for k in ("table", "desk")):
        return 0.55, 0.45, 0.35, 1.0
    if any(k in c for k in ("mug", "cup")):
        return 0.8, 0.8, 0.9, 1.0
    if any(k in c for k in ("bottle", "can")):
        return 0.4, 0.7, 0.9, 1.0
    if any(k in c for k in ("laptop", "monitor", "keyboard")):
        return 0.2, 0.2, 0.25, 1.0
    return 0.7, 0.7, 0.7, 1.0


def _box_inertia_diaginertia(mass: float, size_full: tuple[float, float, float]) -> tuple[float, float, float]:
    x, y, z = size_full
    ixx = (1.0 / 12.0) * mass * (y * y + z * z)
    iyy = (1.0 / 12.0) * mass * (x * x + z * z)
    izz = (1.0 / 12.0) * mass * (x * x + y * y)
    return ixx, iyy, izz


def _cylinder_inertia_diaginertia(mass: float, radius: float, height_full: float) -> tuple[float, float, float]:
    # Cylinder aligned with Z.
    ixx = (1.0 / 12.0) * mass * (3.0 * radius * radius + height_full * height_full)
    iyy = ixx
    izz = 0.5 * mass * radius * radius
    return ixx, iyy, izz


@dataclass
class _Surface:
    name: str
    center_xy: tuple[float, float]
    half_extents_xy: tuple[float, float]
    z_top: float
    occupied: list[tuple[float, float, float, float]] = field(default_factory=list)  # xmin,xmax,ymin,ymax


# Wall-mounting helper: tracks occupied segments along a wall.
@dataclass
class _WallSlot:
    """1-D occupied interval [lo, hi] along a wall."""
    lo: float
    hi: float


@dataclass
class _WallSurface:
    """A vertical wall surface — objects are placed at a fixed depth from the wall."""
    side: str          # "north" | "south" | "east" | "west"
    fixed_coord: float  # world coordinate perpendicular to wall (Y for N/S, X for E/W)
    along_half: float   # half-length of wall in the "along" direction
    depth: float        # how far the object protrudes from the wall
    occupied: list[_WallSlot] = field(default_factory=list)

    def sample_along(self, half_width: float, rng: random.Random, attempts: int = 80) -> float:
        """Sample a non-overlapping position along the wall."""
        lo = -self.along_half + half_width + 0.05
        hi =  self.along_half - half_width - 0.05
        if lo > hi:
            return 0.0
        for _ in range(attempts):
            pos = rng.uniform(lo, hi)
            if not any(s.lo < pos + half_width + 0.05 and pos - half_width - 0.05 < s.hi
                       for s in self.occupied):
                self.occupied.append(_WallSlot(pos - half_width, pos + half_width))
                return pos
        # Fallback: centre
        return 0.0


def _rects_overlap(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> bool:
    ax0, ax1, ay0, ay1 = a
    bx0, bx1, by0, by1 = b
    return (ax0 < bx1) and (bx0 < ax1) and (ay0 < by1) and (by0 < ay1)


def _sample_non_overlapping_xy(
    surface: _Surface,
    half_x: float,
    half_y: float,
    *,
    margin: float,
    rng: random.Random,
    attempts: int = 250,
) -> tuple[float, float]:
    cx, cy = surface.center_xy
    sx, sy = surface.half_extents_xy

    xmin = cx - sx + margin + half_x
    xmax = cx + sx - margin - half_x
    ymin = cy - sy + margin + half_y
    ymax = cy + sy - margin - half_y

    if xmin > xmax:
        xmin, xmax = cx, cx
    if ymin > ymax:
        ymin, ymax = cy, cy

    for _ in range(attempts):
        x = rng.uniform(xmin, xmax)
        y = rng.uniform(ymin, ymax)
        rect = (x - half_x - margin, x + half_x + margin, y - half_y - margin, y + half_y + margin)
        if any(_rects_overlap(rect, r) for r in surface.occupied):
            continue
        surface.occupied.append(rect)
        return x, y

    # Fallback: deterministic scan.
    nx = 10
    ny = 10
    for ix in range(nx):
        for iy in range(ny):
            x = xmin + (xmax - xmin) * (ix + 0.5) / nx
            y = ymin + (ymax - ymin) * (iy + 0.5) / ny
            rect = (x - half_x - margin, x + half_x + margin, y - half_y - margin, y + half_y + margin)
            if any(_rects_overlap(rect, r) for r in surface.occupied):
                continue
            surface.occupied.append(rect)
            return x, y

    # Give up: stack at center.
    return cx, cy


def _compute_room_half_size(graph: SceneGraph) -> float:
    """Compute room half-size from object footprints + circulation factor."""
    if graph.constraints.room_half_size_m is not None:
        return float(graph.constraints.room_half_size_m)

    total_area = 0.0
    for obj in graph.objects:
        size = obj.size_m or list(_default_size_m(obj.class_name))
        sx = float(size[0]) if len(size) > 0 else 0.5
        sy = float(size[1]) if len(size) > 1 else 0.5
        total_area += sx * sy

    # Circulation factor 2.5x, room is square
    half = math.sqrt(max(0.0, total_area) * 2.5)
    return float(_clamp(half, 2.5, 10.0))


def _add_room(
    worldbody: ET.Element,
    half: float,
    wall_h: float,
) -> dict[str, _WallSurface]:
    """Add floor + 4 walls to worldbody. Returns wall surfaces for mounting."""
    wall_t = 0.05  # half-thickness of wall geom

    # Floor (replace later — size updated to match room)
    floor_body = ET.SubElement(worldbody, "body", name="floor", pos="0 0 0")
    ET.SubElement(
        floor_body, "geom",
        name="floor_geom", type="plane",
        size=f"{half:.4f} {half:.4f} 0.01",
        rgba="0.55 0.52 0.48 1",
        contype="1", conaffinity="1",
    )

    wh2 = wall_h / 2.0
    # (name, body_x, body_y, geom_sx, geom_sy, geom_sz)
    wall_defs = [
        ("wall_north",  0.0,    half,  half,    wall_t, wh2),
        ("wall_south",  0.0,   -half,  half,    wall_t, wh2),
        ("wall_east",   half,   0.0,   wall_t,  half,   wh2),
        ("wall_west",  -half,   0.0,   wall_t,  half,   wh2),
    ]
    for bname, bx, by, gsx, gsy, gsz in wall_defs:
        wb = ET.SubElement(worldbody, "body", name=bname, pos=f"{bx:.4f} {by:.4f} {wh2:.4f}")
        ET.SubElement(
            wb, "geom", type="box",
            size=f"{gsx:.4f} {gsy:.4f} {gsz:.4f}",
            rgba="0.85 0.82 0.78 1",
            contype="1", conaffinity="1",
        )

    # Build wall surface descriptors (used for wall_mounted placement)
    wall_surfaces: dict[str, _WallSurface] = {
        "north": _WallSurface(side="north", fixed_coord= half, along_half=half, depth=wall_t),
        "south": _WallSurface(side="south", fixed_coord=-half, along_half=half, depth=wall_t),
        "east":  _WallSurface(side="east",  fixed_coord= half, along_half=half, depth=wall_t),
        "west":  _WallSurface(side="west",  fixed_coord=-half, along_half=half, depth=wall_t),
    }
    return wall_surfaces


def _default_wall_height_for_class(class_name: str) -> float:
    c = (class_name or "").lower()
    if any(k in c for k in ("lamp", "light", "sconce")):
        return 1.8
    if any(k in c for k in ("painting", "picture", "art", "poster", "clock", "mirror")):
        return 1.5
    if any(k in c for k in ("shelf", "floating")):
        return 1.3
    if any(k in c for k in ("whiteboard", "board", "screen", "monitor")):
        return 1.2
    return 1.4  # generic default


def _needs_table(graph: SceneGraph) -> bool:
    # If anything is explicitly on something, a table is a good default anchor.
    if any(r.type in (RelationType.on, RelationType.ontop) for r in graph.relations):
        return True
    if any("table" in (o.class_name or "").lower() or "desk" in (o.class_name or "").lower() for o in graph.objects):
        return True
    return False


def _find_explicit_static_table_id(graph: SceneGraph) -> str | None:
    """Return the id of a static table/desk object if present.

    When the graph already contains a table, we still create a single synthetic
    `anchor_table` support surface for deterministic placement, but we avoid
    emitting a duplicate physical table body.
    """

    for o in graph.objects:
        cls = (o.class_name or "").lower()
        if o.movable:
            continue
        if any(k in cls for k in ("table", "desk")):
            return o.id
    return None


def _find_targets(graph: SceneGraph) -> tuple[set[str], set[str]]:
    by_id = index_objects(graph)
    on_subjects: set[str] = set()
    inside_subjects: set[str] = set()
    for rel in graph.relations:
        if rel.subject not in by_id or rel.object not in by_id:
            continue
        if rel.type in (RelationType.on, RelationType.ontop, RelationType.on_floor):
            on_subjects.add(rel.subject)
        if rel.type == RelationType.inside:
            inside_subjects.add(rel.subject)
    return on_subjects, inside_subjects


def scene_graph_to_mjcf(
    graph: SceneGraph,
    *,
    seed: int | None = None,
) -> str:
    """Convert a SceneGraph into a runnable MJCF (best-effort).

    This generator is deterministic (given the same seed) and does NOT depend on
    the LLM. Physics stabilization is expected to be applied downstream via
    `mujoco_scene_editor.utils.mjcf_physics`.
    """

    rng = random.Random(seed if seed is not None else 0)

    by_id = index_objects(graph)
    on_subjects, _inside_subjects = _find_targets(graph)

    root = ET.Element("mujoco")

    # Keep options minimal; downstream repair may tune solver.
    ET.SubElement(root, "option", timestep="0.002")

    asset = ET.SubElement(root, "asset")
    worldbody = ET.SubElement(root, "worldbody")

    # Compute room dimensions.
    room_half = _compute_room_half_size(graph)
    wall_h = float(graph.constraints.wall_height_m or 3.0)

    # Add room (floor plane + 4 walls) and get wall surface descriptors.
    wall_surfaces = _add_room(worldbody, room_half, wall_h)

    floor_surface = _Surface(
        name="floor",
        center_xy=(0.0, 0.0),
        half_extents_xy=(room_half * 0.85, room_half * 0.85),
        z_top=0.0,
        occupied=[],
    )

    surfaces: dict[str, _Surface] = {"__floor__": floor_surface}

    # Pre-collect wall_mounted relations (subject → relation).
    wall_mounted_rels: dict[str, SceneRelation] = {}
    for rel in graph.relations:
        if rel.type == RelationType.wall_mounted:
            wall_mounted_rels[rel.subject] = rel

    # Default table anchor if useful.
    table_id = None
    explicit_table_id: str | None = None
    if _needs_table(graph):
        explicit_table_id = _find_explicit_static_table_id(graph)
        table_id = "anchor_table"
        table_height = graph.constraints.workspace_height_m or 0.75
        table_height = float(_clamp(table_height, 0.65, 0.9))
        thickness = 0.06
        top_z = table_height
        center_z = top_z - thickness / 2.0

        table_body = ET.SubElement(worldbody, "body", name=table_id, pos=f"0 0 {center_z:.4f}")
        # Static table.
        ET.SubElement(
            table_body,
            "geom",
            name="table_top",
            type="box",
            size="0.6 0.4 0.03",
            rgba="0.55 0.45 0.35 1",
            contype="1",
            conaffinity="1",
        )

        surfaces[table_id] = _Surface(
            name=table_id,
            center_xy=(0.0, 0.0),
            half_extents_xy=(0.55, 0.35),
            z_top=top_z,
            occupied=[],
        )

    # Create meshes in <asset>.
    mesh_name_for_obj: dict[str, str] = {}
    for obj in graph.objects:
        if not obj.mesh_file:
            continue
        mesh_name = f"mesh_{obj.id}"
        mesh_name_for_obj[obj.id] = mesh_name

        scale = obj.mesh_scale
        if scale is None:
            scale = 1.0
        try:
            s = float(scale)
        except (TypeError, ValueError):
            s = 1.0
        ET.SubElement(asset, "mesh", name=mesh_name, file=obj.mesh_file, scale=f"{s} {s} {s}")

    # Determine placement surface per object.
    rel_on: dict[str, str] = {}
    for rel in graph.relations:
        if rel.type not in (RelationType.on, RelationType.ontop, RelationType.on_floor):
            continue
        if rel.subject in by_id and rel.object in by_id:
            target = rel.object
            if explicit_table_id is not None and table_id is not None and target == explicit_table_id:
                target = table_id
            rel_on[rel.subject] = target

    placed: list[PlacedObject] = []

    def _support_surface_for(obj_id: str) -> _Surface:
        target_id = rel_on.get(obj_id)
        if target_id and target_id in surfaces:
            return surfaces[target_id]
        if table_id is not None and obj_id in on_subjects:
            return surfaces[table_id]
        return surfaces["__floor__"]

    # ── Wall-mounted objects (static bodies on walls) ────────────────────────
    for obj in graph.objects:
        if obj.id not in wall_mounted_rels:
            continue
        rel = wall_mounted_rels[obj.id]

        size = obj.size_m
        if size is None:
            size = list(_default_size_m(obj.class_name))
        try:
            sx, sy, sz = float(size[0]), float(size[1]), float(size[2])
        except Exception:
            sx, sy, sz = _default_size_m(obj.class_name)
        sx = float(_clamp(abs(sx), 0.01, 3.0))
        sy = float(_clamp(abs(sy), 0.01, 3.0))
        sz = float(_clamp(abs(sz), 0.01, 3.0))

        side = (rel.wall_side or "north").lower()
        wall_surf = wall_surfaces.get(side, wall_surfaces["north"])
        height_z = rel.height_m if rel.height_m is not None else _default_wall_height_for_class(obj.class_name)
        height_z = float(_clamp(height_z, 0.3, wall_h - sz / 2.0 - 0.05))

        # Which dimension sticks out from the wall?
        # north/south walls are aligned along X, object Y protrudes.
        # east/west walls are aligned along Y, object X protrudes.
        wall_t = 0.05  # half-thickness used in _add_room
        if side in ("north", "south"):
            half_along = sx / 2.0
            half_depth = sy / 2.0
            along = wall_surf.sample_along(half_along, rng)
            sign = 1.0 if side == "north" else -1.0
            wx = along
            wy = sign * (room_half - wall_t - half_depth)
            wz = height_z
        else:  # east / west
            half_along = sy / 2.0
            half_depth = sx / 2.0
            along = wall_surf.sample_along(half_along, rng)
            sign = 1.0 if side == "east" else -1.0
            wx = sign * (room_half - wall_t - half_depth)
            wy = along
            wz = height_z

        body = ET.SubElement(worldbody, "body", name=obj.id, pos=f"{wx:.4f} {wy:.4f} {wz:.4f}")
        # Wall-mounted objects are always static (no freejoint).
        rgba = _rgba_for_class(obj.class_name)
        rgba_s = f"{rgba[0]:.3f} {rgba[1]:.3f} {rgba[2]:.3f} {rgba[3]:.3f}"
        primitive = obj.primitive or _primitive_for_class(obj.class_name)
        half_x, half_y, half_z = sx / 2.0, sy / 2.0, sz / 2.0

        if obj.mesh_file and obj.id in mesh_name_for_obj:
            ET.SubElement(body, "geom", type="mesh", mesh=mesh_name_for_obj[obj.id],
                          rgba=rgba_s, contype="1", conaffinity="1")
        elif primitive == "cylinder":
            radius = 0.5 * min(sx, sy)
            ET.SubElement(body, "geom", type="cylinder",
                          size=f"{radius:.4f} {sz / 2.0:.4f}",
                          rgba=rgba_s, contype="1", conaffinity="1")
        elif primitive == "sphere":
            r = 0.5 * min(sx, sy, sz)
            ET.SubElement(body, "geom", type="sphere", size=f"{r:.4f}",
                          rgba=rgba_s, contype="1", conaffinity="1")
        else:
            ET.SubElement(body, "geom", type="box",
                          size=f"{half_x:.4f} {half_y:.4f} {half_z:.4f}",
                          rgba=rgba_s, contype="1", conaffinity="1")

        # Expose wall-mounted flat tops as support surfaces (e.g. shelf → items on shelf)
        surfaces[obj.id] = _Surface(
            name=obj.id,
            center_xy=(wx, wy),
            half_extents_xy=(half_x, half_y),
            z_top=wz + half_z,
            occupied=[],
        )
        placed.append(PlacedObject(obj=obj, pos_xyz=(wx, wy, wz)))

    # ── Floor / table objects ─────────────────────────────────────────────────
    for obj in graph.objects:
        # Skip wall-mounted objects (already placed above).
        if obj.id in wall_mounted_rels:
            continue
        # Skip anchors that we already created.
        if obj.id == table_id:
            continue
        # If the graph provided an explicit static table/desk, we represent it
        # via the synthetic anchor_table support to avoid duplication.
        if explicit_table_id is not None and obj.id == explicit_table_id:
            continue

        size = obj.size_m
        if size is None:
            size = list(_default_size_m(obj.class_name))

        # Sanitize size.
        try:
            sx, sy, sz = (float(size[0]), float(size[1]), float(size[2]))
        except Exception:
            sx, sy, sz = _default_size_m(obj.class_name)

        sx = float(_clamp(abs(sx), 0.01, 3.0))
        sy = float(_clamp(abs(sy), 0.01, 3.0))
        sz = float(_clamp(abs(sz), 0.01, 3.0))

        primitive = obj.primitive or _primitive_for_class(obj.class_name)

        # Footprint heuristic.
        half_x = sx / 2.0
        half_y = sy / 2.0
        half_z = sz / 2.0

        support = _support_surface_for(obj.id)
        margin = 0.02
        x, y = _sample_non_overlapping_xy(support, half_x, half_y, margin=margin, rng=rng)
        z = support.z_top + half_z + 0.005

        body = ET.SubElement(worldbody, "body", name=obj.id, pos=f"{x:.4f} {y:.4f} {z:.4f}")
        if obj.movable:
            ET.SubElement(body, "freejoint")

        rgba = _rgba_for_class(obj.class_name)
        rgba_s = f"{rgba[0]:.3f} {rgba[1]:.3f} {rgba[2]:.3f} {rgba[3]:.3f}"

        if obj.movable:
            volume = max(1e-9, sx * sy * sz)
            density = _density_kg_m3(obj.material)
            mass = volume * density
            # Clamp to MuJoCo-friendly defaults.
            mass = float(_clamp(mass, 0.05, 20.0))

            if primitive == "cylinder":
                radius = 0.5 * min(sx, sy)
                height = sz
                ixx, iyy, izz = _cylinder_inertia_diaginertia(mass, radius=radius, height_full=height)
            elif primitive == "sphere":
                r = 0.5 * min(sx, sy, sz)
                i = (2.0 / 5.0) * mass * r * r
                ixx, iyy, izz = i, i, i
            else:
                ixx, iyy, izz = _box_inertia_diaginertia(mass, (sx, sy, sz))

            ET.SubElement(
                body,
                "inertial",
                pos="0 0 0",
                mass=f"{mass:.6g}",
                diaginertia=f"{ixx:.6g} {iyy:.6g} {izz:.6g}",
            )

        # Geom.
        if obj.mesh_file and obj.id in mesh_name_for_obj:
            ET.SubElement(
                body,
                "geom",
                type="mesh",
                mesh=mesh_name_for_obj[obj.id],
                rgba=rgba_s,
                contype="1",
                conaffinity="1",
            )
        else:
            if primitive == "cylinder":
                radius = 0.5 * min(sx, sy)
                half_h = 0.5 * sz
                ET.SubElement(
                    body,
                    "geom",
                    type="cylinder",
                    size=f"{radius:.4f} {half_h:.4f}",
                    rgba=rgba_s,
                    contype="1",
                    conaffinity="1",
                )
            elif primitive == "sphere":
                r = 0.5 * min(sx, sy, sz)
                ET.SubElement(
                    body,
                    "geom",
                    type="sphere",
                    size=f"{r:.4f}",
                    rgba=rgba_s,
                    contype="1",
                    conaffinity="1",
                )
            else:
                ET.SubElement(
                    body,
                    "geom",
                    type="box",
                    size=f"{half_x:.4f} {half_y:.4f} {half_z:.4f}",
                    rgba=rgba_s,
                    contype="1",
                    conaffinity="1",
                )

        placed.append(PlacedObject(obj=obj, pos_xyz=(x, y, z)))

    # Minimal camera to match editor defaults.
    ET.SubElement(worldbody, "camera", name="camera", pos="1 0 1.5", xyaxes="0 1 0 -1 0 0")

    return ET.tostring(root, encoding="unicode")
