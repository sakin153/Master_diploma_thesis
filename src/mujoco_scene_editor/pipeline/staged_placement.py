from __future__ import annotations

import random
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from mujoco_scene_editor.pipeline.scene_graph import RelationType, SceneGraph, SceneObject, index_objects


@dataclass(frozen=True)
class PlacementConfig:
    seed: int = 0
    attempts_per_object: int = 40
    clearance_z: float = 0.01
    max_penetration_m: float = 0.005

    # Workspace extents for sampling (rough room footprint).
    room_half_x: float = 2.0
    room_half_y: float = 2.0

    # Table defaults (used when anything needs a tabletop surface).
    table_top_z: float = 0.75
    table_half_x: float = 0.55
    table_half_y: float = 0.35


@dataclass
class _SupportSurface:
    id: str
    xy_center: tuple[float, float]
    half_xy: tuple[float, float]
    top_z: float

    def sample_xy(self, rng: random.Random, half_x: float, half_y: float, *, margin: float = 0.02) -> tuple[float, float]:
        cx, cy = self.xy_center
        hx, hy = self.half_xy
        xmin = cx - hx + half_x + margin
        xmax = cx + hx - half_x - margin
        ymin = cy - hy + half_y + margin
        ymax = cy + hy - half_y - margin
        if xmin > xmax:
            xmin = xmax = cx
        if ymin > ymax:
            ymin = ymax = cy
        return rng.uniform(xmin, xmax), rng.uniform(ymin, ymax)


def _default_size_m(class_name: str) -> tuple[float, float, float]:
    c = (class_name or "").lower()
    if any(k in c for k in ("desk", "table")):
        return 1.2, 0.8, 0.75
    if any(k in c for k in ("shelf", "bookcase")):
        return 1.0, 0.35, 1.8
    if any(k in c for k in ("chair", "stool")):
        return 0.5, 0.5, 0.9
    if any(k in c for k in ("monitor", "screen")):
        return 0.55, 0.2, 0.4
    if any(k in c for k in ("keyboard",)):
        return 0.45, 0.18, 0.04
    if any(k in c for k in ("bottle",)):
        return 0.08, 0.08, 0.26
    if any(k in c for k in ("cup", "mug")):
        return 0.09, 0.09, 0.11
    if any(k in c for k in ("laptop",)):
        return 0.34, 0.24, 0.03
    return 0.25, 0.25, 0.25


def _density_kg_m3(material: str | None) -> float:
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


def _box_inertia_diaginertia(
    mass: float, size_full: tuple[float, float, float]
) -> tuple[float, float, float]:
    x, y, z = size_full
    ixx = (1.0 / 12.0) * mass * (y * y + z * z)
    iyy = (1.0 / 12.0) * mass * (x * x + z * z)
    izz = (1.0 / 12.0) * mass * (x * x + y * y)
    return ixx, iyy, izz


def _primitive_for_obj(obj: SceneObject) -> str:
    if obj.primitive:
        return obj.primitive
    c = (obj.class_name or "").lower()
    if any(k in c for k in ("bottle", "cup", "mug")):
        return "cylinder"
    return "box"


def _iter_relations(graph: SceneGraph, *types: RelationType) -> Iterable[tuple[str, str, RelationType]]:
    for r in graph.relations:
        if r.type in types:
            yield r.subject, r.object, r.type


def _relation_target_map(graph: SceneGraph) -> dict[str, tuple[str, RelationType]]:
    # Prefer explicit targets; keep first match.
    out: dict[str, tuple[str, RelationType]] = {}
    for sub, obj, t in _iter_relations(
        graph,
        RelationType.on_floor,
        RelationType.against_wall,
        RelationType.on,
        RelationType.ontop,
    ):
        out.setdefault(sub, (obj, t))
    return out


def _needs_synthetic_table(graph: SceneGraph) -> bool:
    # Only if a relation needs a support that is missing from the object list.
    ids = {o.id for o in graph.objects}
    for r in graph.relations:
        if r.type not in (RelationType.on, RelationType.ontop):
            continue
        if r.object and r.object not in ids:
            return True
    return False


def _sorted_objects_for_staging(objs: list[SceneObject]) -> list[SceneObject]:
    def _rank(o: SceneObject) -> tuple[int, str]:
        cls = (o.class_name or "").lower()
        # Supports/furniture first, then medium, then small props.
        if any(k in cls for k in ("wall", "floor")):
            return 0, o.id
        if any(k in cls for k in ("table", "desk", "shelf", "bookcase")):
            return 1, o.id
        if any(k in cls for k in ("chair", "stool")):
            return 2, o.id
        return 3, o.id

    return sorted(list(objs), key=_rank)


def _ensure_table_anchor(root: ET.Element, surfaces: dict[str, _SupportSurface], config: PlacementConfig) -> str:
    worldbody = root.find("worldbody")
    assert worldbody is not None

    table_id = "anchor_table"
    if table_id in surfaces:
        return table_id

    thickness = 0.06
    top_z = config.table_top_z
    center_z = top_z - thickness / 2.0

    body = ET.SubElement(worldbody, "body", name=table_id, pos=f"0 0 {center_z:.4f}")
    ET.SubElement(
        body,
        "geom",
        name="table_top",
        type="box",
        size=f"{config.table_half_x:.6g} {config.table_half_y:.6g} {thickness/2.0:.6g}",
        rgba="0.55 0.45 0.35 1",
        contype="1",
        conaffinity="1",
    )

    surfaces[table_id] = _SupportSurface(
        id=table_id,
        xy_center=(0.0, 0.0),
        half_xy=(config.table_half_x, config.table_half_y),
        top_z=top_z,
    )

    return table_id


def _add_mesh_asset(
    root: ET.Element,
    *,
    mesh_name: str,
    mesh_file: str,
    mesh_scale: float | None,
) -> None:
    asset = root.find("asset")
    if asset is None:
        asset = ET.SubElement(root, "asset")

    if any(el.tag == "mesh" and el.get("name") == mesh_name for el in list(asset)):
        return

    s = 1.0
    if mesh_scale is not None:
        try:
            s = float(mesh_scale)
        except (TypeError, ValueError):
            s = 1.0
    ET.SubElement(asset, "mesh", name=mesh_name, file=mesh_file, scale=f"{s} {s} {s}")

def _add_object_body_placeholder(
    root: ET.Element,
    obj: SceneObject,
    *,
    pos_xyz: tuple[float, float, float],
    size_m: tuple[float, float, float],
) -> None:
    worldbody = root.find("worldbody")
    assert worldbody is not None

    x, y, z = pos_xyz
    sx, sy, sz = size_m
    hx, hy, hz = sx / 2.0, sy / 2.0, sz / 2.0

    body = ET.SubElement(worldbody, "body", name=obj.id, pos=f"{x:.4f} {y:.4f} {z:.4f}")
    if obj.movable:
        ET.SubElement(body, "freejoint")

        # Provide explicit inertia so MuJoCo can simulate robustly.
        density = _density_kg_m3(getattr(obj, "material", None))
        mass = max(0.01, min(density * sx * sy * sz, 250.0))
        ixx, iyy, izz = _box_inertia_diaginertia(mass, (sx, sy, sz))
        # Clamp away from zeros.
        ixx = max(ixx, 1e-6)
        iyy = max(iyy, 1e-6)
        izz = max(izz, 1e-6)
        ET.SubElement(
            body,
            "inertial",
            pos="0 0 0",
            mass=f"{mass:.6g}",
            diaginertia=f"{ixx:.6g} {iyy:.6g} {izz:.6g}",
        )

    # Collision placeholder.
    ET.SubElement(
        body,
        "geom",
        name=f"{obj.id}_placeholder_col",
        type="box",
        size=f"{hx:.6g} {hy:.6g} {hz:.6g}",
        rgba="0 0 0 0",
        contype="1",
        conaffinity="1",
    )

    # Visual placeholder (non-colliding).
    prim = _primitive_for_obj(obj)
    if prim == "sphere":
        ET.SubElement(
            body,
            "geom",
            name=f"{obj.id}_placeholder_vis",
            type="sphere",
            size=f"{max(hx, hy, hz):.6g}",
            rgba="0.7 0.7 0.7 1",
            contype="0",
            conaffinity="0",
        )
    elif prim == "cylinder":
        # MuJoCo cylinder size: radius, halfheight.
        r = max(hx, hy)
        ET.SubElement(
            body,
            "geom",
            name=f"{obj.id}_placeholder_vis",
            type="cylinder",
            size=f"{r:.6g} {hz:.6g}",
            rgba="0.7 0.7 0.7 1",
            contype="0",
            conaffinity="0",
        )
    else:
        ET.SubElement(
            body,
            "geom",
            name=f"{obj.id}_placeholder_vis",
            type="box",
            size=f"{hx:.6g} {hy:.6g} {hz:.6g}",
            rgba="0.7 0.7 0.7 1",
            contype="0",
            conaffinity="0",
        )


def _remove_body(root: ET.Element, body_name: str) -> None:
    worldbody = root.find("worldbody")
    if worldbody is None:
        return
    for b in list(worldbody.findall("body")):
        if (b.get("name") or "") == body_name:
            worldbody.remove(b)
            return


def _mj_min_contact_distance(mjcf_xml: str, *, mjcf_dir: Path) -> float | None:
    try:
        import mujoco
    except Exception:
        return None

    try:
        model = mujoco.MjModel.from_xml_string(mjcf_xml, str(mjcf_dir))
    except Exception:
        # If model can't load, treat as invalid.
        return -1e9

    data = mujoco.MjData(model)
    # Forward once; contact distances are meaningful.
    mujoco.mj_forward(model, data)

    if int(data.ncon) <= 0:
        return 0.0

    min_d = 1e9
    for i in range(int(data.ncon)):
        d = float(data.contact[i].dist)
        if d < min_d:
            min_d = d
    return float(min_d)


def _support_for_object(
    obj: SceneObject,
    rel_target: dict[str, tuple[str, RelationType]],
    *,
    by_id: dict[str, SceneObject],
    surfaces: dict[str, _SupportSurface],
    root: ET.Element,
    config: PlacementConfig,
) -> _SupportSurface:
    # Default: floor.
    target = rel_target.get(obj.id)
    if target is None:
        return surfaces["__floor__"]

    obj_id, rel_type = target
    if rel_type == RelationType.on_floor:
        return surfaces["__floor__"]

    # If relation points at an existing surface body, use it.
    if obj_id in surfaces:
        return surfaces[obj_id]

    # If the target is table-like but not yet present as a surface, use a synthetic anchor.
    tgt_obj = by_id.get(obj_id)
    if tgt_obj is not None:
        cls = (tgt_obj.class_name or "").lower()
        if any(k in cls for k in ("table", "desk")):
            table_id = _ensure_table_anchor(root, surfaces, config)
            return surfaces[table_id]

    # Fallback to tabletop if relation says on/ontop.
    if rel_type in (RelationType.on, RelationType.ontop):
        table_id = _ensure_table_anchor(root, surfaces, config)
        return surfaces[table_id]

    return surfaces["__floor__"]


def _candidate_xy_for_against_wall(
    rng: random.Random,
    config: PlacementConfig,
    half_x: float,
    half_y: float,
    *,
    margin: float = 0.02,
) -> tuple[float, float]:
    # Choose one of 4 walls and place near it.
    wall = rng.choice(["+x", "-x", "+y", "-y"])
    if wall in ("+x", "-x"):
        x = (config.room_half_x - half_x - margin) * (1.0 if wall == "+x" else -1.0)
        y = rng.uniform(-config.room_half_y + half_y + margin, config.room_half_y - half_y - margin)
    else:
        y = (config.room_half_y - half_y - margin) * (1.0 if wall == "+y" else -1.0)
        x = rng.uniform(-config.room_half_x + half_x + margin, config.room_half_x - half_x - margin)
    return x, y


def scene_graph_to_mjcf_staged(
    graph: SceneGraph,
    *,
    mjcf_dir: Path,
    config: PlacementConfig | None = None,
    replace_placeholders_with_hipoly: bool = True,
) -> str:
    """Robust staged placement pipeline.

    Stages:
      A) Use relations only (on_floor / against_wall / ontop / on)
      B) Place objects using placeholder collision geometry
      C) After each object, run collision + relation sanity checks
      D) Roll back the object placement and retry if invalid
      E) Replace placeholders with hi-poly visuals (if mesh_file provided)

    Output MJCF is best-effort and intended to be followed by existing
    stabilization passes.
    """

    cfg = config or PlacementConfig()
    rng = random.Random(cfg.seed)

    by_id = index_objects(graph)
    rel_target = _relation_target_map(graph)

    root = ET.Element("mujoco")
    ET.SubElement(root, "option", timestep="0.002")
    ET.SubElement(root, "asset")
    ET.SubElement(root, "worldbody")

    # Floor.
    worldbody = root.find("worldbody")
    assert worldbody is not None
    floor_body = ET.SubElement(worldbody, "body", name="floor", pos="0 0 0")
    ET.SubElement(
        floor_body,
        "geom",
        name="floor_geom",
        type="plane",
        size="2.5 2.5 0.01",
        rgba="0.5 0.5 0.5 1",
        contype="1",
        conaffinity="1",
    )

    surfaces: dict[str, _SupportSurface] = {
        "__floor__": _SupportSurface(
            id="__floor__",
            xy_center=(0.0, 0.0),
            half_xy=(cfg.room_half_x, cfg.room_half_y),
            top_z=0.0,
        )
    }

    if _needs_synthetic_table(graph):
        _ensure_table_anchor(root, surfaces, cfg)

    placed: set[str] = set()

    # Place objects.
    objects = _sorted_objects_for_staging(list(graph.objects))

    for obj in objects:
        if not obj.id:
            continue

        # Skip synthetic anchor ids.
        if obj.id.startswith("anchor_"):
            continue

        # Derive size.
        size = obj.size_m
        if size is None:
            sx, sy, sz = _default_size_m(obj.class_name)
        else:
            try:
                sx, sy, sz = float(size[0]), float(size[1]), float(size[2])
            except Exception:
                sx, sy, sz = _default_size_m(obj.class_name)

        sx = max(0.01, min(abs(sx), 3.0))
        sy = max(0.01, min(abs(sy), 3.0))
        sz = max(0.01, min(abs(sz), 3.0))
        hx, hy, hz = sx / 2.0, sy / 2.0, sz / 2.0

        # Determine support.
        support = _support_for_object(
            obj,
            rel_target,
            by_id=by_id,
            surfaces=surfaces,
            root=root,
            config=cfg,
        )

        # Try placements.
        success = False
        for _attempt in range(max(1, int(cfg.attempts_per_object))):
            # XY.
            tgt = rel_target.get(obj.id)
            if tgt is not None and tgt[1] == RelationType.against_wall:
                x, y = _candidate_xy_for_against_wall(rng, cfg, hx, hy)
            else:
                x, y = support.sample_xy(rng, hx, hy)

            # Z.
            z = support.top_z + hz + max(0.0, cfg.clearance_z)

            # Apply placement.
            _add_object_body_placeholder(root, obj, pos_xyz=(x, y, z), size_m=(sx, sy, sz))
            xml = ET.tostring(root, encoding="unicode")

            # Collision check (MuJoCo).
            min_d = _mj_min_contact_distance(xml, mjcf_dir=Path(mjcf_dir))
            if min_d is not None and float(min_d) < -abs(cfg.max_penetration_m):
                _remove_body(root, obj.id)
                continue

            # Relation check (simple).
            # - on_floor: should be near floor, not floating too high.
            if tgt is not None and tgt[1] == RelationType.on_floor:
                if z - hz > 0.05:
                    _remove_body(root, obj.id)
                    continue

            placed.add(obj.id)
            # Any placed object can act as a support surface for later `on`/`ontop`.
            surfaces[obj.id] = _SupportSurface(
                id=obj.id,
                xy_center=(x, y),
                half_xy=(hx, hy),
                top_z=z + hz,
            )
            success = True
            break

        if not success:
            # Rollback to safe default on floor center (still place, but deterministic).
            _remove_body(root, obj.id)
            x, y = rng.uniform(-0.3, 0.3), rng.uniform(-0.3, 0.3)
            z = surfaces["__floor__"].top_z + hz + 0.05
            _add_object_body_placeholder(root, obj, pos_xyz=(x, y, z), size_m=(sx, sy, sz))
            placed.add(obj.id)
            surfaces[obj.id] = _SupportSurface(
                id=obj.id,
                xy_center=(x, y),
                half_xy=(hx, hy),
                top_z=z + hz,
            )

    # Replace placeholders with hi-poly visuals.
    if replace_placeholders_with_hipoly:
        for obj in graph.objects:
            if obj.id not in placed:
                continue
            if not obj.mesh_file:
                continue

            worldbody = root.find("worldbody")
            if worldbody is None:
                continue
            body = next((b for b in list(worldbody.findall("body")) if (b.get("name") or "") == obj.id), None)
            if body is None:
                continue

            mesh_name = f"mesh_{obj.id}"
            _add_mesh_asset(root, mesh_name=mesh_name, mesh_file=obj.mesh_file, mesh_scale=obj.mesh_scale)

            # Add visual mesh geom (non-colliding). Keep collision placeholder.
            ET.SubElement(
                body,
                "geom",
                name=f"{obj.id}_hipoly_vis",
                type="mesh",
                mesh=mesh_name,
                rgba="0.8 0.8 0.8 1",
                contype="0",
                conaffinity="0",
            )

    return ET.tostring(root, encoding="unicode")
