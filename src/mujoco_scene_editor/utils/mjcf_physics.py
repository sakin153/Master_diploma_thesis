from __future__ import annotations

import math
import re
import zlib
import xml.etree.ElementTree as ET
from pathlib import Path

import trimesh


def _strip_xml_comments(mjcf_xml: str) -> tuple[str, bool]:
    # LLMs often emit decorative comments with repeated dashes like
    # <!-- ----- -->, which is invalid XML (contains "--" inside comment body).
    # Drop comments before parsing so structural repairs can still run.
    cleaned = re.sub(r"<!--[\s\S]*?-->", "", mjcf_xml)
    return cleaned, cleaned != mjcf_xml


def ensure_mesh_asset_scale_is_triplet(mjcf_xml: str) -> str:
    """Ensure every <mesh> scale attribute has 3 values.

    MuJoCo expects mesh scale to be a 3-vector. LLMs often emit a single scalar
    (e.g. scale="0.01"), which causes import failure.
    """

    xml_in, comments_removed = _strip_xml_comments(mjcf_xml)

    try:
        root = ET.fromstring(xml_in)
    except ET.ParseError:
        return mjcf_xml

    changed = False

    for mesh in root.iter("mesh"):
        scale = mesh.get("scale")
        if not scale:
            continue
        parts = [p for p in scale.replace(",", " ").split() if p]
        if len(parts) == 3:
            continue
        if len(parts) == 1:
            s = parts[0]
            mesh.set("scale", f"{s} {s} {s}")
            changed = True
        elif len(parts) == 2:
            a, b = parts
            mesh.set("scale", f"{a} {b} {b}")
            changed = True
        elif len(parts) > 3:
            mesh.set("scale", " ".join(parts[:3]))
            changed = True

    if not changed and not comments_removed:
        return mjcf_xml
    return ET.tostring(root, encoding="unicode")


def _target_max_dim_m(name_or_file: str) -> float:
    s = (name_or_file or "").lower()

    # Tabletop objects
    if any(k in s for k in ("mug", "cup")):
        return 0.12
    if "vase" in s:
        return 0.25
    if any(k in s for k in ("laptop", "notebook", "computer")):
        return 0.36
    if any(k in s for k in ("bottle", "can")):
        return 0.28
    if any(k in s for k in ("plate", "dish")):
        return 0.28

    # Furniture
    if any(k in s for k in ("chair", "armchair")):
        return 1.0
    if any(k in s for k in ("coffee_table", "coffeetable")):
        return 1.0
    if "table" in s:
        return 1.4
    if any(k in s for k in ("sofa", "couch")):
        return 2.0
    if "bed" in s:
        return 2.1

    # Fallback: medium prop
    return 0.6


def _mesh_max_dim(path: Path) -> float | None:
    try:
        loaded = trimesh.load_mesh(path, process=False)
    except Exception:
        return None

    mesh: trimesh.Trimesh | None = None
    if isinstance(loaded, trimesh.Trimesh):
        mesh = loaded
    elif isinstance(loaded, trimesh.Scene):
        geoms = [g for g in loaded.geometry.values() if isinstance(g, trimesh.Trimesh)]
        if not geoms:
            return None
        try:
            mesh = trimesh.util.concatenate(geoms)
        except Exception:
            # Fallback to first geometry
            mesh = geoms[0]
    else:
        return None

    try:
        ext = mesh.bounding_box.extents
        max_dim = float(max(ext))
    except Exception:
        return None

    if not (max_dim > 0):
        return None
    return max_dim


def _mesh_min_max_z(path: Path) -> tuple[float, float] | None:
    try:
        loaded = trimesh.load_mesh(path, process=False)
    except Exception:
        return None

    mesh: trimesh.Trimesh | None = None
    if isinstance(loaded, trimesh.Trimesh):
        mesh = loaded
    elif isinstance(loaded, trimesh.Scene):
        geoms = [g for g in loaded.geometry.values() if isinstance(g, trimesh.Trimesh)]
        if not geoms:
            return None
        try:
            mesh = trimesh.util.concatenate(geoms)
        except Exception:
            mesh = geoms[0]
    else:
        return None

    try:
        bounds = mesh.bounds
        z0 = float(bounds[0][2])
        z1 = float(bounds[1][2])
    except Exception:
        return None

    if z0 > z1:
        z0, z1 = z1, z0
    return z0, z1


def _parse_float_list(value: str | None) -> list[float]:
    if not value:
        return []
    out: list[float] = []
    for p in value.replace(",", " ").split():
        try:
            out.append(float(p))
        except ValueError:
            continue
    return out


def _parse_vec3(value: str | None) -> list[float]:
    vals = _parse_float_list(value)
    if len(vals) >= 3:
        return [vals[0], vals[1], vals[2]]
    while len(vals) < 3:
        vals.append(0.0)
    return vals


def _parse_scale_vec3(value: str | None) -> tuple[float, float, float]:
    vals = _parse_float_list(value)
    if len(vals) == 1:
        s = vals[0]
        return s, s, s
    if len(vals) >= 3:
        return vals[0], vals[1], vals[2]
    return 1.0, 1.0, 1.0


def _mesh_bounds(path: Path) -> tuple[list[float], list[float]] | None:
    try:
        loaded = trimesh.load_mesh(path, process=False)
    except Exception:
        return None

    mesh: trimesh.Trimesh | None = None
    if isinstance(loaded, trimesh.Trimesh):
        mesh = loaded
    elif isinstance(loaded, trimesh.Scene):
        geoms = [g for g in loaded.geometry.values() if isinstance(g, trimesh.Trimesh)]
        if not geoms:
            return None
        try:
            mesh = trimesh.util.concatenate(geoms)
        except Exception:
            mesh = geoms[0]
    else:
        return None

    try:
        bounds = mesh.bounds
        bmin = [float(bounds[0][0]), float(bounds[0][1]), float(bounds[0][2])]
        bmax = [float(bounds[1][0]), float(bounds[1][1]), float(bounds[1][2])]
    except Exception:
        return None

    return bmin, bmax


def _mesh_vertices(path: Path) -> list[list[float]] | None:
    try:
        loaded = trimesh.load_mesh(path, process=False)
    except Exception:
        return None

    mesh: trimesh.Trimesh | None = None
    if isinstance(loaded, trimesh.Trimesh):
        mesh = loaded
    elif isinstance(loaded, trimesh.Scene):
        geoms = [g for g in loaded.geometry.values() if isinstance(g, trimesh.Trimesh)]
        if not geoms:
            return None
        try:
            mesh = trimesh.util.concatenate(geoms)
        except Exception:
            mesh = geoms[0]
    else:
        return None

    try:
        return [[float(v[0]), float(v[1]), float(v[2])] for v in mesh.vertices]
    except Exception:
        return None


def _apply_scale_to_bounds(
    bmin: list[float],
    bmax: list[float],
    sx: float,
    sy: float,
    sz: float,
) -> tuple[list[float], list[float]]:
    mins = [bmin[0] * sx, bmin[1] * sy, bmin[2] * sz]
    maxs = [bmax[0] * sx, bmax[1] * sy, bmax[2] * sz]
    out_min = [min(mins[0], maxs[0]), min(mins[1], maxs[1]), min(mins[2], maxs[2])]
    out_max = [max(mins[0], maxs[0]), max(mins[1], maxs[1]), max(mins[2], maxs[2])]
    return out_min, out_max


def _bottom_top_vertex_ratio(
    verts: list[list[float]],
    *,
    scale: tuple[float, float, float],
    euler_xyz_deg: tuple[float, float, float],
) -> float | None:
    if not verts:
        return None

    rot = _mat_mul(_rot_z(euler_xyz_deg[2]), _mat_mul(_rot_y(euler_xyz_deg[1]), _rot_x(euler_xyz_deg[0])))

    sx, sy, sz = scale
    z_vals: list[float] = []
    for v in verts:
        p = [v[0] * sx, v[1] * sy, v[2] * sz]
        pr = _mat_apply(rot, p)
        z_vals.append(pr[2])

    if not z_vals:
        return None

    z_min = min(z_vals)
    z_max = max(z_vals)
    span = z_max - z_min
    if span <= 1e-9:
        return None

    band = 0.1 * span
    bottom = sum(1 for z in z_vals if z <= z_min + band)
    top = sum(1 for z in z_vals if z >= z_max - band)
    return (bottom + 1.0) / (top + 1.0)


def _mat_mul(a: list[list[float]], b: list[list[float]]) -> list[list[float]]:
    return [
        [a[0][0] * b[0][0] + a[0][1] * b[1][0] + a[0][2] * b[2][0],
         a[0][0] * b[0][1] + a[0][1] * b[1][1] + a[0][2] * b[2][1],
         a[0][0] * b[0][2] + a[0][1] * b[1][2] + a[0][2] * b[2][2]],
        [a[1][0] * b[0][0] + a[1][1] * b[1][0] + a[1][2] * b[2][0],
         a[1][0] * b[0][1] + a[1][1] * b[1][1] + a[1][2] * b[2][1],
         a[1][0] * b[0][2] + a[1][1] * b[1][2] + a[1][2] * b[2][2]],
        [a[2][0] * b[0][0] + a[2][1] * b[1][0] + a[2][2] * b[2][0],
         a[2][0] * b[0][1] + a[2][1] * b[1][1] + a[2][2] * b[2][1],
         a[2][0] * b[0][2] + a[2][1] * b[1][2] + a[2][2] * b[2][2]],
    ]


def _mat_apply(m: list[list[float]], v: list[float]) -> list[float]:
    return [
        m[0][0] * v[0] + m[0][1] * v[1] + m[0][2] * v[2],
        m[1][0] * v[0] + m[1][1] * v[1] + m[1][2] * v[2],
        m[2][0] * v[0] + m[2][1] * v[1] + m[2][2] * v[2],
    ]


def _rot_x(deg: float) -> list[list[float]]:
    r = math.radians(deg)
    c = math.cos(r)
    s = math.sin(r)
    return [[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]]


def _rot_y(deg: float) -> list[list[float]]:
    r = math.radians(deg)
    c = math.cos(r)
    s = math.sin(r)
    return [[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]]


def _rot_z(deg: float) -> list[list[float]]:
    r = math.radians(deg)
    c = math.cos(r)
    s = math.sin(r)
    return [[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]]


def _quat_to_mat3(w: float, x: float, y: float, z: float) -> list[list[float]]:
    n = math.sqrt(w * w + x * x + y * y + z * z)
    if n <= 1e-12:
        return [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    w, x, y, z = w / n, x / n, y / n, z / n
    return [
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ]


def _geom_rotation_mat3(geom_el: ET.Element) -> list[list[float]]:
    quat_vals = _parse_float_list(geom_el.get("quat"))
    if len(quat_vals) >= 4:
        return _quat_to_mat3(quat_vals[0], quat_vals[1], quat_vals[2], quat_vals[3])

    euler_vals = _parse_vec3(geom_el.get("euler"))
    # MuJoCo default eulerseq is xyz; use Rz * Ry * Rx for local point rotation.
    return _mat_mul(_rot_z(euler_vals[2]), _mat_mul(_rot_y(euler_vals[1]), _rot_x(euler_vals[0])))


def _transform_bounds(
    bmin: list[float],
    bmax: list[float],
    rot: list[list[float]],
    trans: list[float],
) -> tuple[list[float], list[float]]:
    corners = [
        [bmin[0], bmin[1], bmin[2]],
        [bmin[0], bmin[1], bmax[2]],
        [bmin[0], bmax[1], bmin[2]],
        [bmin[0], bmax[1], bmax[2]],
        [bmax[0], bmin[1], bmin[2]],
        [bmax[0], bmin[1], bmax[2]],
        [bmax[0], bmax[1], bmin[2]],
        [bmax[0], bmax[1], bmax[2]],
    ]

    mins = [float("inf"), float("inf"), float("inf")]
    maxs = [float("-inf"), float("-inf"), float("-inf")]
    for c in corners:
        p = _mat_apply(rot, c)
        p = [p[0] + trans[0], p[1] + trans[1], p[2] + trans[2]]
        for i in range(3):
            mins[i] = min(mins[i], p[i])
            maxs[i] = max(maxs[i], p[i])

    return mins, maxs


def _collect_mesh_bounds(
    root: ET.Element,
    *,
    mjcf_dir: Path,
) -> dict[str, tuple[list[float], list[float]]]:
    out: dict[str, tuple[list[float], list[float]]] = {}
    asset = root.find("asset")
    if asset is None:
        return out

    for mesh_el in asset.findall("mesh"):
        name = mesh_el.get("name")
        file_attr = mesh_el.get("file")
        if not name or not file_attr:
            continue

        mesh_path = Path(file_attr)
        if not mesh_path.is_absolute():
            mesh_path = (mjcf_dir / mesh_path).resolve()
        if not mesh_path.exists() or not mesh_path.is_file():
            continue

        bounds = _mesh_bounds(mesh_path)
        if bounds is None:
            continue

        sx, sy, sz = _parse_scale_vec3(mesh_el.get("scale"))
        out[name] = _apply_scale_to_bounds(bounds[0], bounds[1], sx, sy, sz)

    return out


def _geom_local_bounds(
    geom_el: ET.Element,
    mesh_bounds: dict[str, tuple[list[float], list[float]]],
) -> tuple[list[float], list[float]] | None:
    geom_type = (geom_el.get("type") or "").strip().lower()
    if not geom_type and geom_el.get("mesh"):
        geom_type = "mesh"

    local: tuple[list[float], list[float]] | None = None
    if geom_type == "mesh":
        mesh_name = geom_el.get("mesh")
        if mesh_name and mesh_name in mesh_bounds:
            local = mesh_bounds[mesh_name]
    elif geom_type == "box":
        size = _parse_float_list(geom_el.get("size"))
        if len(size) >= 3:
            sx, sy, sz = abs(size[0]), abs(size[1]), abs(size[2])
            local = ([-sx, -sy, -sz], [sx, sy, sz])
    elif geom_type == "sphere":
        size = _parse_float_list(geom_el.get("size"))
        if len(size) >= 1:
            r = abs(size[0])
            local = ([-r, -r, -r], [r, r, r])
    elif geom_type in ("cylinder", "capsule"):
        size = _parse_float_list(geom_el.get("size"))
        if len(size) >= 2:
            r = abs(size[0])
            h = abs(size[1])
            local = ([-r, -r, -h], [r, r, h])
    elif geom_type == "ellipsoid":
        size = _parse_float_list(geom_el.get("size"))
        if len(size) >= 3:
            sx, sy, sz = abs(size[0]), abs(size[1]), abs(size[2])
            local = ([-sx, -sy, -sz], [sx, sy, sz])

    if local is None:
        return None

    rot = _geom_rotation_mat3(geom_el)
    pos = _parse_vec3(geom_el.get("pos"))
    return _transform_bounds(local[0], local[1], rot, pos)


def _body_local_bounds(
    body_el: ET.Element,
    mesh_bounds: dict[str, tuple[list[float], list[float]]],
) -> tuple[list[float], list[float]] | None:
    mins: list[float] | None = None
    maxs: list[float] | None = None

    for child in list(body_el):
        if child.tag != "geom":
            continue
        gb = _geom_local_bounds(child, mesh_bounds)
        if gb is None:
            continue
        if mins is None:
            mins = [gb[0][0], gb[0][1], gb[0][2]]
            maxs = [gb[1][0], gb[1][1], gb[1][2]]
        else:
            for i in range(3):
                mins[i] = min(mins[i], gb[0][i])
                maxs[i] = max(maxs[i], gb[1][i])

    if mins is None or maxs is None:
        return None
    return mins, maxs


def _z_overlap(min_a: float, max_a: float, min_b: float, max_b: float, *, eps: float = 1e-3) -> bool:
    return min(max_a, max_b) >= max(min_a, min_b) - eps


def _is_free_motion_joint_element(el: ET.Element) -> bool:
    if el.tag == "freejoint":
        return True
    if el.tag == "joint" and (el.get("type") or "").strip().lower() == "free":
        return True
    return False


def _body_has_free_motion_joint(body_el: ET.Element) -> bool:
    return any(_is_free_motion_joint_element(ch) for ch in list(body_el))


def recenter_freejoint_body_frames(
    mjcf_xml: str,
    *,
    mjcf_dir: Path,
    min_recenter_distance: float = 0.05,
) -> str:
    """Shift freejoint body frames toward local geom centers.

    LLM-generated scenes often place geoms far from body origins while keeping
    inertial at the body origin. This causes unrealistic dynamics. We recenter
    simple freejoint bodies (no nested bodies/articulation) so geom bounds are
    close to local origin.
    """

    xml_in, comments_removed = _strip_xml_comments(mjcf_xml)

    try:
        root = ET.fromstring(xml_in)
    except ET.ParseError:
        return mjcf_xml

    mjcf_dir = Path(mjcf_dir)
    mesh_bounds = _collect_mesh_bounds(root, mjcf_dir=mjcf_dir)
    changed = False

    for body in root.iter("body"):
        direct_children = list(body)
        has_freejoint = _body_has_free_motion_joint(body)
        if not has_freejoint:
            continue

        # Skip articulated and nested composite bodies.
        if any(
            ch.tag == "joint" and (ch.get("type") or "").strip().lower() != "free"
            for ch in direct_children
        ):
            continue
        if any(ch.tag == "body" for ch in direct_children):
            continue

        bb = _body_local_bounds(body, mesh_bounds)
        if bb is None:
            continue

        center = [
            0.5 * (bb[0][0] + bb[1][0]),
            0.5 * (bb[0][1] + bb[1][1]),
            0.5 * (bb[0][2] + bb[1][2]),
        ]
        if max(abs(center[0]), abs(center[1]), abs(center[2])) < max(0.0, min_recenter_distance):
            continue

        body_pos = _parse_vec3(body.get("pos"))
        body_pos[0] += center[0]
        body_pos[1] += center[1]
        body_pos[2] += center[2]
        body.set("pos", f"{body_pos[0]:.6g} {body_pos[1]:.6g} {body_pos[2]:.6g}")

        # Keep world pose unchanged by shifting child geoms in the opposite
        # direction in local frame.
        for child in direct_children:
            if child.tag != "geom":
                continue
            p = _parse_vec3(child.get("pos"))
            p[0] -= center[0]
            p[1] -= center[1]
            p[2] -= center[2]
            child.set("pos", f"{p[0]:.6g} {p[1]:.6g} {p[2]:.6g}")

        # Preserve explicitly authored non-zero inertial offsets, but keep
        # default zero inertials at the recentered origin.
        inertial = next((ch for ch in direct_children if ch.tag == "inertial"), None)
        if inertial is not None:
            ip = _parse_vec3(inertial.get("pos"))
            if max(abs(ip[0]), abs(ip[1]), abs(ip[2])) > 1e-9:
                ip[0] -= center[0]
                ip[1] -= center[1]
                ip[2] -= center[2]
                inertial.set("pos", f"{ip[0]:.6g} {ip[1]:.6g} {ip[2]:.6g}")

        changed = True

    if not changed and not comments_removed:
        return mjcf_xml
    return ET.tostring(root, encoding="unicode")


def ground_large_top_level_bodies(
    mjcf_xml: str,
    *,
    mjcf_dir: Path,
    floor_z: float = 0.0,
    clearance: float = 0.01,
    min_large_dim_m: float = 0.65,
    max_groundable_bottom_z: float = 1.5,
    snap_tolerance: float = 0.03,
) -> str:
    """Snap large top-level bodies near the floor onto the floor level.

    This stabilizes common furniture placement issues where tables/chairs are
    emitted either floating above the ground or intersecting it at generation
    time. Small props are ignored.
    """

    xml_in, comments_removed = _strip_xml_comments(mjcf_xml)

    try:
        root = ET.fromstring(xml_in)
    except ET.ParseError:
        return mjcf_xml

    worldbody = root.find("worldbody")
    if worldbody is None:
        return mjcf_xml if not comments_removed else ET.tostring(root, encoding="unicode")

    mjcf_dir = Path(mjcf_dir)
    mesh_bounds = _collect_mesh_bounds(root, mjcf_dir=mjcf_dir)
    target_floor = floor_z + max(0.0, clearance)
    changed = False

    for body in worldbody.findall("body"):
        name_lc = (body.get("name") or "").strip().lower()
        # Skip synthetic anchors used by the graph placement pipeline.
        # These intentionally define support surfaces (e.g. a tabletop) at
        # a specific height, so snapping them to the floor breaks placement.
        if name_lc.startswith("anchor_"):
            continue

        direct_children = list(body)

        # Skip articulated roots and nested compound structures.
        if any(
            ch.tag == "joint" and (ch.get("type") or "").strip().lower() != "free"
            for ch in direct_children
        ):
            continue
        if any(ch.tag == "body" for ch in direct_children):
            continue

        bb = _body_local_bounds(body, mesh_bounds)
        if bb is None:
            continue

        dx = bb[1][0] - bb[0][0]
        dy = bb[1][1] - bb[0][1]
        dz = bb[1][2] - bb[0][2]
        max_dim = max(abs(dx), abs(dy), abs(dz))
        if max_dim < max(0.0, min_large_dim_m):
            continue

        body_pos = _parse_vec3(body.get("pos"))
        world_min_z = body_pos[2] + bb[0][2]
        if world_min_z > max_groundable_bottom_z:
            continue

        if abs(world_min_z - target_floor) <= max(0.0, snap_tolerance):
            continue

        body_pos[2] += target_floor - world_min_z
        body.set("pos", f"{body_pos[0]:.6g} {body_pos[1]:.6g} {body_pos[2]:.6g}")
        changed = True

    if not changed and not comments_removed:
        return mjcf_xml
    return ET.tostring(root, encoding="unicode")


def _natural_name_key(name: str) -> tuple[str, int, str]:
    m = re.search(r"^(.*?)(\d+)$", name)
    if not m:
        return name, -1, name
    return m.group(1), int(m.group(2)), name


def enforce_top_level_instance_counts(
    mjcf_xml: str,
    *,
    class_counts: dict[str, int],
) -> str:
    """Trim excess top-level bodies for explicitly requested class counts.

    This is intentionally conservative: it only removes surplus bodies for
    classes with a positive requested count and leaves under-count cases as-is.
    """

    if not class_counts:
        return mjcf_xml

    xml_in, comments_removed = _strip_xml_comments(mjcf_xml)

    try:
        root = ET.fromstring(xml_in)
    except ET.ParseError:
        return mjcf_xml

    worldbody = root.find("worldbody")
    if worldbody is None:
        return mjcf_xml if not comments_removed else ET.tostring(root, encoding="unicode")

    class_tokens: dict[str, tuple[str, ...]] = {
        "chair": ("chair", "seat", "stul", "стул"),
        "table": ("table", "desk", "stol", "стол"),
        "sofa": ("sofa", "couch", "диван"),
    }

    changed = False
    bodies = worldbody.findall("body")

    for cls, requested in class_counts.items():
        target = int(requested)
        if target <= 0:
            continue

        tokens = class_tokens.get(cls, (cls,))
        matched: list[ET.Element] = []
        for body in bodies:
            name = (body.get("name") or "").lower()
            if not name:
                continue
            if any(tok in name for tok in tokens):
                matched.append(body)

        if len(matched) <= target:
            continue

        matched.sort(key=lambda b: _natural_name_key((b.get("name") or "").lower()))
        for body in matched[target:]:
            worldbody.remove(body)
            changed = True

    if not changed and not comments_removed:
        return mjcf_xml
    return ET.tostring(root, encoding="unicode")


def normalize_objaverse_geom_orientation(
    mjcf_xml: str,
    *,
    mjcf_dir: Path,
    dominance_ratio: float = 1.2,
) -> str:
    """Rotate Objaverse mesh geoms into a likely Z-up orientation.

    Many Objaverse assets are authored with Y-up/X-up axes. This function adds
    a default geom `euler` only when missing, based on mesh bbox axis dominance.
    """

    xml_in, comments_removed = _strip_xml_comments(mjcf_xml)

    try:
        root = ET.fromstring(xml_in)
    except ET.ParseError:
        return mjcf_xml

    mjcf_dir = Path(mjcf_dir)
    mesh_bounds = _collect_mesh_bounds(root, mjcf_dir=mjcf_dir)
    if not mesh_bounds and not comments_removed:
        return mjcf_xml

    # Build per-mesh orientation hint.
    mesh_hint: dict[str, str] = {}
    mesh_scale: dict[str, tuple[float, float, float]] = {}
    mesh_vertices: dict[str, list[list[float]]] = {}
    furniture_tokens = ("chair", "table", "desk", "stool", "bench", "sofa", "couch", "bed")
    container_tokens = ("mug", "cup", "glass", "bottle", "vase")
    asset = root.find("asset")
    if asset is not None:
        for mesh_el in asset.findall("mesh"):
            mesh_name = mesh_el.get("name")
            file_attr = mesh_el.get("file")
            if not mesh_name or not file_attr:
                continue
            if "objaverse" not in file_attr.replace("\\", "/"):
                continue

            mesh_path = Path(file_attr)
            if not mesh_path.is_absolute():
                mesh_path = (mjcf_dir / mesh_path).resolve()

            mesh_scale[mesh_name] = _parse_scale_vec3(mesh_el.get("scale"))
            if mesh_path.exists() and mesh_path.is_file():
                verts = _mesh_vertices(mesh_path)
                if verts:
                    mesh_vertices[mesh_name] = verts

            b = mesh_bounds.get(mesh_name)
            if b is None:
                continue

            ex = b[1][0] - b[0][0]
            ey = b[1][1] - b[0][1]
            ez = b[1][2] - b[0][2]
            ex, ey, ez = abs(ex), abs(ey), abs(ez)

            # For table-like assets, prefer the flattest orientation (smallest
            # bbox height axis as Z) when it is meaningfully better.
            mesh_key = f"{mesh_name} {Path(file_attr).stem}".lower()
            is_furniture_like = any(tok in mesh_key for tok in furniture_tokens)
            is_container_like = any(tok in mesh_key for tok in container_tokens)
            prefer_more_bottom = not is_container_like
            is_table_like = any(tok in mesh_key for tok in ("table", "desk", "stol", "стол"))
            if is_table_like:
                dims = [ex, ey, ez]
                best_axis = min(range(3), key=lambda i: dims[i])
                if best_axis != 2 and dims[best_axis] <= max(1e-9, ez) * 0.9:
                    if best_axis == 1:
                        mesh_hint[mesh_name] = "90 0 0"
                    elif best_axis == 0:
                        mesh_hint[mesh_name] = "0 -90 0"
                    continue

            axis = 0
            axis_size = ex
            if ey > axis_size:
                axis = 1
                axis_size = ey
            if ez > axis_size:
                axis = 2
                axis_size = ez

            if axis == 2:
                # If Z is already dominant, we can still correct upside-down
                # assets using vertex support asymmetry (identity vs 180° flip).
                verts = mesh_vertices.get(mesh_name)
                scale = mesh_scale.get(mesh_name, (1.0, 1.0, 1.0))
                if verts:
                    scored: list[tuple[str, float]] = []
                    for hint, euler in (
                        ("0 0 0", (0.0, 0.0, 0.0)),
                        ("180 0 0", (180.0, 0.0, 0.0)),
                        ("0 180 0", (0.0, 180.0, 0.0)),
                    ):
                        ratio = _bottom_top_vertex_ratio(verts, scale=scale, euler_xyz_deg=euler)
                        if ratio is not None:
                            scored.append((hint, ratio))

                    if len(scored) >= 2:
                        current_ratio = next((r for h, r in scored if h == "0 0 0"), None)
                        if current_ratio is not None:
                            if prefer_more_bottom:
                                best_hint, best_ratio = max(scored, key=lambda x: x[1])
                                if best_hint != "0 0 0" and best_ratio > current_ratio + 0.05:
                                    mesh_hint[mesh_name] = best_hint
                            else:
                                best_hint, best_ratio = min(scored, key=lambda x: x[1])
                                if best_hint != "0 0 0" and best_ratio + 0.05 < current_ratio:
                                    mesh_hint[mesh_name] = best_hint
                continue

            # Furniture meshes are often almost isotropic between Y and Z after
            # scale normalization, so use a softer threshold there.
            local_dominance_ratio = 1.05 if is_furniture_like else dominance_ratio
            if axis_size <= max(ez, 1e-9) * local_dominance_ratio:
                continue

            if axis == 1:
                # Y-up -> Z-up
                mesh_hint[mesh_name] = "90 0 0"
            elif axis == 0:
                # X-up -> Z-up
                mesh_hint[mesh_name] = "0 -90 0"

            # Resolve ±90 sign ambiguity by preferring the orientation with
            # fewer vertices near the support side than near the top side.
            base_hint = mesh_hint.get(mesh_name)
            if base_hint is None:
                continue

            verts = mesh_vertices.get(mesh_name)
            scale = mesh_scale.get(mesh_name, (1.0, 1.0, 1.0))
            if not verts:
                continue

            if base_hint == "90 0 0":
                base_euler = (90.0, 0.0, 0.0)
                alt_hint = "-90 0 0"
                alt_euler = (-90.0, 0.0, 0.0)
            elif base_hint == "0 -90 0":
                base_euler = (0.0, -90.0, 0.0)
                alt_hint = "0 90 0"
                alt_euler = (0.0, 90.0, 0.0)
            else:
                continue

            r_base = _bottom_top_vertex_ratio(verts, scale=scale, euler_xyz_deg=base_euler)
            r_alt = _bottom_top_vertex_ratio(verts, scale=scale, euler_xyz_deg=alt_euler)
            if r_base is None or r_alt is None:
                continue

            # Small margin avoids noisy flips on nearly symmetric assets.
            if prefer_more_bottom:
                if r_alt > r_base + 0.05:
                    mesh_hint[mesh_name] = alt_hint
            else:
                if r_alt + 0.05 < r_base:
                    mesh_hint[mesh_name] = alt_hint

    changed = False
    for geom in root.iter("geom"):
        if geom.get("quat") is not None or geom.get("euler") is not None:
            continue

        mesh_name = geom.get("mesh")
        if not mesh_name:
            continue

        hint = mesh_hint.get(mesh_name)
        if hint is None:
            continue

        geom.set("euler", hint)
        changed = True

    if not changed and not comments_removed:
        return mjcf_xml
    return ET.tostring(root, encoding="unicode")


def ensure_small_top_level_bodies_are_movable(
    mjcf_xml: str,
    *,
    mjcf_dir: Path,
    max_dynamic_dim_m: float = 1.2,
    max_named_dynamic_dim_m: float = 2.4,
) -> str:
    """Add `<freejoint/>` to likely movable top-level objects missing joints.

    Heuristic:
    - Body is direct child of `<worldbody>`
    - No direct `<joint>`/`<freejoint>` already present
    - No descendant joints (to avoid changing articulated robots)
    - No plane geom
    - Object max bbox dimension is small enough (`max_dynamic_dim_m`)
    """

    xml_in, comments_removed = _strip_xml_comments(mjcf_xml)

    try:
        root = ET.fromstring(xml_in)
    except ET.ParseError:
        return mjcf_xml

    worldbody = root.find("worldbody")
    if worldbody is None:
        return mjcf_xml if not comments_removed else ET.tostring(root, encoding="unicode")

    mjcf_dir = Path(mjcf_dir)
    mesh_bounds = _collect_mesh_bounds(root, mjcf_dir=mjcf_dir)
    changed = False

    for body in worldbody.findall("body"):
        direct_children = list(body)

        body_name = (body.get("name") or "").lower().strip()
        # Skip synthetic anchors created by the graph pipeline (support surfaces,
        # frames, etc.). These should remain static.
        if body_name.startswith("anchor_"):
            continue

        if any(ch.tag in ("freejoint", "joint") for ch in direct_children):
            continue

        # Skip articulated roots.
        has_descendant_joint = any(
            el.tag == "joint" and (el.get("type") or "").strip().lower() != "free"
            for el in body.iter()
            if el is not body
        )
        if has_descendant_joint:
            continue

        # Skip bodies that are explicitly support/ground planes.
        has_plane = False
        for ch in direct_children:
            if ch.tag != "geom":
                continue
            gt = (ch.get("type") or "").lower().strip()
            if gt == "plane":
                has_plane = True
                break
        if has_plane:
            continue

        bb = _body_local_bounds(body, mesh_bounds)
        if bb is None:
            continue

        ex = bb[1][0] - bb[0][0]
        ey = bb[1][1] - bb[0][1]
        ez = bb[1][2] - bb[0][2]
        max_dim = max(abs(ex), abs(ey), abs(ez))

        movable_name_tokens = ("chair", "table", "desk", "stool", "bench", "sofa", "couch")
        is_named_movable = any(tok in body_name for tok in movable_name_tokens)
        dim_limit = max_named_dynamic_dim_m if is_named_movable else max_dynamic_dim_m

        if max_dim > dim_limit:
            continue

        body.insert(0, ET.Element("freejoint"))
        changed = True

    if not changed and not comments_removed:
        return mjcf_xml
    return ET.tostring(root, encoding="unicode")


def ensure_default_gravity(
    mjcf_xml: str,
    *,
    gravity: tuple[float, float, float] = (0.0, 0.0, -9.81),
) -> str:
    """Ensure <option gravity="..."> is present.

    MuJoCo defaults gravity to (0,0,-9.81), but generators sometimes explicitly
    set it to 0 or omit <option> entirely. Keeping it explicit improves
    reproducibility across tools.
    """

    xml_in, comments_removed = _strip_xml_comments(mjcf_xml)
    try:
        root = ET.fromstring(xml_in)
    except ET.ParseError:
        return mjcf_xml

    opt = root.find("option")
    changed = False
    if opt is None:
        opt = ET.Element("option")
        root.insert(0, opt)
        changed = True

    if opt.get("gravity") is None:
        gx, gy, gz = gravity
        opt.set("gravity", f"{gx:.6g} {gy:.6g} {gz:.6g}")
        changed = True

    if not changed and not comments_removed:
        return mjcf_xml
    return ET.tostring(root, encoding="unicode")


def ensure_inertial_top_level_bodies_are_movable(
    mjcf_xml: str,
    *,
    mjcf_dir: Path,
    max_dynamic_dim_m: float = 2.4,
) -> str:
    """If a top-level body declares inertial parameters but has no joint, add a freejoint.

    This targets a common LLM failure mode: emitting <inertial ...> (suggesting
    the object should be dynamic) but forgetting to add <freejoint/>.
    """

    xml_in, comments_removed = _strip_xml_comments(mjcf_xml)

    try:
        root = ET.fromstring(xml_in)
    except ET.ParseError:
        return mjcf_xml

    worldbody = root.find("worldbody")
    if worldbody is None:
        return mjcf_xml if not comments_removed else ET.tostring(root, encoding="unicode")

    mjcf_dir = Path(mjcf_dir)
    mesh_bounds = _collect_mesh_bounds(root, mjcf_dir=mjcf_dir)
    changed = False

    for body in worldbody.findall("body"):
        direct_children = list(body)
        if any(ch.tag in ("freejoint", "joint") for ch in direct_children):
            continue

        inertial = body.find("inertial")
        if inertial is None:
            continue

        # If inertial mass is explicitly 0, treat as static.
        try:
            mass = float(inertial.get("mass") or "0")
        except ValueError:
            mass = 0.0
        if mass <= 0.0:
            continue

        # Avoid touching articulated roots.
        has_descendant_joint = any(
            el.tag == "joint" and (el.get("type") or "").strip().lower() != "free"
            for el in body.iter()
            if el is not body
        )
        if has_descendant_joint:
            continue

        # Skip ground/support planes.
        if any(
            ch.tag == "geom" and (ch.get("type") or "").strip().lower() == "plane"
            for ch in direct_children
        ):
            continue

        bb = _body_local_bounds(body, mesh_bounds)
        if bb is None:
            continue

        dx = abs(bb[1][0] - bb[0][0])
        dy = abs(bb[1][1] - bb[0][1])
        dz = abs(bb[1][2] - bb[0][2])
        if max(dx, dy, dz) > max(0.0, max_dynamic_dim_m):
            continue

        body.insert(0, ET.Element("freejoint"))
        changed = True

    if not changed and not comments_removed:
        return mjcf_xml
    return ET.tostring(root, encoding="unicode")


def ensure_freejoint_bodies_have_collision_proxy(
    mjcf_xml: str,
    *,
    mjcf_dir: Path,
    proxy_name_suffix: str = "_proxy_collision",
    rgba: str = "0 0 0 0",
    friction: str = "1.0 0.02 0.002",
    min_half_extent_m: float = 0.01,
) -> str:
    """Ensure each freejoint body has at least one collidable geom.

    If an object is represented only by visual meshes (e.g. mesh geoms with
    contype=0/conaffinity=0), it will not collide and can look "non-physical".
    This pass adds an invisible primitive box proxy approximating the body's
    bounds to guarantee collisions.
    """

    xml_in, comments_removed = _strip_xml_comments(mjcf_xml)
    try:
        root = ET.fromstring(xml_in)
    except ET.ParseError:
        return mjcf_xml

    worldbody = root.find("worldbody")
    if worldbody is None:
        return mjcf_xml if not comments_removed else ET.tostring(root, encoding="unicode")

    mjcf_dir = Path(mjcf_dir)
    mesh_bounds = _collect_mesh_bounds(root, mjcf_dir=mjcf_dir)
    changed = False

    def _geom_is_collidable(g: ET.Element) -> bool:
        # MuJoCo defaults contype/conaffinity to 1.
        ct = g.get("contype")
        ca = g.get("conaffinity")
        if ct is not None and ct.strip() == "0":
            return False
        if ca is not None and ca.strip() == "0":
            return False
        return True

    for body in worldbody.iter("body"):
        if not _body_has_free_motion_joint(body):
            continue

        direct_children = list(body)
        # Skip if proxy already exists.
        if any(
            ch.tag == "geom" and (ch.get("name") or "").endswith(proxy_name_suffix)
            for ch in direct_children
        ):
            continue

        geoms = [ch for ch in direct_children if ch.tag == "geom"]
        if not geoms:
            continue

        if any(_geom_is_collidable(g) for g in geoms):
            continue

        bb = _body_local_bounds(body, mesh_bounds)
        if bb is None:
            continue

        hx = max(min_half_extent_m, 0.5 * abs(bb[1][0] - bb[0][0]))
        hy = max(min_half_extent_m, 0.5 * abs(bb[1][1] - bb[0][1]))
        hz = max(min_half_extent_m, 0.5 * abs(bb[1][2] - bb[0][2]))
        cx = 0.5 * (bb[0][0] + bb[1][0])
        cy = 0.5 * (bb[0][1] + bb[1][1])
        cz = 0.5 * (bb[0][2] + bb[1][2])

        base_name = (body.get("name") or "obj") + proxy_name_suffix
        proxy = ET.Element(
            "geom",
            {
                "name": base_name,
                "type": "box",
                "size": f"{hx:.6g} {hy:.6g} {hz:.6g}",
                "pos": f"{cx:.6g} {cy:.6g} {cz:.6g}",
                "rgba": rgba,
                "friction": friction,
                "contype": "1",
                "conaffinity": "1",
            },
        )

        # Prefer placing after inertial/joint definitions.
        insert_at = len(direct_children)
        for i, ch in enumerate(direct_children):
            if ch.tag in ("inertial", "joint", "freejoint"):
                insert_at = i + 1
        body.insert(insert_at, proxy)
        changed = True

    if not changed and not comments_removed:
        return mjcf_xml
    return ET.tostring(root, encoding="unicode")


def stabilize_free_motion_joints(
    mjcf_xml: str,
    *,
    damping: float = 0.2,
    armature: float = 0.01,
) -> str:
    """Normalize free-motion joints and add mild damping for stability.

    Replaces `<freejoint/>` with `<joint type="free" .../>` and ensures
    existing free joints have damping/armature unless already specified.
    """

    xml_in, comments_removed = _strip_xml_comments(mjcf_xml)

    try:
        root = ET.fromstring(xml_in)
    except ET.ParseError:
        return mjcf_xml

    changed = False
    damping_s = f"{max(0.0, damping):.6g}"
    armature_s = f"{max(0.0, armature):.6g}"

    for body in root.iter("body"):
        children = list(body)
        for idx, ch in enumerate(children):
            if ch.tag != "freejoint":
                continue

            attrs = {"type": "free"}
            if damping > 0.0:
                attrs["damping"] = damping_s
            if armature > 0.0:
                attrs["armature"] = armature_s
            replacement = ET.Element("joint", attrs)

            body.remove(ch)
            body.insert(idx, replacement)
            changed = True

        for ch in list(body):
            if ch.tag != "joint":
                continue
            if (ch.get("type") or "").strip().lower() != "free":
                continue

            if damping > 0.0 and ch.get("damping") is None:
                ch.set("damping", damping_s)
                changed = True
            if armature > 0.0 and ch.get("armature") is None:
                ch.set("armature", armature_s)
                changed = True

    if not changed and not comments_removed:
        return mjcf_xml
    return ET.tostring(root, encoding="unicode")


def stabilize_contact_friction(
    mjcf_xml: str,
    *,
    floor_friction: str = "1.2 0.02 0.002",
    movable_friction: str = "1.0 0.02 0.002",
) -> str:
    """Set conservative default friction values to reduce uncontrolled drift."""

    xml_in, comments_removed = _strip_xml_comments(mjcf_xml)

    try:
        root = ET.fromstring(xml_in)
    except ET.ParseError:
        return mjcf_xml

    changed = False

    for geom in root.iter("geom"):
        gt = (geom.get("type") or "").strip().lower()
        if gt == "plane" and geom.get("friction") is None:
            geom.set("friction", floor_friction)
            changed = True

    for body in root.iter("body"):
        if not _body_has_free_motion_joint(body):
            continue
        for ch in list(body):
            if ch.tag != "geom":
                continue
            gt = (ch.get("type") or "").strip().lower()
            if gt == "plane":
                continue
            if ch.get("friction") is None:
                ch.set("friction", movable_friction)
                changed = True

    if not changed and not comments_removed:
        return mjcf_xml
    return ET.tostring(root, encoding="unicode")


def stabilize_small_container_bases(
    mjcf_xml: str,
    *,
    mjcf_dir: Path,
    name_tokens: tuple[str, ...] = ("mug", "cup", "glass", "bottle", "vase"),
    min_radius: float = 0.008,
    min_halfheight: float = 0.003,
    max_halfheight: float = 0.02,
) -> str:
    """Add invisible primitive collision bases for small mesh containers.

    This reduces perpetual rocking from uneven mesh bottoms by using the mesh as
    visual geometry and a simple cylinder as collision support.
    """

    xml_in, comments_removed = _strip_xml_comments(mjcf_xml)

    try:
        root = ET.fromstring(xml_in)
    except ET.ParseError:
        return mjcf_xml

    mjcf_dir = Path(mjcf_dir)
    mesh_bounds = _collect_mesh_bounds(root, mjcf_dir=mjcf_dir)
    changed = False

    for body in root.iter("body"):
        if not _body_has_free_motion_joint(body):
            continue

        body_name = (body.get("name") or "").lower()
        if not any(tok in body_name for tok in name_tokens):
            continue

        direct_children = list(body)
        mesh_geoms = [
            ch
            for ch in direct_children
            if ch.tag == "geom" and (((ch.get("type") or "").strip().lower() == "mesh") or ch.get("mesh"))
        ]
        if not mesh_geoms:
            continue

        # Skip if base was already created by this pass.
        if any(
            ch.tag == "geom" and (ch.get("name") or "").endswith("_base_collision")
            for ch in direct_children
        ):
            continue

        bb = _body_local_bounds(body, mesh_bounds)
        if bb is None:
            continue

        dx = abs(bb[1][0] - bb[0][0])
        dy = abs(bb[1][1] - bb[0][1])
        dz = abs(bb[1][2] - bb[0][2])
        radius = max(min_radius, 0.35 * min(dx, dy))
        half_h = max(min_halfheight, min(max_halfheight, 0.06 * max(dz, min(dx, dy))))

        cx = 0.5 * (bb[0][0] + bb[1][0])
        cy = 0.5 * (bb[0][1] + bb[1][1])
        cz = bb[0][2] + half_h

        # Keep mesh as visual-only to avoid jagged-bottom collision artifacts.
        for g in mesh_geoms:
            if g.get("contype") is None:
                g.set("contype", "0")
                changed = True
            if g.get("conaffinity") is None:
                g.set("conaffinity", "0")
                changed = True

        base_name = f"{(body.get('name') or 'obj')}_base_collision"
        base_geom = ET.Element(
            "geom",
            {
                "name": base_name,
                "type": "cylinder",
                "size": f"{radius:.6g} {half_h:.6g}",
                "pos": f"{cx:.6g} {cy:.6g} {cz:.6g}",
                "rgba": "0 0 0 0",
                "friction": "1.1 0.02 0.002",
            },
        )

        # Prefer placing after inertial/joint definitions.
        insert_at = len(direct_children)
        for i, ch in enumerate(direct_children):
            if ch.tag in ("inertial", "joint", "freejoint"):
                insert_at = i + 1
        body.insert(insert_at, base_geom)
        changed = True

    if not changed and not comments_removed:
        return mjcf_xml
    return ET.tostring(root, encoding="unicode")


def resolve_freejoint_xy_overlaps(
    mjcf_xml: str,
    *,
    mjcf_dir: Path,
    margin: float = 0.02,
    iterations: int = 8,
    push_against_static: bool = False,
) -> str:
    """Resolve obvious XY overlaps between movable bodies using circle repulsion.

    Uses per-body XY footprint radius estimated from geom bounds and only pushes
    bodies when Z-ranges overlap.
    """

    xml_in, comments_removed = _strip_xml_comments(mjcf_xml)

    try:
        root = ET.fromstring(xml_in)
    except ET.ParseError:
        return mjcf_xml

    worldbody = root.find("worldbody")
    if worldbody is None:
        return mjcf_xml if not comments_removed else ET.tostring(root, encoding="unicode")

    mjcf_dir = Path(mjcf_dir)
    mesh_bounds = _collect_mesh_bounds(root, mjcf_dir=mjcf_dir)

    info: list[dict[str, object]] = []
    for body in worldbody.findall("body"):
        bb = _body_local_bounds(body, mesh_bounds)
        if bb is None:
            continue

        pos = _parse_vec3(body.get("pos"))
        z_min = pos[2] + bb[0][2]
        z_max = pos[2] + bb[1][2]
        dx = bb[1][0] - bb[0][0]
        dy = bb[1][1] - bb[0][1]
        radius = 0.5 * math.hypot(dx, dy)
        if radius <= 1e-6:
            radius = 0.05

        has_freejoint = _body_has_free_motion_joint(body)
        info.append(
            {
                "body": body,
                "pos": pos,
                "radius": radius,
                "zmin": z_min,
                "zmax": z_max,
                "free": has_freejoint,
            }
        )

    if not info:
        return mjcf_xml if not comments_removed else ET.tostring(root, encoding="unicode")

    dynamic_idx = [i for i, d in enumerate(info) if bool(d["free"])]
    static_idx = [i for i, d in enumerate(info) if not bool(d["free"])]
    if not dynamic_idx:
        return mjcf_xml if not comments_removed else ET.tostring(root, encoding="unicode")

    changed = False

    def _push_pair(i: int, j: int, split: float) -> bool:
        di = info[i]
        dj = info[j]
        if not _z_overlap(float(di["zmin"]), float(di["zmax"]), float(dj["zmin"]), float(dj["zmax"])):
            return False

        pi = di["pos"]
        pj = dj["pos"]
        assert isinstance(pi, list) and isinstance(pj, list)

        dx = float(pj[0]) - float(pi[0])
        dy = float(pj[1]) - float(pi[1])
        dist = math.hypot(dx, dy)
        target = float(di["radius"]) + float(dj["radius"]) + margin
        overlap = target - dist
        if overlap <= 1e-9:
            return False

        if dist <= 1e-9:
            # Deterministic fallback direction from body names.
            bi = di["body"]
            bj = dj["body"]
            assert isinstance(bi, ET.Element) and isinstance(bj, ET.Element)
            seed = (bi.get("name") or "") + "|" + (bj.get("name") or "")
            h = sum(ord(c) for c in seed)
            ang = (h % 360) * math.pi / 180.0
            ux, uy = math.cos(ang), math.sin(ang)
        else:
            ux, uy = dx / dist, dy / dist

        move = overlap * split
        pi[0] = float(pi[0]) - ux * move
        pi[1] = float(pi[1]) - uy * move
        pj[0] = float(pj[0]) + ux * move
        pj[1] = float(pj[1]) + uy * move
        return True

    for _ in range(max(1, iterations)):
        moved = False

        # Dynamic-dynamic: split push.
        for ii, i in enumerate(dynamic_idx):
            for j in dynamic_idx[ii + 1 :]:
                if _push_pair(i, j, 0.5):
                    moved = True

        # Dynamic-static: optional, because coarse static AABBs can over-push
        # valid tabletop placements.
        if push_against_static:
            for i in dynamic_idx:
                for j in static_idx:
                    di = info[i]
                    dj = info[j]
                    if not _z_overlap(float(di["zmin"]), float(di["zmax"]), float(dj["zmin"]), float(dj["zmax"])):
                        continue

                    pi = di["pos"]
                    pj = dj["pos"]
                    assert isinstance(pi, list) and isinstance(pj, list)

                    dx = float(pi[0]) - float(pj[0])
                    dy = float(pi[1]) - float(pj[1])
                    dist = math.hypot(dx, dy)
                    target = float(di["radius"]) + float(dj["radius"]) + margin
                    overlap = target - dist
                    if overlap <= 1e-9:
                        continue

                    if dist <= 1e-9:
                        dx, dy = 1.0, 0.0
                        dist = 1.0
                    ux, uy = dx / dist, dy / dist
                    pi[0] = float(pi[0]) + ux * overlap
                    pi[1] = float(pi[1]) + uy * overlap
                    moved = True

        if not moved:
            break
        changed = True

    if changed:
        for d in info:
            if not bool(d["free"]):
                continue
            body = d["body"]
            pos = d["pos"]
            assert isinstance(body, ET.Element) and isinstance(pos, list)
            body.set("pos", f"{float(pos[0]):.6g} {float(pos[1]):.6g} {float(pos[2]):.6g}")

    if not changed and not comments_removed:
        return mjcf_xml
    return ET.tostring(root, encoding="unicode")


def ensure_freejoint_bodies_above_floor(
    mjcf_xml: str,
    *,
    mjcf_dir: Path,
    floor_z: float = 0.0,
    clearance: float = 0.01,
) -> str:
    """Lift freejoint bodies so their lowest geom point starts above the floor.

    This reduces initial interpenetration for LLM-generated poses, especially for
    mesh-based furniture where body `pos.z=0` often puts geometry below the plane.
    """

    xml_in, comments_removed = _strip_xml_comments(mjcf_xml)

    try:
        root = ET.fromstring(xml_in)
    except ET.ParseError:
        return mjcf_xml

    mjcf_dir = Path(mjcf_dir)
    mesh_bounds = _collect_mesh_bounds(root, mjcf_dir=mjcf_dir)
    changed = False

    target_floor = floor_z + max(0.0, clearance)

    for body in root.iter("body"):
        has_freejoint = _body_has_free_motion_joint(body)
        if not has_freejoint:
            continue

        body_pos = _parse_vec3(body.get("pos"))
        bb = _body_local_bounds(body, mesh_bounds)
        if bb is None:
            continue

        world_min_z = body_pos[2] + bb[0][2]
        if world_min_z < target_floor:
            body_pos[2] += target_floor - world_min_z
            body.set("pos", f"{body_pos[0]:.6g} {body_pos[1]:.6g} {body_pos[2]:.6g}")
            changed = True

    if not changed and not comments_removed:
        return mjcf_xml
    return ET.tostring(root, encoding="unicode")


def ensure_freejoint_bodies_above_supports(
    mjcf_xml: str,
    *,
    mjcf_dir: Path,
    clearance: float = 0.01,
    max_support_gap_m: float = 0.6,
) -> str:
    """Lift freejoint bodies so they start above their nearest support surface.

    This is a more general version of `ensure_freejoint_bodies_above_floor`.
    It helps with common "on table" penetrations after collision proxies are
    introduced (e.g. a laptop proxy intersecting a table top).

    Heuristic: find static support geoms (floor planes + thin horizontal boxes
    that look like table/counter tops). For each freejoint body, if its XY AABB
    overlaps a support and its min Z is below support_top_z + clearance, lift it.

    Assumptions: bodies are upright (no significant rotations). This matches the
    rest of the stabilization pipeline.
    """

    xml_in, comments_removed = _strip_xml_comments(mjcf_xml)

    try:
        root = ET.fromstring(xml_in)
    except ET.ParseError:
        return mjcf_xml

    worldbody = root.find("worldbody")
    if worldbody is None:
        return mjcf_xml if not comments_removed else ET.tostring(root, encoding="unicode")

    mjcf_dir = Path(mjcf_dir)
    mesh_bounds = _collect_mesh_bounds(root, mjcf_dir=mjcf_dir)
    changed = False
    clearance = max(0.0, float(clearance))
    max_support_gap_m = max(0.0, float(max_support_gap_m))

    def _aabb_overlap_2d(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> bool:
        ax0, ax1, ay0, ay1 = a
        bx0, bx1, by0, by1 = b
        return (min(ax1, bx1) >= max(ax0, bx0)) and (min(ay1, by1) >= max(ay0, by0))

    def _iter_bodies_with_world_pos(parent: ET.Element, parent_world_pos: list[float]):
        for body in list(parent):
            if body.tag != "body":
                continue
            p = _parse_vec3(body.get("pos"))
            world_pos = [parent_world_pos[0] + p[0], parent_world_pos[1] + p[1], parent_world_pos[2] + p[2]]
            yield body, world_pos
            yield from _iter_bodies_with_world_pos(body, world_pos)

    # Collect support surfaces.
    supports: list[dict[str, float | tuple[float, float, float, float] | str]] = []

    for body, body_world_pos in _iter_bodies_with_world_pos(worldbody, [0.0, 0.0, 0.0]):
        if _body_has_free_motion_joint(body):
            continue

        for geom in list(body):
            if geom.tag != "geom":
                continue

            gtype = (geom.get("type") or "").strip().lower() or "box"
            gname = (geom.get("name") or "").strip().lower()
            gpos = _parse_vec3(geom.get("pos"))
            gx = body_world_pos[0] + gpos[0]
            gy = body_world_pos[1] + gpos[1]
            gz = body_world_pos[2] + gpos[2]

            if gtype == "plane":
                # Treat any plane as a horizontal support (typically the floor).
                supports.append(
                    {
                        "name": gname,
                        "top_z": gz,
                        # Big footprint; still requires XY overlap check.
                        "xy_aabb": (-1e9, 1e9, -1e9, 1e9),
                    }
                )
                continue

            if gtype != "box":
                continue

            size = _parse_vec3(geom.get("size"))
            hx, hy, hz = abs(size[0]), abs(size[1]), abs(size[2])
            if hx <= 0.0 or hy <= 0.0 or hz <= 0.0:
                continue

            # Heuristic: thin, horizontal-ish box => likely a support surface.
            thin = hz <= min(hx, hy) * 0.6
            wide = (hx >= 0.15 and hy >= 0.15) or ("top" in gname)
            named_like_support = any(tok in gname for tok in ("top", "table", "desk", "counter", "bench"))

            if not (thin and (wide or named_like_support)):
                continue

            top_z = gz + hz
            supports.append(
                {
                    "name": gname,
                    "top_z": float(top_z),
                    "xy_aabb": (gx - hx, gx + hx, gy - hy, gy + hy),
                }
            )

    if not supports:
        return mjcf_xml if not comments_removed else ET.tostring(root, encoding="unicode")

    # Adjust freejoint bodies.
    for body, body_world_pos in _iter_bodies_with_world_pos(worldbody, [0.0, 0.0, 0.0]):
        if not _body_has_free_motion_joint(body):
            continue

        bb = _body_local_bounds(body, mesh_bounds)
        if bb is None:
            continue

        obj_min_z = body_world_pos[2] + bb[0][2]
        obj_xy_aabb = (
            body_world_pos[0] + bb[0][0],
            body_world_pos[0] + bb[1][0],
            body_world_pos[1] + bb[0][1],
            body_world_pos[1] + bb[1][1],
        )

        best_top_z: float | None = None
        for s in supports:
            top_z = float(s["top_z"])  # type: ignore[arg-type]
            if top_z > obj_min_z + max_support_gap_m:
                continue
            if not _aabb_overlap_2d(obj_xy_aabb, s["xy_aabb"]):  # type: ignore[arg-type]
                continue
            if best_top_z is None or top_z > best_top_z:
                best_top_z = top_z

        if best_top_z is None:
            continue

        target_min_z = best_top_z + clearance
        if obj_min_z < target_min_z:
            # Move this body in its local frame. For top-level bodies, this is
            # equivalent to world translation; nested bodies remain approximate.
            local_pos = _parse_vec3(body.get("pos"))
            local_pos[2] += target_min_z - obj_min_z
            body.set("pos", f"{local_pos[0]:.6g} {local_pos[1]:.6g} {local_pos[2]:.6g}")
            changed = True

    if not changed and not comments_removed:
        return mjcf_xml
    return ET.tostring(root, encoding="unicode")


def normalize_objaverse_mesh_scales(mjcf_xml: str, *, mjcf_dir: Path) -> str:
    """Normalize Objaverse mesh scales to consistent real-world sizes.

    For each <asset><mesh file="assets/objaverse/...">, load the referenced mesh,
    compute its bbox max dimension, and set a uniform scale so the final object
    matches a typical size (by name/file heuristics). Preserves proportions.
    """

    xml_in, comments_removed = _strip_xml_comments(mjcf_xml)

    try:
        root = ET.fromstring(xml_in)
    except ET.ParseError:
        return mjcf_xml

    mjcf_dir = Path(mjcf_dir)
    changed = False

    for mesh_el in root.iter("mesh"):
        file_attr = mesh_el.get("file")
        if not file_attr:
            continue

        # Only touch Objaverse assets.
        if "objaverse" not in file_attr.replace("\\", "/"):
            continue

        mesh_path = Path(file_attr)
        if not mesh_path.is_absolute():
            mesh_path = (mjcf_dir / mesh_path).resolve()

        if not mesh_path.exists() or not mesh_path.is_file():
            continue

        max_dim = _mesh_max_dim(mesh_path)
        if max_dim is None:
            continue

        key = (mesh_el.get("name") or "") + " " + mesh_path.stem
        target = _target_max_dim_m(key)

        scale = target / max_dim
        # Clamp to avoid absurd outcomes on corrupted meshes.
        if scale < 1e-4:
            scale = 1e-4
        elif scale > 100.0:
            scale = 100.0

        s = f"{scale:.6g}"
        mesh_el.set("scale", f"{s} {s} {s}")
        changed = True

    if not changed and not comments_removed:
        return mjcf_xml
    return ET.tostring(root, encoding="unicode")


def sanitize_mjcf_schema(mjcf_xml: str) -> str:
    """Fix common LLM schema violations for MuJoCo MJCF.

        Currently handles:
        - <flag ...> frequently contains invalid attributes for the target MuJoCo
            version. We remove all <flag> elements to avoid schema violations.
        - If a removed <flag> had solver="..." and its parent <option> does not,
            we move it to <option solver="...">.
        - Normalize common joint type aliases (URDF-style) into MuJoCo MJCF joint
            types (e.g. revolute->hinge, prismatic->slide).
    """

    xml_in, comments_removed = _strip_xml_comments(mjcf_xml)

    try:
        root = ET.fromstring(xml_in)
    except ET.ParseError:
        return mjcf_xml

    changed = False

    def _parse_scale_triplet(value: str | None) -> tuple[float, float, float] | None:
        if not value:
            return None
        parts = [p for p in value.replace(",", " ").split() if p]
        if len(parts) == 1:
            parts = [parts[0], parts[0], parts[0]]
        elif len(parts) == 2:
            parts = [parts[0], parts[1], parts[1]]
        elif len(parts) >= 3:
            parts = parts[:3]
        else:
            return None

        try:
            return (float(parts[0]), float(parts[1]), float(parts[2]))
        except ValueError:
            return None

    def _fmt_scale_triplet(scale: tuple[float, float, float]) -> str:
        return " ".join(f"{v:.6g}" for v in scale)

    def _scale_suffix(scale: tuple[float, float, float]) -> str:
        # Deterministic, name-safe suffix.
        raw = "x".join(f"{v:.6g}" for v in scale)
        safe = re.sub(r"[^0-9A-Za-z]+", "_", raw).strip("_")
        return f"__s{safe}"

    # Remove <flag> elements (optional; often malformed). Preserve solver by
    # moving flag@solver to option@solver when applicable.
    for opt in root.iter("option"):
        for child in list(opt):
            if child.tag != "flag":
                continue
            solver = child.get("solver")
            if solver is not None and opt.get("solver") is None:
                opt.set("solver", solver)
            opt.remove(child)
            changed = True

    # If any flags exist outside <option> (unexpected), drop their attributes by
    # removing them from their parent is hard without parent pointers; instead,
    # we rewrite by clearing attributes so parsing won't fail.
    for flag in root.iter("flag"):
        flag.attrib.clear()
        changed = True

    # Some generators emit non-schema `ref*` attributes on <mesh> (e.g. refscale).
    # These are optional; dropping them keeps the file importable.
    for mesh_el in root.iter("mesh"):
        for attr in list(mesh_el.attrib.keys()):
            if attr.startswith("ref"):
                del mesh_el.attrib[attr]
                changed = True

    # Normalize non-MuJoCo joint type names.
    joint_type_map = {
        # URDF / common robotics naming
        "revolute": "hinge",
        "continuous": "hinge",
        "prismatic": "slide",
        "floating": "free",
        "float": "free",
        # Some LLMs emit these.
        "linear": "slide",
        "rotary": "hinge",
    }
    for joint_el in root.iter("joint"):
        jtype = (joint_el.get("type") or "").strip().lower()
        if not jtype:
            continue
        mapped = joint_type_map.get(jtype)
        if mapped and mapped != jtype:
            joint_el.set("type", mapped)
            changed = True

    # Some generations reference undefined materials (e.g. material="self" on
    # floor geoms). MuJoCo import fails hard in that case, so drop unresolved
    # material references.
    material_names: set[str] = set()
    asset = root.find("asset")
    if asset is not None:
        for material_el in asset.findall("material"):
            name = material_el.get("name")
            if name:
                material_names.add(name)

    for geom_el in root.iter("geom"):
        material_name = geom_el.get("material")
        if not material_name:
            continue
        if material_name not in material_names:
            del geom_el.attrib["material"]
            changed = True

    # Some generators incorrectly set `scale` on <geom type="mesh" ...>, but MuJoCo
    # expects scaling to happen on the referenced <asset><mesh ... scale="..."/>.
    # We migrate geom@scale into the corresponding asset mesh scale. If multiple
    # different geom scales reference the same mesh, we duplicate the mesh asset
    # with unique names and remap geoms to preserve semantics.
    asset = root.find("asset")
    if asset is not None:
        mesh_assets: dict[str, ET.Element] = {}
        existing_mesh_names: set[str] = set()
        for mesh_el in asset.findall("mesh"):
            name = mesh_el.get("name")
            if not name:
                continue
            mesh_assets[name] = mesh_el
            existing_mesh_names.add(name)

        # Pre-scan geom scales per mesh name.
        scales_by_mesh: dict[str, set[tuple[str, str, str]]] = {}
        for geom in root.iter("geom"):
            if geom.get("scale") is None:
                continue
            mesh_name = geom.get("mesh")
            if not mesh_name:
                continue
            parsed = _parse_scale_triplet(geom.get("scale"))
            if parsed is None:
                continue
            norm = tuple(f"{v:.6g}" for v in parsed)  # type: ignore[assignment]
            scales_by_mesh.setdefault(mesh_name, set()).add(norm)  # type: ignore[arg-type]

        # Build mapping (mesh_name, geom_scale_norm) -> target mesh name.
        remap: dict[tuple[str, tuple[str, str, str]], str] = {}
        for mesh_name, norms in scales_by_mesh.items():
            if mesh_name not in mesh_assets:
                continue

            identity = ("1", "1", "1")
            non_identity = [n for n in norms if n != identity]

            base_el = mesh_assets[mesh_name]
            base_scale = _parse_scale_triplet(base_el.get("scale")) or (1.0, 1.0, 1.0)

            # If only one non-identity scale exists *and* no identity-scale use
            # exists, fold it into the base mesh.
            if len(non_identity) == 1 and identity not in norms:
                s = tuple(float(x) for x in non_identity[0])
                merged = (base_scale[0] * s[0], base_scale[1] * s[1], base_scale[2] * s[2])
                base_el.set("scale", _fmt_scale_triplet(merged))
                remap[(mesh_name, non_identity[0])] = mesh_name
                changed = True
                continue

            # Otherwise create a variant per non-identity scale.
            for norm in non_identity:
                s = tuple(float(x) for x in norm)
                merged = (base_scale[0] * s[0], base_scale[1] * s[1], base_scale[2] * s[2])
                suffix = _scale_suffix(s)

                candidate = f"{mesh_name}{suffix}"
                if candidate in existing_mesh_names:
                    i = 2
                    while f"{candidate}_{i}" in existing_mesh_names:
                        i += 1
                    candidate = f"{candidate}_{i}"

                new_el = ET.Element("mesh", dict(base_el.attrib))
                new_el.set("name", candidate)
                new_el.set("scale", _fmt_scale_triplet(merged))
                asset.append(new_el)

                existing_mesh_names.add(candidate)
                remap[(mesh_name, norm)] = candidate
                changed = True

        # Apply remapping and drop geom@scale.
        for geom in root.iter("geom"):
            scale_attr = geom.get("scale")
            if scale_attr is None:
                continue
            mesh_name = geom.get("mesh")
            parsed = _parse_scale_triplet(scale_attr)
            if mesh_name and parsed is not None:
                norm = tuple(f"{v:.6g}" for v in parsed)  # type: ignore[assignment]
                target = remap.get((mesh_name, norm))
                if target is not None:
                    geom.set("mesh", target)

            # Always remove invalid geom@scale (even if we couldn't migrate it).
            del geom.attrib["scale"]
            changed = True

    if not changed and not comments_removed:
        return mjcf_xml
    return ET.tostring(root, encoding="unicode")


def repair_mesh_file_geoms(mjcf_xml: str, *, mjcf_dir: Path | None = None) -> str:
    """Repair a common non-MJCF pattern: <geom mesh="path/to/file.obj" .../>.

    In MuJoCo MJCF, geom@mesh references an <asset><mesh name="..." file="..."/>.
    Some generators (LLMs) put a file path directly into geom@mesh and omit the
    asset section entirely, which makes MuJoCo/robits import fail.

    This pass:
    - Detects geom@mesh values that look like mesh file paths.
    - Creates corresponding <asset><mesh name=... file=... scale=.../> entries.
    - Rewrites geoms to type="mesh" and mesh="<asset name>".
    - Migrates geom@size (scalar/triplet) into mesh@scale and removes geom@size.
    - If the referenced file cannot be found on disk (when mjcf_dir is provided),
      replaces the geom with a conservative box placeholder to avoid hard import
      failure.
    """

    xml_in, comments_removed = _strip_xml_comments(mjcf_xml)
    try:
        root = ET.fromstring(xml_in)
    except ET.ParseError:
        return mjcf_xml

    mjcf_dir_resolved = None
    if mjcf_dir is not None:
        try:
            mjcf_dir_resolved = Path(mjcf_dir).expanduser().resolve()
        except Exception:
            mjcf_dir_resolved = Path(mjcf_dir).expanduser()

    mesh_exts = {".obj", ".stl", ".ply", ".glb", ".gltf", ".dae"}

    def _looks_like_mesh_file(value: str) -> bool:
        v = (value or "").strip()
        if not v:
            return False
        try:
            suf = Path(v).suffix.lower()
        except Exception:
            return False
        if suf in mesh_exts:
            return True
        # Also treat anything with a directory separator + a dot as a file path.
        if ("/" in v or "\\" in v) and "." in v:
            return True
        return False

    def _parse_floats(value: str | None) -> list[float]:
        if not value:
            return []
        parts = [p for p in value.replace(",", " ").split() if p]
        out: list[float] = []
        for p in parts:
            try:
                out.append(float(p))
            except ValueError:
                return []
        return out

    def _fmt_triplet(values: tuple[float, float, float]) -> str:
        return " ".join(f"{v:.6g}" for v in values)

    def _safe_name(s: str) -> str:
        safe = re.sub(r"[^0-9A-Za-z_]+", "_", (s or "").strip())
        safe = safe.strip("_")
        return safe or "mesh"

    def _resolve_mesh_path(mesh_value: str) -> Path | None:
        if not mesh_value:
            return None
        p = Path(mesh_value)
        if p.is_absolute():
            return p if p.exists() else None
        if mjcf_dir_resolved is None:
            return None
        cand = mjcf_dir_resolved / p
        if cand.exists():
            return cand
        return None

    changed = False

    asset = root.find("asset")
    if asset is None:
        asset = ET.Element("asset")
        # Keep assets close to the top for readability.
        root.insert(0, asset)
        changed = True

    existing_mesh_names: set[str] = set()
    mesh_assets: dict[str, ET.Element] = {}
    for mesh_el in asset.findall("mesh"):
        name = mesh_el.get("name")
        if not name:
            continue
        existing_mesh_names.add(name)
        mesh_assets[name] = mesh_el

    # Reuse same asset for repeated file references.
    by_file_value: dict[str, str] = {}

    for geom in root.iter("geom"):
        mesh_ref = geom.get("mesh")
        if not mesh_ref:
            continue

        # Already references an existing asset name.
        if mesh_ref in mesh_assets:
            if (geom.get("type") or "").strip().lower() != "mesh":
                geom.set("type", "mesh")
                changed = True
            continue

        if not _looks_like_mesh_file(mesh_ref):
            continue

        resolved = _resolve_mesh_path(mesh_ref)
        if mjcf_dir_resolved is not None and resolved is None:
            # Can't find the mesh on disk; avoid hard failure by using a box.
            geom.attrib.pop("mesh", None)
            geom.set("type", "box")
            geom.attrib.pop("size", None)
            geom.set("size", "0.1 0.1 0.1")
            changed = True
            continue

        # Create/reuse an <asset><mesh>.
        asset_name = by_file_value.get(mesh_ref)
        if asset_name is None:
            stem = _safe_name(Path(mesh_ref).stem)
            crc = zlib.crc32(mesh_ref.encode("utf-8")) & 0xFFFFFFFF
            candidate = f"{stem}_{crc:08x}"
            if candidate in existing_mesh_names:
                i = 2
                while f"{candidate}_{i}" in existing_mesh_names:
                    i += 1
                candidate = f"{candidate}_{i}"
            asset_name = candidate
            existing_mesh_names.add(asset_name)
            by_file_value[mesh_ref] = asset_name

            file_value = mesh_ref
            if resolved is not None and mjcf_dir_resolved is not None:
                try:
                    file_value = resolved.relative_to(mjcf_dir_resolved).as_posix()
                except ValueError:
                    file_value = resolved.as_posix()

            mesh_el = ET.Element("mesh", {"name": asset_name, "file": file_value})

            size_vals = _parse_floats(geom.get("size"))
            if len(size_vals) == 1:
                s = size_vals[0]
                mesh_el.set("scale", _fmt_triplet((s, s, s)))
            elif len(size_vals) >= 3:
                mesh_el.set(
                    "scale", _fmt_triplet((size_vals[0], size_vals[1], size_vals[2]))
                )

            asset.append(mesh_el)
            mesh_assets[asset_name] = mesh_el
            changed = True

        # Rewrite geom to reference the asset mesh.
        geom.set("type", "mesh")
        geom.set("mesh", asset_name)
        if geom.get("size") is not None:
            del geom.attrib["size"]
        changed = True

    if not changed and not comments_removed:
        return mjcf_xml
    return ET.tostring(root, encoding="unicode")


def ensure_freejoint_bodies_have_inertial(
    mjcf_xml: str,
    *,
    default_mass: float = 1.0,
    default_diaginertia: tuple[float, float, float] = (0.01, 0.01, 0.01),
) -> str:
    """Ensure bodies with a <freejoint/> include an <inertial .../> tag.

    Many meshes coming from external sources don't encode mass/inertia.
    This post-process step makes objects "physical" in MuJoCo.

    If no changes are needed, returns the input string.
    """

    xml_in, comments_removed = _strip_xml_comments(mjcf_xml)

    try:
        root = ET.fromstring(xml_in)
    except ET.ParseError:
        return mjcf_xml

    changed = False

    for body in root.iter("body"):
        # MuJoCo requires `pos` on <inertial>. Some generated MJCF files include
        # inertials even for static bodies but omit `pos`, which makes import fail.
        inertial_any = body.find("inertial")
        if inertial_any is not None and inertial_any.get("pos") is None:
            inertial_any.set("pos", "0 0 0")
            changed = True

        has_freejoint = _body_has_free_motion_joint(body)
        if not has_freejoint:
            continue

        has_inertial = any(child.tag == "inertial" for child in list(body))
        if has_inertial:
            continue

        inertial = ET.Element(
            "inertial",
            {
                "pos": "0 0 0",
                "mass": f"{default_mass:.6g}",
                "diaginertia": " ".join(f"{v:.6g}" for v in default_diaginertia),
            },
        )

        # Keep ordering stable-ish: after freejoint if present, else at start.
        children = list(body)
        insert_at = 0
        for idx, child in enumerate(children):
            if _is_free_motion_joint_element(child):
                insert_at = idx + 1
                break
        body.insert(insert_at, inertial)
        changed = True

    if not changed and not comments_removed:
        return mjcf_xml

    return ET.tostring(root, encoding="unicode")
