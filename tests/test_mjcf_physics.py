import math
import xml.etree.ElementTree as ET

from mujoco_scene_editor.utils.mjcf_physics import (
  enforce_top_level_instance_counts,
  ensure_default_gravity,
  ensure_freejoint_bodies_above_floor,
  ensure_freejoint_bodies_above_supports,
  ensure_freejoint_bodies_have_collision_proxy,
  ensure_freejoint_bodies_have_inertial,
  ensure_inertial_top_level_bodies_are_movable,
  ensure_small_top_level_bodies_are_movable,
  ensure_mesh_asset_scale_is_triplet,
  ground_large_top_level_bodies,
  normalize_objaverse_geom_orientation,
  normalize_objaverse_mesh_scales,
  recenter_freejoint_body_frames,
  resolve_freejoint_xy_overlaps,
  sanitize_mjcf_schema,
)


def _write_unit_cube_obj(path):
  # A 1x1x1 cube in OBJ (0..1 in each axis)
  path.write_text(
    "\n".join(
      [
        "v 0 0 0",
        "v 1 0 0",
        "v 1 1 0",
        "v 0 1 0",
        "v 0 0 1",
        "v 1 0 1",
        "v 1 1 1",
        "v 0 1 1",
        "f 1 2 3 4",
        "f 5 6 7 8",
        "f 1 5 8 4",
        "f 2 6 7 3",
        "f 1 2 6 5",
        "f 4 3 7 8",
      ]
    ),
    encoding="utf-8",
  )


def _write_rect_prism_obj(path, min_xyz, max_xyz):
  x0, y0, z0 = min_xyz
  x1, y1, z1 = max_xyz
  path.write_text(
    "\n".join(
      [
        f"v {x0} {y0} {z0}",
        f"v {x1} {y0} {z0}",
        f"v {x1} {y1} {z0}",
        f"v {x0} {y1} {z0}",
        f"v {x0} {y0} {z1}",
        f"v {x1} {y0} {z1}",
        f"v {x1} {y1} {z1}",
        f"v {x0} {y1} {z1}",
        "f 1 2 3 4",
        "f 5 6 7 8",
        "f 1 5 8 4",
        "f 2 6 7 3",
        "f 1 2 6 5",
        "f 4 3 7 8",
      ]
    ),
    encoding="utf-8",
  )


def test_ensure_inertial_added_for_freejoint_body():
    xml = """
<mujoco>
  <worldbody>
    <body name="box">
      <freejoint/>
      <geom type="box" size="0.1 0.1 0.1"/>
    </body>
  </worldbody>
</mujoco>
""".strip()

    out = ensure_freejoint_bodies_have_inertial(xml)
    assert "<inertial" in out
    assert "pos=\"0 0 0\"" in out
    assert "mass=\"1\"" in out or "mass=\"1.0\"" in out


def test_backfill_inertial_pos_if_missing():
    xml = """
<mujoco>
  <worldbody>
    <body name="obj">
      <freejoint/>
      <inertial mass="2" diaginertia="0.1 0.1 0.1"/>
      <geom type="sphere" size="0.1"/>
    </body>
  </worldbody>
</mujoco>
""".strip()

    out = ensure_freejoint_bodies_have_inertial(xml)
    assert out != xml
    assert "pos=\"0 0 0\"" in out


def test_ensure_inertial_added_for_joint_type_free():
    xml = """
<mujoco>
  <worldbody>
    <body name="obj">
      <joint type="free"/>
      <geom type="box" size="0.1 0.1 0.1"/>
    </body>
  </worldbody>
</mujoco>
""".strip()

    out = ensure_freejoint_bodies_have_inertial(xml)
    assert out != xml
    assert "<inertial" in out
    assert "pos=\"0 0 0\"" in out


def test_no_change_if_inertial_present_and_has_pos():
    xml = """
<mujoco>
  <worldbody>
    <body name="obj">
      <freejoint/>
      <inertial pos="0 0 0" mass="2" diaginertia="0.1 0.1 0.1"/>
      <geom type="sphere" size="0.1"/>
    </body>
  </worldbody>
</mujoco>
""".strip()

    out = ensure_freejoint_bodies_have_inertial(xml)
    assert out == xml


def test_backfill_inertial_pos_on_static_body():
    xml = """
<mujoco>
  <worldbody>
    <body name="table" pos="0 0 0.7">
      <inertial mass="0" diaginertia="0 0 0"/>
      <geom type="box" size="1 1 0.1"/>
    </body>
  </worldbody>
</mujoco>
""".strip()

    out = ensure_freejoint_bodies_have_inertial(xml)
    assert out != xml
    assert "pos=\"0 0 0\"" in out


def test_parse_error_returns_input():
    bad = "<mujoco><worldbody>"
    assert ensure_freejoint_bodies_have_inertial(bad) == bad


def test_ensure_freejoint_bodies_above_supports_lifts_object_onto_tabletop():
    # Table top: center z=0.73, half-thickness=0.02 => top surface at z=0.75.
    # Object: a thin box centered at local z=0 with half-height 0.01.
    # Body pos=0.74 => min z = 0.73, i.e. penetrating the tabletop.
    xml = """
<mujoco>
  <worldbody>
    <body name="anchor_table" pos="0 0 0">
      <geom name="table_top" type="box" size="0.6 0.4 0.02" pos="0 0 0.73"/>
    </body>
    <body name="laptop_1" pos="0 0 0.74">
      <freejoint/>
      <geom type="box" size="0.2 0.15 0.01"/>
    </body>
  </worldbody>
</mujoco>
""".strip()

    out = ensure_freejoint_bodies_above_supports(xml, mjcf_dir=".", clearance=0.01)
    root = ET.fromstring(out)
    laptop = next(b for b in root.iter("body") if b.get("name") == "laptop_1")
    z = float((laptop.get("pos") or "0 0 0").split()[2])

    # Need min_z >= 0.75 + 0.01, and bb_min_z is -0.01 for the box geom.
    assert z >= 0.75 + 0.01 + 0.01 - 1e-6


def test_mesh_scale_scalar_expands_to_triplet():
    xml = """
<mujoco>
  <asset>
    <mesh name="m" file="a.obj" scale="0.01"/>
  </asset>
</mujoco>
""".strip()
    out = ensure_mesh_asset_scale_is_triplet(xml)
    assert out != xml
    assert "scale=\"0.01 0.01 0.01\"" in out


def test_mesh_scale_pair_expands_to_triplet():
    xml = """
<mujoco>
  <asset>
    <mesh name="m" file="a.obj" scale="1 2"/>
  </asset>
</mujoco>
""".strip()
    out = ensure_mesh_asset_scale_is_triplet(xml)
    assert out != xml
    assert "scale=\"1 2 2\"" in out


def test_mesh_scale_repair_works_with_invalid_decorative_comments():
    xml = """
<mujoco>
  <!-- Assets ----------------------------------------------------------- -->
  <asset>
    <mesh name="m" file="a.obj" scale="0.01"/>
  </asset>
</mujoco>
""".strip()

    out = ensure_mesh_asset_scale_is_triplet(xml)
    assert out != xml


def test_sanitize_mjcf_schema_normalizes_joint_types():
    xml = """
<mujoco>
  <worldbody>
    <body name="arm">
      <joint name="j1" type="revolute" axis="0 0 1"/>
      <joint name="j2" type="prismatic" axis="1 0 0"/>
      <joint name="j3" type="continuous" axis="0 1 0"/>
      <joint name="j4" type="floating"/>
    </body>
  </worldbody>
</mujoco>
""".strip()

    out = sanitize_mjcf_schema(xml)
    assert out != xml
    assert "type=\"hinge\"" in out
    assert "type=\"slide\"" in out
    assert "type=\"free\"" in out


def test_normalize_objaverse_mesh_scales_sets_uniform_scale(tmp_path):
    mjcf_dir = tmp_path
    obj_dir = mjcf_dir / "assets" / "objaverse"
    obj_dir.mkdir(parents=True)
    obj_path = obj_dir / "chair_dummy.obj"
    _write_unit_cube_obj(obj_path)

    xml = f"""
<mujoco>
  <asset>
    <mesh name=\"chair_mesh\" file=\"assets/objaverse/{obj_path.name}\" scale=\"0.01\"/>
  </asset>
</mujoco>
""".strip()

    out = normalize_objaverse_mesh_scales(xml, mjcf_dir=mjcf_dir)
    assert out != xml
    # For a unit cube (max_dim=1) and a chair target (1.0m), scale should become ~1.
    assert "scale=\"1 1 1\"" in out or "scale=\"1.0 1.0 1.0\"" in out


def test_sanitize_mjcf_schema_moves_solver_from_flag_to_option():
    xml = """
<mujoco>
  <option>
    <flag solver="Newton" enableflags="0"/>
  </option>
</mujoco>
""".strip()

    out = sanitize_mjcf_schema(xml)
    assert out != xml
    assert "<option" in out
    assert "solver=\"Newton\"" in out
    assert "<flag" not in out


def test_ensure_default_gravity_adds_option_gravity():
    xml = """
<mujoco>
  <worldbody>
    <body name="floor"><geom type="plane" size="2 2 0.1"/></body>
  </worldbody>
</mujoco>
""".strip()

    out = ensure_default_gravity(xml)
    assert "<option" in out
    assert "gravity=\"0 0 -9.81\"" in out


def test_inertial_top_level_body_gets_freejoint(tmp_path):
    # Has inertial mass but no joint -> should become movable.
    xml = """
<mujoco>
  <worldbody>
    <body name="prop" pos="0 0 1">
      <inertial pos="0 0 0" mass="1" diaginertia="0.01 0.01 0.01"/>
      <geom type="box" size="0.1 0.1 0.1"/>
    </body>
  </worldbody>
</mujoco>
""".strip()

    out = ensure_inertial_top_level_bodies_are_movable(xml, mjcf_dir=tmp_path)
    assert "<freejoint" in out


def test_collision_proxy_added_when_all_geoms_noncollidable(tmp_path):
    mjcf_dir = tmp_path
    obj_dir = mjcf_dir / "assets"
    obj_dir.mkdir(parents=True)
    obj_path = obj_dir / "cube.obj"
    _write_unit_cube_obj(obj_path)

    xml = f"""
<mujoco>
  <asset>
    <mesh name=\"m\" file=\"assets/{obj_path.name}\"/>
  </asset>
  <worldbody>
    <body name=\"cup\">
      <freejoint/>
      <geom type=\"mesh\" mesh=\"m\" contype=\"0\" conaffinity=\"0\"/>
    </body>
  </worldbody>
</mujoco>
""".strip()

    out = ensure_freejoint_bodies_have_collision_proxy(xml, mjcf_dir=mjcf_dir)
    assert "_proxy_collision" in out
    # Proxy must be collidable.
    assert "contype=\"1\"" in out
    assert "conaffinity=\"1\"" in out


def test_sanitize_mjcf_schema_drops_mesh_ref_attrs():
    xml = """
<mujoco>
  <asset>
    <mesh name="m" file="a.obj" refscale="0.01" refpos="0 0 0" scale="1 1 1"/>
  </asset>
</mujoco>
""".strip()

    out = sanitize_mjcf_schema(xml)
    assert out != xml
    assert "refscale" not in out
    assert "refpos" not in out


def test_sanitize_mjcf_schema_drops_undefined_geom_material():
    xml = """
<mujoco>
  <worldbody>
    <geom name="floor" type="plane" size="5 5 0.1" material="self" rgba="0.9 0.9 0.9 1"/>
  </worldbody>
  <asset>
    <material name="other" rgba="0.1 0.1 0.1 1"/>
  </asset>
</mujoco>
""".strip()

    out = sanitize_mjcf_schema(xml)
    assert out != xml
    root = ET.fromstring(out)
    floor = next(g for g in root.iter("geom") if g.get("name") == "floor")
    assert floor.get("material") is None


def test_sanitize_mjcf_schema_migrates_geom_scale_to_asset_mesh():
    xml = """
<mujoco>
  <asset>
    <mesh name="m" file="a.obj" scale="0.5 0.5 0.5"/>
  </asset>
  <worldbody>
    <body>
      <geom type="mesh" mesh="m" scale="2 2 2"/>
    </body>
  </worldbody>
</mujoco>
""".strip()

    out = sanitize_mjcf_schema(xml)
    assert out != xml

    root = ET.fromstring(out)
    assert all(g.get("scale") is None for g in root.iter("geom"))

    mesh = next(iter(root.iter("mesh")))
    assert mesh.get("name") == "m"
    assert mesh.get("scale") in {"1 1 1", "1.0 1.0 1.0"}


def test_sanitize_mjcf_schema_duplicates_mesh_for_multiple_geom_scales():
    xml = """
<mujoco>
  <asset>
    <mesh name="m" file="a.obj" scale="1 1 1"/>
  </asset>
  <worldbody>
    <body>
      <geom type="mesh" mesh="m" scale="1 1 1"/>
      <geom type="mesh" mesh="m" scale="2 2 2"/>
    </body>
  </worldbody>
</mujoco>
""".strip()

    out = sanitize_mjcf_schema(xml)
    assert out != xml

    root = ET.fromstring(out)
    assert all(g.get("scale") is None for g in root.iter("geom"))

    meshes = {m.get("name"): m for m in root.iter("mesh") if m.get("name")}
    assert "m" in meshes
    assert "m__s2x2x2" in meshes

    geom_meshes = [g.get("mesh") for g in root.iter("geom") if g.get("mesh")]
    assert "m" in geom_meshes
    assert "m__s2x2x2" in geom_meshes


def test_ensure_freejoint_bodies_above_floor_for_primitives():
    xml = """
<mujoco>
  <worldbody>
    <geom type="plane" size="5 5 0.1" pos="0 0 0"/>
    <body name="mug" pos="0 0 0.02">
      <freejoint/>
      <geom type="cylinder" size="0.03 0.08" pos="0 0 0"/>
    </body>
  </worldbody>
</mujoco>
""".strip()

    out = ensure_freejoint_bodies_above_floor(xml, mjcf_dir=".")
    assert out != xml

    root = ET.fromstring(out)
    mug = next(b for b in root.iter("body") if b.get("name") == "mug")
    z = float(mug.get("pos").split()[2])
    # Cylinder half-height is 0.08, so center should be lifted to >= 0.09.
    assert z >= 0.09


def test_ensure_freejoint_bodies_above_floor_for_mesh(tmp_path):
    mjcf_dir = tmp_path
    obj_dir = mjcf_dir / "assets" / "objaverse"
    obj_dir.mkdir(parents=True)
    obj_path = obj_dir / "chair_dummy.obj"
    _write_unit_cube_obj(obj_path)

    xml = f"""
<mujoco>
  <asset>
    <mesh name="chair_mesh" file="assets/objaverse/{obj_path.name}" scale="1 1 1"/>
  </asset>
  <worldbody>
    <geom type="plane" size="5 5 0.1" pos="0 0 0"/>
    <body name="chair" pos="0 0 0">
      <freejoint/>
      <geom type="mesh" mesh="chair_mesh" pos="0 0 -0.5"/>
    </body>
  </worldbody>
</mujoco>
""".strip()

    out = ensure_freejoint_bodies_above_floor(xml, mjcf_dir=mjcf_dir)
    assert out != xml

    root = ET.fromstring(out)
    chair = next(b for b in root.iter("body") if b.get("name") == "chair")
    z = float(chair.get("pos").split()[2])
    # With mesh min z at 0 and geom z offset -0.5, body z must be raised to >= 0.51.
    assert z >= 0.51


def test_normalize_objaverse_geom_orientation_adds_z_up_rotation(tmp_path):
    mjcf_dir = tmp_path
    obj_dir = mjcf_dir / "assets" / "objaverse"
    obj_dir.mkdir(parents=True)
    obj_path = obj_dir / "chair_y_up.obj"

    # Y axis dominates (typical Y-up asset): should map Y-up -> Z-up via euler "90 0 0".
    _write_rect_prism_obj(obj_path, (-0.2, 0.0, -0.1), (0.2, 1.2, 0.1))

    xml = f"""
<mujoco>
  <asset>
    <mesh name="chair_mesh" file="assets/objaverse/{obj_path.name}" scale="1 1 1"/>
  </asset>
  <worldbody>
    <body name="chair" pos="0 0 0">
      <freejoint/>
      <geom type="mesh" mesh="chair_mesh"/>
    </body>
  </worldbody>
</mujoco>
""".strip()

    out = normalize_objaverse_geom_orientation(xml, mjcf_dir=mjcf_dir)
    assert out != xml

    root = ET.fromstring(out)
    geom = next(root.iter("geom"))
    assert geom.get("euler") == "90 0 0"


def test_normalize_objaverse_geom_orientation_chair_soft_dominance(tmp_path):
    mjcf_dir = tmp_path
    obj_dir = mjcf_dir / "assets" / "objaverse"
    obj_dir.mkdir(parents=True)
    obj_path = obj_dir / "chair_almost_isotropic.obj"

    # Y only slightly dominates Z; furniture should still orient to Z-up.
    _write_rect_prism_obj(obj_path, (-0.4365, 0.0, -0.446), (0.4365, 1.0, 0.446))

    xml = f"""
<mujoco>
  <asset>
    <mesh name="chair_mesh" file="assets/objaverse/{obj_path.name}" scale="1 1 1"/>
  </asset>
  <worldbody>
    <body name="chair" pos="0 0 0">
      <geom type="mesh" mesh="chair_mesh"/>
    </body>
  </worldbody>
</mujoco>
""".strip()

    out = normalize_objaverse_geom_orientation(xml, mjcf_dir=mjcf_dir)
    assert out != xml

    root = ET.fromstring(out)
    geom = next(root.iter("geom"))
    assert geom.get("euler") == "90 0 0"


def test_normalize_objaverse_geom_orientation_furniture_upside_down_flip(tmp_path):
    mjcf_dir = tmp_path
    obj_dir = mjcf_dir / "assets" / "objaverse"
    obj_dir.mkdir(parents=True)
    obj_path = obj_dir / "chair_top_heavy.obj"

    # Z-dominant mesh with many vertices near top and few near bottom.
    lines = [
        "v -0.5 -0.5 0",
        "v 0.5 -0.5 0",
        "v 0.5 0.5 0",
        "v -0.5 0.5 0",
        "v 0 0 2",
    ]
    for i in range(12):
        ang = i * 3.141592653589793 * 2.0 / 12.0
        x = 0.45 * math.cos(ang)
        y = 0.45 * math.sin(ang)
        lines.append(f"v {x} {y} 2")

    lines.extend(
      [
        "f 1 2 3 4",
        "f 1 2 5",
        "f 2 3 5",
        "f 3 4 5",
        "f 4 1 5",
      ]
    )
    for i in range(6, 18):
        j = 6 if i == 17 else i + 1
        lines.append(f"f 5 {i} {j}")

    obj_path.write_text("\n".join(lines), encoding="utf-8")

    xml = f"""
<mujoco>
  <asset>
    <mesh name="chair_mesh" file="assets/objaverse/{obj_path.name}" scale="1 1 1"/>
  </asset>
  <worldbody>
    <body name="chair" pos="0 0 0">
      <geom type="mesh" mesh="chair_mesh"/>
    </body>
  </worldbody>
</mujoco>
""".strip()

    out = normalize_objaverse_geom_orientation(xml, mjcf_dir=mjcf_dir)
    assert out != xml

    root = ET.fromstring(out)
    geom = next(root.iter("geom"))
    assert geom.get("euler") in {"180 0 0", "0 180 0"}


def test_normalize_objaverse_geom_orientation_table_prefers_flatter_axis(tmp_path):
    mjcf_dir = tmp_path
    obj_dir = mjcf_dir / "assets" / "objaverse"
    obj_dir.mkdir(parents=True)
    obj_path = obj_dir / "dining_table_dummy.obj"

    # X is the thinnest axis; for table-like assets we expect X -> Z mapping.
    _write_rect_prism_obj(obj_path, (-0.5, -1.0, -1.0), (0.5, 1.0, 1.0))

    xml = f"""
<mujoco>
  <asset>
    <mesh name="dining_table_mesh" file="assets/objaverse/{obj_path.name}" scale="1 1 1"/>
  </asset>
  <worldbody>
    <body name="table" pos="0 0 0">
      <geom type="mesh" mesh="dining_table_mesh"/>
    </body>
  </worldbody>
</mujoco>
""".strip()

    out = normalize_objaverse_geom_orientation(xml, mjcf_dir=mjcf_dir)
    assert out != xml

    root = ET.fromstring(out)
    geom = next(root.iter("geom"))
    assert geom.get("euler") == "0 -90 0"


def test_ensure_small_top_level_bodies_are_movable_adds_freejoint(tmp_path):
    xml = """
<mujoco>
  <worldbody>
    <body name="mug" pos="0 0 0.2">
      <geom type="sphere" size="0.05"/>
    </body>
    <body name="cabinet" pos="0 0 0.4">
      <geom type="box" size="1.0 1.0 0.2"/>
    </body>
  </worldbody>
</mujoco>
""".strip()

    out = ensure_small_top_level_bodies_are_movable(
      xml,
      mjcf_dir=tmp_path,
      max_dynamic_dim_m=1.2,
    )
    assert out != xml

    root = ET.fromstring(out)
    mug = next(b for b in root.iter("body") if b.get("name") == "mug")
    cabinet = next(b for b in root.iter("body") if b.get("name") == "cabinet")

    assert any(ch.tag == "freejoint" for ch in list(mug))
    assert not any(ch.tag == "freejoint" for ch in list(cabinet))


def test_ensure_small_top_level_bodies_are_movable_adds_named_table_freejoint(tmp_path):
    xml = """
<mujoco>
  <worldbody>
    <body name="dining_table" pos="0 0 0.4">
      <geom type="box" size="1.0 1.0 0.2"/>
    </body>
  </worldbody>
</mujoco>
""".strip()

    out = ensure_small_top_level_bodies_are_movable(
      xml,
      mjcf_dir=tmp_path,
      max_dynamic_dim_m=1.2,
      max_named_dynamic_dim_m=2.4,
    )
    assert out != xml

    root = ET.fromstring(out)
    table = next(b for b in root.iter("body") if b.get("name") == "dining_table")
    assert any(ch.tag == "freejoint" for ch in list(table))


def test_resolve_freejoint_xy_overlaps_separates_bodies(tmp_path):
    xml = """
<mujoco>
  <worldbody>
    <body name="a" pos="0 0 0.2">
      <freejoint/>
      <geom type="sphere" size="0.1"/>
    </body>
    <body name="b" pos="0.05 0 0.2">
      <freejoint/>
      <geom type="sphere" size="0.1"/>
    </body>
  </worldbody>
</mujoco>
""".strip()

    out = resolve_freejoint_xy_overlaps(xml, mjcf_dir=tmp_path, margin=0.0, iterations=10)
    assert out != xml

    root = ET.fromstring(out)
    bodies = {b.get("name"): b for b in root.iter("body")}
    pa = [float(v) for v in bodies["a"].get("pos").split()]
    pb = [float(v) for v in bodies["b"].get("pos").split()]

    dist = ((pb[0] - pa[0]) ** 2 + (pb[1] - pa[1]) ** 2) ** 0.5
    assert dist >= 0.2 - 1e-6


def test_recenter_freejoint_body_frames_recenters_default_inertial(tmp_path):
    xml = """
<mujoco>
  <worldbody>
    <body name="obj" pos="0 0 0">
      <freejoint/>
      <inertial pos="0 0 0" mass="1" diaginertia="0.01 0.01 0.01"/>
      <geom type="box" size="0.1 0.1 0.1" pos="0.5 0 0.5"/>
    </body>
  </worldbody>
</mujoco>
""".strip()

    out = recenter_freejoint_body_frames(xml, mjcf_dir=tmp_path)
    assert out != xml

    root = ET.fromstring(out)
    body = next(b for b in root.iter("body") if b.get("name") == "obj")
    geom = next(ch for ch in list(body) if ch.tag == "geom")
    inertial = next(ch for ch in list(body) if ch.tag == "inertial")

    assert body.get("pos") == "0.5 0 0.5"
    assert geom.get("pos") == "0 0 0"
    assert inertial.get("pos") == "0 0 0"


def test_recenter_freejoint_body_frames_preserves_nonzero_inertial_world_pose(tmp_path):
    xml = """
<mujoco>
  <worldbody>
    <body name="obj" pos="0 0 0">
      <freejoint/>
      <inertial pos="0.2 0 0" mass="1" diaginertia="0.01 0.01 0.01"/>
      <geom type="box" size="0.1 0.1 0.1" pos="0.5 0 0.5"/>
    </body>
  </worldbody>
</mujoco>
""".strip()

    out = recenter_freejoint_body_frames(xml, mjcf_dir=tmp_path)
    assert out != xml

    root = ET.fromstring(out)
    body = next(b for b in root.iter("body") if b.get("name") == "obj")
    inertial = next(ch for ch in list(body) if ch.tag == "inertial")

    # body shifted by local geom center (0.5, 0, 0.5), inertial shifted back
    # so world-space inertial x remains 0.2
    assert body.get("pos") == "0.5 0 0.5"
    assert inertial.get("pos") == "-0.3 0 -0.5"


def test_ground_large_top_level_bodies_snaps_near_floor(tmp_path):
    xml = """
<mujoco>
  <worldbody>
    <body name="table" pos="0 0 0">
      <geom type="box" size="0.6 0.5 0.3" pos="0 0 1.0"/>
    </body>
  </worldbody>
</mujoco>
""".strip()

    out = ground_large_top_level_bodies(xml, mjcf_dir=tmp_path, floor_z=0.0, clearance=0.01)
    assert out != xml

    root = ET.fromstring(out)
    body = next(b for b in root.iter("body") if b.get("name") == "table")
    z = float(body.get("pos").split()[2])
    # local min z is 0.7, so body z should become -0.69 to place min at 0.01
    assert abs(z + 0.69) < 1e-6


def test_ground_large_top_level_bodies_ignores_small_props(tmp_path):
    xml = """
<mujoco>
  <worldbody>
    <body name="mug" pos="0 0 0">
      <geom type="box" size="0.03 0.03 0.06" pos="0 0 0.8"/>
    </body>
  </worldbody>
</mujoco>
""".strip()

    out = ground_large_top_level_bodies(xml, mjcf_dir=tmp_path)
    assert out == xml


def test_enforce_top_level_instance_counts_trims_excess_chairs():
    xml = """
<mujoco>
  <worldbody>
    <body name="chair_1"/>
    <body name="chair_2"/>
    <body name="chair_3"/>
    <body name="chair_4"/>
    <body name="chair_5"/>
    <body name="chair_6"/>
    <body name="table_1"/>
  </worldbody>
</mujoco>
""".strip()

    out = enforce_top_level_instance_counts(xml, class_counts={"chair": 5})
    assert out != xml

    root = ET.fromstring(out)
    names = [b.get("name") for b in root.find("worldbody").findall("body")]
    chair_names = [n for n in names if n and "chair" in n]
    assert len(chair_names) == 5
    assert "chair_6" not in chair_names
    assert "table_1" in names


def test_enforce_top_level_instance_counts_does_not_touch_under_target():
    xml = """
<mujoco>
  <worldbody>
    <body name="chair_1"/>
    <body name="chair_2"/>
  </worldbody>
</mujoco>
""".strip()

    out = enforce_top_level_instance_counts(xml, class_counts={"chair": 5})
    assert out == xml
