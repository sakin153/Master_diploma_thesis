# Stage 5 - MuJoCo XML Assembly
# Converts placed models (.glb meshes) to MuJoCo XML scene

import math
import os
import re
import shutil
import sys
import xml.etree.ElementTree as ET
from io import StringIO
from pathlib import Path

import trimesh
from obj2mjcf.cli import Args, CoacdArgs, process_obj


# Опционально отключить определение «переда» модели через VLM
# (например в офлайн-окружении). По умолчанию включено.
_ORIENT_DISABLED = os.environ.get(
    "WORLD_CREATOR_DISABLE_ORIENTATION", ""
).lower() in ("1", "true", "yes")


def _get_orientation_offset(model):
    """Вернуть yaw-offset для конкретной GLB модели.

    Полагается на ``project.model_orientation_detector.compute_yaw_offset``,
    у которого внутри персистентный disk-cache по абсолютному пути модели.
    На любой сбой LLM/рендера отдаёт 0 — лучше повёрнутый стул, чем
    сорвавшаяся сборка сцены.
    """
    if _ORIENT_DISABLED:
        return 0
    up_axis = str(model.get("_up_axis", "y"))
    if up_axis != "y":
        # Z-up модели не проходят X-90 коррекцию, у них своя система осей.
        return 0
    model_loc = str(model.get("model_loc") or "").strip()
    if not model_loc or not os.path.exists(model_loc):
        return 0
    try:
        from project.model_orientation_detector import compute_yaw_offset
        model_name = str(
            model.get("_query_name") or model.get("Model") or "object"
        )
        return int(compute_yaw_offset(
            model_loc,
            model_name=re.sub(r"[^a-zA-Z0-9_]+", "_", model_name) or "object",
        )) % 360
    except Exception as e:
        print(f"[mujoco_assembler]   orientation offset failed: {e}; using 0°")
        return 0


_CONCAVE_HINTS = ("crate", "container", "box", "basket", "bin", "drawer", "ящик")

_MASS_TARGETS = (
    (("plate", "dish"),            0.55),
    (("bowl",),                    0.45),
    (("cup", "mug"),               0.35),
    (("glass",),                   0.25),
    (("bottle",),                  0.60),
    (("book",),                    0.60),
    (("laptop",),                  1.80),
    (("keyboard",),                0.80),
    (("phone", "smartphone"),      0.20),
    (("mouse",),                   0.12),
    (("remote", "controller"),     0.15),
    (("pen", "pencil", "marker"),  0.02),
    (("stapler",),                 0.40),
    (("scissors",),                0.10),
    (("tape",),                    0.10),
    (("vase",),                    0.80),
    (("pot", "flower"),            1.20),
    (("candle",),                  0.20),
    (("figurine",),                0.30),
    (("pillow", "cushion"),        0.60),
    (("ball",),                    0.45),
    (("apple", "fruit"),           0.18),
)

_FILL_FACTORS = (
    (("cup", "mug", "glass"),     0.15),
    (("bowl",),                   0.20),
    (("vase", "pot"),             0.15),
    (("bottle",),                 0.25),
    (("plate", "dish"),           0.90),
    (("pillow", "cushion"),       0.30),
    (("book",),                   0.85),
    (("ball",),                   0.10),
    (("laptop",),                 0.35),
    (("keyboard",),               0.50),
    (("phone", "smartphone"),     0.70),
    (("figurine",),               0.60),
    (("apple", "fruit"),          0.90),
    (("candle",),                 0.80),
)


def _get_mass_target(model_name):
    lname = (model_name or "").lower()
    for keywords, mass in _MASS_TARGETS:
        if any(kw in lname for kw in keywords):
            return mass
    return 0.50


def _get_fill_factor(model_name):
    lname = (model_name or "").lower()
    for keywords, fill in _FILL_FACTORS:
        if any(kw in lname for kw in keywords):
            return fill
    return 0.50


def _compute_density(model_name, size):
    if len(size) < 3:
        return 400.0
    w = max(0.01, float(size[0]))
    h = max(0.01, float(size[1]))
    d = max(0.01, float(size[2]))
    bbox_vol = w * h * d
    target_mass = _get_mass_target(model_name)
    fill = _get_fill_factor(model_name)
    effective_vol = bbox_vol * max(0.05, fill)
    density = target_mass / max(1e-6, effective_vol)
    return max(50.0, min(8000.0, density))


def _get_physics_profile(model_name, size, is_static):
    if is_static is None:
        if size is not None and len(size) >= 3:
            volume = float(size[0]) * float(size[1]) * float(size[2])
            is_static = volume >= 0.1
        else:
            is_static = True

    if is_static:
        return {
            "is_static": True,
            "friction": "0.8 0.005 0.0001",
            "density": "600.0",
            "condim": "3",
            "solref": "0.01 1.0",
            "solimp": "0.95 0.99 0.001 0.5 2",
        }
    else:
        density = _compute_density(model_name, size) if size else 400.0
        return {
            "is_static": False,
            "friction": "0.6 0.004 0.0001",
            "density": str(density),
            "condim": "3",
            "solref": "0.01 1.0",
            "solimp": "0.9 0.95 0.001 0.5 2",
        }


def _needs_convex_decomp(model):
    if not model:
        return False
    parts = []
    name = str(model.get("Model") or model.get("name") or "")
    if name:
        parts.append(name)
    for key in ("categories", "tags"):
        raw = model.get(key)
        if isinstance(raw, list):
            parts.extend(str(x) for x in raw if x)
    haystack = " ".join(parts).lower()
    return any(token in haystack for token in _CONCAVE_HINTS)


def _copy_referenced_textures(source_dir, xml_path):
    if not xml_path.exists():
        return
    try:
        xml_root = ET.parse(xml_path).getroot()
    except ET.ParseError:
        return

    target_dir = xml_path.parent
    seen = set()

    for texture in xml_root.findall(".//texture"):
        raw_file = (texture.get("file") or "").strip()
        if not raw_file:
            continue
        texture_rel = raw_file.replace("\\", "/")
        if texture_rel in seen:
            continue
        seen.add(texture_rel)

        src = source_dir / texture_rel
        if not src.exists():
            src = source_dir / Path(texture_rel).name
        if not src.exists() or not src.is_file():
            continue

        dst = target_dir / texture_rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        if src.resolve() == dst.resolve():
            continue
        shutil.copy2(src, dst)


def _modify_default_classes(root, material_map, visual_count):
    for default in root.findall(".//default"):
        cls = default.get("class")
        if cls == "visual" or (isinstance(cls, str) and cls.startswith("visual")):
            material_map[cls] = f"visual{visual_count}"
            default.set("class", material_map[cls])
        elif cls == "collision" or (isinstance(cls, str) and cls.startswith("collision")):
            material_map[cls] = f"collision{visual_count}"
            default.set("class", material_map[cls])


def _modify_body_tag(root, model):
    model_name = str(model.get("Model") or model.get("name") or "")
    size = model.get("size")
    is_static = model.get("is_static")
    profile = _get_physics_profile(model_name, size, is_static)

    yaw_deg = float(model.get("yaw_deg", 0.0))
    pose = model.get("Pose") or {"x": 0.0, "y": 0.0, "z": 0.0}
    px = float(pose.get("x", 0.0))
    py = float(pose.get("y", 0.0))
    pz = float(pose.get("z", 0.0))

    up_axis = str(model.get("_up_axis", "y"))

    yaw_offset = _get_orientation_offset(model)
    final_yaw = (yaw_deg + yaw_offset) % 360.0

    euler_str = f"0 0 {final_yaw}" if up_axis == "z" else f"90 {final_yaw} 0"

    if yaw_offset:
        print(f"[mujoco_assembler]   {model_name}: is_static={profile['is_static']}, "
              f"euler={euler_str} (planner_yaw={yaw_deg:.0f}° + model_offset={yaw_offset}°)")
    else:
        print(f"[mujoco_assembler]   {model_name}: is_static={profile['is_static']}, euler={euler_str}")

    for body in root.findall(".//body"):
        body.set("pos", f"{px} {py} {pz}")
        body.set("euler", euler_str)

        if not profile["is_static"]:
            ET.SubElement(body, "joint", type="free", damping="0.01", stiffness="0")

        for geom in body.findall(".//geom"):
            g_class = geom.get("class", "")
            if "visual" in g_class:
                continue
            geom.set("friction", profile["friction"])
            geom.set("condim", profile["condim"])
            geom.set("density", profile["density"])
            geom.set("solref", profile["solref"])
            geom.set("solimp", profile["solimp"])


def _rewrite_material_names(root, material_map, material_count):
    materials = root.findall(".//material")

    for texture in root.findall(".//texture"):
        old_name = texture.get("name")
        if not old_name:
            continue
        material_map[old_name] = f"material_{material_count}"
        texture.set("name", material_map[old_name])
        material_count += 1

    for material in materials:
        old_name = material.get("name")
        if not old_name:
            continue
        if old_name not in material_map:
            material_map[old_name] = f"material_{material_count}"
            material_count += 1
        material.set("name", material_map[old_name])

        tex_ref = material.get("texture")
        if tex_ref and tex_ref in material_map:
            material.set("texture", material_map[tex_ref])

    for geom in root.findall(".//geom"):
        mat_ref = geom.get("material")
        if mat_ref and mat_ref in material_map:
            geom.set("material", material_map[mat_ref])

        cls_ref = geom.get("class")
        if cls_ref and cls_ref in material_map:
            geom.set("class", material_map[cls_ref])

    return material_count


def _add_physics_options(root):
    ET.SubElement(root, "option",
        timestep="0.002",
        gravity="0 0 -9.81",
        iterations="50",
        tolerance="1e-10",
        integrator="implicitfast",
    ).tail = "\n"

    default = ET.SubElement(root, "default")
    default.tail = "\n"
    ET.SubElement(default, "geom",
        condim="3",
        friction="0.8 0.005 0.0001",
        solref="0.01 1",
        solimp="0.95 0.99 0.001 0.5 2",
        density="600",
    ).tail = "\n"


def _add_environment(asset_elem, worldbody_elem, room_half_size):
    ET.SubElement(asset_elem, "texture",
        type="skybox",
        builtin="gradient",
        rgb1=".3 .5 .7",
        rgb2="0 0 0",
        width="32",
        height="512",
    ).tail = "\n"

    ET.SubElement(asset_elem, "texture",
        name="body",
        type="cube",
        builtin="flat",
        mark="cross",
        width="128",
        height="128",
        rgb1="0.8 0.6 0.4",
        rgb2="0.8 0.6 0.4",
        markrgb="1 1 1",
        random="0.01",
    ).tail = "\n"

    ET.SubElement(asset_elem, "texture",
        name="grid",
        type="2d",
        builtin="checker",
        width="512",
        height="512",
        rgb1=".1 .2 .3",
        rgb2=".2 .3 .4",
    ).tail = "\n"

    ET.SubElement(asset_elem, "material",
        name="grid",
        texture="grid",
        texrepeat="1 1",
        texuniform="true",
        reflectance=".2",
    ).tail = "\n"

    ET.SubElement(worldbody_elem, "geom",
        name="floor",
        type="plane",
        size="0 0 0.05",
        material="grid",
        condim="3",
        friction="1.0 0.005 0.0001",
        solref="0.01 1",
        solimp="0.95 0.99 0.001 0.5 2",
    ).tail = "\n"

    wall_h = 3.0
    ceiling = ET.SubElement(worldbody_elem, "geom",
        name="ceiling",
        type="plane",
        pos=f"0 0 {wall_h}",
        size="0 0 0.05",
        rgba="0.9 0.9 0.9 0.1",
        condim="1",
        contype="1",
        conaffinity="1",
    )
    ceiling.set("euler", "180 0 0")
    ceiling.tail = "\n"

    ET.SubElement(worldbody_elem, "light",
        name="light_main",
        pos="0 0 4",
        dir="0 0 -1",
        diffuse="0.8 0.8 0.8",
        specular="0.2 0.2 0.2",
        castshadow="true",
    ).tail = "\n"

    ET.SubElement(worldbody_elem, "light",
        name="light_fill",
        pos="0 0 6",
        diffuse="0.4 0.4 0.4",
        specular="0.0 0.0 0.0",
        castshadow="false",
    ).tail = "\n"


def _create_fallback_cube(model, path):
    size = model.get("size")
    if size is None:
        width, depth, height = 0.5, 0.5, 0.5
    else:
        width = max(0.1, float(size[0]))
        height = max(0.1, float(size[1])) if len(size) > 1 else width
        depth = max(0.1, float(size[2])) if len(size) > 2 else width

    gx, gy, gz = width / 2.0, depth / 2.0, height / 2.0
    pose = model.get("Pose") or {"x": 0.0, "y": 0.0, "z": gz}
    px = float(pose.get("x", 0.0))
    py = float(pose.get("y", 0.0))
    pz = float(pose.get("z", gz))
    yaw_deg = float(model.get("yaw_deg", 0.0))

    model_name = str(model.get("Model") or model.get("name") or "fallback")
    safe_name = re.sub(r"[^a-zA-Z0-9_]+", "_", model_name)
    save_fn = re.sub(r"[^a-zA-Z0-9_]+", "_", str(model.get("save_fn") or "0"))
    unique_name = f"fallback_{safe_name}_{save_fn}"

    profile = _get_physics_profile(model_name, size, model.get("is_static"))

    root = ET.Element("mujoco", model=unique_name)
    worldbody = ET.SubElement(root, "worldbody")
    body = ET.SubElement(worldbody, "body",
        name=unique_name,
        pos=f"{px} {py} {pz}",
        euler=f"0 0 {yaw_deg}",
    )

    if not profile["is_static"]:
        ET.SubElement(body, "joint", type="free", damping="0.01", stiffness="0")

    ET.SubElement(body, "geom",
        type="box",
        size=f"{gx} {gy} {gz}",
        rgba="0.85 0.2 0.2 1",
        friction=profile["friction"],
        condim=profile["condim"],
        density=profile["density"],
        solref=profile["solref"],
        solimp=profile["solimp"],
    )

    fallback_xml = Path(path) / f"{model.get('save_fn', 'fallback')}_fallback.xml"
    ET.ElementTree(root).write(fallback_xml, encoding="utf-8", xml_declaration=True)
    return fallback_xml


def _model_xml_compiles(xml_path):
    if not xml_path.exists():
        return False
    try:
        import mujoco
        mujoco.MjModel.from_xml_path(str(xml_path))
        return True
    except ImportError:
        return True
    except Exception:
        return False


def _process_obj_file(obj_path, args):
    old_stdout = sys.stdout
    sys.stdout = StringIO()
    try:
        process_obj(Path(obj_path), args)
        output = sys.stdout.getvalue()
    finally:
        sys.stdout = old_stdout
    return output


def _convert_model(model, converted_dir, visual_count, material_count):
    save_fn = model.get("save_fn", "model_0")
    model_dir = converted_dir / save_fn
    model_dir.mkdir(parents=True, exist_ok=True)

    # Fast path: if the per-model MJCF already exists, reuse it and only
    # update pose/physics + re-rename materials/defaults to avoid collisions.
    xml_path = model_dir / save_fn / f"{save_fn}.xml"
    if xml_path.exists():
        material_map = {}
        included_tree = ET.parse(xml_path)
        included_root = included_tree.getroot()

        _modify_default_classes(included_root, material_map, visual_count)
        visual_count += 1

        _modify_body_tag(included_root, model)

        material_count = _rewrite_material_names(included_root, material_map, material_count)
        material_count += 1

        included_tree.write(xml_path, encoding="utf-8", xml_declaration=True)
        return xml_path, visual_count, material_count

    # Load, scale, and center mesh
    mesh = trimesh.load(model["model_loc"], force="mesh")
    mesh.apply_scale(float(model.get("scale", 1.0)))

    bmin, bmax = mesh.bounds
    cx = 0.5 * (float(bmin[0]) + float(bmax[0]))
    cy = 0.5 * (float(bmin[1]) + float(bmax[1]))
    cz = 0.5 * (float(bmin[2]) + float(bmax[2]))
    if all(math.isfinite(v) for v in (cx, cy, cz)):
        mesh.apply_translation([-cx, -cy, -cz])

    # Export to .obj
    obj_data, texture_data = trimesh.exchange.export.export_obj(
        mesh, include_texture=True, return_texture=True
    )

    obj_path = model_dir / f"{save_fn}.obj"
    with open(obj_path, "w") as f:
        f.write(obj_data)

    for filename, data in texture_data.items():
        texture_path = model_dir / filename
        texture_path.parent.mkdir(parents=True, exist_ok=True)
        with open(texture_path, "wb") as f:
            f.write(data)

    # Build obj2mjcf args
    use_decomp = _needs_convex_decomp(model)
    args_kwargs = dict(
        obj_dir=model_dir,
        verbose=True,
        save_mjcf=True,
        compile_model=True,
        overwrite=True,
        decompose=use_decomp,
    )
    if use_decomp:
        args_kwargs["coacd_args"] = CoacdArgs(
            preprocess_resolution=30,
            threshold=0.08,
            max_convex_hull=24,
            mcts_iterations=60,
            mcts_max_depth=3,
            mcts_nodes=16,
            resolution=800,
            pca=False,
            seed=0,
        )
    args = Args(**args_kwargs)

    printed_output = _process_obj_file(str(obj_path), args)

    xml_path = model_dir / save_fn / f"{save_fn}.xml"

    _copy_referenced_textures(model_dir, xml_path)

    if "Error compiling model" in printed_output:
        if not _model_xml_compiles(xml_path):
            raise RuntimeError(f"obj2mjcf compilation failed for {save_fn}")

    if not xml_path.exists():
        raise FileNotFoundError(f"obj2mjcf did not generate {xml_path}")

    # Modify XML: rename defaults, set pose/physics, rename materials
    material_map = {}
    included_tree = ET.parse(xml_path)
    included_root = included_tree.getroot()

    _modify_default_classes(included_root, material_map, visual_count)
    visual_count += 1

    _modify_body_tag(included_root, model)

    material_count = _rewrite_material_names(included_root, material_map, material_count)
    material_count += 1

    included_tree.write(xml_path, encoding="utf-8", xml_declaration=True)

    return xml_path, visual_count, material_count


def assemble_mujoco_scene(placed_models, room_half_size, output_path, cache_dir=None):
    """Stage 5: Assemble MuJoCo XML scene from placed models.

    Args:
        placed_models: List of models with Pose, yaw_deg, size, model_loc
        room_half_size: Room half-size in meters
        output_path: Path to save the MuJoCo XML file
        cache_dir: Cache directory for converted meshes (default: .cache)

    Returns:
        str: Path to the generated MuJoCo XML file
    """
    if cache_dir is None:
        cache_dir = ".cache"

    cache_path = Path(cache_dir).resolve()
    cache_path.mkdir(parents=True, exist_ok=True)
    converted_dir = cache_path / "converted"
    converted_dir.mkdir(parents=True, exist_ok=True)

    print(f"[mujoco_assembler] Assembling scene with {len(placed_models)} objects")
    print(f"[mujoco_assembler] Room: {room_half_size*2:.1f}m x {room_half_size*2:.1f}m")
    print(f"[mujoco_assembler] Cache: {cache_path}")

    main_root = ET.Element("mujoco", model="scene")
    _add_physics_options(main_root)

    asset_elem = ET.SubElement(main_root, "asset")
    worldbody_elem = ET.SubElement(main_root, "worldbody")
    _add_environment(asset_elem, worldbody_elem, room_half_size)

    visual_count = 0
    material_count = 0
    included_paths = set()

    for i, model in enumerate(placed_models):
        model_name = model.get("Model", model.get("name", "unknown"))
        print(f"[mujoco_assembler] Processing {i+1}/{len(placed_models)}: {model_name}")

        save_fn = model.get("save_fn", f"model_{i}")
        model_dir = converted_dir / save_fn
        model_dir.mkdir(parents=True, exist_ok=True)

        model_loc = model.get("model_loc")
        if not model_loc or not os.path.exists(str(model_loc)):
            print(f"[mujoco_assembler]   ⚠ Model not found, using fallback cube")
            fallback_xml = _create_fallback_cube(model, model_dir)
            if str(fallback_xml) not in included_paths:
                included_paths.add(str(fallback_xml))
                ET.SubElement(main_root, "include", file=str(fallback_xml)).tail = "\n"
            continue

        try:
            xml_path, visual_count, material_count = _convert_model(
                model, converted_dir, visual_count, material_count
            )
            if str(xml_path) not in included_paths:
                included_paths.add(str(xml_path))
                ET.SubElement(main_root, "include", file=str(xml_path)).tail = "\n"
            print(f"[mujoco_assembler]   ✓ Converted successfully")
        except Exception as e:
            print(f"[mujoco_assembler]   ✗ Conversion failed: {e}")
            print(f"[mujoco_assembler]   ⚠ Using fallback cube")
            fallback_xml = _create_fallback_cube(model, model_dir)
            if str(fallback_xml) not in included_paths:
                included_paths.add(str(fallback_xml))
                ET.SubElement(main_root, "include", file=str(fallback_xml)).tail = "\n"

    output_path = Path(output_path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ET.ElementTree(main_root).write(output_path, encoding="utf-8", xml_declaration=True)

    print(f"[mujoco_assembler] ✓ Scene saved to: {output_path}")

    try:
        import mujoco
        mujoco.MjModel.from_xml_path(str(output_path))
        print("[mujoco_assembler] ✓ MuJoCo compilation: OK")
    except ImportError:
        print("[mujoco_assembler] ⚠ MuJoCo not installed, skipping validation")
    except Exception as e:
        print(f"[mujoco_assembler] ✗ MuJoCo compilation: FAILED ({e})")

    return str(output_path)
