#!/usr/bin/env python
from mujoco_scene_editor.cli import startup_warning  # noqa: F401

from typing import List
from typing import Optional
from functools import wraps

import logging
import os
import time
import re
from pathlib import Path
import webbrowser


def _read_simple_dotenv(path: Path) -> dict[str, str]:
    """Read a minimal .env file without external dependencies.

    Supports lines in the form `KEY=VALUE`, strips surrounding single/double
    quotes, and ignores empty lines/comments.
    """

    out: dict[str, str] = {}
    if not path.exists() or not path.is_file():
        return out

    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return out

    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].strip()
        if "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not key:
            continue

        if (value.startswith('"') and value.endswith('"')) or (
            value.startswith("'") and value.endswith("'")
        ):
            value = value[1:-1]

        out[key] = value

    return out


def _is_placeholder_api_key(value: str | None) -> bool:
    if not value:
        return True
    normalized = value.strip().lower()
    return (
        normalized in {"replace_with_new_key", "changeme", "your_key_here"}
        or normalized.startswith("replace_with_")
    )


def _load_dotenv_into_environ(dotenv_path: Path | None = None) -> None:
    """Best-effort .env loader.

    This keeps CLI UX consistent: users can put OPENROUTER_* in `.env` without
    manually exporting environment variables.
    """

    path = dotenv_path or (Path.cwd() / ".env")
    for key, value in _read_simple_dotenv(path).items():
        # Real process env wins.
        os.environ.setdefault(key, value)


# Load .env early so key checks work even when not exported.
_load_dotenv_into_environ()

# Keep robits CLI quiet/stable by ensuring a config directory exists.
# Must be set before importing robits modules that may read it at import-time.
if not os.environ.get("ROBITS_CONFIG_DIR"):
    xdg = os.environ.get("XDG_CONFIG_HOME")
    default_cfg = Path(xdg).expanduser() if xdg else (Path.home() / ".config")
    os.environ["ROBITS_CONFIG_DIR"] = str((default_cfg / "robits_config").resolve())

import rich_click as click
from rich.progress import Progress
from click_prompt import filepath_option
from click_prompt import filepath_argument
from click_prompt import input_text_argument

from robits.sim.blueprints import blueprints_from_json
from robits.sim.blueprints import Blueprint
from robits.sim.blueprints import GeomBlueprint
from robits.sim.blueprints import CameraBlueprint
from robits.sim.blueprints import Pose


from robits.utils import camera_intrinsics

from robits.cli.cli_utils import setup_cli

from mujoco_scene_editor.layout import SceneEditorLayout
from mujoco_scene_editor.scene_renderer import ViserSceneRenderer
from mujoco_scene_editor.controller import SceneEditorController
from mujoco_scene_editor.scene_editor import SceneEditor

from mujoco_scene_editor.constants import DEFAULT_ASSET_DIR
from mujoco_scene_editor.constants import DEFAULT_EXPORT_TARGET

logger = logging.getLogger(__name__)

setup_cli(logging.INFO)


def get_scene_editor(blueprints: Optional[List[Blueprint]] = None) -> SceneEditor:
    layout = SceneEditorLayout()
    renderer = ViserSceneRenderer(layout)
    controller = SceneEditorController(renderer)
    if blueprints:
        controller.load_blueprints(blueprints)

    return SceneEditor(controller, layout)


@click.group()
def cli():
    pass


@cli.command()
@click.option("--open-browser/--skip-open-browser", is_flag=True, default=True)
def new(open_browser: bool):
    """
    Start a new, empty scene.
    """
    default_blueprints: List[Blueprint] = [
        GeomBlueprint(
            "/floor",
            geom_type="plane",
            size=[2.5, 2.5, 0.01],
            rgba=[0.5, 0.5, 0.5, 1.0],
            pose=Pose().with_position([0, 0, -0.01]),
            is_static=True,
        ),
        CameraBlueprint(
            "/camera",
            width=640,
            height=480,
            intrinsics=camera_intrinsics.intrinsics_from_fovy(0.785398, 640, 480),
            pose=Pose().with_position([1.0, 0, 1.5]),
        ),
    ]
    viewer = get_scene_editor(default_blueprints)
    viewer.show()

    click.echo(f"Scene editor running at: {viewer.url}")
    click.echo("Press Ctrl+C to stop.")

    if open_browser:
        webbrowser.open_new(viewer.url)

    wait_until_keypress(viewer)


@cli.command()
@filepath_argument("model-name", default=DEFAULT_EXPORT_TARGET)
@click.option("--open-browser/--skip-open-browser", is_flag=True, default=True)
def edit(model_name: Path, open_browser: bool):
    """
    Load a scene from JSON or try to convert it from a MJCF XML.
    """
    path = Path(model_name).expanduser()
    if not path.exists() or not path.is_file():
        raise FileNotFoundError(f"Model file not found: {path}")

    if path.suffix.lower() == ".xml":
        from robits.sim.converters.mujoco_importer import load_mjcf_as_blueprints
        from mujoco_scene_editor.utils.mjcf_physics import (
            repair_mesh_file_geoms,
            ensure_default_gravity,
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
            stabilize_contact_friction,
            stabilize_free_motion_joints,
            stabilize_small_container_bases,
        )

        xml_text = path.read_text(encoding="utf-8")
        fixed = sanitize_mjcf_schema(xml_text)
        fixed = repair_mesh_file_geoms(fixed, mjcf_dir=path.parent)
        fixed = ensure_default_gravity(fixed)
        fixed = ensure_mesh_asset_scale_is_triplet(fixed)
        fixed = normalize_objaverse_mesh_scales(fixed, mjcf_dir=path.parent)
        fixed = normalize_objaverse_geom_orientation(fixed, mjcf_dir=path.parent)
        fixed = ground_large_top_level_bodies(fixed, mjcf_dir=path.parent)
        fixed = ensure_small_top_level_bodies_are_movable(fixed, mjcf_dir=path.parent)
        fixed = ensure_inertial_top_level_bodies_are_movable(fixed, mjcf_dir=path.parent)
        fixed = recenter_freejoint_body_frames(fixed, mjcf_dir=path.parent)
        fixed = stabilize_free_motion_joints(fixed)
        fixed = ensure_freejoint_bodies_have_inertial(fixed)
        fixed = stabilize_small_container_bases(fixed, mjcf_dir=path.parent)
        fixed = ensure_freejoint_bodies_have_collision_proxy(fixed, mjcf_dir=path.parent)
        fixed = ensure_freejoint_bodies_above_supports(fixed, mjcf_dir=path.parent)
        fixed = stabilize_contact_friction(fixed)
        fixed = resolve_freejoint_xy_overlaps(fixed, mjcf_dir=path.parent)
        path_to_load = path
        if fixed != xml_text:
            repaired_path = path.with_name(path.stem + ".repaired.xml")
            repaired_path.write_text(fixed, encoding="utf-8")
            path_to_load = repaired_path
            logger.info(
                "Repaired MJCF XML for import: %s",
                repaired_path,
            )

        blueprints = load_mjcf_as_blueprints(path_to_load)
    else:
        json_data = path.read_text(encoding="utf-8")
        blueprints = blueprints_from_json(json_data)

    viewer = get_scene_editor(blueprints)
    viewer.show()

    click.echo(f"Scene editor running at: {viewer.url}")
    click.echo("Press Ctrl+C to stop.")

    if open_browser:
        webbrowser.open_new(viewer.url)

    wait_until_keypress(viewer)


@cli.command()
@click.option(
    "--root",
    default=DEFAULT_ASSET_DIR,
    help="Root directory to scan for assets (obj, stl, ply, glb, gltf, usd)",
)
def list_assets(root: str):
    """
    List all available assets found under a directory.
    """

    from mujoco_scene_editor.inventory.local_assets import Inventory

    inv = Inventory()
    root_path = Path(root).expanduser()

    if not root_path.exists() or not root_path.is_dir():
        raise FileNotFoundError(
            f"Folder does not exist or is not a directory: {root_path}"
        )

    items = inv.list(root=root_path)

    if not items:
        click.echo(f"No assets found under {root_path.resolve()}.")
        return

    click.echo(f"Found {len(items)} assets under {root_path.resolve()}:")
    for m in items:
        click.echo(f"- {m.name}: {m.path}")


def validate_has_openrouter_key(func):
    """Compatibility decorator.

    For local Ollama usage an API key is not required, so this decorator no
    longer enforces OPENROUTER_API_KEY.
    """

    @wraps(func)
    def _wrapper(*args, **kwargs):
        # No API-key enforcement for local Ollama.
        return func(*args, **kwargs)

    return _wrapper


def _require_openrouter_key(
    ctx: click.Context, param: click.Parameter, value: object
) -> object:
    # Retained for compatibility with existing option callbacks.
    # For Ollama-backed usage we allow missing OPENROUTER_API_KEY.
    if ctx.resilient_parsing:
        return value
    return value


def _scene_graph_from_prompt_offline(prompt_text: str):
    """Best-effort offline prompt->SceneGraph.

    This is intentionally simple and deterministic. It exists to let users run
    the full placement + physics pipeline without an OpenRouter key.
    """

    from mujoco_scene_editor.pipeline.scene_graph import (
        SceneGraph,
        SceneObject,
        SceneRelation,
        RelationType,
    )

    text = (prompt_text or "").lower()

    counts = _extract_requested_instance_counts(prompt_text)

    def _has_any(*tokens: str) -> bool:
        return any(t in text for t in tokens)

    objects: list[SceneObject] = []
    relations: list[SceneRelation] = []

    # Anchor / support.
    want_table = _has_any("table", "desk", "стол") or ("on" in text) or ("на " in text)
    if want_table:
        objects.append(
            SceneObject(
                id="table_1",
                class_name="table",
                movable=False,
                material="wood",
                size_m=[1.2, 0.8, 0.75],
            )
        )

    # Furniture counts.
    chair_n = int(counts.get("chair", 0))
    if chair_n <= 0 and _has_any("chair", "chairs", "стул", "стуль"):
        chair_n = 2
    chair_n = max(0, min(chair_n, 12))
    for i in range(chair_n):
        objects.append(
            SceneObject(
                id=f"chair_{i+1}",
                class_name="chair",
                movable=True,
                material="wood",
                size_m=[0.5, 0.5, 0.9],
            )
        )

    # Common props.
    prop_specs: list[tuple[str, str, list[float], str]] = []
    if _has_any("cup", "mug", "круж", "чаш"):
        prop_specs.append(("cup", "cup_1", [0.09, 0.09, 0.11], "plastic"))
    if _has_any("bottle", "бутыл"):
        prop_specs.append(("bottle", "bottle_1", [0.08, 0.08, 0.26], "plastic"))
    if _has_any("laptop", "ноут"):
        prop_specs.append(("laptop", "laptop_1", [0.34, 0.24, 0.025], "metal"))

    if not prop_specs and want_table:
        # Default demo props so the scene looks alive.
        prop_specs = [
            ("cup", "cup_1", [0.09, 0.09, 0.11], "plastic"),
            ("bottle", "bottle_1", [0.08, 0.08, 0.26], "plastic"),
        ]

    for cls, obj_id, size_m, material in prop_specs:
        objects.append(
            SceneObject(
                id=obj_id,
                class_name=cls,
                movable=True,
                material=material,  # type: ignore[arg-type]
                size_m=size_m,
            )
        )
        if want_table:
            relations.append(
                SceneRelation(type=RelationType.on, subject=obj_id, object="table_1")
            )

    return SceneGraph(scene_type="offline", objects=objects, relations=relations)


def _extract_requested_instance_counts(prompt_text: str) -> dict[str, int]:
    text = (prompt_text or "").lower()
    out: dict[str, int] = {}

    patterns = {
        "chair": r"chairs?|seat(?:s)?|стул(?:а|ов|ья|ьев)?|стуль(?:я|ев)",
        "table": r"tables?|desks?|стол(?:а|ов)?",
        "sofa": r"sofas?|couches?|диван(?:а|ов)?",
    }

    for cls, noun_pat in patterns.items():
        for m in re.finditer(rf"\b(\d+)\s*(?:x\s*)?(?:{noun_pat})\b", text):
            try:
                out[cls] = int(m.group(1))
            except (TypeError, ValueError):
                continue

    return out


@cli.command()
@click.option(
    "--model",
    "openrouter_model",
    default=None,
    help="OpenRouter model id (overrides OPENROUTER_MODEL)",
)
@click.option(
    "--pipeline",
    "pipeline_mode",
    type=click.Choice(["xml", "graph", "staged"], case_sensitive=False),
    default="graph",
    show_default=True,
    help="Generation pipeline: xml=LLM outputs MJCF directly; graph=SceneGraph JSON then deterministic layout; staged=SceneGraph then rollback-capable staged placement",
)
@click.option(
    "--offline",
    is_flag=True,
    default=False,
    help="Run without OpenRouter: use a simple deterministic prompt->SceneGraph heuristic",
)
@click.option(
    "--verify",
    is_flag=True,
    default=False,
    help="Run a short MuJoCo simulation to sanity-check physics",
)
@click.option(
    "--skip-verify",
    is_flag=True,
    default=False,
    help="Skip physics verification (overrides --verify and graph default)",
)
@filepath_option(
    "--output-model-name",
    default=str(Path(DEFAULT_EXPORT_TARGET).with_name("scene_prompt.xml")),
)
@input_text_argument(
    "prompt", default="A detailed kitchen with a robot.", prompt="Describe your scene."
)
def prompt(
    output_model_name: str,
    openrouter_model: str | None,
    pipeline_mode: str,
    offline: bool,
    verify: bool,
    skip_verify: bool,
    prompt: str,
) -> None:
    """
    Ask an LLM via OpenRouter/Ollama to generate a scene.
    """
    if not offline and _is_placeholder_api_key(os.environ.get("OPENROUTER_API_KEY")):
        logger.warning(
            "OPENROUTER_API_KEY is not set; continuing for local Ollama usage."
        )

    if not offline:
        from mujoco_scene_editor.llm.openrouter import query_openrouter
        from mujoco_scene_editor.llm.objaverse_keywords import extract_objaverse_keywords
        from mujoco_scene_editor.inventory.objaverse_prompt import (
            PreparedObjaverseMesh,
            prepare_objaverse_meshes_for_mjcf,
        )
    from mujoco_scene_editor.pipeline.graph_to_mjcf import scene_graph_to_mjcf
    from mujoco_scene_editor.pipeline.staged_placement import scene_graph_to_mjcf_staged
    from mujoco_scene_editor.pipeline.scene_graph_llm import (
        build_scene_graph_prefix,
        meshes_from_prepared,
        parse_scene_graph_from_llm,
    )
    from mujoco_scene_editor.utils.mjcf_physics import (
        enforce_top_level_instance_counts,
        repair_mesh_file_geoms,
        ensure_default_gravity,
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
        stabilize_contact_friction,
        stabilize_free_motion_joints,
        stabilize_small_container_bases,
    )
    from mujoco_scene_editor.utils.mjcf_verify import verify_mjcf_physics
    import re

    prefix = """
Generate a MuJoCo 3.3.7 XML. Here are some guidelines:
- Don't use global tags for colors or other elements
- Don't use textures/materials. Use a color instead assigned to each element
- If the scene should contain a robot try match it first to an existing description in the robot_description package (https://github.com/robot-descriptions/robot_descriptions.py). If found copy the content.
- For moving objects add a freejoint
- There is no geom cone element
- Avoid accelerometer and sensor tags
- MJCF schema: avoid using <flag> entirely. If you need to set the solver, use <option solver="Newton"|"PGS"|...> attributes (never <flag solver=...>).
- Keep furniture and objects upright relative to +Z (do not place tables/chairs upside-down or lying on their side).
- Avoid initial intersections: do not place objects penetrating floor, table, or each other.
Output the complete XML file. The scene is as follows:
"""
    output_path = Path(output_model_name).expanduser()

    pipeline = (pipeline_mode or "xml").lower().strip()
    do_verify = False
    if not skip_verify:
        do_verify = bool(verify) or pipeline in {"graph", "staged"}

    prepared_meshes = []
    content = ""

    if not offline:
        with Progress() as progress:
            progress.add_task("Querying.", total=None)
            try:
                if pipeline in {"graph", "staged"}:
                    # Pass 1: discover scene objects/classes without mesh bias.
                    graph_prefix_initial = build_scene_graph_prefix(meshes=[])
                    first_content = query_openrouter(
                        graph_prefix_initial, prompt, model=openrouter_model
                    )

                    keywords: list[str] = []
                    try:
                        first_graph = parse_scene_graph_from_llm(first_content)
                        keywords = sorted(
                            {
                                (o.class_name or "").strip().lower()
                                for o in first_graph.objects
                                if (o.class_name or "").strip()
                            }
                        )
                    except ValueError:
                        # Fallback: derive search keywords from the user prompt.
                        keywords = extract_objaverse_keywords(prompt)

                    if keywords:
                        prepared_meshes = prepare_objaverse_meshes_for_mjcf(
                            keywords=keywords,
                            mjcf_path=output_path,
                            max_labels=6,
                            limit_per_label=1,
                            max_total=8,
                        )
                        if prepared_meshes:
                            logger.info(
                                "Prepared %d meshes for objects: %s",
                                len(prepared_meshes),
                                ", ".join(keywords),
                            )
                        else:
                            logger.warning(
                                "No meshes found for objects: %s",
                                ", ".join(keywords),
                            )

                    # Pass 2: regenerate graph with known local mesh list.
                    graph_prefix_final = build_scene_graph_prefix(
                        meshes=meshes_from_prepared(prepared_meshes)
                    )
                    content = query_openrouter(
                        graph_prefix_final, prompt, model=openrouter_model
                    )
                else:
                    # XML pipeline.
                    # Pass 1: generate a first scene draft and extract object names.
                    first_content = query_openrouter(prefix, prompt, model=openrouter_model)
                    keywords = sorted(
                        {
                            name.strip().lower()
                            for name in re.findall(r'<body[^>]*name="([^"]+)"', first_content)
                            if name and name.strip()
                        }
                    )
                    if not keywords:
                        keywords = extract_objaverse_keywords(prompt)

                    if keywords:
                        prepared_meshes = prepare_objaverse_meshes_for_mjcf(
                            keywords=keywords,
                            mjcf_path=output_path,
                            max_labels=6,
                            limit_per_label=1,
                            max_total=8,
                        )
                        if prepared_meshes:
                            logger.info(
                                "Prepared %d meshes for objects: %s",
                                len(prepared_meshes),
                                ", ".join(keywords),
                            )
                            meshes_txt = "\n".join(
                                f"- name: {m.name}\n  uid: {m.uid}\n  file: {m.rel_file}\n  suggested_scale: {m.scale}"
                                for m in prepared_meshes
                            )
                            prefix_with_meshes = (
                                prefix
                                + "\n\n"
                                + "Objaverse meshes are available locally next to this XML. "
                                + "If any of them match the requested scene, you MUST use them instead of primitive geoms. "
                                + "Only use mesh files listed below; do not invent file paths.\n\n"
                                + "Meshes:\n"
                                + meshes_txt
                                + "\n\n"
                                + "Physics requirements for movable objects:\n"
                                + "- Every movable object must be a <body> with a <freejoint/>.\n"
                                + "- Every movable object body must include an <inertial mass=... diaginertia=.../> tag.\n"
                                + "- Use collision geoms; do not make movable objects static.\n"
                                + "- Keep requested counts exact (e.g. 5 chairs means exactly 5 chairs total).\n"
                                + "- Reuse one mesh per object class when instancing repeats (do not mix multiple chair mesh types unless explicitly requested).\n"
                                + "- If a matching table mesh is provided and the prompt asks for a table, use that table mesh instead of primitive table boxes.\n"
                            )
                            # Pass 2: regenerate final XML with concrete local mesh hints.
                            content = query_openrouter(
                                prefix_with_meshes, prompt, model=openrouter_model
                            )
                        else:
                            logger.warning(
                                "No meshes found for objects: %s",
                                ", ".join(keywords),
                            )
                            content = first_content
                    else:
                        content = first_content
            except RuntimeError as e:
                raise click.ClickException(str(e)) from e

    if pipeline in {"graph", "staged"}:
        if offline:
            graph = _scene_graph_from_prompt_offline(prompt)
        else:
            try:
                graph = parse_scene_graph_from_llm(content)
            except ValueError as e:
                click.echo(content)
                raise click.ClickException(str(e)) from e

        # Best-effort: enforce simple requested counts (chairs/tables/sofas).
        requested_counts = _extract_requested_instance_counts(prompt)
        if requested_counts:
            from mujoco_scene_editor.pipeline.scene_graph import SceneObject, SceneRelation, SceneGraph

            by_class: dict[str, list[SceneObject]] = {}
            for o in graph.objects:
                key = (o.class_name or "").lower().strip()
                by_class.setdefault(key, []).append(o)

            objects = list(graph.objects)
            relations = list(graph.relations)

            def _drop_object(obj_id: str) -> None:
                nonlocal objects, relations
                objects = [o for o in objects if o.id != obj_id]
                relations = [r for r in relations if r.subject != obj_id and r.object != obj_id]

            for cls, desired in requested_counts.items():
                cls_lc = cls.lower().strip()
                current = [o for o in objects if (o.class_name or "").lower().strip() == cls_lc]
                if len(current) > desired:
                    for o in current[desired:]:
                        _drop_object(o.id)
                elif len(current) < desired:
                    template = current[-1] if current else None
                    for i in range(desired - len(current)):
                        new_id = f"{cls_lc}_{len(current) + i + 1}"
                        if template is not None:
                            clone = template.model_copy(update={"id": new_id})
                        else:
                            clone = SceneObject(id=new_id, class_name=cls_lc)
                        objects.append(clone)

            graph = SceneGraph(
                scene_type=graph.scene_type,
                objects=objects,
                relations=relations,
                constraints=graph.constraints,
                llm_notes=graph.llm_notes,
                metadata=graph.metadata,
            )

        if pipeline == "staged":
            xml = scene_graph_to_mjcf_staged(graph, mjcf_dir=output_path.parent)
        else:
            xml = scene_graph_to_mjcf(graph)
    else:
        pattern = r"<mujoco\b[^>]*>.*?</mujoco>"
        match = re.search(pattern, content, flags=re.DOTALL)

        if match is None:
            click.echo(content)
            logger.error("Unable to parse response")
            return

        xml = match.group(0)

    xml = sanitize_mjcf_schema(xml)
    xml = repair_mesh_file_geoms(xml, mjcf_dir=output_path.parent)
    xml = ensure_default_gravity(xml)
    xml = ensure_mesh_asset_scale_is_triplet(xml)
    xml = normalize_objaverse_mesh_scales(xml, mjcf_dir=output_path.parent)
    xml = normalize_objaverse_geom_orientation(xml, mjcf_dir=output_path.parent)
    xml = ground_large_top_level_bodies(xml, mjcf_dir=output_path.parent)
    xml = ensure_small_top_level_bodies_are_movable(xml, mjcf_dir=output_path.parent)
    xml = ensure_inertial_top_level_bodies_are_movable(xml, mjcf_dir=output_path.parent)
    xml = recenter_freejoint_body_frames(xml, mjcf_dir=output_path.parent)
    xml = stabilize_free_motion_joints(xml)
    xml = ensure_freejoint_bodies_have_inertial(xml)
    xml = stabilize_small_container_bases(xml, mjcf_dir=output_path.parent)
    xml = ensure_freejoint_bodies_have_collision_proxy(xml, mjcf_dir=output_path.parent)
    xml = ensure_freejoint_bodies_above_supports(xml, mjcf_dir=output_path.parent)
    xml = stabilize_contact_friction(xml)
    xml = resolve_freejoint_xy_overlaps(xml, mjcf_dir=output_path.parent)

    requested_counts = _extract_requested_instance_counts(prompt)
    if requested_counts and pipeline != "graph":
        xml = enforce_top_level_instance_counts(xml, class_counts=requested_counts)

    if do_verify:
        res = verify_mjcf_physics(xml, mjcf_dir=output_path.parent, steps=220)
        if not res.ok:
            logger.warning("Physics verification failed: %s", "; ".join(res.errors))
        for w in res.warnings:
            logger.warning("Physics verification warning: %s", w)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(xml)

    click.echo(f"Edit the model with mjedit {output_path}")


def wait_until_keypress(viewer) -> None:
    try:
        while viewer.is_running:
            time.sleep(0.1)
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt.")
        viewer.quit_server(None)


if __name__ == "__main__":
    cli()
