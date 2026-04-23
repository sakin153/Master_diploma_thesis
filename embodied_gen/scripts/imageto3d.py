import os
import random
from glob import glob
from shutil import copy, copytree, rmtree

import numpy as np
import trimesh
from PIL import Image
from embodied_gen.data.utils import delete_dir
from embodied_gen.models.segment_model import RembgRemover
from embodied_gen.utils.gpt_clients import GPT_CLIENT
from embodied_gen.utils.inference import image3d_model_infer
from embodied_gen.utils.log import logger
from embodied_gen.utils.process_media import combine_images_to_grid
from embodied_gen.utils.tags import VERSION
from embodied_gen.utils.vram_utils import free_vram, log_vram
from embodied_gen.validators.quality_checkers import (
    BaseChecker,
    ImageAestheticChecker,
    ImageSegChecker,
    MeshGeoChecker,
)
from embodied_gen.validators.urdf_convertor import URDFGenerator

# Quality checkers: GPT/Ollama-based, no VRAM, safe at module level.
RBG_REMOVER = RembgRemover()
SEG_CHECKER = ImageSegChecker(GPT_CLIENT)
GEO_CHECKER = MeshGeoChecker(GPT_CLIENT)
_AESTHETIC_CHECKER = None


def _get_aesthetic_checker():
    global _AESTHETIC_CHECKER
    if _AESTHETIC_CHECKER is None:
        _AESTHETIC_CHECKER = ImageAestheticChecker()
    return _AESTHETIC_CHECKER


def _get_checkers():
    return [GEO_CHECKER, SEG_CHECKER, _get_aesthetic_checker()]


# ── Lazy 3-D generation pipeline (Hunyuan3D-2mini) ───────────────────────────
_PIPELINE = None


def _release_pipeline():
    global _PIPELINE
    if _PIPELINE is not None:
        del _PIPELINE
        _PIPELINE = None
        free_vram()
        log_vram("after 3D pipeline release")


def _get_pipeline():
    global _PIPELINE
    if _PIPELINE is None:
        log_vram("before 3D pipeline load")
        logger.info("Loading Hunyuan3D-2mini pipeline...")
        from embodied_gen.models.hunyuan3d import Hunyuan3DInference
        _PIPELINE = Hunyuan3DInference()
        log_vram("after 3D pipeline load")
    return _PIPELINE


def process_single_image(
    image_path: str,
    output_root: str,
    asset_type: str = None,
    seed: int = 0,
    n_retry: int = 3,
    keep_intermediate: bool = False,
    disable_decompose_convex: bool = False,
    texture_size: int = 1024,
) -> dict:
    """Process one image → OBJ + GLB + URDF. Returns dict with file paths."""
    filename = os.path.basename(image_path).split(".")[0]
    os.makedirs(output_root, exist_ok=True)

    image = Image.open(image_path)
    image.save(f"{output_root}/{filename}_raw.png")

    seg_path = f"{output_root}/{filename}_cond.png"
    seg_image = RBG_REMOVER(image) if image.mode != "RGBA" else image
    seg_image.save(seg_path)

    mesh_model = None
    trimesh_result = None

    # Rotation matrices to align Hunyuan3D output to MuJoCo convention
    rot_matrix = [[0, 0, -1], [0, 1, 0], [1, 0, 0]]
    mesh_add_rot = [[1, 0, 0], [0, 0, -1], [0, 1, 0]]

    # ── Stage 1: Image → 3D ──────────────────────────────────────────────────
    current_seed = seed
    for try_idx in range(n_retry):
        logger.info(
            f"Try: {try_idx + 1}/{n_retry}, "
            f"Seed: {current_seed}, Input: {seg_path}"
        )
        try:
            outputs = image3d_model_infer(
                _get_pipeline(), seg_image, current_seed
            )
        except Exception as e:
            logger.error(
                f"[Image3D Failed] {image_path}: {e}, "
                f"retry {try_idx+1}/{n_retry}"
            )
            current_seed = random.randint(0, 100000)
            continue

        # Hunyuan3D: gaussian is None, accept first result
        mesh_model = outputs["mesh"][0]
        trimesh_result = outputs.get("trimesh", [None])[0]
        logger.info("Hunyuan3D generation succeeded.")
        break

    if mesh_model is None:
        logger.error(f"Exceeded retry limit for {image_path}, skipping.")
        return {}

    # ── Stage 2: Mesh export ─────────────────────────────────────────────────
    mesh = trimesh_result
    mesh.vertices = (
        mesh.vertices @ np.array(mesh_add_rot) @ np.array(rot_matrix)
    )

    mesh_obj_path = os.path.join(output_root, f"{filename}.obj")
    mesh.export(mesh_obj_path)

    del mesh_model
    free_vram()
    log_vram("after releasing 3D outputs")

    # ── Stage 3: URDF ────────────────────────────────────────────────────────
    urdf_convertor = URDFGenerator(
        GPT_CLIENT,
        render_view_num=4,
        decompose_convex=not disable_decompose_convex,
    )
    asset_attrs = {"version": VERSION}
    if asset_type:
        asset_attrs["category"] = asset_type

    urdf_root = f"{output_root}/URDF_{filename}"
    urdf_path = urdf_convertor(
        mesh_path=mesh_obj_path,
        output_root=urdf_root,
        **asset_attrs,
    )

    # Export GLB inside URDF mesh dir too
    mesh_out_final = (
        f"{urdf_root}/{urdf_convertor.output_mesh_dir}/{filename}.obj"
    )
    trimesh.load(mesh_out_final).export(mesh_out_final.replace(".obj", ".glb"))

    # ── Stage 4: Quality check ───────────────────────────────────────────────
    image_dir = f"{urdf_root}/{urdf_convertor.output_render_dir}/image_color"
    image_paths = glob(f"{image_dir}/*.png")
    images_list = []
    checkers = _get_checkers()
    for checker in checkers:
        images = combine_images_to_grid(image_paths)
        if isinstance(checker, ImageSegChecker):
            images = [
                f"{output_root}/{filename}_raw.png",
                f"{output_root}/{filename}_cond.png",
            ]
        images_list.append(images)
    qa_results = BaseChecker.validate(checkers, images_list)
    urdf_convertor.add_quality_tag(urdf_path, qa_results)

    # ── Stage 5: Organize results ────────────────────────────────────────────
    result_dir = f"{output_root}/result"
    if os.path.exists(result_dir):
        rmtree(result_dir, ignore_errors=True)
    os.makedirs(result_dir, exist_ok=True)
    copy(urdf_path, f"{result_dir}/{os.path.basename(urdf_path)}")
    copytree(
        f"{urdf_root}/{urdf_convertor.output_mesh_dir}",
        f"{result_dir}/{urdf_convertor.output_mesh_dir}",
    )
    if os.path.exists(video_path):
        copy(video_path, f"{result_dir}/video.mp4")

    # Copy renders so textto3d QA can reuse them without a second render pass
    render_src = os.path.join(urdf_root, urdf_convertor.output_render_dir, "image_color")
    render_dst = os.path.join(result_dir, "renders", "image_color")
    saved_renders: list[str] = []
    if os.path.exists(render_src):
        copytree(render_src, render_dst)
        saved_renders = sorted(glob(f"{render_dst}/*.png"))

    if not keep_intermediate:
        delete_dir(output_root, keep_subs=["result"])

    logger.info(f"Saved results for {image_path} in {result_dir}")

    # Collect output file paths
    final_obj = f"{result_dir}/{urdf_convertor.output_mesh_dir}/{filename}.obj"
    final_glb = f"{result_dir}/{urdf_convertor.output_mesh_dir}/{filename}.glb"
    final_urdf = f"{result_dir}/{os.path.basename(urdf_path)}"
    return {
        "obj": final_obj if os.path.exists(final_obj) else None,
        "glb": final_glb if os.path.exists(final_glb) else None,
        "urdf": final_urdf if os.path.exists(final_urdf) else None,
        "result_dir": result_dir,
        "renders": saved_renders,
    }


def entrypoint(**kwargs):
    """Legacy entrypoint for CLI usage. Processes a list of images."""
    image_path = kwargs.get("image_path") or []
    image_root = kwargs.get("image_root")
    output_root = kwargs.get("output_root", "outputs")
    asset_type = kwargs.get("asset_type") or []
    seed = kwargs.get("seed", 0)
    n_retry = kwargs.get("n_retry", 3)
    keep_intermediate = kwargs.get("keep_intermediate", False)
    disable_decompose_convex = kwargs.get("disable_decompose_convex", False)
    skip_exists = kwargs.get("skip_exists", False)

    if not image_path:
        if image_root:
            image_path = (
                glob(os.path.join(image_root, "*.png"))
                + glob(os.path.join(image_root, "*.jpg"))
                + glob(os.path.join(image_root, "*.jpeg"))
            )
        else:
            raise ValueError("Provide either image_path or image_root.")

    for idx, img_path in enumerate(image_path):
        try:
            filename = os.path.basename(img_path).split(".")[0]
            out_root = output_root
            if image_root or len(image_path) > 1:
                out_root = os.path.join(output_root, filename)

            mesh_out = f"{out_root}/{filename}.obj"
            if skip_exists and os.path.exists(mesh_out):
                logger.warning(
                    f"Skip {img_path}, already processed in {mesh_out}"
                )
                continue

            a_type = (
                asset_type[idx]
                if isinstance(asset_type, list) and idx < len(asset_type)
                else None
            )
            process_single_image(
                image_path=img_path,
                output_root=out_root,
                asset_type=a_type,
                seed=seed,
                n_retry=n_retry,
                keep_intermediate=keep_intermediate,
                disable_decompose_convex=disable_decompose_convex,
            )
        except Exception as e:
            logger.error(f"Failed to process {img_path}: {e}, skipping.")

    logger.info(f"Processing complete. Outputs saved to {output_root}")
