import os
import random
from glob import glob
from shutil import copy, copytree, rmtree

import numpy as np
import trimesh
from PIL import Image
from asset_gen.data.utils import delete_dir
from asset_gen.models.segment_model import (
    RembgRemover,
    SAMRemover,
    get_segmented_image_by_agent,
)
from asset_gen.utils.gpt_clients import GPT_CLIENT
from asset_gen.utils.inference import image3d_model_infer
from asset_gen.utils.log import logger
from asset_gen.utils.process_media import (
    combine_images_to_grid,
)
from asset_gen.utils.tags import VERSION
from asset_gen.utils.vram_utils import free_vram, log_vram
from asset_gen.validators.quality_checkers import (
    BaseChecker,
    ImageAestheticChecker,
    ImageSegChecker,
    MeshGeoChecker,
    TrellisOutputChecker,
)
from asset_gen.validators.urdf_convertor import URDFGenerator

# GPT-based quality checkers: no VRAM, safe at module level.
SEG_CHECKER = ImageSegChecker(GPT_CLIENT)
GEO_CHECKER = MeshGeoChecker(GPT_CLIENT)
_AESTHETIC_CHECKER = None
_TRELLIS_CHECKER = TrellisOutputChecker()


def _get_aesthetic_checker():
    global _AESTHETIC_CHECKER
    if _AESTHETIC_CHECKER is None:
        _AESTHETIC_CHECKER = ImageAestheticChecker()
    return _AESTHETIC_CHECKER


def _get_checkers():
    return [GEO_CHECKER, SEG_CHECKER, _get_aesthetic_checker()]


# ── Lazy background removers (RembgRemover: CPU-only, safe to keep loaded) ───
_RBG_REMOVER = None
_SAM_REMOVER = None


def _get_rbg_remover() -> RembgRemover:
    global _RBG_REMOVER
    if _RBG_REMOVER is None:
        _RBG_REMOVER = RembgRemover()
    return _RBG_REMOVER


def _get_sam_remover() -> SAMRemover:
    global _SAM_REMOVER
    if _SAM_REMOVER is None:
        log_vram("before SAM load")
        logger.info("Loading SAMRemover (vit_h)...")
        _SAM_REMOVER = SAMRemover()
        log_vram("after SAM load")
    return _SAM_REMOVER


def _release_sam() -> None:
    global _SAM_REMOVER
    if _SAM_REMOVER is not None:
        del _SAM_REMOVER
        _SAM_REMOVER = None
        free_vram()
        log_vram("after SAM release")


# ── TRELLIS client (no VRAM — remote API) ────────────────────────────────────
_TRELLIS_CLIENT = None


def _get_pipeline():
    global _TRELLIS_CLIENT
    if _TRELLIS_CLIENT is None:
        from asset_gen.models.trellis_client import TrellisClient
        _TRELLIS_CLIENT = TrellisClient()
    return _TRELLIS_CLIENT


def _release_pipeline():
    global _TRELLIS_CLIENT
    _TRELLIS_CLIENT = None  # no VRAM, just drop the reference


def process_single_image(
    image_path: str,
    output_root: str,
    asset_type: str = None,
    seed: int = 0,
    n_retry: int = 3,
    keep_intermediate: bool = False,
    disable_decompose_convex: bool = False,
    texture_size: int = 1024,
    skip_qa: bool = False,
    delight_model=None,
) -> dict:
    """Process one image → OBJ + GLB (textured) + URDF. Returns dict with paths.

    Args:
        image_path: Path to the source image.
        output_root: Directory where all outputs will be written.
        asset_type: Semantic category hint for GPT-based URDF parameter estimation.
        seed: 3D generation seed (passed to TRELLIS).
        n_retry: How many times to retry if TRELLIS fails or QA check fails.
        keep_intermediate: Keep intermediate files (images, logs) after success.
        disable_decompose_convex: Skip CoACD convex decomposition in URDF.
        texture_size: Texture atlas resolution for TRELLIS (pixels).
        skip_qa: Skip all GPT/CLIP quality checks.
        delight_model: Optional DelightingModel instance. If provided, applied
            to the segmented image before sending to TRELLIS.

    Returns:
        dict with keys: ``obj``, ``glb``, ``urdf``, ``result_dir``.
        Empty dict on total failure.
    """
    filename = os.path.basename(image_path).split(".")[0]
    os.makedirs(output_root, exist_ok=True)

    image = Image.open(image_path)
    image.save(f"{output_root}/{filename}_raw.png")

    seg_path = f"{output_root}/{filename}_cond.png"

    # ── Stage 1: Background removal + segmentation ───────────────────────────
    # get_segmented_image_by_agent() tries SAM → inverted SAM → rembg and
    # applies trellis_preprocess() internally → returns RGB 518×518.
    if image.mode != "RGBA":
        seg_image = get_segmented_image_by_agent(
            image,
            sam_remover=_get_sam_remover(),
            rbg_remover=_get_rbg_remover(),
            seg_checker=SEG_CHECKER,
            save_path=seg_path,
            mode="loose",
        )
    else:
        from asset_gen.data.utils import trellis_preprocess
        seg_image = trellis_preprocess(image)
        seg_image.save(seg_path)

    # Release SAM before de-lighting to free VRAM
    _release_sam()

    # ── Stage 1b: De-lighting (optional) ────────────────────────────────────
    if delight_model is not None:
        logger.info(f"Applying de-lighting to {filename}...")
        try:
            seg_image.save(
                f"{output_root}/{filename}_pre_delight.png"
            )
            # delight_model expects RGBA; seg_image from trellis_preprocess is RGB
            rgba_for_delight = seg_image.convert("RGBA") if seg_image.mode == "RGB" else seg_image
            seg_image = delight_model(rgba_for_delight)
            seg_image.save(f"{output_root}/{filename}_delight.png")
            logger.info(f"De-lighting done for {filename}.")
        except Exception as exc:
            logger.warning(
                f"De-lighting failed for {filename}: {exc}. Continuing without it."
            )

    mesh_model = None
    trimesh_result = None
    glb_bytes_raw = None

    # ── Stage 2: Image → 3D (TRELLIS API) ────────────────────────────────────
    current_seed = seed
    for try_idx in range(n_retry):
        logger.info(
            f"Try: {try_idx + 1}/{n_retry}, "
            f"Seed: {current_seed}, Input: {seg_path}"
        )
        try:
            outputs = image3d_model_infer(
                _get_pipeline(),
                seg_image,
                current_seed,
                texture_size=texture_size,
            )
        except Exception as e:
            logger.error(
                f"[TRELLIS Failed] {image_path}: {e}, "
                f"retry {try_idx + 1}/{n_retry}"
            )
            current_seed = random.randint(0, 100000)
            continue

        trimesh_result = outputs.get("trimesh", [None])[0]
        glb_bytes_raw = outputs.get("glb_bytes")
        mesh_model = outputs.get("mesh", [None])[0]  # always None for TRELLIS

        # ── Geometry QA (fast, no GPT) ────────────────────────────────────
        if trimesh_result is not None and not skip_qa:
            ok, msg = _TRELLIS_CHECKER(trimesh_result)
            if not ok:
                logger.warning(
                    f"[TrellisOutputChecker] {msg}. "
                    f"Retrying {try_idx + 1}/{n_retry}."
                )
                current_seed = random.randint(0, 100000)
                trimesh_result = None
                continue

        logger.info("TRELLIS generation succeeded.")
        break

    if trimesh_result is None:
        logger.error(f"Exceeded retry limit for {image_path}, skipping.")
        return {}

    # ── Stage 3: Mesh export ─────────────────────────────────────────────────
    mesh_obj_path = os.path.join(output_root, f"{filename}.obj")
    mesh_glb_path = os.path.join(output_root, f"{filename}.glb")

    # Save original GLB (with baked texture) from TRELLIS directly.
    if glb_bytes_raw:
        with open(mesh_glb_path, "wb") as f:
            f.write(glb_bytes_raw)
    else:
        trimesh_result.export(mesh_glb_path)

    # Export OBJ for URDF/MJCF pipeline (trimesh also writes .mtl + texture PNG).
    trimesh_result.export(mesh_obj_path)

    # ── Stage 4: URDF ────────────────────────────────────────────────────────
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

    # Export GLB inside the URDF mesh dir too (used by API download endpoints)
    mesh_out_final = (
        f"{urdf_root}/{urdf_convertor.output_mesh_dir}/{filename}.obj"
    )
    if os.path.exists(mesh_out_final):
        glb_final = mesh_out_final.replace(".obj", ".glb")
        if glb_bytes_raw:
            with open(glb_final, "wb") as f:
                f.write(glb_bytes_raw)
        else:
            trimesh.load(mesh_out_final).export(glb_final)

    # ── Stage 5: GPT-based quality check ─────────────────────────────────────
    if skip_qa:
        logger.info("Skipping GPT QA checks (skip_qa=True).")
    else:
        image_dir = (
            f"{urdf_root}/{urdf_convertor.output_render_dir}/image_color"
        )
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

    # ── Stage 6: Organize results ─────────────────────────────────────────────
    result_dir = f"{output_root}/result"
    if os.path.exists(result_dir):
        rmtree(result_dir, ignore_errors=True)
    os.makedirs(result_dir, exist_ok=True)
    copy(urdf_path, f"{result_dir}/{os.path.basename(urdf_path)}")
    copytree(
        f"{urdf_root}/{urdf_convertor.output_mesh_dir}",
        f"{result_dir}/{urdf_convertor.output_mesh_dir}",
    )

    if not keep_intermediate:
        delete_dir(output_root, keep_subs=["result"])

    logger.info(f"Saved results for {image_path} in {result_dir}")

    final_obj = f"{result_dir}/{urdf_convertor.output_mesh_dir}/{filename}.obj"
    final_glb = f"{result_dir}/{urdf_convertor.output_mesh_dir}/{filename}.glb"
    final_urdf = f"{result_dir}/{os.path.basename(urdf_path)}"
    return {
        "obj": final_obj if os.path.exists(final_obj) else None,
        "glb": final_glb if os.path.exists(final_glb) else None,
        "urdf": final_urdf if os.path.exists(final_urdf) else None,
        "result_dir": result_dir,
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
            fn = os.path.basename(img_path).split(".")[0]
            out_root = output_root
            if image_root or len(image_path) > 1:
                out_root = os.path.join(output_root, fn)

            mesh_out = f"{out_root}/{fn}.obj"
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
