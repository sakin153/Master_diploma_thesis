"""Text/image → 3D pipeline with batch-optimised VRAM usage.

Batch strategy:
  Phase 1  — generate ALL images (text2img model loaded once, then released).
  Phase 1.5 — (if enable_delight) de-light ALL images (delight model loaded
               once after text2img is released, then released).
  Phase 2  — generate ALL 3D meshes via TRELLIS API (no local VRAM needed),
               runs QA, writes URDF per object.

Phases are strictly sequential so heavy models never coexist in VRAM.
"""

import os
import random
from collections import defaultdict
from typing import Optional

from PIL import Image

from asset_gen.models.segment_model import RembgRemover
from asset_gen.scripts.imageto3d import (
    _release_pipeline as _release_3d_pipeline,
    process_single_image,
)
from asset_gen.scripts.text2image import (
    GenerateItem,
    generate_images_batch,
)
from asset_gen.utils.gpt_clients import GPT_CLIENT
from asset_gen.utils.log import logger
from asset_gen.utils.process_media import (
    combine_images_to_grid,
    render_asset3d,
)
from asset_gen.utils.vram_utils import free_vram, log_vram
from asset_gen.validators.quality_checkers import (
    SemanticConsistChecker,
    ImageSegChecker,
    TextGenAlignChecker,
)

os.environ["TOKENIZERS_PARALLELISM"] = "false"

SEMANTIC_CHECKER = SemanticConsistChecker(GPT_CLIENT)
SEG_CHECKER = ImageSegChecker(GPT_CLIENT)
TXTGEN_CHECKER = TextGenAlignChecker(GPT_CLIENT)
BG_REMOVER = RembgRemover()

__all__ = ["text_to_3d", "GenerateItem"]

# Re-export GenerateItem so callers that import it from textto3d keep working.
GenerateItem = GenerateItem  # noqa: F811 (re-export)


def text_to_3d(
    items: list[GenerateItem],
    output_root: str,
    n_image_retry: int = 2,
    n_asset_retry: int = 2,
    n_pipe_retry: int = 1,
    img_denoise_step: int = 25,
    text_guidance_scale: float = 7.0,
    n_img_sample: int = 1,
    image_height: int = 768,
    image_width: int = 768,
    keep_intermediate: bool = True,
    disable_decompose_convex: bool = False,
    enable_texture: bool = False,   # deprecated: TRELLIS bakes texture natively
    enable_delight: bool = False,
    skip_qa: bool = False,
) -> dict:
    """Batch-generate 3D assets from a list of GenerateItem.

    Phase 1:   Generate all images (text2img loaded once).
    Phase 1.5: De-light all images (delight model loaded once, optional).
    Phase 2:   Generate all 3D meshes via TRELLIS API.

    Args:
        items: List of GenerateItem (text prompt or pre-supplied image).
        output_root: Root directory for all outputs.
        n_image_retry: Max retries for text-to-image generation.
        n_asset_retry: Max retries for 3D generation per object.
        n_pipe_retry: Max pipeline-level retries per object.
        img_denoise_step: Diffusion denoising steps.
        text_guidance_scale: CFG guidance scale for text-to-image.
        n_img_sample: Images generated per prompt attempt.
        image_height: Diffusion output height in pixels.
        image_width: Diffusion output width in pixels.
        keep_intermediate: Keep intermediate files after success.
        disable_decompose_convex: Skip CoACD in URDF generation.
        enable_texture: Deprecated — TRELLIS generates texture natively.
            Accepted for API backwards-compatibility but has no effect.
        enable_delight: Apply Hunyuan3D-Delight to remove lighting from
            generated images before sending to TRELLIS.
        skip_qa: Skip all GPT/CLIP quality checks.

    Returns:
        {
          "assets": { name: "asset3d/<name>/result" },
          "files":  { name: {"obj": ..., "glb": ..., "urdf": ...} },
          "quality": { name: qa_result },
        }
    """
    if enable_texture:
        logger.warning(
            "enable_texture is deprecated. TRELLIS generates textured GLB "
            "natively. The flag is accepted but ignored."
        )

    _release_3d_pipeline()

    img_save_dir = os.path.join(output_root, "images")
    asset_save_dir = os.path.join(output_root, "asset3d")
    os.makedirs(img_save_dir, exist_ok=True)
    os.makedirs(asset_save_dir, exist_ok=True)

    results: dict = defaultdict(dict)

    logger.info(
        "text_to_3d settings: "
        f"model={os.environ.get('TEXT_MODEL', 'sd15')}, "
        f"image_size={image_width}x{image_height}, "
        f"img_steps={img_denoise_step}, "
        f"n_image_retry={n_image_retry}, n_asset_retry={n_asset_retry}, "
        f"n_pipe_retry={n_pipe_retry}, n_img_sample={n_img_sample}, "
        f"enable_delight={enable_delight}, skip_qa={skip_qa}"
    )

    # ── Phase 1: generate ALL images ──────────────────────────────────────────
    image_paths: dict[str, Optional[str]] = generate_images_batch(
        items=items,
        img_save_dir=img_save_dir,
        bg_remover=BG_REMOVER,
        seg_checker=SEG_CHECKER,
        semantic_checker=SEMANTIC_CHECKER,
        n_retry=n_image_retry,
        img_denoise_step=img_denoise_step,
        guidance_scale=text_guidance_scale,
        n_sample=n_img_sample,
        height=image_height,
        width=image_width,
        skip_qa=skip_qa,
        model_name=os.environ.get("TEXT_MODEL", "sd15"),
    )
    # Text2ImagePipeline is released automatically by generate_images_batch.

    # ── Phase 1.5: de-light ALL images (optional) ─────────────────────────────
    use_delight = enable_delight or os.environ.get("ENABLE_DELIGHT", "0") == "1"
    delight_model = None

    if use_delight:
        log_vram("before delight load")
        from asset_gen.models.delight_model import DelightingModel
        delight_model = DelightingModel()
        delight_model._lazy_init()
        logger.info("Phase 1.5: applying de-lighting to all images...")
        for item in items:
            img_path = image_paths.get(item.name)
            if not img_path or not os.path.exists(img_path):
                continue
            try:
                img = Image.open(img_path)
                # Save pre-delight version for debugging
                img.save(img_path.replace(".png", "_pre_delight.png"))
                delighted = delight_model(img.convert("RGBA"))
                delighted.save(img_path)
                logger.info(f"De-lit '{item.name}' → {img_path}")
            except Exception as exc:
                logger.warning(
                    f"De-lighting failed for '{item.name}': {exc}. Skipping."
                )
        delight_model.release()
        log_vram("after delight release")

    # ── Phase 2: generate ALL 3D meshes ──────────────────────────────────────
    for item in items:
        img_path = image_paths.get(item.name)
        if not img_path or not os.path.exists(img_path):
            logger.error(
                f"No image for '{item.name}', skipping 3D generation."
            )
            continue

        save_node = item.name.replace(" ", "_")
        node_save_dir = os.path.join(asset_save_dir, save_node)

        success = False
        current_seed_3d = item.seed_3d
        for pipe_try in range(n_pipe_retry):
            logger.info(
                f"3D pipeline for '{item.name}' "
                f"attempt {pipe_try + 1}/{n_pipe_retry}"
            )
            file_paths = process_single_image(
                image_path=img_path,
                output_root=node_save_dir,
                asset_type=item.asset_type,
                seed=current_seed_3d,
                n_retry=n_asset_retry,
                keep_intermediate=keep_intermediate,
                disable_decompose_convex=disable_decompose_convex,
                skip_qa=skip_qa,
                delight_model=None,  # delight already applied in Phase 1.5
            )
            if not file_paths:
                current_seed_3d = random.randint(0, 100000)
                continue

            # Post-generation QA on rendered mesh views
            result_dir = file_paths.get("result_dir", "")
            obj_path = file_paths.get("obj")
            if obj_path and os.path.exists(obj_path) and not skip_qa:
                image_path_list = render_asset3d(
                    obj_path,
                    output_root=result_dir,
                    num_images=4,
                    elevation=(30, -30),
                    output_subdir="renders",
                    no_index_file=True,
                )
                grid = combine_images_to_grid(image_path_list)
                check_text = item.asset_type or item.prompt or item.name
                qa_flag, qa_result = TXTGEN_CHECKER(check_text, grid)
                logger.warning(f"'{item.name}' QA: {qa_result}")
                results["quality"][item.name] = qa_result
                if qa_flag is None or qa_flag is True:
                    success = True
            else:
                success = True

            results["assets"][item.name] = f"asset3d/{save_node}/result"
            results["files"][item.name] = file_paths
            if success:
                break

            current_seed_3d = random.randint(0, 100000)

        free_vram()

    _release_3d_pipeline()

    return results
