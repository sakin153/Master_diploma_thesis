"""Text/image → 3D pipeline with batch-optimised VRAM usage.

Batch strategy:
  Phase 1 — generate ALL images (text prompts only), text2image model
             stays loaded in VRAM for the whole batch, then is released.
  Phase 2 — generate ALL meshes (Hunyuan3D shape), 3D model stays loaded
             for the whole batch, then is released.
  Phase 3 — (if HUNYUAN3D_TEXTURE=1) apply texture to ALL meshes,
             paint model loaded once for the whole batch, then released.

This avoids reloading heavy models between objects.
"""

import base64
import os
import random
from collections import defaultdict
from dataclasses import dataclass
from typing import Optional

import torch
from PIL import Image
from asset_gen.models.image_comm_model import build_hf_image_pipeline
from asset_gen.models.segment_model import RembgRemover
from asset_gen.models.text_model import PROMPT_APPEND
from asset_gen.scripts.imageto3d import (
    _release_pipeline as _release_3d_pipeline,
    _get_texture_pipeline,
    _release_texture_pipeline,
    process_single_image,
)
from asset_gen.utils.gpt_clients import GPT_CLIENT
from asset_gen.utils.log import logger
from asset_gen.utils.process_media import (
    check_object_edge_truncated,
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

_PIPE_IMG = None


def _load_pipe_img():
    global _PIPE_IMG
    if _PIPE_IMG is None:
        log_vram("before text2img load")
        logger.info("Loading TEXT2IMG model...")
        _PIPE_IMG = build_hf_image_pipeline(
            os.environ.get("TEXT_MODEL", "sd15")
        )
        log_vram("after text2img load")
    return _PIPE_IMG


def _release_pipe_img():
    global _PIPE_IMG
    if _PIPE_IMG is not None:
        del _PIPE_IMG
        _PIPE_IMG = None
        free_vram()
        log_vram("after text2img release")


__all__ = ["text_to_3d", "GenerateItem"]


@dataclass
class GenerateItem:
    """One item to generate: either text prompt or ready image."""
    name: str
    prompt: Optional[str] = None        # text prompt
    image_path: Optional[str] = None    # path to existing image
    image_b64: Optional[str] = None     # base64-encoded image
    asset_type: Optional[str] = None    # semantic category hint
    seed_img: Optional[int] = None
    seed_3d: int = 0


def _generate_image_for_item(
    item: GenerateItem,
    img_save_dir: str,
    n_image_retry: int,
    img_denoise_step: int,
    text_guidance_scale: float,
    n_img_sample: int,
    image_height: int,
    image_width: int,
) -> Optional[str]:
    """
    Return path to a ready background-removed PNG for this item.
    For image inputs: copy/decode and return. For text: generate via SD.
    """
    save_node = item.name.replace(" ", "_")
    out_path = os.path.join(img_save_dir, f"{save_node}.png")

    # ── Already have an image ────────────────────────────────────────────────
    if item.image_b64:
        b64_str = item.image_b64.strip()
        logger.info(
            f"Processing image_b64 for '{item.name}': "
            f"length={len(b64_str)}, first 100 chars: {b64_str[:100]}..."
        )
        try:
            img_data = base64.b64decode(b64_str, validate=True)
        except Exception as e:
            logger.warning(f"Base64 validation failed: {e}, trying with padding fix...")
            # Add padding if needed
            padding = 4 - (len(b64_str) % 4)
            if padding and padding != 4:
                b64_str += "=" * padding
            img_data = base64.b64decode(b64_str, validate=False)

        logger.info(f"Decoded image data: {len(img_data)} bytes, first 20 bytes: {img_data[:20]}")

        # Debug: save raw data for inspection
        debug_bin = f"{out_path}.bin"
        with open(debug_bin, "wb") as f:
            f.write(img_data)
        logger.info(f"Saved raw image data to: {debug_bin}")

        try:
            raw = Image.open(__import__("io").BytesIO(img_data)).convert("RGB")
        except Exception as e:
            logger.error(
                f"Failed to load image from base64 for '{item.name}'. "
                f"Data size: {len(img_data)} bytes. First 100 bytes: {img_data[:100]!r}. "
                f"Error: {e}"
            )
            raise ValueError(
                f"Invalid image data for '{item.name}' - data corrupted/incomplete "
                f"({len(img_data)} bytes). Check {debug_bin} for raw data."
            )

        seg = BG_REMOVER(raw)
        seg.save(out_path)
        return out_path

    if item.image_path:
        raw = Image.open(item.image_path).convert("RGB")
        seg = BG_REMOVER(raw) if raw.mode != "RGBA" else raw
        seg.save(out_path)
        return out_path

    # ── Generate from text ───────────────────────────────────────────────────
    if not item.prompt:
        logger.error(f"Item '{item.name}' has no prompt or image, skipping.")
        return None

    f_prompt = PROMPT_APPEND.format(object=item.prompt)
    seed = item.seed_img
    select_image = None

    for try_idx in range(n_image_retry):
        if select_image is not None:
            break
        logger.info(
            f"Image GEN for '{item.name}' "
            f"try {try_idx + 1}/{n_image_retry}, "
            f"seed={seed}, size={image_width}x{image_height}, "
            f"steps={img_denoise_step}, prompt={f_prompt}"
        )
        torch.cuda.empty_cache()
        images = _load_pipe_img().run(
            f_prompt,
            num_inference_steps=img_denoise_step,
            guidance_scale=text_guidance_scale,
            num_images_per_prompt=n_img_sample,
            height=image_height,
            width=image_width,
            generator=(
                torch.Generator().manual_seed(seed)
                if seed is not None else None
            ),
        )
        for raw_image in images:
            seg_image = BG_REMOVER(raw_image)
            semantic_flag, sem_res = SEMANTIC_CHECKER(
                item.prompt, [seg_image.convert("RGB")]
            )
            seg_flag, seg_res = SEG_CHECKER(
                [raw_image, seg_image.convert("RGB")]
            )
            edge_flag = check_object_edge_truncated(
                __import__("numpy").array(seg_image)[..., -1]
            )
            logger.warning(
                f"SEMANTIC: {sem_res}. SEG: {seg_res}. EDGE: {edge_flag}"
            )
            if (
                (edge_flag and semantic_flag and seg_flag)
                or (edge_flag and semantic_flag is None)
                or (edge_flag and seg_flag is None)
            ):
                raw_image.save(out_path.replace(".png", "_raw.png"))
                seg_image.save(out_path)
                select_image = seg_image
                break

        seed = random.randint(0, 100000) if seed is not None else None

    if select_image is None:
        logger.warning(
            f"Image generation for '{item.name}' did not pass QA, "
            "using last generated image."
        )
        if images:
            seg_image = BG_REMOVER(images[-1])
            seg_image.save(out_path)
            return out_path
        return None

    return out_path


def text_to_3d(
    items: list[GenerateItem],
    output_root: str,
    n_image_retry: int = 2,
    n_asset_retry: int = 2,
    n_pipe_retry: int = 1,
    img_denoise_step: int = 50,
    text_guidance_scale: float = 7.0,
    n_img_sample: int = 3,
    image_height: int = 768,
    image_width: int = 768,
    keep_intermediate: bool = True,
    disable_decompose_convex: bool = False,
    enable_texture: bool = False,
) -> dict:
    """
    Batch generate 3D assets from a list of GenerateItem.

    Phase 1: Generate all images (text2img loaded once).
    Phase 2: Generate all 3D meshes (Hunyuan3D loaded once).

    Returns:
        {
          "assets": { name: "asset3d/<name>/result" },
          "files":  { name: {"obj": ..., "glb": ..., "urdf": ...} },
          "quality": { name: qa_result },
        }
    """
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
        f"n_pipe_retry={n_pipe_retry}, n_img_sample={n_img_sample}"
    )

    # ── Phase 1: generate ALL images ────────────────────────────────────────
    image_paths: dict[str, Optional[str]] = {}
    for item in items:
        image_paths[item.name] = _generate_image_for_item(
            item=item,
            img_save_dir=img_save_dir,
            n_image_retry=n_image_retry,
            img_denoise_step=img_denoise_step,
            text_guidance_scale=text_guidance_scale,
            n_img_sample=n_img_sample,
            image_height=image_height,
            image_width=image_width,
        )

    _release_pipe_img()

    # ── Phase 2: generate ALL 3D meshes ─────────────────────────────────────
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
            )
            if not file_paths:
                current_seed_3d = random.randint(0, 100000)
                continue

            # QA on rendered views
            result_dir = file_paths.get("result_dir", "")
            obj_path = file_paths.get("obj")
            if obj_path and os.path.exists(obj_path):
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
                logger.warning(
                    f"'{item.name}' QA: {qa_result}"
                )
                results["quality"][item.name] = qa_result
                if qa_flag is None or qa_flag is True:
                    success = True
            else:
                success = True

            results["assets"][item.name] = (
                f"asset3d/{save_node}/result"
            )
            results["files"][item.name] = file_paths
            if success:
                break

            current_seed_3d = random.randint(0, 100000)

        free_vram()

    _release_3d_pipeline()

    # ── Phase 3: texture ALL meshes (paint model loaded once) ───────────────
    # enable_texture from request body takes priority; env var is a global fallback.
    use_texture = enable_texture or os.environ.get("HUNYUAN3D_TEXTURE", "0") == "1"
    if use_texture:
        logger.info("Phase 3: applying Hunyuan3D-Paint-Turbo texture to all meshes...")
        log_vram("before texture pipeline load")
        try:
            import trimesh as _trimesh
            from PIL import Image as _PILImage
            tex_pipe = _get_texture_pipeline()
            for item in items:
                file_paths = results["files"].get(item.name)
                img_path = image_paths.get(item.name)
                if not file_paths or not img_path or not os.path.exists(img_path):
                    continue
                obj_path = file_paths.get("obj")
                glb_path = file_paths.get("glb")
                if not obj_path or not os.path.exists(obj_path):
                    continue
                logger.info(f"Texturing '{item.name}'...")
                try:
                    mesh = _trimesh.load(obj_path)
                    image = _PILImage.open(img_path)
                    textured = tex_pipe.run(mesh, image)
                    if glb_path:
                        textured.export(glb_path)
                        logger.info(
                            f"'{item.name}' textured GLB saved: {glb_path}"
                        )
                except Exception as exc:
                    logger.warning(
                        f"Texture failed for '{item.name}': {exc}, "
                        "keeping untextured mesh."
                    )
        finally:
            _release_texture_pipeline()
        log_vram("after texture pipeline release")

    return results
