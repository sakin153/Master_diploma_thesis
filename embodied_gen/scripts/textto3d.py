"""Text/image → 3D pipeline with batch-optimised VRAM usage.

Batch strategy:
  Phase 1 — generate ALL images (text prompts only), text2image model
             stays loaded in VRAM for the whole batch, then is released.
  Phase 2 — generate ALL meshes (Hunyuan3D), 3D model stays loaded in
             VRAM for the whole batch, then is released.

This avoids reloading heavy models for every object.
"""

import base64
import os
import random
from collections import defaultdict
from dataclasses import dataclass
from typing import Optional

import torch
from PIL import Image
from embodied_gen.models.image_comm_model import build_hf_image_pipeline
from embodied_gen.models.segment_model import RembgRemover
from embodied_gen.models.text_model import PROMPT_APPEND
from embodied_gen.scripts.imageto3d import (
    _release_pipeline as _release_3d_pipeline,
    process_single_image,
)
from embodied_gen.utils.gpt_clients import GPT_CLIENT
from embodied_gen.utils.log import logger
from embodied_gen.utils.process_media import (
    check_object_edge_truncated,
    combine_images_to_grid,
    render_asset3d,
)
from embodied_gen.utils.vram_utils import free_vram, log_vram
from embodied_gen.validators.quality_checkers import (
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
) -> Optional[str]:
    """
    Return path to a ready background-removed PNG for this item.
    For image inputs: copy/decode and return. For text: generate via SD.
    """
    save_node = item.name.replace(" ", "_")
    out_path = os.path.join(img_save_dir, f"{save_node}.png")

    # ── Already have an image ────────────────────────────────────────────────
    if item.image_b64:
        img_data = base64.b64decode(item.image_b64)
        raw = Image.open(__import__("io").BytesIO(img_data)).convert("RGB")
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
            f"seed={seed}, prompt={f_prompt}"
        )
        torch.cuda.empty_cache()
        images = _load_pipe_img().run(
            f_prompt,
            num_inference_steps=img_denoise_step,
            guidance_scale=text_guidance_scale,
            num_images_per_prompt=n_img_sample,
            height=1024,
            width=1024,
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
    img_denoise_step: int = 25,
    text_guidance_scale: float = 7.0,
    n_img_sample: int = 1,
    keep_intermediate: bool = False,
    disable_decompose_convex: bool = False,
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
    return results
