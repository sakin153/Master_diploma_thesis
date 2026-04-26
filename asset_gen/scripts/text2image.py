"""Batch text-to-image generation module.

Loads the diffusion model once for the whole batch, generates all images,
then releases VRAM. This avoids the expensive load/unload cycle that would
occur if each item were processed independently.

Extracted from textto3d.py to keep responsibilities separate.
"""

import base64
import io
import os
import random
from dataclasses import dataclass
from typing import Optional

import torch
from PIL import Image

from asset_gen.models.text_model import PROMPT_APPEND
from asset_gen.utils.log import logger
from asset_gen.utils.process_media import check_object_edge_truncated
from asset_gen.utils.vram_utils import free_vram, log_vram

__all__ = ["Text2ImagePipeline", "generate_images_batch"]


# ---------------------------------------------------------------------------
# Pipeline lifecycle wrapper
# ---------------------------------------------------------------------------

class Text2ImagePipeline:
    """Lifecycle manager for a HuggingFace text-to-image model.

    Supports use as a context manager so VRAM is always released:

    Example:
        ```py
        with Text2ImagePipeline("sdxl-turbo") as pipe:
            images = pipe.generate("a wooden chair", ...)
        # VRAM freed here
        ```
    """

    def __init__(self, model_name: str = "sd15") -> None:
        self._model_name = model_name
        self._runner = None

    def load(self) -> "Text2ImagePipeline":
        """Load model into VRAM (or CPU offload)."""
        from asset_gen.models.model_image_runtime import build_hf_image_pipeline
        log_vram("before text2img load")
        logger.info(f"Loading TEXT2IMG model: {self._model_name}")
        self._runner = build_hf_image_pipeline(self._model_name)
        log_vram("after text2img load")
        return self

    def release(self) -> None:
        """Delete model and free VRAM."""
        if self._runner is not None:
            del self._runner
            self._runner = None
            free_vram()
            log_vram("after text2img release")

    def generate(
        self,
        prompt: str,
        num_inference_steps: int = 25,
        guidance_scale: float = 7.0,
        num_images_per_prompt: int = 1,
        height: int = 768,
        width: int = 768,
        seed: Optional[int] = None,
    ) -> list[Image.Image]:
        """Run inference and return a list of PIL images.

        Args:
            prompt: Formatted text prompt (already includes PROMPT_APPEND).
            num_inference_steps: Denoising steps.
            guidance_scale: CFG scale.
            num_images_per_prompt: How many images to generate per call.
            height: Image height in pixels.
            width: Image width in pixels.
            seed: Optional reproducibility seed.

        Returns:
            List of RGB PIL.Image objects.
        """
        if self._runner is None:
            raise RuntimeError("Pipeline not loaded. Call load() first.")
        if seed is not None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            generator = torch.Generator(device=device).manual_seed(seed)
        else:
            generator = None
        return self._runner.run(
            prompt,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            num_images_per_prompt=num_images_per_prompt,
            height=height,
            width=width,
            generator=generator,
        )

    def __enter__(self) -> "Text2ImagePipeline":
        return self.load()

    def __exit__(self, *_) -> None:
        self.release()


# ---------------------------------------------------------------------------
# Per-item generation
# ---------------------------------------------------------------------------

@dataclass
class GenerateItem:
    """One item to generate: either text prompt or ready image."""
    name: str
    prompt: Optional[str] = None
    image_path: Optional[str] = None
    image_b64: Optional[str] = None
    asset_type: Optional[str] = None
    seed_img: Optional[int] = None
    seed_3d: int = 0


def _generate_one(
    item: GenerateItem,
    pipe: Text2ImagePipeline,
    bg_remover,
    seg_checker,
    semantic_checker,
    img_save_dir: str,
    n_retry: int,
    img_denoise_step: int,
    guidance_scale: float,
    n_sample: int,
    height: int,
    width: int,
    skip_qa: bool,
) -> Optional[str]:
    """Generate or decode one image and return the path to the BG-removed PNG.

    For image inputs (image_b64 / image_path): decode, remove background, save.
    For text inputs: generate via diffusion, apply QA checks, save best.
    """
    save_node = item.name.replace(" ", "_")
    out_path = os.path.join(img_save_dir, f"{save_node}.png")

    # ── Already have an image ────────────────────────────────────────────────
    if item.image_b64:
        b64_str = item.image_b64.strip()
        try:
            img_data = base64.b64decode(b64_str, validate=True)
        except Exception:
            padding = 4 - (len(b64_str) % 4)
            if padding and padding != 4:
                b64_str += "=" * padding
            img_data = base64.b64decode(b64_str, validate=False)

        try:
            raw = Image.open(io.BytesIO(img_data)).convert("RGB")
        except Exception as exc:
            logger.error(
                f"Failed to decode base64 image for '{item.name}': {exc}"
            )
            raise ValueError(
                f"Invalid image data for '{item.name}' ({len(img_data)} bytes)."
            ) from exc

        seg = bg_remover(raw)
        seg.save(out_path)
        return out_path

    if item.image_path:
        raw = Image.open(item.image_path).convert("RGB")
        seg = bg_remover(raw) if raw.mode != "RGBA" else raw
        seg.save(out_path)
        return out_path

    # ── Generate from text ───────────────────────────────────────────────────
    if not item.prompt:
        logger.error(f"Item '{item.name}' has no prompt or image, skipping.")
        return None

    f_prompt = PROMPT_APPEND.format(object=item.prompt)
    seed = item.seed_img
    select_image = None
    images = []

    for try_idx in range(n_retry):
        if select_image is not None:
            break
        logger.info(
            f"Image GEN for '{item.name}' "
            f"try {try_idx + 1}/{n_retry}, "
            f"seed={seed}, size={width}x{height}, "
            f"steps={img_denoise_step}, prompt={f_prompt}"
        )
        images = pipe.generate(
            f_prompt,
            num_inference_steps=img_denoise_step,
            guidance_scale=guidance_scale,
            num_images_per_prompt=n_sample,
            height=height,
            width=width,
            seed=seed,
        )
        for raw_image in images:
            seg_image = bg_remover(raw_image)
            if skip_qa:
                raw_image.save(out_path.replace(".png", "_raw.png"))
                seg_image.save(out_path)
                select_image = seg_image
                break

            import numpy as np
            semantic_flag, sem_res = semantic_checker(
                item.prompt, [seg_image.convert("RGB")]
            )
            seg_flag, seg_res = seg_checker([raw_image, seg_image.convert("RGB")])
            edge_flag = check_object_edge_truncated(
                np.array(seg_image)[..., -1]
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
            seg_image = bg_remover(images[-1])
            seg_image.save(out_path)
            return out_path
        return None

    return out_path


# ---------------------------------------------------------------------------
# Batch entry point
# ---------------------------------------------------------------------------

def generate_images_batch(
    items: list[GenerateItem],
    img_save_dir: str,
    bg_remover,
    seg_checker,
    semantic_checker,
    n_retry: int = 2,
    img_denoise_step: int = 25,
    guidance_scale: float = 7.0,
    n_sample: int = 1,
    height: int = 768,
    width: int = 768,
    skip_qa: bool = False,
    model_name: str = "sd15",
) -> dict[str, Optional[str]]:
    """Generate images for all items, loading the model only once.

    Args:
        items: List of GenerateItem (text prompt, image path, or base64).
        img_save_dir: Directory where generated PNGs are saved.
        bg_remover: RembgRemover instance (CPU, already created).
        seg_checker: ImageSegChecker instance.
        semantic_checker: SemanticConsistChecker instance.
        n_retry: Retry attempts per item.
        img_denoise_step: Diffusion denoising steps.
        guidance_scale: CFG guidance scale.
        n_sample: Images generated per prompt per attempt.
        height: Output image height in pixels.
        width: Output image width in pixels.
        skip_qa: Skip semantic/segmentation quality checks.
        model_name: Text-to-image model key (must match PIPELINE_REGISTRY).

    Returns:
        Dict mapping ``item.name`` → path to background-removed PNG, or None.
        Items with pre-supplied images skip diffusion and are decoded directly.
    """
    os.makedirs(img_save_dir, exist_ok=True)

    # Items that already have images don't need the diffusion model at all.
    pre_supplied = {i.name for i in items if i.image_b64 or i.image_path}
    needs_diffusion = [i for i in items if i.name not in pre_supplied]

    result: dict[str, Optional[str]] = {}

    # Process pre-supplied images without loading diffusion model
    if pre_supplied:
        for item in items:
            if item.name in pre_supplied:
                result[item.name] = _generate_one(
                    item=item,
                    pipe=None,  # not called for image inputs
                    bg_remover=bg_remover,
                    seg_checker=seg_checker,
                    semantic_checker=semantic_checker,
                    img_save_dir=img_save_dir,
                    n_retry=n_retry,
                    img_denoise_step=img_denoise_step,
                    guidance_scale=guidance_scale,
                    n_sample=n_sample,
                    height=height,
                    width=width,
                    skip_qa=skip_qa,
                )

    # Process text-prompt items with diffusion model loaded once
    if needs_diffusion:
        with Text2ImagePipeline(model_name) as pipe:
            for item in needs_diffusion:
                result[item.name] = _generate_one(
                    item=item,
                    pipe=pipe,
                    bg_remover=bg_remover,
                    seg_checker=seg_checker,
                    semantic_checker=semantic_checker,
                    img_save_dir=img_save_dir,
                    n_retry=n_retry,
                    img_denoise_step=img_denoise_step,
                    guidance_scale=guidance_scale,
                    n_sample=n_sample,
                    height=height,
                    width=width,
                    skip_qa=skip_qa,
                )

    return result
