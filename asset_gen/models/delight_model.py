"""De-lighting model: removes specular highlights and shadows from RGBA images.

Based on Hunyuan3D-Delight v2.0 (tencent/Hunyuan3D-2/hunyuan3d-delight-v2-0).
Adapted from EmbodiedGen/embodied_gen/models/delight_model.py with:
  - spaces.GPU decorator removed (not running in HF Spaces)
  - enable_model_cpu_offload() for 8 GB VRAM (RTX 2070 Super)
  - Explicit release() for sequential VRAM management
"""

import os
from typing import Union

import cv2
import numpy as np
import torch
from PIL import Image

from asset_gen.utils.log import logger
from asset_gen.utils.vram_utils import free_vram, log_vram

__all__ = ["DelightingModel"]


class DelightingModel:
    """Removes lighting from RGBA images using Hunyuan3D-Delight.

    Call ``_lazy_init()`` (or just ``__call__``) to load the model.
    Call ``release()`` immediately after use to free VRAM.

    Example:
        ```py
        from asset_gen.models.delight_model import DelightingModel
        from PIL import Image

        model = DelightingModel()
        rgba = Image.open("chair_seg.png")  # RGBA, background removed
        result = model(rgba)               # RGBA, lighting removed
        model.release()
        ```
    """

    MODEL_PATH = "tencent/Hunyuan3D-2"
    SUBFOLDER = "hunyuan3d-delight-v2-0"

    def __init__(
        self,
        model_path: str = None,
        num_infer_step: int = 50,
        mask_erosion_size: int = 3,
        image_guide_scale: float = 1.5,
        text_guide_scale: float = 1.0,
        seed: int = 0,
    ) -> None:
        self.num_infer_step = num_infer_step
        self.mask_erosion_size = mask_erosion_size
        self.image_guide_scale = image_guide_scale
        self.text_guide_scale = text_guide_scale
        self.seed = seed
        self.kernel = np.ones(
            (self.mask_erosion_size, self.mask_erosion_size), np.uint8
        )
        self._pipe = None

        if model_path is None:
            from huggingface_hub import snapshot_download
            local = snapshot_download(
                repo_id=self.MODEL_PATH,
                allow_patterns=f"{self.SUBFOLDER}/*",
            )
            model_path = os.path.join(local, self.SUBFOLDER)

        self._model_path = model_path

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def _lazy_init(self) -> None:
        if self._pipe is not None:
            return
        from diffusers import (
            EulerAncestralDiscreteScheduler,
            StableDiffusionInstructPix2PixPipeline,
        )
        log_vram("before delight model load")
        logger.info("Loading DelightingModel (Hunyuan3D-Delight-v2-0)...")
        pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(
            self._model_path,
            torch_dtype=torch.float16,
            safety_checker=None,
        )
        pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(
            pipe.scheduler.config
        )
        pipe.set_progress_bar_config(disable=True)
        # CPU offload keeps the model weights on CPU and moves layers to GPU
        # one at a time — fits in 8 GB without explicit .to("cuda").
        pipe.enable_model_cpu_offload()
        self._pipe = pipe
        log_vram("after delight model load")

    def release(self) -> None:
        """Unload model and free VRAM."""
        if self._pipe is not None:
            del self._pipe
            self._pipe = None
            free_vram()
            log_vram("after delight model release")

    # ------------------------------------------------------------------
    # Core
    # ------------------------------------------------------------------

    def recenter_image(
        self, image: Image.Image, border_ratio: float = 0.2
    ) -> Image.Image:
        """Crop to object bounding box and add proportional padding."""
        if image.mode in ("RGB", "L"):
            return image.convert("RGBA")

        alpha = np.array(image)[:, :, 3]
        nz = np.argwhere(alpha > 0)
        if nz.size == 0:
            raise ValueError("Image is fully transparent — nothing to de-light.")

        r0, c0 = nz.min(axis=0)
        r1, c1 = nz.max(axis=0)
        cropped = image.crop((c0, r0, c1 + 1, r1 + 1))

        w, h = cropped.size
        bw = int(w * border_ratio)
        bh = int(h * border_ratio)
        new_w = w + 2 * bw
        new_h = h + 2 * bh
        sq = max(new_w, new_h)

        canvas = Image.new("RGBA", (sq, sq), (255, 255, 255, 0))
        px = (sq - new_w) // 2 + bw
        py = (sq - new_h) // 2 + bh
        canvas.paste(cropped, (px, py))
        return canvas

    @torch.no_grad()
    def __call__(
        self,
        image: Union[str, np.ndarray, Image.Image],
        border_ratio: float = 0.2,
        target_wh: tuple = None,
    ) -> Image.Image:
        """Remove lighting from image.

        Args:
            image: RGBA PIL image (background already removed).
            border_ratio: Padding fraction added around the object before
                inference. Reduces edge artifacts.
            target_wh: Optional (width, height) to resize result to.
                Defaults to input size.

        Returns:
            RGBA PIL.Image with lighting removed. Alpha channel is restored
            from the (eroded) input alpha.
        """
        self._lazy_init()

        if isinstance(image, str):
            image = Image.open(image)
        elif isinstance(image, np.ndarray):
            image = Image.fromarray(image)

        image = self.recenter_image(image, border_ratio=border_ratio)

        if target_wh is None:
            target_wh = image.size

        image = image.resize(target_wh)
        arr = np.array(image)
        assert arr.shape[-1] == 4, "Image must have alpha channel after recenter."

        raw_alpha = arr[:, :, 3]
        alpha = cv2.erode(raw_alpha, self.kernel, iterations=1)

        # White background where alpha is 0, as required by InstructPix2Pix
        arr_rgb = arr.copy()
        arr_rgb[alpha == 0, :3] = 255
        arr_rgb[:, :, 3] = alpha

        result = self._pipe(
            prompt="",
            image=Image.fromarray(arr_rgb).convert("RGB"),
            generator=torch.manual_seed(self.seed),
            num_inference_steps=self.num_infer_step,
            image_guidance_scale=self.image_guide_scale,
            guidance_scale=self.text_guide_scale,
        ).images[0]

        # Restore alpha from eroded mask
        rgba = result.convert("RGBA").resize(target_wh)
        rgba.putalpha(Image.fromarray(alpha))
        return rgba
