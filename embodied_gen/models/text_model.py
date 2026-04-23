# Project EmbodiedGen
#
# Copyright (c) 2025 Horizon Robotics. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
# implied. See the License for the specific language governing
# permissions and limitations under the License.
"""Prompt templates + optional Kolors IP-Adapter pipeline.

The heavy IP-Adapter path is imported lazily — plain text-to-image flows
only need ``PROMPT_APPEND`` and do not require the ``kolors`` package.
"""

from __future__ import annotations

import logging
import os
import subprocess

logger = logging.getLogger(__name__)

__all__ = [
    "PROMPT_APPEND",
    "PROMPT_KAPPEND",
    "build_text2img_ip_pipeline",
    "download_kolors_weights",
]

PROMPT_APPEND = (
    "Angled 3D view of one {object}, centered, no cropping, no occlusion, "
    "isolated product photo, placed horizontally, no surroundings, "
    "high-quality appearance, vivid colors, on a plain clean surface, "
    "3D style revealing multiple surfaces"
)

PROMPT_KAPPEND = (
    "Single {object}, in the center of the image, white background, "
    "3D style, best quality"
)


def download_kolors_weights(local_dir: str = "weights/Kolors") -> None:
    """Fetch Kolors + IP-Adapter checkpoints via huggingface-cli."""
    os.makedirs(local_dir, exist_ok=True)
    logger.info("Downloading Kolors weights to %s", local_dir)
    subprocess.run(
        ["huggingface-cli", "download", "--resume-download",
         "Kwai-Kolors/Kolors", "--local-dir", local_dir],
        check=True,
    )
    ip_adapter_path = f"{local_dir}/../Kolors-IP-Adapter-Plus"
    subprocess.run(
        ["huggingface-cli", "download", "--resume-download",
         "Kwai-Kolors/Kolors-IP-Adapter-Plus",
         "--local-dir", ip_adapter_path],
        check=True,
    )


def build_text2img_ip_pipeline(
    ckpt_dir: str,
    ref_scale: float,
    device: str = "cuda",
):
    """Kolors + IP-Adapter pipeline for image-conditioned text-to-image.

    Imported lazily so a missing ``kolors`` package does not break the
    default text-only flow.
    """
    import torch
    from diffusers import AutoencoderKL, EulerDiscreteScheduler
    from kolors.models.modeling_chatglm import ChatGLMModel
    from kolors.models.tokenization_chatglm import ChatGLMTokenizer
    from kolors.models.unet_2d_condition import (
        UNet2DConditionModel as UNet2DConditionModelIP,
    )
    from kolors.pipelines.pipeline_stable_diffusion_xl_chatglm_256_ipadapter import (  # noqa: E501
        StableDiffusionXLPipeline as StableDiffusionXLPipelineIP,
    )
    from transformers import (
        CLIPImageProcessor,
        CLIPVisionModelWithProjection,
    )

    download_kolors_weights(ckpt_dir)

    text_encoder = ChatGLMModel.from_pretrained(
        f"{ckpt_dir}/text_encoder", torch_dtype=torch.float16
    ).half()
    tokenizer = ChatGLMTokenizer.from_pretrained(f"{ckpt_dir}/text_encoder")
    vae = AutoencoderKL.from_pretrained(f"{ckpt_dir}/vae").half()
    scheduler = EulerDiscreteScheduler.from_pretrained(f"{ckpt_dir}/scheduler")
    unet = UNet2DConditionModelIP.from_pretrained(f"{ckpt_dir}/unet").half()
    image_encoder = CLIPVisionModelWithProjection.from_pretrained(
        f"{ckpt_dir}/../Kolors-IP-Adapter-Plus/image_encoder",
        ignore_mismatched_sizes=True,
    ).to(dtype=torch.float16)
    clip_image_processor = CLIPImageProcessor(size=336, crop_size=336)

    pipe = StableDiffusionXLPipelineIP(
        vae=vae,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        unet=unet,
        scheduler=scheduler,
        image_encoder=image_encoder,
        feature_extractor=clip_image_processor,
        force_zeros_for_empty_prompt=False,
    )
    if hasattr(pipe.unet, "encoder_hid_proj"):
        pipe.unet.text_encoder_hid_proj = pipe.unet.encoder_hid_proj

    pipe.load_ip_adapter(
        f"{ckpt_dir}/../Kolors-IP-Adapter-Plus",
        subfolder="",
        weight_name=["ip_adapter_plus_general.bin"],
    )
    pipe.set_ip_adapter_scale([ref_scale])

    pipe = pipe.to(device)
    pipe.image_encoder = pipe.image_encoder.to(device)
    pipe.enable_model_cpu_offload()
    return pipe
