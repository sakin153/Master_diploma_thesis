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
"""Text-to-image pipelines curated for 8–12 GB consumer GPUs."""

from abc import ABC, abstractmethod

import torch
from diffusers import (
    AutoPipelineForText2Image,
    DPMSolverMultistepScheduler,
    KolorsPipeline,
    StableDiffusionPipeline,
)
from PIL import Image

__all__ = ["build_hf_image_pipeline"]


class BasePipelineLoader(ABC):
    def __init__(self, device: str = "cuda"):
        self.device = device

    @abstractmethod
    def load(self):
        ...


class BasePipelineRunner(ABC):
    def __init__(self, pipe):
        self.pipe = pipe

    @abstractmethod
    def run(self, prompt: str, **kwargs) -> list[Image.Image]:
        ...


# ── Kolors (EmbodiedGen default — best quality for robotics assets) ─────────
class KolorsLoader(BasePipelineLoader):
    def load(self):
        pipe = KolorsPipeline.from_pretrained(
            "Kwai-Kolors/Kolors-diffusers",
            torch_dtype=torch.float16,
            variant="fp16",
        ).to(self.device)
        pipe.enable_model_cpu_offload()
        pipe.enable_xformers_memory_efficient_attention()
        pipe.enable_vae_slicing()
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(
            pipe.scheduler.config, use_karras_sigmas=True
        )
        return pipe


class KolorsRunner(BasePipelineRunner):
    def run(self, prompt: str, **kwargs) -> list[Image.Image]:
        return self.pipe(prompt=prompt, **kwargs).images


# ── SDXL-Turbo (fastest, 512×512, 4 steps, for 8 GB GPU) ────────────────────
class SDXLTurboLoader(BasePipelineLoader):
    def load(self):
        pipe = AutoPipelineForText2Image.from_pretrained(
            "stabilityai/sdxl-turbo",
            torch_dtype=torch.float16,
            variant="fp16",
        )
        pipe.enable_model_cpu_offload()
        pipe.enable_xformers_memory_efficient_attention()
        pipe.enable_attention_slicing()
        pipe.enable_vae_slicing()
        return pipe


_SDXL_NEGATIVE_PROMPT = (
    "blurry, deformed, distorted, ugly, bad anatomy, extra limbs, "
    "cropped, out of frame, worst quality, low quality, watermark, "
    "text, logo, floating, background clutter, multiple objects"
)


class SDXLTurboRunner(BasePipelineRunner):
    def run(self, prompt: str, **kwargs) -> list[Image.Image]:
        kwargs["num_inference_steps"] = 4
        kwargs["guidance_scale"] = 0.0
        kwargs.setdefault("negative_prompt", _SDXL_NEGATIVE_PROMPT)
        kwargs["height"] = min(kwargs.get("height", 512), 512)
        kwargs["width"] = min(kwargs.get("width", 512), 512)
        return self.pipe(prompt=prompt, **kwargs).images


# ── SDXL Base 1.0 (highest quality, 1024×1024, ~6 GB VRAM + CPU offload) ────
class SDXLBaseLoader(BasePipelineLoader):
    def load(self):
        from diffusers import StableDiffusionXLPipeline
        pipe = StableDiffusionXLPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0",
            torch_dtype=torch.float16,
            variant="fp16",
            use_safetensors=True,
            low_cpu_mem_usage=True,
        )
        pipe.enable_model_cpu_offload()
        pipe.enable_xformers_memory_efficient_attention()
        pipe.enable_attention_slicing()
        pipe.enable_vae_slicing()
        return pipe


class SDXLBaseRunner(BasePipelineRunner):
    def run(self, prompt: str, **kwargs) -> list[Image.Image]:
        kwargs.setdefault("num_inference_steps", 30)
        kwargs.setdefault("guidance_scale", 7.5)
        kwargs.setdefault("negative_prompt", _SDXL_NEGATIVE_PROMPT)
        kwargs["height"] = min(kwargs.get("height", 1024), 1024)
        kwargs["width"] = min(kwargs.get("width", 1024), 1024)
        return self.pipe(prompt=prompt, **kwargs).images


# ── SD 1.5 (lightweight fallback, 512×512, ~1.7 GB via CPU offload) ─────────
class SD15Loader(BasePipelineLoader):
    def load(self):
        pipe = StableDiffusionPipeline.from_pretrained(
            "sd-legacy/stable-diffusion-v1-5",
            torch_dtype=torch.float16,
            safety_checker=None,
        )
        pipe.enable_model_cpu_offload()
        pipe.enable_attention_slicing()
        pipe.enable_vae_slicing()
        return pipe


_SD15_NEGATIVE_PROMPT = _SDXL_NEGATIVE_PROMPT + (
    ", duplicate, disfigured, extra arms, mutated, artifacts, noise"
)


class SD15Runner(BasePipelineRunner):
    def run(self, prompt: str, **kwargs) -> list[Image.Image]:
        kwargs.setdefault("num_inference_steps", 30)
        kwargs.setdefault("guidance_scale", 8.5)
        kwargs.setdefault("negative_prompt", _SD15_NEGATIVE_PROMPT)
        kwargs["height"] = 512
        kwargs["width"] = 512
        return self.pipe(prompt=prompt, **kwargs).images


PIPELINE_REGISTRY = {
    "kolors": (KolorsLoader, KolorsRunner),
    "sdxl-turbo": (SDXLTurboLoader, SDXLTurboRunner),
    "sdxl": (SDXLBaseLoader, SDXLBaseRunner),
    "sd15": (SD15Loader, SD15Runner),
}


def build_hf_image_pipeline(name: str, device: str = "cuda") -> BasePipelineRunner:
    """Build a text-to-image pipeline runner by registry key.

    Available keys: ``kolors``, ``sdxl-turbo``, ``sdxl``, ``sd15``.
    """
    if name not in PIPELINE_REGISTRY:
        raise ValueError(
            f"Unsupported model: {name}. "
            f"Available: {sorted(PIPELINE_REGISTRY)}"
        )
    loader_cls, runner_cls = PIPELINE_REGISTRY[name]
    return runner_cls(loader_cls(device=device).load())
