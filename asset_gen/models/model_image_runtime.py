# Универсальный модуль-обёртка для запуска разных text-to-image моделей из Hugging Face (diffusers).
# Loader → загружает модель, Runner → запускает генерацию,
# Registry → хранит список моделей, build_hf_image_pipeline → фабрика.
#
# Профиль оптимизации: RTX 2070 Super (Turing, sm_75, 8 ГБ VRAM).
# Приоритет — качество, во вторую очередь — скорость.

import os

os.environ.setdefault(
    "PYTORCH_CUDA_ALLOC_CONF",
    "expandable_segments:True,garbage_collection_threshold:0.6,max_split_size_mb:512",
)

import torch
from abc import ABC, abstractmethod

torch.set_float32_matmul_precision("high")
torch.backends.cuda.matmul.allow_tf32 = True  # игнорируется на Turing, но не вредит
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True

from diffusers import (
    AutoencoderKL,
    AutoPipelineForText2Image,
    DPMSolverMultistepScheduler,
    FluxPipeline,
    KolorsPipeline,
    StableDiffusion3Pipeline,
    StableDiffusionPipeline,
    StableDiffusionXLPipeline,
)
from huggingface_hub import snapshot_download
from PIL import Image
from transformers import AutoModelForCausalLM, SiglipProcessor

__all__ = ["build_hf_image_pipeline"]


# ------------------------------------------------------------------ #
# Определение возможностей GPU
# ------------------------------------------------------------------ #
if torch.cuda.is_available():
    _CC = torch.cuda.get_device_capability()
    IS_AMPERE_OR_NEWER = _CC[0] >= 8          # flash-attn 2, нативный bf16
    IS_TURING = _CC == (7, 5)                 # RTX 2070/2080 Super
else:
    _CC = (0, 0)
    IS_AMPERE_OR_NEWER = False
    IS_TURING = False

# На Turing bf16 эмулируется → выбираем fp16 для скорости и качества.
PREFERRED_DTYPE = torch.bfloat16 if IS_AMPERE_OR_NEWER else torch.float16

try:
    import xformers  # noqa: F401
    HAS_XFORMERS = True
except Exception:
    HAS_XFORMERS = False


def _attn_kwargs() -> dict:
    """attn_implementation для transformers-компонент пайплайна (text encoders)."""
    if IS_AMPERE_OR_NEWER:
        return {"attn_implementation": "flash_attention_2"}
    return {}  # на Turing diffusers/transformers сами выберут SDPA


def _apply_mem_opts(pipe, *, vae_tile: bool = False) -> None:
    """Включает xformers/SDPA-attention и VAE-оптимизации."""
    import logging
    _log = logging.getLogger(__name__)
    if HAS_XFORMERS:
        try:
            pipe.enable_xformers_memory_efficient_attention()
            _log.info("Attention: xformers (fast)")
        except Exception:
            _log.warning("xformers import OK but enable failed — using SDPA")
    else:
        _log.info("Attention: PyTorch SDPA (xformers not installed — slower)")
    try:
        pipe.enable_vae_slicing()
    except Exception:
        pass
    if vae_tile:
        try:
            pipe.enable_vae_tiling()
        except Exception:
            pass


# ------------------------------------------------------------------ #
# Базовые классы
# ------------------------------------------------------------------ #
class BasePipelineLoader(ABC):
    def __init__(self, device="cuda"):
        self.device = device

    @abstractmethod
    def load(self):
        ...


class BasePipelineRunner(ABC):
    def __init__(self, pipe):
        self.pipe = pipe

    @abstractmethod
    def run(self, prompt: str, **kwargs) -> Image.Image:
        ...


# ===== SD 3.5-medium (тяжёлый, ~10 ГБ fp16 → нужен cpu_offload на 8 ГБ) =====
class SD35Loader(BasePipelineLoader):
    def load(self):
        pipe = StableDiffusion3Pipeline.from_pretrained(
            "stabilityai/stable-diffusion-3.5-medium",
            torch_dtype=torch.float16,
            **_attn_kwargs(),
        )
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(
            pipe.scheduler.config,
            use_karras_sigmas=True,
        )
        # Не зовём .to(device) — это конфликтует с cpu_offload и тратит время.
        pipe.enable_model_cpu_offload()
        _apply_mem_opts(pipe, vae_tile=True)
        return pipe


class SD35Runner(BasePipelineRunner):
    def run(self, prompt: str, **kwargs) -> Image.Image:
        return self.pipe(prompt=prompt, **kwargs).images


# ===== Cosmos2 (4-bit квантизация, помещается в 8 ГБ) =====
class CosmosLoader(BasePipelineLoader):
    def __init__(
        self,
        model_id="nvidia/Cosmos-Predict2-2B-Text2Image",
        local_dir="weights/cosmos2",
        device="cuda",
    ):
        super().__init__(device)
        self.model_id = model_id
        self.local_dir = local_dir

    def _patch(self):
        def patch_model(cls):
            orig = cls.from_pretrained

            def new(*args, **kwargs):
                if IS_AMPERE_OR_NEWER:
                    kwargs.setdefault("attn_implementation", "flash_attention_2")
                kwargs.setdefault("torch_dtype", PREFERRED_DTYPE)
                return orig(*args, **kwargs)

            cls.from_pretrained = new

        def patch_processor(cls):
            orig = cls.from_pretrained

            def new(*args, **kwargs):
                kwargs.setdefault("use_fast", True)
                return orig(*args, **kwargs)

            cls.from_pretrained = new

        patch_model(AutoModelForCausalLM)
        patch_processor(SiglipProcessor)

    def load(self):
        self._patch()
        snapshot_download(
            repo_id=self.model_id,
            local_dir=self.local_dir,
            local_dir_use_symlinks=False,
            resume_download=True,
        )

        from diffusers import Cosmos2TextToImagePipeline
        from diffusers.quantizers import PipelineQuantizationConfig

        config = PipelineQuantizationConfig(
            quant_backend="bitsandbytes_4bit",
            quant_kwargs={
                "load_in_4bit": True,
                "bnb_4bit_quant_type": "nf4",
                "bnb_4bit_compute_dtype": PREFERRED_DTYPE,
                "bnb_4bit_use_double_quant": True,
            },
            components_to_quantize=["text_encoder", "transformer", "unet"],
        )

        pipe = Cosmos2TextToImagePipeline.from_pretrained(
            self.model_id,
            torch_dtype=PREFERRED_DTYPE,
            quantization_config=config,
            use_safetensors=True,
            safety_checker=None,
            requires_safety_checker=False,
        ).to(self.device)
        _apply_mem_opts(pipe, vae_tile=True)
        return pipe


class CosmosRunner(BasePipelineRunner):
    def run(self, prompt: str, negative_prompt=None, **kwargs) -> Image.Image:
        return self.pipe(prompt=prompt, negative_prompt=negative_prompt, **kwargs).images


# ===== Kolors (~5B параметров, на 8 ГБ — только с offload) =====
class KolorsLoader(BasePipelineLoader):
    def load(self):
        pipe = KolorsPipeline.from_pretrained(
            "Kwai-Kolors/Kolors-diffusers",
            torch_dtype=torch.float16,
            variant="fp16",
            **_attn_kwargs(),
        )
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(
            pipe.scheduler.config,
            use_karras_sigmas=True,
        )
        pipe.enable_model_cpu_offload()
        _apply_mem_opts(pipe, vae_tile=True)
        return pipe


class KolorsRunner(BasePipelineRunner):
    def run(self, prompt: str, **kwargs) -> Image.Image:
        return self.pipe(prompt=prompt, **kwargs).images


# ===== Flux.1-schnell (12B → требует sequential offload на 8 ГБ) =====
class FluxLoader(BasePipelineLoader):
    def load(self):
        # На Turing bf16 эмулируется → fp16 заметно быстрее без потери качества.
        dtype = torch.bfloat16 if IS_AMPERE_OR_NEWER else torch.float16
        pipe = FluxPipeline.from_pretrained(
            "black-forest-labs/FLUX.1-schnell",
            torch_dtype=dtype,
            **_attn_kwargs(),
        )
        # ВАЖНО: схему планировщика для Flux не подменяем — он использует
        # FlowMatch, и DPM-Solver ломает качество. Оставляем дефолт.
        # 12B не помещается даже в bf16 (≈24 ГБ) — нужен sequential offload.
        pipe.enable_sequential_cpu_offload()
        _apply_mem_opts(pipe, vae_tile=True)
        return pipe


class FluxRunner(BasePipelineRunner):
    def run(self, prompt: str, **kwargs) -> Image.Image:
        return self.pipe(prompt=prompt, **kwargs).images


# ===== Chroma =====
class ChromaLoader(BasePipelineLoader):
    def load(self):
        from diffusers import ChromaPipeline

        dtype = torch.bfloat16 if IS_AMPERE_OR_NEWER else torch.float16
        pipe = ChromaPipeline.from_pretrained("lodestones/Chroma", torch_dtype=dtype)
        pipe.enable_sequential_cpu_offload()
        _apply_mem_opts(pipe, vae_tile=True)
        return pipe


class ChromaRunner(BasePipelineRunner):
    def run(self, prompt: str, negative_prompt=None, **kwargs) -> Image.Image:
        return self.pipe(prompt=prompt, negative_prompt=negative_prompt, **kwargs).images


# ===== SDXL-Turbo (8 ГБ, дистилляция: 4 шага, без CFG) =====
class SDXLTurboLoader(BasePipelineLoader):
    def load(self):
        # FP16-safe VAE — иначе на ряде промптов получаем чёрный/мутный результат.
        vae = AutoencoderKL.from_pretrained(
            "madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16
        )
        pipe = AutoPipelineForText2Image.from_pretrained(
            "stabilityai/sdxl-turbo",
            vae=vae,
            torch_dtype=torch.float16,
            variant="fp16",
            use_safetensors=True,
            **_attn_kwargs(),
        )
        # SDXL-Turbo fp16 ≈ 5 ГБ — помещается в 8 ГБ целиком.
        # .to(device) в 5-10× быстрее cpu_offload при 4-шаговой дистилляции.
        pipe = pipe.to(self.device)
        _apply_mem_opts(pipe, vae_tile=False)
        return pipe


_SDXL_NEGATIVE_PROMPT = (
    "blurry, deformed, distorted, ugly, bad anatomy, extra limbs, "
    "cropped, out of frame, worst quality, low quality, watermark, "
    "text, logo, floating, background clutter, multiple objects"
)


class SDXLTurboRunner(BasePipelineRunner):
    """Runner for SDXL-Turbo. 4 steps, guidance_scale=0 (distilled)."""

    def run(self, prompt: str, **kwargs) -> list[Image.Image]:
        kwargs["num_inference_steps"] = 4
        kwargs["guidance_scale"] = 0.0
        kwargs.setdefault("negative_prompt", _SDXL_NEGATIVE_PROMPT)
        kwargs["height"] = min(kwargs.get("height", 512), 512)
        kwargs["width"] = min(kwargs.get("width", 512), 512)
        return self.pipe(prompt=prompt, **kwargs).images


# ===== Stable Diffusion 1.5 (лёгкий ~2 ГБ, помещается на GPU целиком) =====
class SD15Loader(BasePipelineLoader):
    def load(self):
        pipe = StableDiffusionPipeline.from_pretrained(
            "sd-legacy/stable-diffusion-v1-5",
            torch_dtype=torch.float16,
            safety_checker=None,
            requires_safety_checker=False,
            use_safetensors=True,
        )
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(
            pipe.scheduler.config,
            use_karras_sigmas=True,
        )
        # Полностью на GPU — у Turing 8 ГБ хватает с запасом.
        pipe = pipe.to(self.device)
        _apply_mem_opts(pipe, vae_tile=False)
        return pipe


_SD15_NEGATIVE_PROMPT = (
    "blurry, deformed, distorted, ugly, bad anatomy, extra limbs, "
    "cropped, out of frame, worst quality, low quality, watermark, "
    "text, logo, floating, background clutter, multiple objects, "
    "duplicate, disfigured, extra arms, mutated, artifacts, noise"
)


class SD15Runner(BasePipelineRunner):
    """Runner for SD 1.5. Native resolution 512x512."""

    def run(self, prompt: str, **kwargs) -> list[Image.Image]:
        # 25 шагов с DPM-Solver Karras уже даёт максимум на SD1.5;
        # выше — диминишн ретёрн без видимого прироста качества.
        kwargs.setdefault("num_inference_steps", 25)
        kwargs.setdefault("guidance_scale", 7.5)
        kwargs.setdefault("negative_prompt", _SD15_NEGATIVE_PROMPT)
        kwargs["height"] = 512
        kwargs["width"] = 512
        return self.pipe(prompt=prompt, **kwargs).images


# ===== SDXL Base 1.0 (1024×1024, 8 ГБ — только с offload + fp16-VAE-fix) =====
_SDXL_BASE_NEGATIVE_PROMPT = (
    "blurry, deformed, distorted, ugly, bad anatomy, extra limbs, "
    "cropped, out of frame, worst quality, low quality, watermark, "
    "text, logo, floating, background clutter, multiple objects, "
    "duplicate, disfigured, extra arms, mutated, artifacts, noise, "
    "lowres, bad composition, unrealistic lighting, blurry background"
)


class SDXLBaseLoader(BasePipelineLoader):
    def load(self):
        # Без этого VAE при fp16 даёт чёрные изображения / NaN — критично для качества.
        vae = AutoencoderKL.from_pretrained(
            "madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16
        )

        pipe = StableDiffusionXLPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0",
            vae=vae,
            torch_dtype=torch.float16,
            variant="fp16",
            use_safetensors=True,
            **_attn_kwargs(),
        )

        pipe.scheduler = DPMSolverMultistepScheduler.from_config(
            pipe.scheduler.config,
            algorithm_type="dpmsolver++",
            solver_order=2,
            use_karras_sigmas=True,
            final_sigmas_type="sigma_min",
        )

        # SDXL Base fp16 ≈ 5.5 ГБ — помещается в 8 ГБ целиком.
        # .to(device) в 5-10× быстрее cpu_offload при 25-30 шагах.
        pipe = pipe.to(self.device)
        _apply_mem_opts(pipe, vae_tile=False)
        return pipe


class SDXLBaseRunner(BasePipelineRunner):
    """Runner for SDXL Base 1.0. Native resolution 1024x1024."""

    def run(self, prompt: str, **kwargs) -> list[Image.Image]:
        kwargs.setdefault("num_inference_steps", 30)
        kwargs.setdefault("guidance_scale", 7.5)
        kwargs.setdefault("negative_prompt", _SDXL_BASE_NEGATIVE_PROMPT)
        kwargs["height"] = min(kwargs.get("height", 1024), 1024)
        kwargs["width"] = min(kwargs.get("width", 1024), 1024)
        return self.pipe(prompt=prompt, **kwargs).images


PIPELINE_REGISTRY = {
    "sd35": (SD35Loader, SD35Runner),
    "cosmos": (CosmosLoader, CosmosRunner),
    "kolors": (KolorsLoader, KolorsRunner),
    "flux": (FluxLoader, FluxRunner),
    "chroma": (ChromaLoader, ChromaRunner),
    "sdxl-turbo": (SDXLTurboLoader, SDXLTurboRunner),
    "sdxl": (SDXLBaseLoader, SDXLBaseRunner),
    "sd15": (SD15Loader, SD15Runner),
}


def build_hf_image_pipeline(name: str, device="cuda") -> BasePipelineRunner:
    """Build a Hugging Face image generation pipeline runner by name.

    Example:
        ```py
        runner = build_hf_image_pipeline("sdxl")
        images = runner.run(prompt="A robot holding a sign that says 'Hello'")
        ```
    """
    if name not in PIPELINE_REGISTRY:
        raise ValueError(f"Unsupported model: {name}")
    loader_cls, runner_cls = PIPELINE_REGISTRY[name]
    pipe = loader_cls(device=device).load()
    return runner_cls(pipe)


if __name__ == "__main__":
    model_name = "sdxl"
    runner = build_hf_image_pipeline(model_name)
    images = runner.run(
        prompt="A robot holding a sign that says 'Hello'",
        height=1024,
        width=1024,
        num_inference_steps=30,
        guidance_scale=6,
        num_images_per_prompt=4,
    )

    for i, img in enumerate(images):
        img.save(f"image_{model_name}_{i}.jpg")
