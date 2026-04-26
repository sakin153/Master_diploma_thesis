    """
    Универсальный модуль-обёртка для запуска разных text-to-image моделей из Hugging Face (diffusers).
    Данный модуль даёт единый интерфейс для загрузки и запуска разных моделей генерации изображений.
    Файл реализует паттерн:
    = Loader → загружает модель
    = Runner → запускает генерацию
    = Registry → хранит список доступных моделей
    = Factory (build_hf_image_pipeline) → создаёт нужную модель по имени
    """
    import os
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True,garbage_collection_threshold:0.6,max_split_size_mb:512'

    import torch
    from abc import ABC, abstractmethod

    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True # полезно для матричных операций в моделях, ускоряет на Ampere+ без заметной потери качества
    torch.backends.cudnn.allow_tf32 = True  # тоже полезно
    torch.backends.cudnn.benchmark = True 

    from diffusers import (
        AutoPipelineForText2Image,
        DPMSolverMultistepScheduler,
        FluxPipeline,
        KolorsPipeline,
        StableDiffusion3Pipeline,
        StableDiffusionPipeline,
    )
    from huggingface_hub import snapshot_download
    from PIL import Image
    from transformers import AutoModelForCausalLM, SiglipProcessor

    __all__ = [
        "build_hf_image_pipeline",
    ]


    class BasePipelineLoader(ABC):
        """Abstract base class for loading Hugging Face image generation pipelines.

        Attributes:
            device (str): Device to load the pipeline on.

        Methods:
            load(): Loads and returns the pipeline.
        """

        def __init__(self, device="cuda"):
            self.device = device

        @abstractmethod
        def load(self):
            """Load and return the pipeline instance."""
            pass


    class BasePipelineRunner(ABC):
        """Abstract base class for running image generation pipelines.

        Attributes:
            pipe: The loaded pipeline.

        Methods:
            run(prompt, **kwargs): Runs the pipeline with a prompt.
        """

        def __init__(self, pipe):
            self.pipe = pipe

        @abstractmethod
        def run(self, prompt: str, **kwargs) -> Image.Image:
            """Run the pipeline with the given prompt.

            Args:
                prompt (str): Text prompt for image generation.
                **kwargs: Additional pipeline arguments.

            Returns:
                Image.Image: Generated image(s).
            """
            pass


    # ===== SD3.5-medium =====
    class SD35Loader(BasePipelineLoader):
        def load(self):
            kwargs = {
                "torch_dtype": torch.float16
            }
            if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8:
                kwargs["attn_implementation"] = "flash_attention_2"

            pipe = StableDiffusion3Pipeline.from_pretrained(
                "stabilityai/stable-diffusion-3.5-medium",
                **kwargs
            )

            pipe.scheduler = DPMSolverMultistepScheduler.from_config(
                pipe.scheduler.config,
                use_karras_sigmas=True
            )

            pipe = pipe.to(self.device)
            pipe.enable_model_cpu_offload()
            # pipe.enable_xformers_memory_efficient_attention()
            pipe.enable_attention_slicing()
            pipe.enable_vae_slicing()
            return pipe


    class SD35Runner(BasePipelineRunner):
        """Runner for Stable Diffusion 3.5 medium pipeline."""

        def run(self, prompt: str, **kwargs) -> Image.Image:
            """Generate images using Stable Diffusion 3.5 medium.

            Args:
                prompt (str): Text prompt.
                **kwargs: Additional arguments.

            Returns:
                Image.Image: Generated image(s).
            """
            return self.pipe(prompt=prompt, **kwargs).images


    # ===== Cosmos2 =====
    class CosmosLoader(BasePipelineLoader):
        """Loader for Cosmos2 text-to-image pipeline."""

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
            """Patch model and processor for optimized loading."""

            def patch_model(cls):
                orig = cls.from_pretrained

                def new(*args, **kwargs):
                    kwargs.setdefault("attn_implementation", "flash_attention_2")
                    kwargs.setdefault("torch_dtype", torch.bfloat16)
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
            """Load the Cosmos2 text-to-image pipeline.

            Returns:
                Cosmos2TextToImagePipeline: Loaded pipeline.
            """
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
                    "bnb_4bit_compute_dtype": torch.bfloat16,
                    "bnb_4bit_use_double_quant": True,
                },
                components_to_quantize=["text_encoder", "transformer", "unet"],
            )

            pipe = Cosmos2TextToImagePipeline.from_pretrained(
                self.model_id,
                torch_dtype=torch.bfloat16,
                quantization_config=config,
                use_safetensors=True,
                safety_checker=None,
                requires_safety_checker=False,
            ).to(self.device)
            return pipe


    class CosmosRunner(BasePipelineRunner):
        """Runner for Cosmos2 text-to-image pipeline."""

        def run(self, prompt: str, negative_prompt=None, **kwargs) -> Image.Image:
            """Generate images using Cosmos2 pipeline.

            Args:
                prompt (str): Text prompt.
                negative_prompt (str, optional): Negative prompt.
                **kwargs: Additional arguments.

            Returns:
                Image.Image: Generated image(s).
            """
            return self.pipe(
                prompt=prompt, negative_prompt=negative_prompt, **kwargs
            ).images


    # ===== Kolors =====
    class KolorsLoader(BasePipelineLoader):
        def load(self):
            kwargs = {
                "torch_dtype": torch.float16,
                "variant": "fp16"
            }

            if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8:
                kwargs["attn_implementation"] = "flash_attention_2"

            pipe = KolorsPipeline.from_pretrained(
                "Kwai-Kolors/Kolors-diffusers",
                **kwargs
            ).to(self.device)

            # pipe.enable_xformers_memory_efficient_attention()
            pipe.enable_vae_slicing()

            pipe.scheduler = DPMSolverMultistepScheduler.from_config(
                pipe.scheduler.config,
                use_karras_sigmas=True
            )
            return pipe


    class KolorsRunner(BasePipelineRunner):
        """Runner for Kolors pipeline."""

        def run(self, prompt: str, **kwargs) -> Image.Image:
            """Generate images using Kolors pipeline.

            Args:
                prompt (str): Text prompt.
                **kwargs: Additional arguments.

            Returns:
                Image.Image: Generated image(s).
            """
            return self.pipe(prompt=prompt, **kwargs).images


    # ===== Flux =====
    class FluxLoader(BasePipelineLoader):
        def load(self):
            os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

            kwargs = {
                "torch_dtype": torch.bfloat16
            }

            if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8:
                kwargs["attn_implementation"] = "flash_attention_2"

            pipe = FluxPipeline.from_pretrained(
                "black-forest-labs/FLUX.1-schnell",
                **kwargs
            )

            pipe.scheduler = DPMSolverMultistepScheduler.from_config(
                pipe.scheduler.config,
                use_karras_sigmas=True
            )

            # pipe.enable_xformers_memory_efficient_attention()
            pipe.enable_attention_slicing()
            pipe.enable_vae_slicing()

            return pipe.to(self.device)


    class FluxRunner(BasePipelineRunner):
        """Runner for Flux pipeline."""

        def run(self, prompt: str, **kwargs) -> Image.Image:
            """Generate images using Flux pipeline.

            Args:
                prompt (str): Text prompt.
                **kwargs: Additional arguments.

            Returns:
                Image.Image: Generated image(s).
            """
            return self.pipe(prompt=prompt, **kwargs).images


    # ===== Chroma =====
    class ChromaLoader(BasePipelineLoader):
        """Loader for Chroma pipeline."""

        def load(self):
            """Load the Chroma pipeline.

            Returns:
                ChromaPipeline: Loaded pipeline.
            """
            from diffusers import ChromaPipeline
            return ChromaPipeline.from_pretrained(
                "lodestones/Chroma", torch_dtype=torch.bfloat16
            ).to(self.device)


    class ChromaRunner(BasePipelineRunner):
        """Runner for Chroma pipeline."""

        def run(self, prompt: str, negative_prompt=None, **kwargs) -> Image.Image:
            """Generate images using Chroma pipeline.

            Args:
                prompt (str): Text prompt.
                negative_prompt (str, optional): Negative prompt.
                **kwargs: Additional arguments.

            Returns:
                Image.Image: Generated image(s).
            """
            return self.pipe(
                prompt=prompt, negative_prompt=negative_prompt, **kwargs
            ).images


    # ===== SDXL-Turbo (8GB GPU, e.g. RTX 2070 Super) =====
    class SDXLTurboLoader(BasePipelineLoader):
        def load(self):
            kwargs = {
                "torch_dtype": torch.float16,
                "variant": "fp16"
            }

            if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8:
                kwargs["attn_implementation"] = "flash_attention_2"

            pipe = AutoPipelineForText2Image.from_pretrained(
                "stabilityai/sdxl-turbo",
                **kwargs
            )

            # pipe.enable_xformers_memory_efficient_attention()
            pipe.enable_attention_slicing()
            pipe.enable_vae_slicing()

            return pipe

    _SDXL_NEGATIVE_PROMPT = (
        "blurry, deformed, distorted, ugly, bad anatomy, extra limbs, "
        "cropped, out of frame, worst quality, low quality, watermark, "
        "text, logo, floating, background clutter, multiple objects"
    )


    class SDXLTurboRunner(BasePipelineRunner):
        """Runner for SDXL-Turbo. 4 steps, guidance_scale=0 (distilled)."""

        def run(self, prompt: str, **kwargs) -> list[Image.Image]:
            # SDXL-Turbo is distilled: fixed 4 steps, no CFG guidance
            kwargs["num_inference_steps"] = 4
            kwargs["guidance_scale"] = 0.0
            kwargs.setdefault("negative_prompt", _SDXL_NEGATIVE_PROMPT)
            # Native resolution is 512x512; clamp to avoid OOM
            kwargs["height"] = min(kwargs.get("height", 512), 512)
            kwargs["width"] = min(kwargs.get("width", 512), 512)
            return self.pipe(prompt=prompt, **kwargs).images


    # ===== Stable Diffusion 1.5 (lightweight, ~1.7 GB RAM via cpu_offload) =====
    class SD15Loader(BasePipelineLoader):
        """SD 1.5 with cpu_offload — minimal VRAM footprint."""

        def load(self):
            pipe = StableDiffusionPipeline.from_pretrained(
                "sd-legacy/stable-diffusion-v1-5",
                torch_dtype=torch.float16,
                safety_checker=None
            )
            pipe.scheduler = DPMSolverMultistepScheduler.from_config(
                pipe.scheduler.config,
                use_karras_sigmas=True
            )

            # pipe.enable_model_cpu_offload()
            pipe.enable_attention_slicing()
            pipe.enable_vae_slicing()
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
            kwargs.setdefault("num_inference_steps", 30)
            kwargs.setdefault("guidance_scale", 8.5)
            kwargs.setdefault("negative_prompt", _SD15_NEGATIVE_PROMPT)
            # SD 1.5 native res is 512x512 — force it to avoid artifacts
            kwargs["height"] = 512
            kwargs["width"] = 512
            return self.pipe(prompt=prompt, **kwargs).images


    _SDXL_BASE_NEGATIVE_PROMPT = (
        "blurry, deformed, distorted, ugly, bad anatomy, extra limbs, "
        "cropped, out of frame, worst quality, low quality, watermark, "
        "text, logo, floating, background clutter, multiple objects, "
        "duplicate, disfigured, extra arms, mutated, artifacts, noise, "
        "lowres, bad composition, unrealistic lighting, blurry background"
    )


    class SDXLBaseLoader(BasePipelineLoader):
        def load(self):
            kwargs = {
                "torch_dtype": torch.float16,
                "variant": "fp16",
                "use_safetensors": True
            }

            if torch.cuda.is_available():
                cap = torch.cuda.get_device_capability()
                kwargs["attn_implementation"] = "flash_attention_2" if cap[0] >= 8 else "sdpa"

                pipe = StableDiffusionXLPipeline.from_pretrained(
                    "stabilityai/stable-diffusion-xl-base-1.0",
                    **kwargs
                )

                pipe = pipe.to(self.device)
                
                pipe.scheduler = DPMSolverMultistepScheduler.from_config(
                    pipe.scheduler.config,
                    algorithm_type="dpmsolver++",
                    solver_order=2,
                    use_karras_sigmas=True,
                    final_sigmas_type="sigma_min"  # 🔥 Улучшает сходимость на 30 шагах
                )

                # ⚖️ Для 8 ГБ: cpu_offload + tiling = стабильность без OOM
                pipe.enable_model_cpu_offload()
                pipe.enable_vae_tiling()  # 🔸 Режет декодирование 1024x1024 чанками
                # ❌ attention_slicing и vae_slicing УБРАНЫ: они уже внутри cpu_offload

                return pipe

    class SDXLBaseRunner(BasePipelineRunner):
        """Runner for SDXL Base 1.0. Native resolution 1024x1024."""

        def run(self, prompt: str, **kwargs) -> list[Image.Image]:
            kwargs.setdefault("num_inference_steps", 30)
            kwargs.setdefault("guidance_scale", 7.5)
            kwargs.setdefault("negative_prompt", _SDXL_BASE_NEGATIVE_PROMPT)
            # SDXL native resolution is 1024x1024
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

        Args:
            name (str): Name of the pipeline (e.g., "sd35", "cosmos").
            device (str): Device to load the pipeline on.

        Returns:
            BasePipelineRunner: Pipeline runner instance.

        Example:
            ```py
            from asset_gen.models.image_comm_model import (
                build_hf_image_pipeline,
            )
            runner = build_hf_image_pipeline("sd35")
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
        # NOTE: generation quality at low resolution is poor.
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
