import torch
from PIL import Image
from asset_gen.utils.vram_utils import free_vram

__all__ = ["image3d_model_infer"]


def image3d_model_infer(
    pipe,
    seg_image: Image.Image,
    seed: int = None,
    **kwargs,
) -> dict:
    from asset_gen.models.hunyuan3d import Hunyuan3DInference

    if not isinstance(pipe, Hunyuan3DInference):
        raise ValueError(f"Unsupported pipeline type: {type(pipe)}")

    with torch.inference_mode():
        outputs = pipe.run(seg_image, seed=seed, **kwargs)

    free_vram()
    return outputs
