import random

import torch
from PIL import Image
from embodied_gen.models.sam3d import Sam3dInference
from embodied_gen.utils.trender import pack_state, unpack_state
from embodied_gen.utils.vram_utils import free_vram

__all__ = [
    "image3d_model_infer",
]


def image3d_model_infer(
    pipe,
    seg_image: Image.Image,
    seed: int = None,
    **kwargs: dict,
) -> dict[str, any]:
    try:
        from thirdparty.TRELLIS.trellis.pipelines import (
            TrellisImageTo3DPipeline,
        )
        _trellis_available = True
    except ImportError:
        _trellis_available = False
        TrellisImageTo3DPipeline = type(None)

    try:
        from embodied_gen.models.hunyuan3d import Hunyuan3DInference
        _hunyuan_available = True
    except ImportError:
        _hunyuan_available = False
        Hunyuan3DInference = type(None)

    _seed = random.randint(0, 100000) if seed is None else seed

    if _trellis_available and isinstance(pipe, TrellisImageTo3DPipeline):
        from embodied_gen.data.utils import trellis_preprocess
        pipe.cuda()
        seg_image = trellis_preprocess(seg_image)
        with torch.inference_mode():
            outputs = pipe.run(
                seg_image,
                preprocess_image=False,
                seed=_seed,
                **kwargs,
            )
        pipe.cpu()

    elif _hunyuan_available and isinstance(pipe, Hunyuan3DInference):
        # Pipeline already uses @torch.inference_mode() internally.
        outputs = pipe.run(seg_image, seed=_seed, **kwargs)

    elif isinstance(pipe, Sam3dInference):
        # torch.inference_mode reduces peak activation memory ~25-35%.
        with torch.inference_mode():
            outputs = pipe.run(seg_image, seed=_seed, **kwargs)
        state = pack_state(outputs["gaussian"][0], outputs["mesh"][0])
        outputs["gaussian"][0], _ = unpack_state(state, device="cuda")

    else:
        raise ValueError(f"Unsupported pipeline type: {type(pipe)}")

    free_vram()

    return outputs
