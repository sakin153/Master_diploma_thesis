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
    """Unified inference wrapper for image-to-3D backends.

    Supported backends:
      - TrellisClient: calls remote REST API, returns GLB bytes + trimesh.

    Returns dict with keys:
      ``trimesh``   — list with one trimesh.Trimesh (Y-up→Z-up already applied).
      ``mesh``      — list with None (no local mesh adapter).
      ``glb_bytes`` — raw GLB bytes from the API (with baked texture).
    """
    from asset_gen.models.trellis_client import TrellisClient

    if isinstance(pipe, TrellisClient):
        mesh, glb_bytes = pipe.generate(seg_image, seed=seed, **kwargs)
        free_vram()
        return {"mesh": [None], "trimesh": [mesh], "glb_bytes": glb_bytes}

    raise ValueError(
        f"Unsupported pipeline type: {type(pipe).__name__}. "
        "Expected TrellisClient."
    )
