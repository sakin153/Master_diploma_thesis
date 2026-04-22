"""Wrapper around the local Hunyuan3D-2 repository.

Expects the repo at <project_root>/Hunyuan3D-2  (already present).
Only shape generation (no texture) to stay within 8 GB VRAM.
"""

import os
import sys

import torch

# Add local Hunyuan3D-2 to path so hy3dgen can be imported without install
_PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../..")
)
_HY3D_LOCAL = os.path.join(_PROJECT_ROOT, "Hunyuan3D-2")
if os.path.isdir(_HY3D_LOCAL) and _HY3D_LOCAL not in sys.path:
    sys.path.insert(0, _HY3D_LOCAL)

__all__ = ["Hunyuan3DInference"]


class _MeshAdapter:
    """Wraps trimesh.Trimesh so render_video() can read .vertices/.faces."""

    def __init__(self, mesh):
        self.vertices = torch.from_numpy(
            mesh.vertices.astype("float32")
        ).cuda()
        self.faces = torch.from_numpy(
            mesh.faces.astype("int64")
        ).cuda()
        self._mesh = mesh


class Hunyuan3DInference:
    """Image → mesh via Hunyuan3D-2mini (shape only, no texture).

    Uses the local Hunyuan3D-2 repo at <project_root>/Hunyuan3D-2.
    Model subfolder: hunyuan3d-dit-v2-mini (~0.6B, fits in 8 GB VRAM).
    """

    def __init__(
        self,
        model_path: str = "tencent/Hunyuan3D-2mini",
        subfolder: str = "hunyuan3d-dit-v2-mini",
    ) -> None:
        from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline

        self.model_path = model_path
        self._shape = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
            model_path,
            subfolder=subfolder,
            use_safetensors=True,
            variant="fp16",
            device="cuda",
        )

    def run(
        self,
        image,
        seed: int = None,
        num_inference_steps: int = 30,
        octree_resolution: int = 256,
        guidance_scale: float = 5.0,
        **kwargs,
    ) -> dict:
        gen = (
            torch.Generator(device="cuda").manual_seed(seed)
            if seed is not None
            else None
        )

        meshes = self._shape(
            image=image,
            num_inference_steps=num_inference_steps,
            octree_resolution=octree_resolution,
            guidance_scale=guidance_scale,
            generator=gen,
            output_type="trimesh",
        )
        mesh = meshes[0]

        return {
            "gaussian": [None],
            "mesh": [_MeshAdapter(mesh)],
            "trimesh": [mesh],
        }
