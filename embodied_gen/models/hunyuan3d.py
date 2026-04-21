"""Wrapper around Hunyuan3D-2 (tencent/Hunyuan3D-2mini) for EmbodiedGen.

Returns the same dict structure as Sam3dInference.run() so the rest of
the pipeline needs only conditional checks on gaussian being None.

Usage (set in imageto3d.py):
    IMAGE3D_MODEL = "HUNYUAN3D"
"""

import os

import torch

__all__ = ["Hunyuan3DInference"]

# Use the paint (texture) pipeline by default; set to "0" to disable.
_ENABLE_TEXTURE = os.environ.get("HUNYUAN3D_TEXTURE", "1") == "1"


class _MeshAdapter:
    """Wraps a trimesh.Trimesh so render_video() can read .vertices/.faces."""

    def __init__(self, mesh):
        self.vertices = torch.from_numpy(
            mesh.vertices.astype("float32")
        ).cuda()
        self.faces = torch.from_numpy(
            mesh.faces.astype("int64")
        ).cuda()
        self._mesh = mesh


class Hunyuan3DInference:
    """Image → textured mesh via Hunyuan3D-2mini.

    Args:
        model_path: HuggingFace model ID for the shape DiT.
        enable_texture: If True, run Hunyuan3DPaintPipeline after shape gen.
    """

    def __init__(
        self,
        model_path: str = "tencent/Hunyuan3D-2mini",
        enable_texture: bool = _ENABLE_TEXTURE,
    ) -> None:
        from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline

        self.model_path = model_path
        self.enable_texture = enable_texture

        self._shape = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
            model_path,
            device="cuda",
            dtype=torch.float16,
        )

        self._paint = None
        if enable_texture:
            from hy3dgen.texgen import Hunyuan3DPaintPipeline
            self._paint = Hunyuan3DPaintPipeline.from_pretrained(
                "tencent/Hunyuan3D-2",
            )

    def run(
        self,
        image,
        seed: int = None,
        num_inference_steps: int = 50,
        octree_resolution: int = 380,
        guidance_scale: float = 5.0,
        **kwargs,
    ) -> dict:
        """Run shape (+ optional texture) generation.

        Returns a dict compatible with the rest of the pipeline:
            {
                "gaussian": [None],          # no GS — callers must check
                "mesh":     [_MeshAdapter],  # for render_video normals
                "trimesh":  [trimesh.Trimesh],
            }
        """
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
            **{k: v for k, v in kwargs.items() if k not in (
                "stage1_inference_steps",
                "stage2_inference_steps",
                "use_stage1_distillation",
                "use_stage2_distillation",
                "pointmap",
            )},
        )
        mesh = meshes[0][0]

        if self._paint is not None:
            mesh = self._paint(mesh=mesh, image=image)

        return {
            "gaussian": [None],
            "mesh": [_MeshAdapter(mesh)],
            "trimesh": [mesh],
        }
