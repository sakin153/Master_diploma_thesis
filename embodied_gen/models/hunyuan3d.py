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
    print(f"[DEBUG] Added Hunyuan3D-2 to sys.path: {_HY3D_LOCAL}")
else:
    if not os.path.isdir(_HY3D_LOCAL):
        print(f"[ERROR] Hunyuan3D-2 not found at: {_HY3D_LOCAL}")
    if _HY3D_LOCAL in sys.path:
        print(f"[DEBUG] Hunyuan3D-2 already in sys.path")

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
        # Ensure Hunyuan3D-2 is in path (may be needed in different execution contexts)
        if _HY3D_LOCAL not in sys.path:
            sys.path.insert(0, _HY3D_LOCAL)

        try:
            from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline
        except ImportError as e:
            print(f"[ERROR] Failed to import hy3dgen.shapegen: {e}")
            print(f"[DEBUG] sys.path (first 3): {sys.path[:3]}")
            print(f"[DEBUG] Hunyuan3D-2 path exists: {os.path.isdir(_HY3D_LOCAL)}")
            hy3dgen_path = os.path.join(_HY3D_LOCAL, 'hy3dgen')
            print(f"[DEBUG] hy3dgen exists: {os.path.isdir(hy3dgen_path)}")
            if os.path.isdir(hy3dgen_path):
                print(f"[DEBUG] hy3dgen contents: {os.listdir(hy3dgen_path)}")
            raise

        self.model_path = model_path

        # Check GPU memory before loading
        print("[DEBUG] Checking GPU memory...")
        if torch.cuda.is_available():
            gpu_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            allocated = torch.cuda.memory_allocated(0) / (1024**3)
            print(f"[DEBUG] GPU Memory: {gpu_mem:.1f} GB total, {allocated:.1f} GB allocated")
        else:
            print("[WARNING] CUDA not available, loading on CPU (very slow!)")

        try:
            print(f"[DEBUG] Loading Hunyuan3D model from {model_path}/{subfolder}...")
            self._shape = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
                model_path,
                subfolder=subfolder,
                use_safetensors=True,
                variant="fp16",
                device="cuda" if torch.cuda.is_available() else "cpu",
            )
            print("[DEBUG] ✓ Hunyuan3D model loaded successfully")
        except RuntimeError as e:
            if "out of memory" in str(e).lower() or "cuda" in str(e).lower():
                print(f"[ERROR] GPU Memory Error: {e}")
                print("[ERROR] Try clearing GPU cache or reducing batch size")
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                raise RuntimeError(
                    f"Failed to load Hunyuan3D model due to GPU memory: {e}"
                )
            raise
        except Exception as e:
            print(f"[ERROR] Failed to load Hunyuan3D model: {e}")
            import traceback
            traceback.print_exc()
            raise

    def run(
        self,
        image,
        seed: int = None,
        num_inference_steps: int = 30,
        octree_resolution: int = 256,
        guidance_scale: float = 5.0,
        **kwargs,
    ) -> dict:
        try:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            gen = (
                torch.Generator(device=device).manual_seed(seed)
                if seed is not None
                else None
            )

            print(f"[DEBUG] Running Hunyuan3D inference on {device}...")
            meshes = self._shape(
                image=image,
                num_inference_steps=num_inference_steps,
                octree_resolution=octree_resolution,
                guidance_scale=guidance_scale,
                generator=gen,
                output_type="trimesh",
            )
            mesh = meshes[0]
            print("[DEBUG] ✓ Hunyuan3D inference completed")

            return {
                "gaussian": [None],
                "mesh": [_MeshAdapter(mesh)],
                "trimesh": [mesh],
            }
        except RuntimeError as e:
            if "cuda" in str(e).lower() or "out of memory" in str(e).lower():
                print(f"[ERROR] CUDA/GPU Error: {e}")
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    print("[DEBUG] GPU cache cleared")
            raise
