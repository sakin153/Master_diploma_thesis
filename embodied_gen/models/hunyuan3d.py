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
        msg = f"[ERROR] Hunyuan3D-2 not found at: {_HY3D_LOCAL}"
        print(msg)
    if _HY3D_LOCAL in sys.path:
        print("[DEBUG] Hunyuan3D-2 already in sys.path")

__all__ = ["Hunyuan3DInference"]


class _MeshAdapter:
    """Wraps trimesh.Trimesh so render_video() can read vertices/faces."""

    def __init__(self, mesh):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        try:
            vertices = mesh.vertices.astype("float32")
            faces = mesh.faces.astype("int64")

            if vertices.size == 0 or faces.size == 0:
                print("[WARNING] Empty mesh detected")

            self.vertices = torch.from_numpy(vertices).to(device)
            self.faces = torch.from_numpy(faces).to(device)
            self._mesh = mesh
        except Exception as e:
            print(f"[ERROR] Failed to adapt mesh: {e}")
            raise


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
            from hy3dgen.shapegen import (
                Hunyuan3DDiTFlowMatchingPipeline
            )
        except ImportError as e:
            print(f"[ERROR] Failed to import hy3dgen.shapegen: {e}")
            print(f"[DEBUG] sys.path (first 3): {sys.path[:3]}")
            exists = os.path.isdir(_HY3D_LOCAL)
            print(f"[DEBUG] Hunyuan3D-2 path exists: {exists}")
            hy3dgen_path = os.path.join(_HY3D_LOCAL, "hy3dgen")
            exists_hy = os.path.isdir(hy3dgen_path)
            print(f"[DEBUG] hy3dgen exists: {exists_hy}")
            if exists_hy:
                contents = os.listdir(hy3dgen_path)
                print(f"[DEBUG] hy3dgen contents: {contents}")
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
            # Allow forcing CPU via env var (for debugging CUDA issues)
            force_cpu_env = os.environ.get("HY3D_FORCE_CPU", "0")
            force_cpu = force_cpu_env.lower() in ("1", "true", "yes")
            if force_cpu:
                device = "cpu"
            else:
                device = "cuda" if torch.cuda.is_available() else "cpu"

            print(
                f"[DEBUG] Loading Hunyuan3D model "
                f"from {model_path}/{subfolder}..."
            )
            print(f"[DEBUG] Using device: {device}")

            variant = "fp16" if device == "cuda" else "fp32"
            self._shape = (
                Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
                    model_path,
                    subfolder=subfolder,
                    use_safetensors=True,
                    variant=variant,
                    device=device,
                )
            )
            print("[DEBUG] ✓ Hunyuan3D model loaded successfully")
        except RuntimeError as e:
            err_str = str(e).lower()
            if "out of memory" in err_str or "cuda" in err_str:
                print(f"[ERROR] GPU Memory Error: {e}")
                msg = "[ERROR] Try clearing GPU cache or reducing batch size"
                print(msg)
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                raise RuntimeError(
                    f"Failed to load Hunyuan3D model due to GPU: {e}"
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
