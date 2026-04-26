"""TRELLIS REST API client for image → textured 3D mesh generation.

Wraps the remote TRELLIS server (BASE_URL) with blocking polling.
The returned trimesh is already reoriented from GLB Y-up to MuJoCo Z-up.
"""

import base64
import io
import time
from typing import Optional

import numpy as np
import requests
import trimesh
from PIL import Image

from asset_gen.utils.log import logger

__all__ = ["TrellisClient", "TrellisAPIError", "TrellisTimeoutError", "TrellisGenerationError"]


class TrellisAPIError(RuntimeError):
    pass


class TrellisTimeoutError(TrellisAPIError):
    pass


class TrellisGenerationError(TrellisAPIError):
    pass


class TrellisClient:
    """REST API client for the remote TRELLIS 3D generation server.

    Encodes a PIL image to base64, submits a generation job to
    ``/generate_no_preview``, polls ``/status`` until completion,
    downloads the GLB, and returns a ``trimesh.Trimesh`` reoriented
    from GLTF Y-up to MuJoCo Z-up.

    Example:
        ```py
        from asset_gen.models.trellis_client import TrellisClient
        from PIL import Image

        client = TrellisClient()
        img = Image.open("chair_seg.png")          # RGB 518x518 after trellis_preprocess
        mesh, glb_bytes = client.generate(img, seed=42)
        mesh.export("chair.obj")
        with open("chair.glb", "wb") as f:
            f.write(glb_bytes)
        ```
    """

    BASE_URL = "http://94.19.29.186:443"
    # Stall detection: if progress doesn't change for this long, abort.
    _STALL_TIMEOUT = 120.0

    def __init__(
        self,
        base_url: str = BASE_URL,
        poll_interval: float = 10.0,
        timeout: float = 600.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.poll_interval = poll_interval
        self.timeout = timeout

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def generate(
        self,
        image: Image.Image,
        seed: Optional[int] = None,
        ss_guidance_strength: float = 7.5,
        ss_sampling_steps: int = 30,
        slat_guidance_strength: float = 7.5,
        slat_sampling_steps: int = 30,
        mesh_simplify_ratio: float = 0.95,
        texture_size: int = 1024,
    ) -> tuple["trimesh.Trimesh", bytes]:
        """Submit image, wait for completion, return (trimesh, raw_glb_bytes).

        Args:
            image: RGB PIL image (should be 518×518 after trellis_preprocess).
            seed: Random seed for reproducibility.
            ss_guidance_strength: Structure sampling guidance strength.
            ss_sampling_steps: Structure sampling steps.
            slat_guidance_strength: SLAT sampling guidance strength.
            slat_sampling_steps: SLAT sampling steps.
            mesh_simplify_ratio: Mesh simplification ratio (0–1).
            texture_size: Output texture resolution in pixels.

        Returns:
            Tuple of (trimesh.Trimesh, raw GLB bytes).
            Trimesh is already rotated from Y-up (GLB) to Z-up (MuJoCo).

        Raises:
            TrellisGenerationError: Server reported generation failure.
            TrellisTimeoutError: Generation exceeded timeout or stalled.
            TrellisAPIError: Any other API communication error.
        """
        image_b64 = self._encode_image(image)

        params: dict = {
            "image_base64": image_b64,
            "ss_guidance_strength": ss_guidance_strength,
            "ss_sampling_steps": ss_sampling_steps,
            "slat_guidance_strength": slat_guidance_strength,
            "slat_sampling_steps": slat_sampling_steps,
            "mesh_simplify_ratio": mesh_simplify_ratio,
            "texture_size": texture_size,
            "output_format": "glb",
        }
        if seed is not None:
            params["seed"] = seed

        logger.info(
            f"[TRELLIS] Submitting generation: "
            f"ss_steps={ss_sampling_steps}, slat_steps={slat_sampling_steps}, "
            f"texture={texture_size}px"
        )
        self._submit(params)
        self._poll_until_done()

        logger.info("[TRELLIS] Downloading GLB...")
        glb_bytes = self._download_glb()
        logger.info(f"[TRELLIS] Downloaded {len(glb_bytes) / 1024:.1f} KB")

        mesh = self._glb_bytes_to_trimesh(glb_bytes)
        return mesh, glb_bytes

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _encode_image(image: Image.Image) -> str:
        """Encode PIL image to base64 string (no data: prefix)."""
        buf = io.BytesIO()
        # Convert to RGB before encoding (drop alpha if present — TRELLIS does its own bg handling)
        image.convert("RGB").save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode("utf-8")

    def _submit(self, params: dict) -> None:
        url = f"{self.base_url}/generate_no_preview"
        try:
            resp = requests.post(url, data=params, timeout=30)
            resp.raise_for_status()
        except requests.RequestException as e:
            raise TrellisAPIError(f"Failed to submit TRELLIS job: {e}") from e
        logger.info("[TRELLIS] Job submitted successfully.")

    def _poll_until_done(self) -> None:
        url = f"{self.base_url}/status"
        deadline = time.monotonic() + self.timeout
        last_progress = -1
        last_progress_time = time.monotonic()

        while True:
            if time.monotonic() > deadline:
                raise TrellisTimeoutError(
                    f"TRELLIS generation timed out after {self.timeout:.0f}s"
                )

            try:
                resp = requests.get(url, timeout=15)
                resp.raise_for_status()
                data = resp.json()
            except requests.RequestException as e:
                logger.warning(f"[TRELLIS] Status poll error (will retry): {e}")
                time.sleep(self.poll_interval)
                continue

            status = data.get("status", "")
            progress = data.get("progress", 0)

            logger.info(f"[TRELLIS] Status: {status}, progress: {progress}%")

            if status == "COMPLETE":
                return

            if status == "FAILED":
                msg = data.get("message", "unknown error")
                raise TrellisGenerationError(f"TRELLIS generation failed: {msg}")

            # Stall detection
            if progress != last_progress:
                last_progress = progress
                last_progress_time = time.monotonic()
            elif time.monotonic() - last_progress_time > self._STALL_TIMEOUT:
                raise TrellisTimeoutError(
                    f"TRELLIS generation stalled at {progress}% "
                    f"for {self._STALL_TIMEOUT:.0f}s"
                )

            time.sleep(self.poll_interval)

    def _download_glb(self) -> bytes:
        url = f"{self.base_url}/download/model"
        try:
            resp = requests.get(url, timeout=120)
            resp.raise_for_status()
            return resp.content
        except requests.RequestException as e:
            raise TrellisAPIError(f"Failed to download TRELLIS GLB: {e}") from e

    @staticmethod
    def _glb_bytes_to_trimesh(data: bytes) -> "trimesh.Trimesh":
        """Parse GLB bytes → trimesh, apply Y-up (GLTF) → Z-up (MuJoCo) rotation."""
        scene_or_mesh = trimesh.load(io.BytesIO(data), file_type="glb")

        if isinstance(scene_or_mesh, trimesh.Scene):
            mesh = scene_or_mesh.dump(concatenate=True)
        else:
            mesh = scene_or_mesh

        # GLTF standard: Y-up, right-handed.
        # MuJoCo: Z-up, right-handed.
        # Transform: +90° rotation around X axis maps Y→Z and Z→-Y.
        # det(R) = +1 → proper rotation → winding order preserved.
        R = np.array(
            [[1, 0,  0],
             [0, 0, -1],
             [0, 1,  0]],
            dtype=np.float32,
        )
        mesh.vertices = (mesh.vertices @ R.T).astype(np.float32)

        return mesh
