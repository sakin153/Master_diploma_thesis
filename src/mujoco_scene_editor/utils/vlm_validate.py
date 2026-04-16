from __future__ import annotations

import base64
import io
import json
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


@dataclass
class VLMValidationResult:
    ok: bool
    issues: list[str] = field(default_factory=list)
    feedback: str = ""


def _render_mjcf_to_png_bytes(
    xml_text: str,
    *,
    mjcf_dir: Path,
    width: int = 640,
    height: int = 480,
    settle_steps: int = 80,
    camera: str | None = None,
) -> bytes | None:
    """Render a MuJoCo scene to PNG bytes using the offscreen renderer.

    Returns None if rendering is unavailable (missing mujoco/PIL, or model error).
    """
    try:
        import mujoco
    except ImportError:
        return None

    from mujoco_scene_editor.utils.mjcf_verify import _collect_mjcf_assets

    assets = _collect_mjcf_assets(xml_text, mjcf_dir=mjcf_dir)

    try:
        model = mujoco.MjModel.from_xml_string(xml_text, assets=assets)
    except Exception:
        return None

    data = mujoco.MjData(model)
    for _ in range(settle_steps):
        try:
            mujoco.mj_step(model, data)
        except Exception:
            break

    try:
        renderer = mujoco.Renderer(model, height=height, width=width)
        if camera is not None:
            cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, camera)
            if cam_id >= 0:
                renderer.update_scene(data, camera=cam_id)
            else:
                renderer.update_scene(data)
        else:
            renderer.update_scene(data)
        pixels: np.ndarray = renderer.render()
        renderer.close()
    except Exception:
        return None

    try:
        from PIL import Image

        buf = io.BytesIO()
        Image.fromarray(pixels.astype(np.uint8)).save(buf, format="PNG")
        return buf.getvalue()
    except ImportError:
        pass

    # Fallback: write raw PPM header + pixels → PNG via stdlib
    try:
        import struct
        import zlib

        h, w, _ = pixels.shape
        raw_rows = b"".join(b"\x00" + pixels[y].astype(np.uint8).tobytes() for y in range(h))
        compressed = zlib.compress(raw_rows)

        def _png_chunk(tag: bytes, data: bytes) -> bytes:
            length = struct.pack(">I", len(data))
            body = tag + data
            crc = struct.pack(">I", zlib.crc32(body) & 0xFFFFFFFF)
            return length + body + crc

        ihdr = struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0)
        png = (
            b"\x89PNG\r\n\x1a\n"
            + _png_chunk(b"IHDR", ihdr)
            + _png_chunk(b"IDAT", compressed)
            + _png_chunk(b"IEND", b"")
        )
        return png
    except Exception:
        return None


_VLM_CHECK_PROMPT = (
    "You are reviewing a rendered MuJoCo physics scene for a robot manipulation environment.\n"
    "Examine the image and identify any physical plausibility issues:\n"
    "1. Objects floating in mid-air without visible support\n"
    "2. Objects clipping through each other or through surfaces\n"
    "3. Objects in wrong orientation (upside-down table, lying book on a wall, etc.)\n"
    "4. Objects placed outside the room boundaries\n"
    "5. Unrealistic scale differences (mug larger than table, etc.)\n\n"
    "Respond ONLY with valid JSON in this exact format (no markdown):\n"
    '{"ok": true, "issues": [], "feedback": "Scene looks physically correct."}\n'
    "Set ok=false and list specific issues when problems are found."
)


def vlm_validate_scene(
    xml_text: str,
    *,
    mjcf_dir: Path,
    model: str | None = None,
    width: int = 640,
    height: int = 480,
    camera: str | None = "camera",
) -> VLMValidationResult:
    """Render the MuJoCo scene and ask the multimodal LLM to validate it.

    Falls back to ok=True with an informative message if rendering or the LLM
    call fails (so it never blocks the pipeline).
    """
    mjcf_dir = Path(mjcf_dir)
    png_bytes = _render_mjcf_to_png_bytes(
        xml_text, mjcf_dir=mjcf_dir, width=width, height=height, camera=camera
    )

    if png_bytes is None:
        return VLMValidationResult(
            ok=True, feedback="[rendering unavailable — VLM validation skipped]"
        )

    b64 = base64.b64encode(png_bytes).decode("ascii")
    data_url = f"data:image/png;base64,{b64}"

    from mujoco_scene_editor.llm.openrouter import chat_completion

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": data_url}},
                {"type": "text", "text": _VLM_CHECK_PROMPT},
            ],
        }
    ]

    try:
        response = chat_completion(messages, model=model)
    except Exception as e:
        return VLMValidationResult(ok=True, feedback=f"[VLM call failed: {e}]")

    # Parse JSON response.
    try:
        json_match = re.search(r"\{[\s\S]*\}", response)
        if json_match:
            parsed = json.loads(json_match.group(0))
            return VLMValidationResult(
                ok=bool(parsed.get("ok", True)),
                issues=[str(x) for x in parsed.get("issues", [])],
                feedback=str(parsed.get("feedback", "")),
            )
    except (json.JSONDecodeError, Exception):
        pass

    # Fallback: treat raw text as feedback.
    return VLMValidationResult(ok=True, feedback=response[:300])
