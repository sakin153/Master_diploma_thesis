"""VLM-based scene validation (LayoutVLM-style).

Approach:
1. Format the placed scene as a structured text description
   (object names, positions, sizes, relations).
2. Pass to an LLM/VLM with a validation prompt asking:
   "Are all objects correctly placed? List any problems."
3. Parse the response to extract a list of issues.
4. Optionally: run a repair loop (pass issues back to placement solver).

This works even without image rendering because modern LLMs reason well
over spatial descriptions (bounding box coordinates + relation text).
If the underlying model is multimodal, can also pass rendered images.

References:
- LayoutVLM (2024): closed-loop constraint-feedback prompting
- LayoutGPT (Feng et al., 2023): text-based spatial reasoning
"""
from __future__ import annotations

import base64
import io
import json
import re
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# MuJoCo offscreen renderer
# ---------------------------------------------------------------------------

def render_mujoco_scene(
    world_path: str,
    *,
    width: int = 640,
    height: int = 480,
    camera_distance: float = 10.0,
    camera_azimuth: float = 45.0,
    camera_elevation: float = -30.0,
) -> Optional[str]:
    """Render the MuJoCo XML scene to a base64-encoded PNG string.

    Returns None if rendering fails (mujoco not installed, bad XML, etc.).
    The camera is placed at an isometric-ish angle to see the full room layout.
    """
    try:
        import mujoco  # type: ignore
        import numpy as np
        from PIL import Image

        model = mujoco.MjModel.from_xml_path(world_path)
        data = mujoco.MjData(model)
        # Step a few frames to resolve initial penetrations before snapshot
        for _ in range(100):
            mujoco.mj_step(model, data)

        renderer = mujoco.Renderer(model, height=height, width=width)

        cam = mujoco.MjvCamera()
        cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        cam.distance = camera_distance
        cam.azimuth = camera_azimuth
        cam.elevation = camera_elevation
        cam.lookat[:] = [0.0, 0.0, 0.8]  # look at ~waist height

        renderer.update_scene(data, camera=cam)

        # Enable shadows for a cleaner scene preview
        renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = True
        renderer.scene.flags[mujoco.mjtRndFlag.mjRND_REFLECTION] = False

        pixels = renderer.render()  # uint8 H×W×3

        img = Image.fromarray(pixels.astype("uint8"))
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode("utf-8")

    except Exception as exc:
        print(f"[vlm_validator] render failed: {exc}")
        return None


def save_scene_preview(
    world_path: str,
    output_path: Optional[str] = None,
    **render_kwargs: Any,
) -> Optional[str]:
    """Render the scene and save a PNG preview to disk.

    If output_path is None, saves next to the world file as scene_latest.png.
    Returns the saved path or None on failure.
    """
    if output_path is None:
        import os
        output_path = os.path.splitext(world_path)[0] + "_preview.png"

    b64 = render_mujoco_scene(world_path, **render_kwargs)
    if b64 is None:
        return None

    try:
        raw = base64.b64decode(b64)
        with open(output_path, "wb") as f:
            f.write(raw)
        print(f"[vlm_validator] Preview saved → {output_path}")
        return output_path
    except Exception as exc:
        print(f"[vlm_validator] Failed to save preview: {exc}")
        return None


# ---------------------------------------------------------------------------
# Scene description formatter
# ---------------------------------------------------------------------------

def _format_scene_for_llm(
    placed_models: List[Dict[str, Any]],
    room_half_size: float = 5.0,
) -> str:
    """Convert placed models to a compact text description for LLM reasoning."""
    room_size = room_half_size * 2
    lines = [
        f"Room: {room_size:.1f}m × {room_size:.1f}m (half_size={room_half_size:.1f}m)",
        f"Objects ({len(placed_models)} total):",
    ]
    for i, m in enumerate(placed_models):
        name = str(m.get("Model") or m.get("name") or f"obj_{i}")
        pose = m.get("Pose") or {}
        size = m.get("size") or [0, 0, 0]
        x = float(pose.get("x", 0.0))
        y = float(pose.get("y", 0.0))
        z = float(pose.get("z", 0.0))
        # size[0]=width, size[1]=height, size[2]=depth
        w = float(size[0]) if len(size) > 0 else 0.0
        h = float(size[1]) if len(size) > 1 else 0.0
        d = float(size[2]) if len(size) > 2 else 0.0
        yaw = float(m.get("yaw_deg", 0.0))
        lines.append(
            f"  {i+1:2d}. {name:25s} pos=({x:5.2f},{y:5.2f},{z:5.2f})m  "
            f"size={w:.2f}×{h:.2f}×{d:.2f}m  yaw={yaw:.0f}°"
        )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Validation prompts
# ---------------------------------------------------------------------------

_VALIDATE_PROMPT = """\
You are a 3D scene layout validator for a robotics simulator.

Scene description:
{scene_description}

Original user request: "{query}"

Your task: Evaluate if this scene layout is correct and realistic.
Check for:
1. Objects outside room bounds (|x| or |y| > room_half_size)
2. Objects at wrong height (z should equal half their height for floor objects)
3. Unrealistic placement (e.g., sofa floating in air, book inside table)
4. Missing expected objects for this type of scene
5. Objects too close together (overlapping)
6. Objects that should be near each other but are far apart

Respond with JSON:
```json
{{
  "overall_quality": "good|acceptable|poor",
  "score": 0.0,
  "issues": [
    {{
      "severity": "critical|warning|info",
      "object": "object name",
      "issue": "brief description of the problem",
      "suggestion": "how to fix it"
    }}
  ],
  "summary": "One sentence overall assessment"
}}
```
"""

_REPAIR_PROMPT = """\
You are a 3D scene layout repair assistant.

The scene has these placement issues:
{issues_text}

Current scene:
{scene_description}

For each critical issue, provide a correction.
Return a JSON list of corrections:
```json
[
  {{
    "object": "object name",
    "new_x": 0.0,
    "new_y": 0.0,
    "new_z": 0.0,
    "new_yaw_deg": 0.0,
    "reason": "why this position is better"
  }}
]
```
Only include objects that need to move. Use null for unchanged coordinates.
"""


# ---------------------------------------------------------------------------
# Main validation function
# ---------------------------------------------------------------------------

def validate_scene_with_llm(
    placed_models: List[Dict[str, Any]],
    query: str,
    prompt_model_fn: Any,
    llm_model: str = "",
    room_half_size: float = 5.0,
    *,
    world_path: Optional[str] = None,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Validate scene layout using LLM reasoning (LayoutVLM-style).

    When world_path is provided, renders the scene via MuJoCo offscreen renderer
    and passes the image to the VLM alongside the text description.

    Returns:
        {
            "overall_quality": "good|acceptable|poor",
            "score": float,
            "issues": [...],
            "summary": str,
        }
    """
    scene_text = _format_scene_for_llm(placed_models, room_half_size)
    prompt = _VALIDATE_PROMPT.format(
        scene_description=scene_text,
        query=query,
    )

    try:
        # Try multimodal (image) validation when world_path is given
        if world_path is not None:
            image_b64 = render_mujoco_scene(world_path)
            if image_b64 is not None:
                from creator.llm.model import prompt_model_with_image
                if verbose:
                    print("[vlm_validator] Using multimodal validation (rendered image)")
                raw = prompt_model_with_image(
                    _VALIDATE_PROMPT.format(
                        scene_description=scene_text + "\n\nThe rendered scene image is attached.",
                        query=query,
                    ),
                    "Validate the scene shown in the image and the description above.",
                    image_b64,
                )
                result = _parse_validation_output(raw)
                result["_used_image"] = True
                if verbose:
                    q = result.get("overall_quality", "?")
                    s = result.get("score", 0.0)
                    n = len(result.get("issues", []))
                    print(f"[vlm_validator] (image) Quality={q} score={s:.2f} issues={n}")
                return result

        # Fallback: text-only validation
        raw = prompt_model_fn(prompt, query, llm_model)
        result = _parse_validation_output(raw)
        result["_used_image"] = False
        if verbose:
            q = result.get("overall_quality", "?")
            s = result.get("score", 0.0)
            n = len(result.get("issues", []))
            print(f"[vlm_validator] Quality={q} score={s:.2f} issues={n}")
            for issue in result.get("issues", [])[:3]:
                sev = issue.get("severity", "?")
                print(f"  [{sev}] {issue.get('object','?')}: {issue.get('issue','')}")
        return result
    except Exception as exc:
        if verbose:
            print(f"[vlm_validator] LLM validation failed ({exc})")
        return {
            "overall_quality": "unknown",
            "score": 0.5,
            "issues": [],
            "summary": f"Validation skipped: {exc}",
            "_used_image": False,
        }


def repair_scene_with_llm(
    placed_models: List[Dict[str, Any]],
    issues: List[Dict[str, Any]],
    query: str,
    prompt_model_fn: Any,
    llm_model: str = "",
    room_half_size: float = 5.0,
    *,
    verbose: bool = True,
) -> List[Dict[str, Any]]:
    """Apply LLM-suggested corrections to placed_models.

    Returns updated placed_models.
    """
    critical = [i for i in issues if i.get("severity") == "critical"]
    if not critical:
        return placed_models

    issues_text = "\n".join(
        f"- [{i['severity']}] {i.get('object','?')}: {i.get('issue','')} → {i.get('suggestion','')}"
        for i in critical[:5]
    )
    scene_text = _format_scene_for_llm(placed_models, room_half_size)
    prompt = _REPAIR_PROMPT.format(
        issues_text=issues_text,
        scene_description=scene_text,
    )

    try:
        raw = prompt_model_fn(prompt, query, llm_model)
        corrections = _parse_repair_output(raw)
        updated = _apply_corrections(placed_models, corrections, room_half_size)
        if verbose:
            print(f"[vlm_validator] Applied {len(corrections)} corrections")
        return updated
    except Exception as exc:
        if verbose:
            print(f"[vlm_validator] LLM repair failed ({exc}), keeping original layout")
        return placed_models


# ---------------------------------------------------------------------------
# Validation + repair loop
# ---------------------------------------------------------------------------

def validate_and_repair_loop(
    placed_models: List[Dict[str, Any]],
    query: str,
    prompt_model_fn: Any,
    llm_model: str = "",
    room_half_size: float = 5.0,
    *,
    world_path: Optional[str] = None,
    max_iterations: int = 2,
    quality_threshold: float = 0.6,
    verbose: bool = True,
) -> List[Dict[str, Any]]:
    """Run validate → repair → validate loop up to max_iterations.

    Stops early if quality reaches quality_threshold.
    When world_path is given, each validation iteration uses a rendered image.
    """
    models = placed_models
    for iteration in range(max_iterations):
        result = validate_scene_with_llm(
            models, query, prompt_model_fn, llm_model,
            room_half_size=room_half_size,
            world_path=world_path,
            verbose=verbose,
        )
        score = float(result.get("score", 0.0))
        quality = result.get("overall_quality", "unknown")

        if quality == "good" or score >= quality_threshold:
            if verbose:
                print(f"[vlm_validator] Scene accepted at iteration {iteration} (score={score:.2f})")
            break

        issues = result.get("issues", [])
        if not any(i.get("severity") == "critical" for i in issues):
            if verbose:
                print("[vlm_validator] No critical issues, stopping repair loop")
            break

        models = repair_scene_with_llm(
            models, issues, query, prompt_model_fn, llm_model,
            room_half_size=room_half_size, verbose=verbose,
        )

    return models


# ---------------------------------------------------------------------------
# Parsers
# ---------------------------------------------------------------------------

def _parse_validation_output(raw: Any) -> Dict[str, Any]:
    text = str(raw) if not isinstance(raw, str) else raw
    match = re.search(r"```json\s*(.*?)\s*```", text, re.DOTALL)
    if not match:
        match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return {"overall_quality": "unknown", "score": 0.5, "issues": [], "summary": text[:200]}

    data = json.loads(match.group(1) if "```" in text else match.group(0))
    return {
        "overall_quality": str(data.get("overall_quality", "unknown")),
        "score": float(data.get("score", 0.5)),
        "issues": data.get("issues", []),
        "summary": str(data.get("summary", "")),
    }


def _parse_repair_output(raw: Any) -> List[Dict[str, Any]]:
    text = str(raw) if not isinstance(raw, str) else raw
    match = re.search(r"```json\s*(.*?)\s*```", text, re.DOTALL)
    if not match:
        match = re.search(r"\[.*\]", text, re.DOTALL)
    if not match:
        return []
    data = json.loads(match.group(1) if "```" in text else match.group(0))
    return data if isinstance(data, list) else []


def _apply_corrections(
    placed_models: List[Dict[str, Any]],
    corrections: List[Dict[str, Any]],
    room_half_size: float,
) -> List[Dict[str, Any]]:
    """Apply position/yaw corrections from LLM to placed_models."""
    name_index: Dict[str, List[int]] = {}
    for i, m in enumerate(placed_models):
        n = str(m.get("Model") or m.get("name") or "")
        name_index.setdefault(n, []).append(i)

    updated = [dict(m) for m in placed_models]
    applied_per_name: Dict[str, int] = {}

    for corr in corrections:
        if not isinstance(corr, dict):
            continue
        obj_name = str(corr.get("object", ""))
        indices = name_index.get(obj_name, [])
        if not indices:
            continue
        # Apply to next unused instance
        inst = applied_per_name.get(obj_name, 0)
        if inst >= len(indices):
            continue
        idx = indices[inst]
        applied_per_name[obj_name] = inst + 1

        m = updated[idx]
        pose = dict(m.get("Pose") or {})
        size = m.get("size") or [0.5, 0.5, 0.5]

        if corr.get("new_x") is not None:
            x = float(corr["new_x"])
            x = max(-room_half_size + float(size[0])/2, min(room_half_size - float(size[0])/2, x))
            pose["x"] = x
        if corr.get("new_y") is not None:
            y = float(corr["new_y"])
            y = max(-room_half_size + float(size[2])/2, min(room_half_size - float(size[2])/2, y))
            pose["y"] = y
        if corr.get("new_z") is not None:
            pose["z"] = max(0.0, float(corr["new_z"]))
        if corr.get("new_yaw_deg") is not None:
            m["yaw_deg"] = float(corr["new_yaw_deg"])

        m["Pose"] = pose
        updated[idx] = m

    return updated
