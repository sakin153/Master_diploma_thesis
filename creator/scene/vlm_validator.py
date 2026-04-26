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
    """Convert placed models to a rich text description for LLM reasoning.

    Columns:
      pos=(x,y,z)  — centre position in metres
      size=W×H×D   — width(X) × height(Z) × depth(Y) in metres
      yaw          — rotation around Z axis in degrees (0=faces +Y)
      floor_z_err  — how far z deviates from the correct floor resting height (h/2)
      wall_dist    — distance from object centre to the nearest room wall
      footprint    — approximate floor footprint area W×D
    """
    import math

    room_size = room_half_size * 2
    lines = [
        f"Room: {room_size:.1f}m × {room_size:.1f}m  "
        f"(half_size={room_half_size:.1f}m, origin at centre/floor)",
        f"Objects ({len(placed_models)} total):",
        f"{'#':>3}  {'Name':<26} {'pos (x,y,z)':>20}  {'size W×H×D':>18}  "
        f"{'yaw':>5}  {'floor_z_err':>11}  {'wall_dist':>9}",
        "    " + "-" * 100,
    ]
    for i, m in enumerate(placed_models):
        name = str(m.get("Model") or m.get("name") or f"obj_{i}")
        pose = m.get("Pose") or {}
        size = m.get("size") or [0, 0, 0]
        x = float(pose.get("x", 0.0))
        y = float(pose.get("y", 0.0))
        z = float(pose.get("z", 0.0))
        w = float(size[0]) if len(size) > 0 else 0.0  # width  (X)
        h = float(size[1]) if len(size) > 1 else 0.0  # height (Z)
        d = float(size[2]) if len(size) > 2 else 0.0  # depth  (Y)
        yaw = float(m.get("yaw_deg", 0.0))

        # How far z is from the correct floor-resting height
        expected_z = h / 2.0
        z_err = z - expected_z  # positive = floating, negative = sinking

        # Distance from centre to nearest wall
        wall_dist = min(
            room_half_size - abs(x),
            room_half_size - abs(y),
        )

        lines.append(
            f"  {i+1:2d}. {name:<26} "
            f"({x:+6.2f},{y:+6.2f},{z:+6.2f})m  "
            f"{w:.2f}×{h:.2f}×{d:.2f}m  "
            f"{yaw:5.0f}°  "
            f"z_err={z_err:+.3f}m  "
            f"wall={wall_dist:.2f}m"
        )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Validation prompts
# ---------------------------------------------------------------------------

_VALIDATE_PROMPT = """\
You are an expert 3D scene layout validator for a MuJoCo robotics simulator.
Your job is to rigorously audit a generated scene and identify every spatial,
semantic, and ergonomic problem.

=== COORDINATE SYSTEM ===
- Origin (0, 0, 0) is the room centre at floor level.
- X axis = left/right, Y axis = front/back, Z axis = up.
- Floor objects: correct z = height/2  (object rests on floor).
- Wall-mounted objects (shelves, paintings, TVs): z > 1.0 m, must be near a wall (|x| or |y| ≈ room_half_size).
- Surface objects (cup on desk, book on shelf): z = parent_surface_top + own_height/2.
- room_half_size is the maximum allowed |x| and |y|.  A safe margin is 0.1 m from the wall.

=== SCENE TO AUDIT ===
{scene_description}

=== USER REQUEST ===
"{query}"

=== CHECKLIST — evaluate EVERY item ===

GEOMETRY / BOUNDS
G1. Out-of-bounds: any object whose |x| + width/2 > room_half_size  OR  |y| + depth/2 > room_half_size.
G2. Floating: floor object whose z differs from height/2 by more than 0.05 m (upward = floating, downward = sinking).
G3. Overlap: two objects whose bounding boxes intersect on the floor plane
    (approximate check: distance between centres < (w1+w2)/2  AND  < (d1+d2)/2 simultaneously).
G4. Wall-mount height: wall-mounted object z < 0.8 m (too low) or z > 2.5 m (unreachably high).
G5. Wall-mount proximity: wall-mounted object whose nearest wall distance > 0.3 m (not actually on a wall).

SEMANTIC / FUNCTIONAL
S1. Object cluster coherence: objects that must be near each other (e.g., chair and desk, lamp and nightstand,
    TV and sofa) are farther apart than 2.0 m — score as warning.
S2. Orientation realism: object yaw makes the item face an implausible direction
    (e.g., sofa facing the wall instead of the room centre, desk facing the corner).
S3. Missing key objects: for the stated scene type, identify standard objects that are absent
    (e.g., a "bedroom" without a bed, an "office" without a desk).
S4. Duplicate crowding: more than one large object (footprint > 0.8 m × 0.8 m) placed within 0.5 m of another.
S5. Surface hierarchy: a small object (cup, book, plant) placed at floor level when it should be on a surface.
S6. Scale plausibility: any object whose size differs by more than 3× from real-world expectation
    (e.g., a chair that is 2.5 m tall, a wardrobe that is 0.2 m wide).

SCENE-LEVEL
L1. Room utilisation: more than 60 % of the floor area is empty while objects are clustered in one corner.
L2. Circulation space: no clear 0.8 m-wide path between the room entrance (assume one of the walls) and
    the main furniture group.
L3. Symmetry / balance: for formal rooms (dining, conference), highly asymmetric layouts are penalised.

=== SEVERITY RULES ===
- critical  : G1, G2 (>0.1 m), G3, G5, S3, S5  — scene is physically invalid or functionally broken.
- warning   : G4, G2 (0.05–0.1 m), S1, S2, S4, S6, L2  — scene works but looks wrong or is ergonomically poor.
- info      : S3 (optional objects), L1, L3  — cosmetic or minor improvements.

=== SCORING ===
Start at 1.0. Deduct per issue:
  critical → −0.20 each (min 0.0)
  warning  → −0.08 each
  info     → −0.02 each
Round to two decimal places.

overall_quality:
  score ≥ 0.80 → "good"
  score ≥ 0.55 → "acceptable"
  score  < 0.55 → "poor"

=== RESPONSE FORMAT ===
Reply with ONLY a JSON block, no prose before or after:
```json
{{
  "overall_quality": "good|acceptable|poor",
  "score": 0.0,
  "issues": [
    {{
      "rule": "G1",
      "severity": "critical|warning|info",
      "object": "exact object name from the scene list",
      "issue": "Concise description: what is wrong and by how much (include numbers).",
      "suggestion": "Concrete fix: target coordinates or yaw value if applicable."
    }}
  ],
  "missing_objects": ["list of object types absent but expected for this scene"],
  "positive_aspects": ["list of things done well"],
  "summary": "One sentence overall verdict."
}}
```
"""

_REPAIR_PROMPT = """\
You are an expert 3D scene layout repair assistant for a MuJoCo simulator.
Your task is to compute corrected positions/orientations that resolve the listed
critical issues while keeping every other object in place.

=== COORDINATE SYSTEM ===
- Origin at room centre / floor level.  X = left/right, Y = front/back, Z = up.
- Floor objects must satisfy  z = object_height / 2  (resting on floor).
- All corrected positions must satisfy  |new_x| + width/2 ≤ room_half_size − 0.05  (stay inside room with 5 cm margin).
- All corrected positions must satisfy  |new_y| + depth/2 ≤ room_half_size − 0.05.
- To resolve an overlap between two objects A and B, move ONLY the lower-priority object
  (prefer moving the smaller one, or the one listed later in the scene).
- Prefer positions near walls for large furniture (sofa, bookcase, wardrobe).
- Prefer positions near the centre for tables/desks.
- Maintain functional groupings: keep chairs close to their desk/table,
  keep nightstands adjacent to the bed, keep the TV facing the sofa.
- yaw_deg = 0   → object faces +Y direction.
- yaw_deg = 90  → object faces +X direction.
- yaw_deg = 180 → object faces −Y direction.
- yaw_deg = 270 → object faces −X direction.
  Choose yaw so the object's "front" faces the room interior, not the wall.

=== CRITICAL ISSUES TO FIX ===
{issues_text}

=== CURRENT SCENE ===
{scene_description}

=== INSTRUCTIONS ===
For each critical issue above:
1. Identify the precise cause (out-of-bounds, floating, overlap, wrong surface, etc.).
2. Compute a specific corrected position that resolves the problem.
3. Verify your proposed position does not create a NEW overlap with any other object.
4. If fixing an overlap requires chaining moves (object A pushes B which pushes C),
   include all affected objects in the response.
5. Do NOT move objects that are already correctly placed.

=== RESPONSE FORMAT ===
Reply with ONLY a JSON array, no prose before or after:
```json
[
  {{
    "object": "exact object name as listed in the scene",
    "new_x": 1.23,
    "new_y": -0.50,
    "new_z": null,
    "new_yaw_deg": 180.0,
    "rule_fixed": "G1",
    "reason": "Was at x=6.1 m, outside room_half_size=5.0 m. Moved to x=1.23 m near the north wall, clear of other objects."
  }}
]
```
Rules:
- Use null for any coordinate that does NOT change.
- Provide numeric values rounded to 2 decimal places.
- new_z = null unless the object is floating/sinking (then set to height/2).
- Include a "reason" that quotes the old value, the problem, and the chosen new value.
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
        return {
            "overall_quality": "unknown",
            "score": 0.5,
            "issues": [],
            "missing_objects": [],
            "positive_aspects": [],
            "summary": text[:200],
        }

    data = json.loads(match.group(1) if "```" in text else match.group(0))
    return {
        "overall_quality": str(data.get("overall_quality", "unknown")),
        "score": float(data.get("score", 0.5)),
        "issues": data.get("issues", []),
        "missing_objects": data.get("missing_objects", []),
        "positive_aspects": data.get("positive_aspects", []),
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
