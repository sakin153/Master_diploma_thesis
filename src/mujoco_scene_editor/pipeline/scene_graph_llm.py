from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Iterable, Sequence

from mujoco_scene_editor.pipeline.scene_graph import SceneGraph


@dataclass(frozen=True)
class AvailableMesh:
    file: str
    suggested_scale: float | None = None


def build_scene_graph_prefix(*, meshes: Sequence[AvailableMesh] | None = None) -> str:
    meshes = list(meshes or [])

    meshes_block = ""
    if meshes:
        lines: list[str] = []
        for m in meshes:
            if m.suggested_scale is None:
                lines.append(f"- file: {m.file}")
            else:
                lines.append(f"- file: {m.file}\n  suggested_scale: {m.suggested_scale}")
        meshes_txt = "\n".join(lines)
        meshes_block = (
            "\n\n"
            "Available local meshes (you may reference ONLY these exact file paths in objects[].mesh_file):\n"
            f"{meshes_txt}\n"
        )

    # Keep this prompt stable and model-agnostic.
    return (
        "You are generating a structured scene graph for a MuJoCo scene.\n"
        "Return ONLY valid JSON (no markdown, no code fences, no commentary).\n"
        "All distances are in meters. Use +Z as up.\n"
        "\n"
        "JSON format (strict):\n"
        "{\n"
        "  \"scene_type\": string,\n"
        "  \"objects\": [\n"
        "    {\n"
        "      \"id\": string,\n"
        "      \"class\": string,\n"
        "      \"name\": string|null,\n"
        "      \"affordances\": [string],\n"
        "      \"movable\": boolean,\n"
        "      \"material\": \"plastic\"|\"wood\"|\"metal\"|\"glass\"|\"rubber\"|\"paper\"|\"generic\",\n"
        "      \"primitive\": \"box\"|\"cylinder\"|\"sphere\"|null,\n"
        "      \"size_m\": [number, number, number]|null,\n"
        "      \"mesh_file\": string|null,\n"
        "      \"mesh_scale\": number|null\n"
        "    }\n"
        "  ],\n"
        "  \"relations\": [\n"
        "    {\n"
        "      \"type\": \"on\"|\"ontop\"|\"on_floor\"|\"against_wall\"|\"wall_mounted\"|\"inside\"|\"next_to\"|\"aligned_with\"|\"reachable_by_robot\",\n"
        "      \"subject\": string,\n"
        "      \"object\": string,\n"
        "      \"wall_side\": \"north\"|\"south\"|\"east\"|\"west\"|null,\n"
        "      \"height_m\": number|null,\n"
        "      \"distance_m\": number|null\n"
        "    }\n"
        "  ],\n"
        "  \"constraints\": {\n"
        "    \"keep_clear_radius_m\": number|null,\n"
        "    \"workspace_height_m\": number|null,\n"
        "    \"room_half_size_m\": number|null,\n"
        "    \"wall_height_m\": number|null\n"
        "  },\n"
        "  \"llm_notes\": string|null,\n"
        "  \"metadata\": { }\n"
        "}\n"
        "\n"
        "Rules:\n"
        "- ids must be unique and referenced by relations.\n"
        "- Prefer expressing placement via relations (on/inside/next_to).\n"
        "- Use `on` for tabletop placement (mug on table, monitor on desk, etc.).\n"
        "- IMPORTANT: Use `wall_mounted` (NOT on_floor, NOT against_wall) for ANY object physically\n"
        "  attached to a wall: wall shelves, floating shelves, paintings, clocks, lamps, mirrors.\n"
        "  Example wall_mounted relation:\n"
        "    {\"type\": \"wall_mounted\", \"subject\": \"shelf1\", \"object\": \"\", \"wall_side\": \"north\", \"height_m\": 1.3}\n"
        "- Objects resting ON a wall_mounted shelf use `on`: {\"type\": \"on\", \"subject\": \"book1\", \"object\": \"shelf1\"}\n"
        "- If you reference a mesh, set `mesh_file` to one of the provided paths exactly; otherwise leave it null.\n"
        "- If unsure about exact sizes, leave size_m null.\n"
        "- Ensure the scene is plausible and robot-relevant (clear workspace, reachable items).\n"
        + meshes_block
        + "\nThe scene prompt is:\n"
    )


def _extract_first_json_object(text: str) -> str | None:
    if not text:
        return None

    # Prefer JSON in fenced blocks if present.
    fence = re.search(r"```(?:json)?\s*(\{[\s\S]*?\})\s*```", text, flags=re.IGNORECASE)
    if fence:
        return fence.group(1).strip()

    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    return text[start : end + 1].strip()


def parse_scene_graph_from_llm(text: str) -> SceneGraph:
    raw = _extract_first_json_object(text)
    if raw is None:
        raise ValueError("LLM output does not contain a JSON object")

    try:
        data = json.loads(raw)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON from LLM: {e}") from e

    try:
        return SceneGraph.model_validate(data)
    except Exception as e:
        raise ValueError(f"SceneGraph validation failed: {e}") from e


def meshes_from_prepared(prepared: Iterable[object]) -> list[AvailableMesh]:
    out: list[AvailableMesh] = []
    for m in prepared:
        file = getattr(m, "rel_file", None)
        scale = getattr(m, "scale", None)
        if isinstance(file, str) and file:
            suggested: float | None = None
            try:
                if scale is not None:
                    suggested = float(scale)
            except (TypeError, ValueError):
                suggested = None
            out.append(AvailableMesh(file=file, suggested_scale=suggested))
    return out
