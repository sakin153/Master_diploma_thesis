"""Scene specification models and parsing.

ObjectHint and SceneSpec are the core data structures that represent
a scene before any model lookup or placement happens.
"""
from __future__ import annotations

import json
import re
from typing import Any, List, Optional

_MAX_TOTAL_OBJECTS = 50


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

class ObjectHint:
    """Hint for one object type that should appear in the scene."""

    def __init__(self, name: str, quantity: int = 1, notes: str = "") -> None:
        self.name = name
        self.quantity = max(1, quantity)
        self.notes = notes

    def __repr__(self) -> str:
        return f"ObjectHint({self.name!r} ×{self.quantity})"


class SceneSpec:
    """High-level scene specification produced by Stage 0."""

    def __init__(
        self,
        original_query: str,
        expanded_description: str,
        estimated_objects: Optional[List[ObjectHint]] = None,
        room_half_size: float = 2.5,
    ) -> None:
        self.original_query = original_query
        self.expanded_description = expanded_description
        self.estimated_objects: List[ObjectHint] = estimated_objects or []
        self.room_half_size = room_half_size

    @property
    def effective_query(self) -> str:
        """Return expanded description when it is richer than the original."""
        if len(self.expanded_description) > len(self.original_query) + 10:
            return self.expanded_description
        return self.original_query

    def __repr__(self) -> str:
        return (
            f"SceneSpec("
            f"room={self.room_half_size*2:.0f}m×{self.room_half_size*2:.0f}m, "
            f"objects={self.estimated_objects})"
        )


# ---------------------------------------------------------------------------
# Parsing LLM output → SceneSpec
# ---------------------------------------------------------------------------

def parse_llm_response(raw: Any, original_query: str) -> SceneSpec:
    """Parse LLM JSON output into a SceneSpec.

    `raw` is usually already a dict (upstream caller ran parse_output_to_json).
    Falls back to extracting JSON from raw text when needed.
    """
    if isinstance(raw, dict):
        data = raw
    else:
        text = raw if isinstance(raw, str) else str(raw)
        match = re.search(r"```json\s*(.*?)\s*```", text, re.DOTALL)
        if match:
            json_str = match.group(1)
        else:
            match = re.search(r"\{.*\}", text, re.DOTALL)
            json_str = match.group(0) if match else ""

        if not json_str:
            raise ValueError("LLM returned no JSON in output")

        try:
            data = json.loads(json_str)
        except json.JSONDecodeError as exc:
            raise ValueError(f"LLM returned invalid JSON: {exc}") from exc

    objects: List[ObjectHint] = []
    for o in data.get("estimated_objects", []):
        if isinstance(o, dict):
            objects.append(ObjectHint(
                name=str(o.get("name", "")),
                quantity=int(o.get("quantity", 1)),
                notes=str(o.get("notes", "")),
            ))
        elif isinstance(o, str) and o.strip():
            objects.append(ObjectHint(name=o.strip(), quantity=1, notes=""))

    if not objects:
        raise ValueError("LLM returned no objects in estimated_objects")

    total = sum(o.quantity for o in objects)
    if total > _MAX_TOTAL_OBJECTS:
        scale = _MAX_TOTAL_OBJECTS / total
        for o in objects:
            o.quantity = max(1, int(o.quantity * scale))

    room_half = _parse_room_dim_hint(str(data.get("room_dimensions_hint", "medium (5x5m)")))

    expanded_description = str(data.get("expanded_description", original_query)).strip()
    if not expanded_description:
        expanded_description = original_query

    return SceneSpec(
        original_query=original_query,
        expanded_description=expanded_description,
        estimated_objects=objects,
        room_half_size=room_half,
    )


def _parse_room_dim_hint(hint: str) -> float:
    m = re.search(r"(\d+)\s*x\s*\d+", hint)
    if m:
        return float(m.group(1)) / 2.0
    hint_l = hint.lower()
    if "extra" in hint_l and "large" in hint_l:
        return 6.0
    if "large" in hint_l:
        return 4.0
    if "small" in hint_l:
        return 2.0
    return 2.5  # medium default
