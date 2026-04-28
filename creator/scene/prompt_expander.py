"""Stage 0: Prompt expansion and scene understanding.

If the user gives a short query ("a bedroom", "офис с 5 столами"),
we expand it into a full scene specification before anything else.

Also extracts a structured SceneSpec (room_type, style, key requirements).
"""
from __future__ import annotations

import re
from typing import Any, List, Optional


# ---------------------------------------------------------------------------
# Prompt expansion
# ---------------------------------------------------------------------------

from creator.contexts_prompts.expand import fmt_expand_system as _EXPAND_SYSTEM

_SCENE_TOO_SHORT_WORDS = 8   # expand if fewer than this many words


def _word_count(text: str) -> int:
    return len(text.split())


def _tokenize(text: str) -> List[str]:
    return re.findall(r"[a-zа-яё0-9]+", (text or "").lower())


def _singularize(word: str) -> str:
    w = (word or "").strip().lower()
    if len(w) > 3 and w.endswith("ves"):
        return w[:-3] + "f"   # shelves→shelf, knives→knife
    if len(w) > 3 and w.endswith("ies"):
        return w[:-3] + "y"
    if len(w) > 2 and w.endswith("s"):
        return w[:-1]
    return w



def expand_prompt(
    query: str,
    prompt_model_fn: Any,
    llm_model: str = "",
    *,
    force: bool = False,
    verbose: bool = True,
) -> "SceneSpec":
    """Expand a short query into a full SceneSpec.

    Always uses LLM to properly extract objects and quantities.
    """
    if verbose:
        print(f"[prompt_expander] Expanding query: '{query}'")
    raw = prompt_model_fn(_EXPAND_SYSTEM, query, llm_model)
    spec = _parse_expand_output(raw, original_query=query)
    if verbose:
        print(f"[prompt_expander] Expanded: {spec.expanded_description[:80]}...")
    return spec


def _parse_expand_output(raw: Any, original_query: str) -> "SceneSpec":
    """Parse LLM output into SceneSpec.

    `prompt_model` already runs `parse_output_to_json`, so `raw` is usually
    a dict. Fall back to text-extraction only when the LLM returned a
    plain string (e.g. JSON wasn't fenced and the upstream parser raised).
    """
    import json

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

    room_type = str(data.get("room_type", "other"))
    if room_type not in {
        "bedroom", "office", "classroom", "kitchen",
        "living_room", "warehouse", "lab", "outdoor", "other",
    }:
        raise ValueError(f"LLM returned unknown room_type: {room_type!r}")

    objects = []
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

    dim_hint = str(data.get("room_dimensions_hint", "medium (5x5m)"))
    room_half = _parse_room_dim_hint(dim_hint)

    expanded_description = str(data.get("expanded_description", original_query)).strip()
    if not expanded_description:
        expanded_description = original_query

    return SceneSpec(
        original_query=original_query,
        expanded_description=expanded_description,
        room_type=room_type,
        room_style=str(data.get("room_style", "modern")),
        estimated_objects=objects,
        room_half_size=room_half,
    )


def _create_minimal_spec(query: str) -> "SceneSpec":
    """Create minimal SceneSpec for simple, specific requests."""
    import re
    
    # Extract objects from simple patterns
    objects = []
    query_lower = query.lower()
    
    # Common object patterns
    object_patterns = {
        r'стол|table': 'table',
        r'яблок|apple': 'apple', 
        r'банан|banana': 'banana',
        r'книг|book': 'book',
        r'чашк|cup': 'cup',
        r'ящик|коробк|box': 'box',
    }
    
    for pattern, obj_name in object_patterns.items():
        if re.search(pattern, query_lower):
            # Count quantity if specified
            qty_match = re.search(rf'(\d+).*{pattern}', query_lower)
            if qty_match:
                qty = int(qty_match.group(1))
            else:
                qty = 1
            
            objects.append(ObjectHint(name=obj_name, quantity=qty))
    
    # Default room settings for simple requests
    return SceneSpec(
        original_query=query,
        expanded_description=query,  # Don't expand
        room_type="other",
        room_style="simple", 
        estimated_objects=objects,
        room_half_size=2.0,  # Small room for simple scenes
    )


def _parse_room_dim_hint(hint: str) -> float:
    """Extract room half-size in metres from dimension hint string."""
    # Try to extract numbers like "5x5" or "8x8"
    m = re.search(r"(\d+)\s*x\s*\d+", hint)
    if m:
        return float(m.group(1)) / 2.0
    if "small" in hint.lower():
        return 2.0
    if "large" in hint.lower() and "extra" in hint.lower():
        return 6.0
    if "large" in hint.lower():
        return 4.0
    return 2.5  # medium default



# ---------------------------------------------------------------------------
# SceneSpec dataclass
# ---------------------------------------------------------------------------

class ObjectHint:
    """Lightweight hint for an object that should appear in the scene."""
    def __init__(self, name: str, quantity: int = 1, notes: str = ""):
        self.name = name
        self.quantity = max(1, quantity)
        self.notes = notes

    def __repr__(self) -> str:
        return f"ObjectHint({self.name!r} ×{self.quantity})"


class SceneSpec:
    """Result of prompt expansion — high-level scene specification."""

    def __init__(
        self,
        original_query: str,
        expanded_description: str,
        room_type: str = "other",
        room_style: str = "modern",
        estimated_objects: Optional[List[ObjectHint]] = None,
        room_half_size: float = 2.5,
    ) -> None:
        self.original_query = original_query
        self.expanded_description = expanded_description
        self.room_type = room_type
        self.room_style = room_style
        self.estimated_objects: List[ObjectHint] = estimated_objects or []
        self.room_half_size = room_half_size   # initial estimate; may be revised by room_planner

    @property
    def effective_query(self) -> str:
        """Use expanded description if it's richer than the original."""
        if len(self.expanded_description) > len(self.original_query) + 10:
            return self.expanded_description
        return self.original_query

    def __repr__(self) -> str:
        return (
            f"SceneSpec(type={self.room_type!r}, "
            f"room={self.room_half_size*2:.0f}m×{self.room_half_size*2:.0f}m, "
            f"objects={self.estimated_objects})"
        )
