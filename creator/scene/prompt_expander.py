"""Stage 0: Prompt expansion and scene understanding.

If the user gives a short query ("a bedroom", "офис с 5 столами"),
we expand it into a full scene specification before anything else.

Also extracts a structured SceneSpec (room_type, style, key requirements).
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# Prompt expansion
# ---------------------------------------------------------------------------

_EXPAND_PROMPT = """\
You are a 3D scene designer for a robotics simulator (MuJoCo).
Your task: take a short scene description and expand it into a detailed, realistic scene specification.

User query: "{query}"

Respond with JSON only:
```json
{{
  "expanded_description": "Full detailed description of the scene (3-5 sentences). Include room type, style, specific objects with quantities, arrangement logic.",
  "room_type": "bedroom|office|classroom|kitchen|living_room|warehouse|lab|outdoor|other",
  "room_style": "modern|minimalist|cozy|industrial|academic|other",
  "estimated_objects": [
    {{"name": "Object Name", "quantity": 1, "notes": "brief description or placement hint"}}
  ],
  "room_dimensions_hint": "small (3x3m)|medium (5x5m)|large (8x8m)|extra_large (12x12m)"
}}
```

Rules:
- Be specific about quantities (e.g., "10 desks" not "some desks")
- Include typical supporting objects (a bedroom always has a bed, nightstand, lamp, wardrobe)
- Keep the style coherent and realistic
- Respond ONLY with valid JSON inside ```json ... ```
"""

_SCENE_TOO_SHORT_WORDS = 8   # expand if fewer than this many words

_ROOM_DEFAULTS: Dict[str, List[tuple]] = {
    "classroom": [
        ("desk", 10, "student desks arranged in rows"),
        ("chair", 10, "chairs aligned with desks"),
        ("teacher desk", 1, "at front of the room"),
        ("whiteboard", 1, "mounted on front wall"),
    ],
    "office": [
        ("desk", 1, "main work desk"),
        ("office chair", 1, "near desk"),
        ("bookshelf", 1, "against wall"),
    ],
    "bedroom": [
        ("bed", 1, "main bed"),
        ("nightstand", 2, "one on each side of bed"),
        ("wardrobe", 1, "against wall"),
    ],
    "living_room": [
        ("sofa", 1, "main seating"),
        ("coffee table", 1, "in front of sofa"),
        ("armchair", 2, "around coffee table"),
    ],
}


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


def _default_object_hints(room_type: str) -> List["ObjectHint"]:
    return [
        ObjectHint(name=name, quantity=qty, notes=notes)
        for name, qty, notes in _ROOM_DEFAULTS.get(room_type, [])
    ]


def expand_prompt(
    query: str,
    prompt_model_fn: Any,
    llm_model: str = "",
    *,
    force: bool = False,
    verbose: bool = True,
) -> "SceneSpec":
    """Expand a short query into a full SceneSpec.

    If the query is already detailed (>= _SCENE_TOO_SHORT_WORDS words),
    we still parse a minimal SceneSpec from it.
    Always returns a SceneSpec.
    """
    needs_expansion = force or _word_count(query.strip()) < _SCENE_TOO_SHORT_WORDS

    if needs_expansion:
        if verbose:
            print(f"[prompt_expander] Expanding short query: '{query}'")
        prompt = _EXPAND_PROMPT.format(query=query)
        try:
            raw = prompt_model_fn(prompt, query, llm_model)
            spec = _parse_expand_output(raw, original_query=query)
            if verbose:
                print(f"[prompt_expander] Expanded: {spec.expanded_description[:80]}...")
            return spec
        except Exception as exc:
            if verbose:
                print(f"[prompt_expander] LLM expansion failed ({exc}), using heuristic")

    return _heuristic_scene_spec(query)


def _parse_expand_output(raw: Any, original_query: str) -> "SceneSpec":
    """Parse LLM JSON output into SceneSpec."""
    text = str(raw) if not isinstance(raw, str) else raw

    # Extract JSON block
    match = re.search(r"```json\s*(.*?)\s*```", text, re.DOTALL)
    if not match:
        match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return _heuristic_scene_spec(original_query)

    import json
    try:
        data = json.loads(match.group(1) if "```" in text else match.group(0))
    except json.JSONDecodeError:
        return _heuristic_scene_spec(original_query)

    room_type = str(data.get("room_type", "other"))
    if room_type not in {
        "bedroom", "office", "classroom", "kitchen",
        "living_room", "warehouse", "lab", "outdoor", "other",
    }:
        room_type = _heuristic_scene_spec(original_query).room_type

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
        objects = _default_object_hints(room_type)

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


def _heuristic_scene_spec(query: str) -> "SceneSpec":
    """Build a minimal SceneSpec from the query without LLM."""
    lq = query.lower()

    # Guess room type
    room_type = "other"
    for rt, keywords in {
        "bedroom": ["bedroom", "bed room", "спальн"],
        "office": ["office", "офис", "workspace"],
        "classroom": ["classroom", "class", "школьн", "аудитори"],
        "kitchen": ["kitchen", "кухн"],
        "living_room": ["living room", "гостин", "lounge"],
        "warehouse": ["warehouse", "склад"],
        "lab": ["lab", "laboratory", "лаборатори"],
    }.items():
        if any(kw in lq for kw in keywords):
            room_type = rt
            break

    hints = _default_object_hints(room_type)
    obj_text = ", ".join(f"{h.quantity} {h.name}" for h in hints[:6])
    expanded = (
        f"{query}. This is a {room_type.replace('_', ' ')} scene with coherent furnishing. "
        f"Suggested objects: {obj_text}. Arrange objects with realistic spacing and circulation."
    )

    return SceneSpec(
        original_query=query,
        expanded_description=expanded,
        room_type=room_type,
        room_style="modern",
        estimated_objects=hints,
        room_half_size=2.5,
    )


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
