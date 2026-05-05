# Стадия 0 - расширяем запрос пользователя.

import json
import re
from pathlib import Path

from project.llm_request import DEFAULT_MODEL, request as _default_request

_VALID_ROOM_TYPES = {"bedroom", "office", "classroom", "kitchen", "living_room", "dining_room", "warehouse", "lab", "outdoor", "other"}
_MAX_TOTAL_OBJECTS = 20


class ObjectHint:
    def __init__(self, name, quantity=1, notes=""):
        self.name = name
        self.quantity = max(1, quantity)
        self.notes = notes

    def __repr__(self):
        return f"ObjectHint({self.name!r} x{self.quantity})"


class SceneSpec:
    def __init__(self, original_query, expanded_description, room_type="other",
                 room_style="modern", estimated_objects=None, room_half_size=2.5,
                 ):
        self.original_query = original_query
        self.expanded_description = expanded_description
        self.room_type = room_type
        self.room_style = room_style
        self.estimated_objects = estimated_objects or []
        self.room_half_size = room_half_size

    def __repr__(self):
        objects_str = "\n".join(f"    - {o.name} x{o.quantity}" for o in self.estimated_objects)
        return (
            f"SceneSpec:\n"
            f"  тип комнаты : {self.room_type}\n"
            f"  стиль       : {self.room_style}\n"
            f"  размер      : {self.room_half_size * 2:.0f}m x {self.room_half_size * 2:.0f}m\n"
            f"  объекты     :\n{objects_str}"
        )


def _parse_room_dim(hint):
    m = re.search(r"(\d+)\s*x\s*\d+", hint)
    if m:
        return float(m.group(1)) / 2.0
    if "extra" in hint and "large" in hint:
        return 6.0
    if "large" in hint:
        return 4.0
    if "small" in hint:
        return 2.0
    return 2.5


def _parse_llm_response(data, original_query):
    room_type = str(data.get("room_type", "other"))
    if room_type not in _VALID_ROOM_TYPES:
        raise ValueError(f"Неизвестный тип комнаты: {room_type!r}")

    objects = []
    for o in data.get("estimated_objects", []):
        if isinstance(o, dict):
            objects.append(ObjectHint(
                name=str(o.get("name", "")),
                quantity=int(o.get("quantity", 1)),
                notes=str(o.get("notes", "")),
            ))

    if not objects:
        raise ValueError("LLM не вернул объекты")

    total = sum(o.quantity for o in objects)
    if total > _MAX_TOTAL_OBJECTS:
        scale = _MAX_TOTAL_OBJECTS / total
        for o in objects:
            o.quantity = max(1, int(o.quantity * scale))

    expanded_description = str(data.get("expanded_description", original_query)).strip() or original_query
    room_half = _parse_room_dim(str(data.get("room_dimensions_hint", "medium (5x5m)")))

    return SceneSpec(
        original_query=original_query,
        expanded_description=expanded_description,
        room_type=room_type,
        room_style=str(data.get("room_style", "modern")),
        estimated_objects=objects,
        room_half_size=room_half,
    )


def _load_expand_prompt_from_file():
    prompt_file = Path(__file__).parent / "prompts" / "expand_prompt.txt"
    if prompt_file.exists():
        return prompt_file.read_text(encoding="utf-8")
    raise FileNotFoundError(f"Файл промпта не найден: {prompt_file}")


_EXPAND_SYSTEM = _load_expand_prompt_from_file()


def expand_prompt(query, prompt_model_fn=None, llm_model=DEFAULT_MODEL, verbose=True):
    llm = prompt_model_fn if prompt_model_fn is not None else _default_request

    if verbose:
        print(f"[prompt_expander] Расширяем запрос: '{query}'")

    raw = llm(_EXPAND_SYSTEM, query, llm_model)

    if not isinstance(raw, dict):
        match = re.search(r"\{.*\}", str(raw), re.DOTALL)
        raw = json.loads(match.group(0)) if match else {}

    spec = _parse_llm_response(raw, original_query=query)

    if verbose:
        print(f"[prompt_expander] Готово: {spec}")

    return spec
