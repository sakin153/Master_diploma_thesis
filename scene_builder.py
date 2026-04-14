from __future__ import annotations

import argparse
import json
import math
import random
import re
import threading
import time
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

try:
    from request import request as llm_request
except Exception:
    llm_request = None


DEFAULT_LLM_TIMEOUT_S = 8.0
DEFAULT_LLM_MAX_ATTEMPTS = 3


@dataclass
class MaterialSpec:
    name: str
    density_kg_m3: float
    friction: Tuple[float, float, float]
    rgba: Tuple[float, float, float, float]


@dataclass
class ObjectTemplate:
    template_name: str
    category: str
    size_m: Tuple[float, float, float]
    material: str
    support: str
    is_static: bool
    solid_fraction: float


@dataclass
class PlannedObject:
    name: str
    template_name: str
    required: bool = True
    confidence: float = 1.0
    relation: Optional[str] = None


@dataclass
class SceneSpec:
    prompt_ru: str
    room_type: str
    room_size_m: Tuple[float, float, float]
    objects: List[PlannedObject]


@dataclass
class PlacedObject:
    name: str
    template_name: str
    category: str
    size_m: Tuple[float, float, float]
    material: str
    support: str
    is_static: bool
    position_m: Tuple[float, float, float]
    yaw_rad: float
    parent: Optional[str]
    mass_kg: float
    inertia_kgm2: Tuple[float, float, float]
    friction: Tuple[float, float, float]
    rgba: Tuple[float, float, float, float]


@dataclass
class ValidationResult:
    success: bool
    issues: List[Dict[str, Any]]
    metrics: Dict[str, float]


MATERIAL_DB: Dict[str, MaterialSpec] = {
    "wood": MaterialSpec("wood", density_kg_m3=650.0, friction=(0.60, 0.005, 0.0001), rgba=(0.55, 0.42, 0.28, 1.0)),
    "steel": MaterialSpec("steel", density_kg_m3=7850.0, friction=(0.45, 0.005, 0.0001), rgba=(0.60, 0.62, 0.65, 1.0)),
    "plastic": MaterialSpec("plastic", density_kg_m3=1150.0, friction=(0.35, 0.005, 0.0001), rgba=(0.22, 0.22, 0.25, 1.0)),
    "rubber": MaterialSpec("rubber", density_kg_m3=1100.0, friction=(1.20, 0.010, 0.0002), rgba=(0.05, 0.05, 0.05, 1.0)),
    "concrete": MaterialSpec("concrete", density_kg_m3=2400.0, friction=(0.90, 0.005, 0.0001), rgba=(0.80, 0.80, 0.80, 1.0)),
}


OBJECT_TEMPLATES: Dict[str, ObjectTemplate] = {
    "table": ObjectTemplate("table", "furniture", (1.40, 0.80, 0.75), "wood", "floor", True, 0.18),
    "chair": ObjectTemplate("chair", "furniture", (0.50, 0.50, 0.90), "wood", "floor", True, 0.16),
    "cabinet": ObjectTemplate("cabinet", "furniture", (1.20, 0.50, 1.80), "wood", "floor", True, 0.14),
    "shelf": ObjectTemplate("shelf", "furniture", (1.00, 0.40, 1.80), "steel", "floor", True, 0.05),
    "workbench": ObjectTemplate("workbench", "furniture", (1.80, 0.70, 0.90), "steel", "floor", True, 0.03),
    "car": ObjectTemplate("car", "vehicle", (4.20, 1.90, 1.60), "steel", "floor", True, 0.012),
    "tire": ObjectTemplate("tire", "vehicle_part", (0.70, 0.70, 0.25), "rubber", "floor", True, 0.12),
    "toolbox": ObjectTemplate("toolbox", "tool", (0.45, 0.25, 0.25), "steel", "surface", False, 0.08),
    "wrench": ObjectTemplate("wrench", "tool", (0.25, 0.05, 0.02), "steel", "surface", False, 0.70),
    "box": ObjectTemplate("box", "storage", (0.55, 0.40, 0.40), "wood", "floor", True, 0.08),
    "crate": ObjectTemplate("crate", "storage", (0.60, 0.45, 0.35), "wood", "floor", True, 0.12),
    "monitor": ObjectTemplate("monitor", "electronics", (0.50, 0.20, 0.35), "plastic", "surface", True, 0.10),
    "computer": ObjectTemplate("computer", "electronics", (0.45, 0.22, 0.35), "plastic", "surface", True, 0.16),
    "robot_arm": ObjectTemplate("robot_arm", "robotics", (0.70, 0.70, 1.10), "steel", "floor", True, 0.05),
    "battery": ObjectTemplate("battery", "electronics", (0.35, 0.25, 0.20), "steel", "surface", False, 0.16),
}


ROOM_PRESETS: Dict[str, Tuple[float, float, float]] = {
    "garage": (8.0, 6.0, 3.5),
    "office": (6.0, 5.0, 3.0),
    "lab": (8.0, 6.0, 3.2),
    "classroom": (9.0, 7.0, 3.2),
    "restaurant": (12.0, 9.0, 3.2),
    "warehouse": (12.0, 8.0, 4.0),
    "generic": (6.0, 6.0, 3.0),
}


ROOM_DEFAULTS: Dict[str, List[Tuple[str, str, bool, float, Optional[str]]]] = {
    "garage": [
        ("car", "center", True, 1.0, "car_1"),
        ("workbench", "near_wall", True, 0.95, "workbench_1"),
    ],
    "office": [
        ("table", "center", True, 1.0, "table_1"),
        ("chair", "near:table_1", True, 0.95, "chair_1"),
    ],
    "lab": [
        ("workbench", "center", True, 1.0, "workbench_1"),
        ("shelf", "near_wall", True, 0.9, "shelf_1"),
    ],
    "classroom": [
        ("table", "center", True, 1.0, "table_1"),
        ("chair", "around:table_1", True, 0.9, "chair_1"),
        ("shelf", "near_wall", False, 0.8, "shelf_1"),
    ],
    "restaurant": [
        ("table", "center", True, 1.0, "table_1"),
        ("chair", "around:table_1", True, 0.95, "chair_1"),
    ],
    "warehouse": [
        ("crate", "center", True, 0.9, "crate_1"),
        ("shelf", "near_wall", True, 0.9, "shelf_1"),
    ],
    "generic": [
        ("table", "center", True, 0.8, "table_1"),
        ("chair", "near:table_1", True, 0.8, "chair_1"),
    ],
}


KEYWORDS_ROOM = {
    "garage": ["гараж", "машина", "автомобиль", "верстак", "шина"],
    "office": ["кабинет", "офис", "рабочее место", "стол", "компьютер"],
    "lab": ["лаборат", "лаборатория", "эксперимент", "робот", "манипулятор"],
    "classroom": ["класс", "аудитория", "школ", "учеб", "парт", "учительск"],
    "restaurant": ["ресторан", "кафе", "обеденн", "столик", "официант", "зал"],
    "warehouse": ["склад", "паллет", "короб", "ящик"],
}


KEYWORDS_OBJECT = {
    "table": ["стол", "столик", "table", "парт"],
    "chair": ["стул", "chair"],
    "workbench": ["верстак", "workbench"],
    "car": ["машин", "автомоб", "car"],
    "tire": ["шин", "колес", "tire"],
    "toolbox": ["инструмент", "ящик с инструментами", "toolbox"],
    "wrench": ["ключ", "гаечн", "wrench"],
    "cabinet": ["шкаф", "cabinet"],
    "shelf": ["полк", "стеллаж", "shelf"],
    "monitor": ["монитор", "экран", "monitor"],
    "computer": ["компьютер", "пк", "computer"],
    "robot_arm": ["манипулятор", "робот", "robot arm"],
    "crate": ["ящик", "короб", "crate"],
    "battery": ["батаре", "аккумулятор", "battery"],
}


TEMPLATE_ALIASES = {
    "автомобиль": "car",
    "машина": "car",
    "car": "car",
    "верстак": "workbench",
    "workbench": "workbench",
    "стол": "table",
    "столик": "table",
    "парта": "table",
    "парт": "table",
    "парты": "table",
    "партой": "table",
    "партами": "table",
    "table": "table",
    "стул": "chair",
    "chair": "chair",
    "шкаф": "cabinet",
    "полка": "shelf",
    "стеллаж": "shelf",
    "монитор": "monitor",
    "компьютер": "computer",
    "ключ": "wrench",
    "гаечный ключ": "wrench",
    "инструменты": "toolbox",
    "toolbox": "toolbox",
    "шина": "tire",
    "ящик": "crate",
    "коробка": "box",
    "манипулятор": "robot_arm",
    "робот": "robot_arm",
    "аккумулятор": "battery",
}


SUPPORT_SURFACES = {"table", "workbench", "cabinet", "shelf", "crate"}


RUSSIAN_NUMBER_WORDS: Dict[str, int] = {
    "ноль": 0,
    "один": 1,
    "одна": 1,
    "одно": 1,
    "одну": 1,
    "одного": 1,
    "одному": 1,
    "одним": 1,
    "одной": 1,
    "два": 2,
    "две": 2,
    "двух": 2,
    "двум": 2,
    "двумя": 2,
    "пара": 2,
    "пару": 2,
    "три": 3,
    "трех": 3,
    "трёх": 3,
    "трем": 3,
    "трём": 3,
    "тремя": 3,
    "четыре": 4,
    "четырех": 4,
    "четырёх": 4,
    "четырем": 4,
    "четырём": 4,
    "четырьмя": 4,
    "пять": 5,
    "пяти": 5,
    "пятью": 5,
    "шесть": 6,
    "шести": 6,
    "шестью": 6,
    "семь": 7,
    "семи": 7,
    "семью": 7,
    "восемь": 8,
    "восьми": 8,
    "восемью": 8,
    "девять": 9,
    "девяти": 9,
    "девятью": 9,
    "десять": 10,
    "десяти": 10,
    "десятью": 10,
    "одиннадцать": 11,
    "одиннадцати": 11,
    "одиннадцатью": 11,
    "двенадцать": 12,
    "двенадцати": 12,
    "двенадцатью": 12,
    "тринадцать": 13,
    "тринадцати": 13,
    "тринадцатью": 13,
    "четырнадцать": 14,
    "четырнадцати": 14,
    "четырнадцатью": 14,
    "пятнадцать": 15,
    "пятнадцати": 15,
    "пятнадцатью": 15,
    "шестнадцать": 16,
    "шестнадцати": 16,
    "шестнадцатью": 16,
    "семнадцать": 17,
    "семнадцати": 17,
    "семнадцатью": 17,
    "восемнадцать": 18,
    "восемнадцати": 18,
    "восемнадцатью": 18,
    "девятнадцать": 19,
    "девятнадцати": 19,
    "девятнадцатью": 19,
    "двадцать": 20,
    "двадцати": 20,
    "двадцатью": 20,
    "несколько": 3,
}

NUMBER_TOKEN_PATTERN = r"(?:\d+|" + "|".join(sorted(RUSSIAN_NUMBER_WORDS.keys(), key=len, reverse=True)) + r")"


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def normalize_room_type(value: Any) -> str:
    if not isinstance(value, str):
        return "generic"
    lowered = value.strip().lower()
    if lowered in ROOM_PRESETS:
        return lowered
    for room_type, keywords in KEYWORDS_ROOM.items():
        if any(k in lowered for k in keywords):
            return room_type
    return "generic"


def normalize_template_name(value: Any) -> Optional[str]:
    if not isinstance(value, str):
        return None
    lowered = value.strip().lower().replace("-", "_").replace(" ", "_")
    lowered = TEMPLATE_ALIASES.get(lowered, lowered)
    if lowered in OBJECT_TEMPLATES:
        return lowered
    # Try alias lookup by partial match to help with slightly noisy model outputs.
    for alias, canonical in TEMPLATE_ALIASES.items():
        if alias in lowered:
            return canonical
    return None


def parse_json_object(text: str) -> Optional[Dict[str, Any]]:
    if not text:
        return None
    stripped = text.strip()
    if not stripped:
        return None

    try:
        obj = json.loads(stripped)
        if isinstance(obj, dict):
            return obj
    except json.JSONDecodeError:
        pass

    decoder = json.JSONDecoder()
    for idx, char in enumerate(stripped):
        if char != "{":
            continue
        try:
            candidate, _ = decoder.raw_decode(stripped[idx:])
            if isinstance(candidate, dict):
                return candidate
        except json.JSONDecodeError:
            continue
    return None


def template_volume(template_name: str) -> float:
    sx, sy, sz = OBJECT_TEMPLATES[template_name].size_m
    return sx * sy * sz


def _count_by_template(objects: Sequence[PlannedObject]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for obj in objects:
        counts[obj.template_name] = counts.get(obj.template_name, 0) + 1
    return counts


def _name_exists(objects: Sequence[PlannedObject], name: str) -> bool:
    return any(obj.name == name for obj in objects)


def make_unique_name(template_name: str, objects: Sequence[PlannedObject], preferred: Optional[str] = None) -> str:
    if preferred and not _name_exists(objects, preferred):
        return preferred
    counts = _count_by_template(objects)
    return f"{template_name}_{counts.get(template_name, 0) + 1}"


def add_planned_object(
    objects: List[PlannedObject],
    template_name: str,
    relation: Optional[str],
    required: bool,
    confidence: float,
    preferred_name: Optional[str] = None,
) -> Optional[str]:
    if template_name not in OBJECT_TEMPLATES:
        return None
    name = make_unique_name(template_name, objects, preferred_name)
    objects.append(
        PlannedObject(
            name=name,
            template_name=template_name,
            required=required,
            confidence=clamp(float(confidence), 0.0, 1.0),
            relation=relation,
        )
    )
    return name


def find_first_by_template(objects: Sequence[PlannedObject], template_name: str) -> Optional[str]:
    for obj in objects:
        if obj.template_name == template_name:
            return obj.name
    return None


def infer_room_type(prompt_ru: str) -> str:
    lowered = prompt_ru.lower()
    best_room = "generic"
    best_score = 0
    for room_type, keywords in KEYWORDS_ROOM.items():
        score = sum(1 for keyword in keywords if keyword in lowered)
        if score > best_score:
            best_score = score
            best_room = room_type
    return best_room


def parse_count_token(token: str) -> Optional[int]:
    normalized = token.strip().lower().strip(".,;:!?()[]{}\"'")
    if not normalized:
        return None
    if normalized.isdigit():
        return int(normalized)
    return RUSSIAN_NUMBER_WORDS.get(normalized)


def infer_classroom_counts(prompt_ru: str) -> Dict[str, Any]:
    lowered = prompt_ru.lower()
    teacher_pattern = re.compile(
        rf"(?P<count>{NUMBER_TOKEN_PATTERN})\s+учительск\w*\s+(?:парт\w*|стол\w*)",
        re.IGNORECASE,
    )
    student_pattern = re.compile(
        rf"(?P<count>{NUMBER_TOKEN_PATTERN})\s+(?:школьн\w*\s+)?(?:парт\w*|стол\w*)",
        re.IGNORECASE,
    )
    chair_pattern = re.compile(
        rf"(?P<count>{NUMBER_TOKEN_PATTERN})\s+стул\w*",
        re.IGNORECASE,
    )

    teacher_count = 0
    for match in teacher_pattern.finditer(lowered):
        parsed = parse_count_token(match.group("count"))
        if parsed is not None:
            teacher_count += parsed

    lowered_without_teacher = teacher_pattern.sub(" ", lowered)

    student_count = 0
    for match in student_pattern.finditer(lowered_without_teacher):
        parsed = parse_count_token(match.group("count"))
        if parsed is not None:
            student_count += parsed

    chair_count = 0
    for match in chair_pattern.finditer(lowered):
        parsed = parse_count_token(match.group("count"))
        if parsed is not None:
            chair_count += parsed

    return {
        "student_desks": student_count,
        "teacher_desks": teacher_count,
        "chairs": chair_count,
        "chairs_mentioned": "стул" in lowered,
    }


def estimate_classroom_room_size(
    current_room_size_m: Tuple[float, float, float],
    desk_count: int,
    chair_count: int,
    teacher_desk_count: int = 0,
) -> Tuple[float, float, float]:
    if desk_count <= 0:
        return current_room_size_m

    min_area = current_room_size_m[0] * current_room_size_m[1]
    estimated_area = desk_count * 3.2 + chair_count * 1.0 + 16.0
    required_area = max(min_area, estimated_area)

    aspect_ratio = 1.35
    room_x = clamp(math.sqrt(required_area * aspect_ratio), 8.0, 30.0)
    room_y = clamp(required_area / room_x, 7.0, 24.0)

    if desk_count >= 10:
        room_x = max(room_x, 12.0)
        room_y = max(room_y, 9.0)
    elif desk_count >= 6:
        room_x = max(room_x, 10.0)
        room_y = max(room_y, 8.0)

    # Keep room sizing consistent with classroom row-layout geometry.
    table_template = OBJECT_TEMPLATES["table"]
    chair_template = OBJECT_TEMPLATES["chair"]
    table_hx = table_template.size_m[0] / 2.0
    table_hy = table_template.size_m[1] / 2.0

    side_margin = 0.8
    wall_margin = 0.55
    rear_margin = 0.85
    teacher_to_students_gap = 1.35
    min_col_spacing = table_template.size_m[0] + 0.55
    min_row_spacing = table_template.size_m[1] + chair_template.size_m[1] + 0.6

    teacher_tables = max(0, int(teacher_desk_count))
    student_tables = max(0, int(desk_count) - teacher_tables)
    if student_tables <= 0:
        student_tables = int(desk_count)

    if student_tables <= 1:
        cols = 1
        rows = 1 if student_tables > 0 else 0
    else:
        initial_cols = min(6, max(2, int(math.ceil(math.sqrt(student_tables * 1.4)))))
        cols = min(max(1, student_tables), initial_cols)
        rows = int(math.ceil(student_tables / cols))

    required_span_x = min_col_spacing * (cols - 1) if cols > 1 else 0.0
    required_room_x = required_span_x + 2.0 * (side_margin + table_hx)

    if teacher_tables > 1:
        teacher_step = table_template.size_m[0] + 0.8
        teacher_span = teacher_step * (teacher_tables - 1)
        required_room_x = max(required_room_x, teacher_span + 2.0 * (side_margin + table_hx))

    required_span_y = min_row_spacing * (rows - 1) if rows > 1 else 0.0
    if teacher_tables > 0:
        base_depth = wall_margin + rear_margin + 2.0 * table_hy + teacher_to_students_gap
    else:
        base_depth = wall_margin + rear_margin + 2.0 * table_hy
    required_room_y = base_depth + required_span_y

    room_x = max(room_x, required_room_x + 0.25)
    room_y = max(room_y, required_room_y + 0.25)
    room_x = clamp(room_x, 8.0, 30.0)
    room_y = clamp(room_y, 7.0, 24.0)

    room_z = max(current_room_size_m[2], 3.2)
    return (room_x, room_y, room_z)


def count_mentions(prompt_ru: str, noun_patterns: Sequence[str]) -> int:
    lowered = prompt_ru.lower()
    total = 0
    for noun_pattern in noun_patterns:
        pattern = re.compile(
            rf"(?P<count>{NUMBER_TOKEN_PATTERN})\s+(?:[а-яё\-]+\s+){{0,3}}(?:{noun_pattern})",
            re.IGNORECASE,
        )
        for match in pattern.finditer(lowered):
            parsed = parse_count_token(match.group("count"))
            if parsed is not None:
                total += parsed
    return total


def infer_restaurant_seats_per_table(prompt_ru: str) -> int:
    lowered = prompt_ru.lower()

    digit_match = re.search(r"(?P<count>\d+)\s*[- ]?местн", lowered)
    if digit_match:
        return int(clamp(float(digit_match.group("count")), 1.0, 12.0))

    seat_words = {
        "двух": 2,
        "двоих": 2,
        "трех": 3,
        "троих": 3,
        "четырех": 4,
        "пяти": 5,
        "шести": 6,
        "семи": 7,
        "восьми": 8,
        "девяти": 9,
        "десяти": 10,
    }

    for stem, value in seat_words.items():
        if re.search(rf"{stem}\w*\s*[- ]?местн", lowered):
            return value

    if "на двоих" in lowered:
        return 2
    if "на троих" in lowered:
        return 3
    if "на четверых" in lowered:
        return 4

    return 2


def infer_restaurant_counts(prompt_ru: str) -> Dict[str, Any]:
    lowered = prompt_ru.lower()
    table_count = count_mentions(prompt_ru, [r"стол\w*", r"столик\w*"])
    chair_count = count_mentions(prompt_ru, [r"стул\w*"])
    seats_per_table = infer_restaurant_seats_per_table(prompt_ru)
    chairs_mentioned = (
        "стул" in lowered
        or "местн" in lowered
        or "на двоих" in lowered
        or "на троих" in lowered
        or "на четверых" in lowered
    )
    return {
        "tables": table_count,
        "chairs": chair_count,
        "seats_per_table": seats_per_table,
        "chairs_mentioned": chairs_mentioned,
    }


def infer_generic_object_counts(prompt_ru: str) -> Dict[str, int]:
    pattern_map: Dict[str, Sequence[str]] = {
        "table": [r"стол\w*", r"столик\w*"],
        "chair": [r"стул\w*"],
        "car": [r"автомоб\w*", r"машин\w*"],
        "workbench": [r"верстак\w*"],
        "shelf": [r"стеллаж\w*", r"полк\w*"],
        "cabinet": [r"шкаф\w*"],
        "crate": [r"ящик\w*", r"короб\w*"],
        "tire": [r"шин\w*", r"колес\w*"],
        "robot_arm": [r"манипулятор\w*"],
    }

    inferred: Dict[str, int] = {}
    for template_name, patterns in pattern_map.items():
        count = count_mentions(prompt_ru, patterns)
        if count > 0:
            inferred[template_name] = int(clamp(float(count), 1.0, 80.0))
    return inferred


def estimate_restaurant_room_size(
    current_room_size_m: Tuple[float, float, float],
    table_count: int,
    chair_count: int,
) -> Tuple[float, float, float]:
    if table_count <= 0:
        return current_room_size_m

    min_area = current_room_size_m[0] * current_room_size_m[1]
    estimated_area = table_count * 6.0 + chair_count * 0.9 + 22.0
    required_area = max(min_area, estimated_area)

    aspect_ratio = 1.4
    room_x = clamp(math.sqrt(required_area * aspect_ratio), 10.0, 30.0)
    room_y = clamp(required_area / room_x, 8.0, 24.0)

    if table_count >= 10:
        room_x = max(room_x, 14.0)
        room_y = max(room_y, 10.0)
    elif table_count >= 6:
        room_x = max(room_x, 12.0)
        room_y = max(room_y, 9.0)

    room_z = max(current_room_size_m[2], 3.2)
    return (room_x, room_y, room_z)


def estimate_generic_room_size_from_objects(
    current_room_size_m: Tuple[float, float, float],
    objects: Sequence[PlannedObject],
) -> Tuple[float, float, float]:
    floor_templates = [OBJECT_TEMPLATES[obj.template_name] for obj in objects if OBJECT_TEMPLATES[obj.template_name].support != "surface"]
    if not floor_templates:
        return current_room_size_m

    min_area = current_room_size_m[0] * current_room_size_m[1]
    estimated_area = 10.0 + sum(template.size_m[0] * template.size_m[1] * 2.0 for template in floor_templates)
    required_area = max(min_area, estimated_area)

    aspect_ratio = 1.25
    room_x = clamp(math.sqrt(required_area * aspect_ratio), current_room_size_m[0], 24.0)
    room_y = clamp(required_area / room_x, current_room_size_m[1], 20.0)
    room_z = max(current_room_size_m[2], 3.0)
    return (room_x, room_y, room_z)


def apply_prompt_quantity_constraints(spec: SceneSpec) -> SceneSpec:
    prompt_lower = spec.prompt_ru.lower()
    room_type = spec.room_type
    objects = list(spec.objects)

    if room_type == "generic" and any(keyword in prompt_lower for keyword in KEYWORDS_ROOM["classroom"]):
        room_type = "classroom"
    elif room_type == "generic" and any(keyword in prompt_lower for keyword in KEYWORDS_ROOM["restaurant"]):
        room_type = "restaurant"

    if room_type == "classroom":
        counts = infer_classroom_counts(spec.prompt_ru)
        student_desks_target = counts["student_desks"]
        teacher_desks_target = counts["teacher_desks"]

        def trim_tables(predicate: Any, target_count: int) -> None:
            if target_count < 0:
                return
            candidates = [obj for obj in objects if obj.template_name == "table" and predicate(obj)]
            if len(candidates) <= target_count:
                return
            keep = sorted(candidates, key=planned_name_sort_key)[:target_count]
            keep_names = {obj.name for obj in keep}
            objects[:] = [
                obj
                for obj in objects
                if not (obj.template_name == "table" and predicate(obj) and obj.name not in keep_names)
            ]

        def trim_chairs(target_count: int) -> None:
            if target_count < 0:
                return
            candidates = [obj for obj in objects if obj.template_name == "chair"]
            if len(candidates) <= target_count:
                return
            keep = sorted(candidates, key=planned_name_sort_key)[:target_count]
            keep_names = {obj.name for obj in keep}
            objects[:] = [
                obj
                for obj in objects
                if not (obj.template_name == "chair" and obj.name not in keep_names)
            ]

        existing_teacher = [obj for obj in objects if obj.template_name == "table" and obj.name.startswith("teacher_desk")]
        existing_student = [obj for obj in objects if obj.template_name == "table" and not obj.name.startswith("teacher_desk")]

        if teacher_desks_target > 0:
            # Prefer promoting existing tables to teacher desks before adding new ones.
            if len(existing_teacher) < teacher_desks_target and existing_student:
                promote_needed = min(teacher_desks_target - len(existing_teacher), len(existing_student))
                promotion_candidates = sorted(
                    existing_student,
                    key=lambda obj: (0 if (obj.relation or "") == "near_wall" else 1, planned_name_sort_key(obj)),
                )
                next_idx = 1
                for promoted in promotion_candidates[:promote_needed]:
                    while _name_exists(objects, f"teacher_desk_{next_idx}"):
                        next_idx += 1
                    promoted.name = f"teacher_desk_{next_idx}"
                    promoted.relation = "near_wall"
                    promoted.required = True
                    promoted.confidence = max(promoted.confidence, 0.98)
                    next_idx += 1

            existing_teacher = [obj for obj in objects if obj.template_name == "table" and obj.name.startswith("teacher_desk")]
            for idx in range(len(existing_teacher) + 1, teacher_desks_target + 1):
                add_planned_object(
                    objects,
                    template_name="table",
                    relation="near_wall",
                    required=True,
                    confidence=0.98,
                    preferred_name=f"teacher_desk_{idx}",
                )

            trim_tables(lambda obj: obj.name.startswith("teacher_desk"), teacher_desks_target)

        if student_desks_target > 0:
            existing_student = [obj for obj in objects if obj.template_name == "table" and not obj.name.startswith("teacher_desk")]
            for idx in range(len(existing_student) + 1, student_desks_target + 1):
                add_planned_object(
                    objects,
                    template_name="table",
                    relation="free",
                    required=True,
                    confidence=0.98,
                    preferred_name=f"desk_{idx}",
                )

            trim_tables(lambda obj: not obj.name.startswith("teacher_desk"), student_desks_target)

        # Re-anchor classroom tables for realistic spread when many desks are present.
        if student_desks_target >= 3:
            for obj in objects:
                if obj.template_name == "table" and not obj.name.startswith("teacher_desk"):
                    obj.relation = "free"
                    obj.required = True

        for obj in objects:
            if obj.template_name == "table" and obj.name.startswith("teacher_desk"):
                obj.relation = "near_wall"
                obj.required = True

        all_desk_names = [obj.name for obj in objects if obj.template_name == "table"]
        target_chairs = counts["chairs"]
        if target_chairs <= 0 and counts["chairs_mentioned"] and (student_desks_target + teacher_desks_target) > 0:
            target_chairs = max(1, student_desks_target + teacher_desks_target)

        if target_chairs > 0:
            existing_chairs = [obj for obj in objects if obj.template_name == "chair"]
            for idx in range(len(existing_chairs) + 1, target_chairs + 1):
                relation = None
                if all_desk_names:
                    desk_name = all_desk_names[(idx - 1) % len(all_desk_names)]
                    relation = f"around:{desk_name}"
                add_planned_object(
                    objects,
                    template_name="chair",
                    relation=relation,
                    required=True,
                    confidence=0.95,
                    preferred_name=f"chair_{idx}",
                )

            trim_chairs(target_chairs)

        table_count = len([obj for obj in objects if obj.template_name == "table"])
        teacher_table_count = len([obj for obj in objects if obj.template_name == "table" and obj.name.startswith("teacher_desk")])
        chair_count = len([obj for obj in objects if obj.template_name == "chair"])
        room_size = estimate_classroom_room_size(
            spec.room_size_m,
            desk_count=table_count,
            chair_count=chair_count,
            teacher_desk_count=teacher_table_count,
        )

        return SceneSpec(
            prompt_ru=spec.prompt_ru,
            room_type=room_type,
            room_size_m=room_size,
            objects=objects,
        )

    if room_type == "restaurant":
        counts = infer_restaurant_counts(spec.prompt_ru)
        tables_target = counts["tables"]
        existing_tables = [obj for obj in objects if obj.template_name == "table"]

        if tables_target <= 0:
            if existing_tables:
                tables_target = len(existing_tables)
            elif "стол" in prompt_lower or "столик" in prompt_lower:
                tables_target = 4

        if tables_target > 0:
            for idx in range(len(existing_tables) + 1, tables_target + 1):
                add_planned_object(
                    objects,
                    template_name="table",
                    relation="free",
                    required=True,
                    confidence=0.98,
                    preferred_name=f"table_{idx}",
                )

        table_names = [obj.name for obj in objects if obj.template_name == "table"]
        for obj in objects:
            if obj.template_name == "table":
                obj.relation = "free"
                obj.required = True

        chairs_target = counts["chairs"]
        if chairs_target <= 0 and table_names and counts["chairs_mentioned"]:
            chairs_target = len(table_names) * counts["seats_per_table"]
        if chairs_target <= 0 and table_names and any(keyword in prompt_lower for keyword in KEYWORDS_ROOM["restaurant"]):
            chairs_target = len(table_names) * counts["seats_per_table"]

        if chairs_target > 0:
            existing_chairs = [obj for obj in objects if obj.template_name == "chair"]
            for idx in range(len(existing_chairs) + 1, chairs_target + 1):
                relation = None
                if table_names:
                    relation = f"around:{table_names[(idx - 1) % len(table_names)]}"
                add_planned_object(
                    objects,
                    template_name="chair",
                    relation=relation,
                    required=True,
                    confidence=0.95,
                    preferred_name=f"chair_{idx}",
                )

        table_count = len([obj for obj in objects if obj.template_name == "table"])
        chair_count = len([obj for obj in objects if obj.template_name == "chair"])
        room_size = estimate_restaurant_room_size(spec.room_size_m, table_count=table_count, chair_count=chair_count)

        return SceneSpec(
            prompt_ru=spec.prompt_ru,
            room_type=room_type,
            room_size_m=room_size,
            objects=objects,
        )

    inferred_counts = infer_generic_object_counts(spec.prompt_ru)
    for template_name, target_count in inferred_counts.items():
        existing = [obj for obj in objects if obj.template_name == template_name]
        if target_count <= len(existing):
            continue

        default_relation = "free" if template_name in {"table", "chair", "car", "workbench", "crate", "tire"} else None
        for idx in range(len(existing) + 1, target_count + 1):
            add_planned_object(
                objects,
                template_name=template_name,
                relation=default_relation,
                required=True,
                confidence=0.92,
                preferred_name=f"{template_name}_{idx}",
            )

    if inferred_counts.get("table", 0) > 0 and inferred_counts.get("chair", 0) == 0 and "местн" in prompt_lower:
        implied_chairs = inferred_counts["table"] * infer_restaurant_seats_per_table(spec.prompt_ru)
        existing_chairs = [obj for obj in objects if obj.template_name == "chair"]
        table_names = [obj.name for obj in objects if obj.template_name == "table"]
        for idx in range(len(existing_chairs) + 1, implied_chairs + 1):
            relation = f"around:{table_names[(idx - 1) % len(table_names)]}" if table_names else None
            add_planned_object(
                objects,
                template_name="chair",
                relation=relation,
                required=True,
                confidence=0.9,
                preferred_name=f"chair_{idx}",
            )

    room_size = estimate_generic_room_size_from_objects(spec.room_size_m, objects)

    return SceneSpec(
        prompt_ru=spec.prompt_ru,
        room_type=room_type,
        room_size_m=room_size,
        objects=objects,
    )


def heuristic_scene_spec(prompt_ru: str) -> SceneSpec:
    room_type = infer_room_type(prompt_ru)
    room_size = ROOM_PRESETS.get(room_type, ROOM_PRESETS["generic"])
    objects: List[PlannedObject] = []
    lowered = prompt_ru.lower()

    for template_name, keywords in KEYWORDS_OBJECT.items():
        if any(keyword in lowered for keyword in keywords):
            add_planned_object(
                objects,
                template_name=template_name,
                relation="center" if template_name in {"car", "table", "workbench"} else None,
                required=True,
                confidence=0.95,
            )

    for template_name, relation, required, confidence, preferred_name in ROOM_DEFAULTS.get(room_type, ROOM_DEFAULTS["generic"]):
        if not find_first_by_template(objects, template_name):
            add_planned_object(
                objects,
                template_name=template_name,
                relation=relation,
                required=required,
                confidence=confidence,
                preferred_name=preferred_name,
            )

    return SceneSpec(
        prompt_ru=prompt_ru,
        room_type=room_type,
        room_size_m=room_size,
        objects=objects,
    )


def llm_scene_spec(
    prompt_ru: str,
    timeout_s: float = DEFAULT_LLM_TIMEOUT_S,
    max_attempts: int = DEFAULT_LLM_MAX_ATTEMPTS,
) -> Optional[SceneSpec]:
    if llm_request is None:
        return None

    timeout_s = max(0.5, float(timeout_s))
    max_attempts = max(1, int(max_attempts))

    templates = ", ".join(sorted(OBJECT_TEMPLATES.keys()))
    room_types = ", ".join(sorted(ROOM_PRESETS.keys()))
    prompt = f"""
Ты проектируешь 3D-сцену для MuJoCo.
Нужно разобрать русский текст и вернуть ТОЛЬКО JSON без пояснений.

Формат JSON:
{{
  "room_type": "один из [{room_types}]",
  "room_size_m": [x, y, z],
  "objects": [
    {{
      "name": "уникальное_имя",
      "template_name": "один из [{templates}]",
      "required": true,
      "confidence": 0.0,
      "relation": "center | near_wall | corner | near:object_name | around:object_name | on:object_name"
    }}
  ],
  "context_objects": [
    {{
      "template_name": "один из [{templates}]",
      "required": false,
      "confidence": 0.7,
      "relation": "on:object_name"
    }}
  ]
}}

Правила:
1) room_size_m должен быть в метрах, реалистичный.
2) Добавь недостающие, но полезные контекстные объекты.
3) Если указываешь relation с object_name, этот object_name должен существовать среди objects.
4) Не выдумывай template_name вне списка.

Текст пользователя:
{prompt_ru}
""".strip()

    def _parse_response_to_spec(response: Any) -> Optional[SceneSpec]:
        parsed = parse_json_object(response)
        if not parsed:
            return None

        room_type = normalize_room_type(parsed.get("room_type"))
        room_size_raw = parsed.get("room_size_m")
        if isinstance(room_size_raw, list) and len(room_size_raw) == 3:
            try:
                room_size = (
                    clamp(float(room_size_raw[0]), 4.0, 20.0),
                    clamp(float(room_size_raw[1]), 4.0, 20.0),
                    clamp(float(room_size_raw[2]), 2.5, 6.0),
                )
            except (TypeError, ValueError):
                room_size = ROOM_PRESETS.get(room_type, ROOM_PRESETS["generic"])
        else:
            room_size = ROOM_PRESETS.get(room_type, ROOM_PRESETS["generic"])

        planned: List[PlannedObject] = []

        def parse_planned(raw_items: Any, default_required: bool) -> None:
            if not isinstance(raw_items, list):
                return
            for raw in raw_items:
                if not isinstance(raw, dict):
                    continue
                template_name = normalize_template_name(raw.get("template_name"))
                if not template_name:
                    continue
                name = raw.get("name")
                if not isinstance(name, str) or not name.strip():
                    name = make_unique_name(template_name, planned)
                if _name_exists(planned, name):
                    name = make_unique_name(template_name, planned)
                relation = raw.get("relation")
                if not isinstance(relation, str):
                    relation = None
                required = bool(raw.get("required", default_required))
                try:
                    confidence_raw = float(raw.get("confidence", 0.85 if required else 0.7))
                except (TypeError, ValueError):
                    confidence_raw = 0.85 if required else 0.7
                planned.append(
                    PlannedObject(
                        name=name,
                        template_name=template_name,
                        required=required,
                        confidence=clamp(confidence_raw, 0.0, 1.0),
                        relation=relation,
                    )
                )

        parse_planned(parsed.get("objects"), default_required=True)
        parse_planned(parsed.get("context_objects"), default_required=False)

        if not planned:
            return None

        return SceneSpec(prompt_ru=prompt_ru, room_type=room_type, room_size_m=room_size, objects=planned)

    for attempt in range(1, max_attempts + 1):
        response_holder: Dict[str, Any] = {"response": None, "error": None}

        def _llm_call() -> None:
            try:
                response_holder["response"] = llm_request(prompt)
            except Exception as exc:
                response_holder["error"] = exc

        worker = threading.Thread(target=_llm_call, daemon=True)
        worker.start()
        worker.join(timeout=timeout_s)

        if worker.is_alive():
            print(f"LLM planning timeout ({attempt}/{max_attempts}) after {timeout_s:.1f}s.")
            continue

        if response_holder["error"] is not None:
            error_name = type(response_holder["error"]).__name__
            print(f"LLM planning error ({attempt}/{max_attempts}): {error_name}.")
            continue

        spec = _parse_response_to_spec(response_holder["response"])
        if spec is not None:
            return spec

        print(f"LLM planning returned invalid JSON/schema ({attempt}/{max_attempts}).")

    print("LLM planning failed after retries.")
    return None


def expand_context(spec: SceneSpec) -> SceneSpec:
    objects = list(spec.objects)
    room_type = spec.room_type
    prompt_lower = spec.prompt_ru.lower()

    for template_name, relation, required, confidence, preferred_name in ROOM_DEFAULTS.get(room_type, ROOM_DEFAULTS["generic"]):
        if find_first_by_template(objects, template_name):
            continue
        add_planned_object(
            objects,
            template_name=template_name,
            relation=relation,
            required=required,
            confidence=confidence,
            preferred_name=preferred_name,
        )

    if room_type == "garage":
        car_name = find_first_by_template(objects, "car")
        workbench_name = find_first_by_template(objects, "workbench")
        if car_name and not find_first_by_template(objects, "tire"):
            add_planned_object(objects, "tire", relation=f"around:{car_name}", required=False, confidence=0.85)
            add_planned_object(objects, "tire", relation=f"around:{car_name}", required=False, confidence=0.82)
        if workbench_name and not find_first_by_template(objects, "toolbox"):
            add_planned_object(objects, "toolbox", relation=f"on:{workbench_name}", required=False, confidence=0.85)
        if workbench_name and not find_first_by_template(objects, "wrench"):
            add_planned_object(objects, "wrench", relation=f"on:{workbench_name}", required=False, confidence=0.84)

    if room_type == "office":
        table_name = find_first_by_template(objects, "table")
        if table_name and not find_first_by_template(objects, "monitor"):
            add_planned_object(objects, "monitor", relation=f"on:{table_name}", required=False, confidence=0.8)
        if table_name and not find_first_by_template(objects, "computer"):
            add_planned_object(objects, "computer", relation=f"on:{table_name}", required=False, confidence=0.8)

    if room_type == "lab":
        workbench_name = find_first_by_template(objects, "workbench")
        if workbench_name and not find_first_by_template(objects, "robot_arm"):
            add_planned_object(objects, "robot_arm", relation=f"near:{workbench_name}", required=False, confidence=0.85)
        if workbench_name and not find_first_by_template(objects, "toolbox"):
            add_planned_object(objects, "toolbox", relation=f"on:{workbench_name}", required=False, confidence=0.75)

    if any(keyword in prompt_lower for keyword in ["инструмент", "ключ", "гаечн"]):
        workbench_name = find_first_by_template(objects, "workbench") or find_first_by_template(objects, "table")
        if workbench_name and not find_first_by_template(objects, "toolbox"):
            add_planned_object(objects, "toolbox", relation=f"on:{workbench_name}", required=False, confidence=0.78)
        if workbench_name and not find_first_by_template(objects, "wrench"):
            add_planned_object(objects, "wrench", relation=f"on:{workbench_name}", required=False, confidence=0.78)

    if any(keyword in prompt_lower for keyword in ["аккумулятор", "батаре"]):
        workbench_name = find_first_by_template(objects, "workbench") or find_first_by_template(objects, "table")
        if workbench_name and not find_first_by_template(objects, "battery"):
            add_planned_object(objects, "battery", relation=f"on:{workbench_name}", required=False, confidence=0.7)

    return SceneSpec(
        prompt_ru=spec.prompt_ru,
        room_type=room_type,
        room_size_m=spec.room_size_m,
        objects=objects,
    )


def build_scene_spec_with_source(
    prompt_ru: str,
    use_llm: bool,
    require_llm: bool = False,
    llm_timeout_s: float = DEFAULT_LLM_TIMEOUT_S,
    llm_max_attempts: int = DEFAULT_LLM_MAX_ATTEMPTS,
) -> Tuple[SceneSpec, str]:
    planning_source = "heuristic"
    parsed = llm_scene_spec(prompt_ru, timeout_s=llm_timeout_s, max_attempts=llm_max_attempts) if use_llm else None
    if parsed is not None:
        planning_source = "llm"
    else:
        if use_llm and require_llm:
            raise RuntimeError("LLM planning failed and heuristic fallback is disabled (--require-llm).")
        if use_llm:
            print("Falling back to heuristic parser.")
        parsed = heuristic_scene_spec(prompt_ru)

    constrained = apply_prompt_quantity_constraints(parsed)
    enriched = expand_context(constrained)
    enriched = apply_prompt_quantity_constraints(enriched)
    return enriched, planning_source


def build_scene_spec(
    prompt_ru: str,
    use_llm: bool,
    require_llm: bool = False,
    llm_timeout_s: float = DEFAULT_LLM_TIMEOUT_S,
    llm_max_attempts: int = DEFAULT_LLM_MAX_ATTEMPTS,
) -> SceneSpec:
    spec, _ = build_scene_spec_with_source(
        prompt_ru=prompt_ru,
        use_llm=use_llm,
        require_llm=require_llm,
        llm_timeout_s=llm_timeout_s,
        llm_max_attempts=llm_max_attempts,
    )
    return spec


def summarize_template_counts(objects: Sequence[Any]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for obj in objects:
        template_name = getattr(obj, "template_name", None)
        if not isinstance(template_name, str):
            continue
        counts[template_name] = counts.get(template_name, 0) + 1
    return dict(sorted(counts.items(), key=lambda item: item[0]))


def format_template_counts(counts: Dict[str, int]) -> str:
    if not counts:
        return "none"
    return ", ".join(f"{name}:{count}" for name, count in counts.items())


def layout_strategy_name(room_type: str) -> str:
    if room_type == "classroom":
        return "classroom_rows"
    if room_type == "restaurant":
        return "restaurant_grid"
    if room_type == "office":
        return "office_grid"
    return "constraint_sampling"


def relation_kind(relation: Optional[str]) -> Tuple[str, Optional[str]]:
    if not relation:
        return "free", None
    rel = relation.strip().lower()
    if rel in {"center", "near_wall", "corner"}:
        return rel, None
    for prefix in ["near:", "around:", "on:"]:
        if rel.startswith(prefix) and len(rel) > len(prefix):
            return prefix[:-1], rel[len(prefix) :]
    return "free", None


def oriented_half_extents(size_m: Tuple[float, float, float], yaw_rad: float) -> Tuple[float, float, float]:
    sx, sy, sz = size_m
    # We only sample 90-degree rotations, so axis-aligned swap is enough.
    quarter_turn = int(round((yaw_rad % (2.0 * math.pi)) / (math.pi / 2.0))) % 4
    if quarter_turn in {1, 3}:
        sx, sy = sy, sx
    return sx / 2.0, sy / 2.0, sz / 2.0


def overlap_3d(
    a_pos: Tuple[float, float, float],
    a_h: Tuple[float, float, float],
    b_pos: Tuple[float, float, float],
    b_h: Tuple[float, float, float],
    margin_xy: float = 0.02,
    margin_z: float = 0.005,
) -> bool:
    ax, ay, az = a_pos
    ahx, ahy, ahz = a_h
    bx, by, bz = b_pos
    bhx, bhy, bhz = b_h
    return (
        abs(ax - bx) < (ahx + bhx - margin_xy)
        and abs(ay - by) < (ahy + bhy - margin_xy)
        and abs(az - bz) < (ahz + bhz - margin_z)
    )


def compute_mass_and_inertia(template: ObjectTemplate) -> Tuple[float, Tuple[float, float, float]]:
    material = MATERIAL_DB[template.material]
    sx, sy, sz = template.size_m
    volume = sx * sy * sz
    mass = material.density_kg_m3 * volume * template.solid_fraction
    ixx = mass * (sy ** 2 + sz ** 2) / 12.0
    iyy = mass * (sx ** 2 + sz ** 2) / 12.0
    izz = mass * (sx ** 2 + sy ** 2) / 12.0
    return mass, (ixx, iyy, izz)


def planned_name_sort_key(planned: PlannedObject) -> Tuple[str, int, str]:
    match = re.search(r"_(\d+)$", planned.name)
    if not match:
        return (planned.name, 10**9, planned.name)
    prefix = planned.name[: match.start()]
    return (prefix, int(match.group(1)), planned.name)


def build_placed_object(
    planned: PlannedObject,
    position_m: Tuple[float, float, float],
    yaw_rad: float,
    parent: Optional[str] = None,
) -> PlacedObject:
    template = OBJECT_TEMPLATES[planned.template_name]
    material = MATERIAL_DB[template.material]
    mass, inertia = compute_mass_and_inertia(template)
    return PlacedObject(
        name=planned.name,
        template_name=template.template_name,
        category=template.category,
        size_m=template.size_m,
        material=template.material,
        support=template.support,
        is_static=template.is_static,
        position_m=position_m,
        yaw_rad=yaw_rad,
        parent=parent,
        mass_kg=mass,
        inertia_kgm2=inertia,
        friction=material.friction,
        rgba=material.rgba,
    )


def place_classroom_objects(spec: SceneSpec, seed: int) -> Optional[List[PlacedObject]]:
    if spec.room_type != "classroom":
        return None

    rng = random.Random(seed)
    room_x, room_y, _ = spec.room_size_m
    half_x, half_y = room_x / 2.0, room_y / 2.0

    floor_objects = [obj for obj in spec.objects if OBJECT_TEMPLATES[obj.template_name].support != "surface"]
    surface_objects = [obj for obj in spec.objects if OBJECT_TEMPLATES[obj.template_name].support == "surface"]

    tables = sorted([obj for obj in floor_objects if obj.template_name == "table"], key=planned_name_sort_key)
    chairs = sorted([obj for obj in floor_objects if obj.template_name == "chair"], key=planned_name_sort_key)
    other_floor = sorted(
        [obj for obj in floor_objects if obj.template_name not in {"table", "chair"}],
        key=lambda item: (not item.required, -template_volume(item.template_name), planned_name_sort_key(item)),
    )

    if not tables:
        return None

    teacher_tables = [obj for obj in tables if obj.name.startswith("teacher_desk")]
    student_tables = [obj for obj in tables if not obj.name.startswith("teacher_desk")]

    if not student_tables:
        student_tables = list(tables)
        teacher_tables = []

    table_template = OBJECT_TEMPLATES["table"]
    chair_template = OBJECT_TEMPLATES["chair"]
    table_hx, table_hy, table_hz = table_template.size_m[0] / 2.0, table_template.size_m[1] / 2.0, table_template.size_m[2] / 2.0
    chair_hx, chair_hy, chair_hz = chair_template.size_m[0] / 2.0, chair_template.size_m[1] / 2.0, chair_template.size_m[2] / 2.0

    side_margin = 0.8
    wall_margin = 0.55
    rear_margin = 0.85
    teacher_to_students_gap = 1.35
    min_col_spacing = table_template.size_m[0] + 0.55
    min_row_spacing = table_template.size_m[1] + chair_template.size_m[1] + 0.6

    usable_x = room_x - 2.0 * (side_margin + table_hx)
    if usable_x <= 0.0:
        raise RuntimeError("Classroom width is too small for table layout")

    teacher_y = half_y - wall_margin - table_hy
    available_top = teacher_y - teacher_to_students_gap if teacher_tables else half_y - wall_margin - table_hy
    available_bottom = -half_y + rear_margin + table_hy
    usable_y = available_top - available_bottom

    if student_tables and usable_y <= 0.0:
        raise RuntimeError("Classroom depth is too small for student desk rows")

    num_students = len(student_tables)
    max_cols_fit = max(1, int(usable_x / min_col_spacing) + 1)
    initial_cols = min(6, max(2, int(math.ceil(math.sqrt(num_students * 1.4))))) if num_students > 1 else 1
    cols = min(max_cols_fit, initial_cols, max(1, num_students))
    rows = int(math.ceil(num_students / cols)) if cols > 0 else 0

    max_rows_fit = max(1, int(usable_y / min_row_spacing) + 1) if num_students > 0 else 1
    while rows > max_rows_fit and cols < min(max_cols_fit, max(1, num_students)):
        cols += 1
        rows = int(math.ceil(num_students / cols))

    if rows > max_rows_fit:
        raise RuntimeError("Classroom layout does not fit room dimensions")

    if cols == 1:
        x_positions = [0.0]
    else:
        needed_span_x = min_col_spacing * (cols - 1)
        span_x = min(needed_span_x, usable_x)
        x_start = -span_x / 2.0
        x_step = span_x / (cols - 1)
        x_positions = [x_start + idx * x_step for idx in range(cols)]

    if rows == 1:
        y_positions = [(available_top + available_bottom) / 2.0]
    else:
        needed_span_y = min_row_spacing * (rows - 1)
        span_y = min(needed_span_y, usable_y)
        y_mid = (available_top + available_bottom) / 2.0
        y_start = y_mid + span_y / 2.0
        y_step = span_y / (rows - 1)
        y_positions = [y_start - ridx * y_step for ridx in range(rows)]

    placed: List[PlacedObject] = []
    placed_by_name: Dict[str, PlacedObject] = {}
    desk_positions: Dict[str, Tuple[float, float, float]] = {}

    if teacher_tables:
        if len(teacher_tables) == 1:
            teacher_x_positions = [0.0]
        else:
            teacher_step = table_template.size_m[0] + 0.8
            teacher_max_span = max(0.0, usable_x)
            teacher_span = min(teacher_step * (len(teacher_tables) - 1), teacher_max_span)
            t_start = -teacher_span / 2.0
            t_step = teacher_span / (len(teacher_tables) - 1)
            teacher_x_positions = [t_start + idx * t_step for idx in range(len(teacher_tables))]

        for idx, obj in enumerate(teacher_tables):
            x = clamp(teacher_x_positions[idx], -half_x + side_margin + table_hx, half_x - side_margin - table_hx)
            y = teacher_y
            placed_obj = build_placed_object(obj, (x, y, table_hz), yaw_rad=0.0)
            placed.append(placed_obj)
            placed_by_name[obj.name] = placed_obj
            desk_positions[obj.name] = (x, y, table_hz)

    for idx, obj in enumerate(student_tables):
        row = idx // cols
        col = idx % cols
        x = x_positions[col]
        y = y_positions[row]
        placed_obj = build_placed_object(obj, (x, y, table_hz), yaw_rad=0.0)
        placed.append(placed_obj)
        placed_by_name[obj.name] = placed_obj
        desk_positions[obj.name] = (x, y, table_hz)

    def chair_candidate_collides(x: float, y: float, ignore_table_name: str) -> bool:
        candidate_h = (chair_hx, chair_hy, chair_hz)
        for other in placed:
            if other.name == ignore_table_name:
                continue
            other_h = oriented_half_extents(other.size_m, other.yaw_rad)
            if overlap_3d((x, y, chair_hz), candidate_h, other.position_m, other_h):
                return True
        return False

    teacher_chair_count = min(len(chairs), len(teacher_tables))
    teacher_chairs = chairs[:teacher_chair_count]
    student_chairs = chairs[teacher_chair_count:]

    for idx, chair_plan in enumerate(teacher_chairs):
        teacher_pos = desk_positions[teacher_tables[idx].name]
        x = teacher_pos[0]
        y = teacher_pos[1] - (table_hy + chair_hy + 0.22)
        min_y = -half_y + rear_margin + chair_hy
        max_y = half_y - wall_margin - chair_hy
        y = clamp(y, min_y, max_y)
        placed_obj = build_placed_object(chair_plan, (x, y, chair_hz), yaw_rad=0.0)
        placed.append(placed_obj)
        placed_by_name[chair_plan.name] = placed_obj

    student_anchor_tables = [obj for obj in student_tables]
    for idx, chair_plan in enumerate(student_chairs):
        if idx < len(student_anchor_tables):
            anchor_name = student_anchor_tables[idx].name
            anchor_pos = desk_positions[anchor_name]
            x = anchor_pos[0]
            south_y = anchor_pos[1] - (table_hy + chair_hy + 0.15)
            north_y = anchor_pos[1] + (table_hy + chair_hy + 0.15)
            min_y = -half_y + rear_margin + chair_hy
            max_y = half_y - wall_margin - chair_hy

            x_min = -half_x + side_margin + chair_hx
            x_max = half_x - side_margin - chair_hx

            candidate_poses: List[Tuple[float, float, float]] = [
                (clamp(x, x_min, x_max), clamp(south_y, min_y, max_y), 0.0),
                (clamp(x, x_min, x_max), clamp(north_y, min_y, max_y), math.pi),
            ]

            # If default slots collide in dense layouts, try slight side shifts near the desk.
            for offset in (-0.18, 0.18, -0.32, 0.32):
                candidate_poses.append((clamp(x + offset, x_min, x_max), clamp(south_y, min_y, max_y), 0.0))
                candidate_poses.append((clamp(x + offset, x_min, x_max), clamp(north_y, min_y, max_y), math.pi))

            chosen_pose = candidate_poses[0]
            for cand_x, cand_y, cand_yaw in candidate_poses:
                if not chair_candidate_collides(cand_x, cand_y, anchor_name):
                    chosen_pose = (cand_x, cand_y, cand_yaw)
                    break
            x, y, yaw = chosen_pose
        else:
            extra_idx = idx - len(student_anchor_tables)
            extra_step = chair_template.size_m[0] + 0.25
            max_line_span = max(0.0, room_x - 2.0 * (side_margin + chair_hx))
            line_capacity = max(1, int(max_line_span / extra_step) + 1)
            row_idx = extra_idx // line_capacity
            col_idx = extra_idx % line_capacity
            if line_capacity == 1:
                x = 0.0
            else:
                line_span = min(max_line_span, extra_step * (line_capacity - 1))
                line_start = -line_span / 2.0
                line_step = line_span / (line_capacity - 1)
                x = line_start + col_idx * line_step
            y = -half_y + rear_margin + chair_hy + row_idx * (chair_template.size_m[1] + 0.15)
            y = clamp(y, -half_y + rear_margin + chair_hy, half_y - wall_margin - chair_hy)
            yaw = 0.0

        placed_obj = build_placed_object(chair_plan, (x, y, chair_hz), yaw_rad=yaw)
        placed.append(placed_obj)
        placed_by_name[chair_plan.name] = placed_obj

    for obj in other_floor:
        placed_obj = place_floor_object(obj, spec.room_size_m, placed, placed_by_name, rng)
        if placed_obj is None:
            if obj.required:
                raise RuntimeError(f"Failed to place required classroom floor object: {obj.name}")
            continue
        placed.append(placed_obj)
        placed_by_name[placed_obj.name] = placed_obj

    for obj in surface_objects:
        placed_obj = place_surface_object(obj, spec.room_size_m, placed, placed_by_name, rng)
        if placed_obj is None:
            if obj.required:
                raise RuntimeError(f"Failed to place required classroom surface object: {obj.name}")
            continue
        placed.append(placed_obj)
        placed_by_name[placed_obj.name] = placed_obj

    return placed


def place_restaurant_objects(spec: SceneSpec, seed: int) -> Optional[List[PlacedObject]]:
    if spec.room_type != "restaurant":
        return None

    rng = random.Random(seed)
    room_x, room_y, _ = spec.room_size_m
    half_x, half_y = room_x / 2.0, room_y / 2.0

    floor_objects = [obj for obj in spec.objects if OBJECT_TEMPLATES[obj.template_name].support != "surface"]
    surface_objects = [obj for obj in spec.objects if OBJECT_TEMPLATES[obj.template_name].support == "surface"]

    tables = sorted([obj for obj in floor_objects if obj.template_name == "table"], key=planned_name_sort_key)
    chairs = sorted([obj for obj in floor_objects if obj.template_name == "chair"], key=planned_name_sort_key)
    other_floor = sorted(
        [obj for obj in floor_objects if obj.template_name not in {"table", "chair"}],
        key=lambda item: (not item.required, -template_volume(item.template_name), planned_name_sort_key(item)),
    )

    if not tables:
        return None

    table_template = OBJECT_TEMPLATES["table"]
    chair_template = OBJECT_TEMPLATES["chair"]
    table_hx, table_hy, table_hz = table_template.size_m[0] / 2.0, table_template.size_m[1] / 2.0, table_template.size_m[2] / 2.0
    chair_hx, chair_hy, chair_hz = chair_template.size_m[0] / 2.0, chair_template.size_m[1] / 2.0, chair_template.size_m[2] / 2.0

    seats_per_table = max(2, min(6, infer_restaurant_seats_per_table(spec.prompt_ru)))

    side_margin = 0.9
    front_margin = 0.9
    rear_margin = 0.9
    chair_gap = 0.2
    chair_ring_x = table_hx + chair_hx + chair_gap
    chair_ring_y = table_hy + chair_hy + chair_gap
    table_extent_x = table_hx if seats_per_table <= 2 else (chair_ring_x + chair_hx)
    table_extent_y = chair_ring_y + chair_hy

    min_col_spacing = table_template.size_m[0] + 0.8
    if seats_per_table > 2:
        min_col_spacing = max(min_col_spacing, 2.0 * chair_ring_x + 2.0 * chair_hx + 0.12)
    min_row_spacing = max(table_template.size_m[1] + 1.0, 2.0 * chair_ring_y + 2.0 * chair_hy + 0.12)

    usable_x = room_x - 2.0 * (side_margin + table_extent_x)
    usable_y = room_y - (front_margin + rear_margin + 2.0 * table_extent_y)
    if usable_x <= 0.0 or usable_y <= 0.0:
        raise RuntimeError("Restaurant room is too small for table layout")

    num_tables = len(tables)
    max_cols_fit = max(1, int(usable_x / min_col_spacing) + 1)
    cols = min(max_cols_fit, max(1, int(math.ceil(math.sqrt(num_tables * 1.3)))))
    rows = int(math.ceil(num_tables / cols))
    max_rows_fit = max(1, int(usable_y / min_row_spacing) + 1)

    while rows > max_rows_fit and cols < min(max_cols_fit, num_tables):
        cols += 1
        rows = int(math.ceil(num_tables / cols))

    if rows > max_rows_fit:
        raise RuntimeError("Restaurant layout does not fit room dimensions")

    if cols == 1:
        x_positions = [0.0]
    else:
        needed_span_x = min_col_spacing * (cols - 1)
        span_x = min(needed_span_x, usable_x)
        x_start = -span_x / 2.0
        x_step = span_x / (cols - 1)
        x_positions = [x_start + idx * x_step for idx in range(cols)]

    top_y = half_y - front_margin - table_extent_y
    bottom_y = -half_y + rear_margin + table_extent_y

    if rows == 1:
        y_positions = [(top_y + bottom_y) / 2.0]
    else:
        needed_span_y = min_row_spacing * (rows - 1)
        span_y = min(needed_span_y, top_y - bottom_y)
        y_start = top_y
        y_step = span_y / (rows - 1)
        y_positions = [y_start - ridx * y_step for ridx in range(rows)]

    placed: List[PlacedObject] = []
    placed_by_name: Dict[str, PlacedObject] = {}
    table_positions: Dict[str, Tuple[float, float, float]] = {}

    for idx, table_plan in enumerate(tables):
        row = idx // cols
        col = idx % cols
        x = clamp(x_positions[col], -half_x + side_margin + table_extent_x, half_x - side_margin - table_extent_x)
        y = clamp(y_positions[row], -half_y + rear_margin + table_extent_y, half_y - front_margin - table_extent_y)
        yaw = 0.0
        placed_obj = build_placed_object(table_plan, (x, y, table_hz), yaw_rad=yaw)
        placed.append(placed_obj)
        placed_by_name[table_plan.name] = placed_obj
        table_positions[table_plan.name] = (x, y, table_hz)

    def chair_slots_for_table(seat_count: int) -> List[Tuple[float, float, float]]:
        slots = [
            (0.0, -chair_ring_y, 0.0),
            (0.0, chair_ring_y, math.pi),
            (-chair_ring_x, 0.0, math.pi / 2.0),
            (chair_ring_x, 0.0, -math.pi / 2.0),
            (-chair_ring_x * 0.82, -chair_ring_y * 0.82, math.pi / 4.0),
            (chair_ring_x * 0.82, -chair_ring_y * 0.82, -math.pi / 4.0),
        ]
        return slots[: max(1, min(len(slots), seat_count))]

    chair_index = 0
    slots = chair_slots_for_table(seats_per_table)
    for table_plan in tables:
        tx, ty, _ = table_positions[table_plan.name]
        for slot_x, slot_y, yaw in slots:
            if chair_index >= len(chairs):
                break
            chair_plan = chairs[chair_index]
            chair_index += 1
            x = clamp(tx + slot_x, -half_x + side_margin + chair_hx, half_x - side_margin - chair_hx)
            y = clamp(ty + slot_y, -half_y + rear_margin + chair_hy, half_y - front_margin - chair_hy)
            placed_obj = build_placed_object(chair_plan, (x, y, chair_hz), yaw_rad=yaw)
            placed.append(placed_obj)
            placed_by_name[chair_plan.name] = placed_obj

    if chair_index < len(chairs):
        extra_chairs = chairs[chair_index:]
        line_step = chair_template.size_m[0] + 0.22
        max_line_span = max(0.0, room_x - 2.0 * (side_margin + chair_hx))
        line_capacity = max(1, int(max_line_span / line_step) + 1)

        for extra_idx, chair_plan in enumerate(extra_chairs):
            row_idx = extra_idx // line_capacity
            col_idx = extra_idx % line_capacity
            if line_capacity == 1:
                x = 0.0
            else:
                line_span = min(max_line_span, line_step * (line_capacity - 1))
                line_start = -line_span / 2.0
                line_step_eff = line_span / (line_capacity - 1)
                x = line_start + col_idx * line_step_eff
            y = -half_y + rear_margin + chair_hy + row_idx * (chair_template.size_m[1] + 0.18)
            y = clamp(y, -half_y + rear_margin + chair_hy, half_y - front_margin - chair_hy)
            placed_obj = build_placed_object(chair_plan, (x, y, chair_hz), yaw_rad=0.0)
            placed.append(placed_obj)
            placed_by_name[chair_plan.name] = placed_obj

    for obj in other_floor:
        placed_obj = place_floor_object(obj, spec.room_size_m, placed, placed_by_name, rng)
        if placed_obj is None:
            if obj.required:
                raise RuntimeError(f"Failed to place required restaurant floor object: {obj.name}")
            continue
        placed.append(placed_obj)
        placed_by_name[placed_obj.name] = placed_obj

    for obj in surface_objects:
        placed_obj = place_surface_object(obj, spec.room_size_m, placed, placed_by_name, rng)
        if placed_obj is None:
            if obj.required:
                raise RuntimeError(f"Failed to place required restaurant surface object: {obj.name}")
            continue
        placed.append(placed_obj)
        placed_by_name[placed_obj.name] = placed_obj

    return placed


def place_office_objects(spec: SceneSpec, seed: int) -> Optional[List[PlacedObject]]:
    if spec.room_type != "office":
        return None

    rng = random.Random(seed)
    room_x, room_y, _ = spec.room_size_m
    half_x, half_y = room_x / 2.0, room_y / 2.0

    floor_objects = [obj for obj in spec.objects if OBJECT_TEMPLATES[obj.template_name].support != "surface"]
    surface_objects = [obj for obj in spec.objects if OBJECT_TEMPLATES[obj.template_name].support == "surface"]

    tables = sorted([obj for obj in floor_objects if obj.template_name == "table"], key=planned_name_sort_key)
    chairs = sorted([obj for obj in floor_objects if obj.template_name == "chair"], key=planned_name_sort_key)
    other_floor = sorted(
        [obj for obj in floor_objects if obj.template_name not in {"table", "chair"}],
        key=lambda item: (not item.required, -template_volume(item.template_name), planned_name_sort_key(item)),
    )

    if not tables:
        return None

    table_template = OBJECT_TEMPLATES["table"]
    chair_template = OBJECT_TEMPLATES["chair"]
    table_hx, table_hy, table_hz = table_template.size_m[0] / 2.0, table_template.size_m[1] / 2.0, table_template.size_m[2] / 2.0
    chair_hx, chair_hy, chair_hz = chair_template.size_m[0] / 2.0, chair_template.size_m[1] / 2.0, chair_template.size_m[2] / 2.0

    side_margin = 0.75
    front_margin = 0.75
    rear_margin = 0.8
    min_col_spacing = table_template.size_m[0] + 1.0
    min_row_spacing = table_template.size_m[1] + chair_template.size_m[1] + 0.9

    usable_x = room_x - 2.0 * (side_margin + table_hx)
    usable_y = room_y - (front_margin + rear_margin + 2.0 * table_hy)
    if usable_x <= 0.0 or usable_y <= 0.0:
        raise RuntimeError("Office room is too small for table layout")

    num_tables = len(tables)
    max_cols_fit = max(1, int(usable_x / min_col_spacing) + 1)
    cols = min(max_cols_fit, max(1, int(math.ceil(math.sqrt(num_tables * 1.2)))))
    rows = int(math.ceil(num_tables / cols))
    max_rows_fit = max(1, int(usable_y / min_row_spacing) + 1)

    while rows > max_rows_fit and cols < min(max_cols_fit, num_tables):
        cols += 1
        rows = int(math.ceil(num_tables / cols))

    if rows > max_rows_fit:
        raise RuntimeError("Office layout does not fit room dimensions")

    if cols == 1:
        x_positions = [0.0]
    else:
        needed_span_x = min_col_spacing * (cols - 1)
        span_x = min(needed_span_x, usable_x)
        x_start = -span_x / 2.0
        x_step = span_x / (cols - 1)
        x_positions = [x_start + idx * x_step for idx in range(cols)]

    top_y = half_y - front_margin - table_hy
    bottom_y = -half_y + rear_margin + table_hy
    if rows == 1:
        y_positions = [(top_y + bottom_y) / 2.0]
    else:
        needed_span_y = min_row_spacing * (rows - 1)
        span_y = min(needed_span_y, top_y - bottom_y)
        y_start = top_y
        y_step = span_y / (rows - 1)
        y_positions = [y_start - ridx * y_step for ridx in range(rows)]

    placed: List[PlacedObject] = []
    placed_by_name: Dict[str, PlacedObject] = {}
    table_positions: Dict[str, Tuple[float, float, float]] = {}

    for idx, table_plan in enumerate(tables):
        row = idx // cols
        col = idx % cols
        x = clamp(x_positions[col], -half_x + side_margin + table_hx, half_x - side_margin - table_hx)
        y = clamp(y_positions[row], -half_y + rear_margin + table_hy, half_y - front_margin - table_hy)
        placed_obj = build_placed_object(table_plan, (x, y, table_hz), yaw_rad=0.0)
        placed.append(placed_obj)
        placed_by_name[table_plan.name] = placed_obj
        table_positions[table_plan.name] = (x, y, table_hz)

    south_slot = (0.0, -(table_hy + chair_hy + 0.18), 0.0)
    north_slot = (0.0, (table_hy + chair_hy + 0.18), math.pi)
    west_slot = (-(table_hx + chair_hx + 0.15), 0.0, math.pi / 2.0)
    east_slot = ((table_hx + chair_hx + 0.15), 0.0, -math.pi / 2.0)

    table_seq = [table_plan.name for table_plan in tables]
    slots_per_table: Dict[str, List[Tuple[float, float, float]]] = {}
    for table_name in table_seq:
        tx, ty, _ = table_positions[table_name]
        south_space = (ty + south_slot[1]) - (-half_y + rear_margin + chair_hy)
        north_space = (half_y - front_margin - chair_hy) - (ty + north_slot[1])
        if south_space >= north_space:
            slots_per_table[table_name] = [south_slot, north_slot, west_slot, east_slot]
        else:
            slots_per_table[table_name] = [north_slot, south_slot, west_slot, east_slot]

    for idx, chair_plan in enumerate(chairs):
        table_idx = idx % len(table_seq)
        slot_idx = idx // len(table_seq)
        table_name = table_seq[table_idx]
        chair_slots = slots_per_table[table_name]
        if slot_idx >= len(chair_slots):
            break
        tx, ty, _ = table_positions[table_name]
        sx, sy, yaw = chair_slots[slot_idx]
        x = clamp(tx + sx, -half_x + side_margin + chair_hx, half_x - side_margin - chair_hx)
        y = clamp(ty + sy, -half_y + rear_margin + chair_hy, half_y - front_margin - chair_hy)
        placed_obj = build_placed_object(chair_plan, (x, y, chair_hz), yaw_rad=yaw)
        placed.append(placed_obj)
        placed_by_name[chair_plan.name] = placed_obj

    # Any extra chairs are placed as reserve along rear wall.
    used_chairs = min(len(chairs), 4 * len(table_seq))
    if used_chairs < len(chairs):
        extra = chairs[used_chairs:]
        step = chair_template.size_m[0] + 0.25
        line_span = max(0.0, room_x - 2.0 * (side_margin + chair_hx))
        capacity = max(1, int(line_span / step) + 1)
        for i, chair_plan in enumerate(extra):
            row = i // capacity
            col = i % capacity
            if capacity == 1:
                x = 0.0
            else:
                used_span = min(line_span, step * (capacity - 1))
                x = -used_span / 2.0 + col * (used_span / (capacity - 1))
            y = -half_y + rear_margin + chair_hy + row * (chair_template.size_m[1] + 0.16)
            y = clamp(y, -half_y + rear_margin + chair_hy, half_y - front_margin - chair_hy)
            placed_obj = build_placed_object(chair_plan, (x, y, chair_hz), yaw_rad=0.0)
            placed.append(placed_obj)
            placed_by_name[chair_plan.name] = placed_obj

    for obj in other_floor:
        placed_obj = place_floor_object(obj, spec.room_size_m, placed, placed_by_name, rng)
        if placed_obj is None:
            if obj.required:
                raise RuntimeError(f"Failed to place required office floor object: {obj.name}")
            continue
        placed.append(placed_obj)
        placed_by_name[placed_obj.name] = placed_obj

    for obj in surface_objects:
        placed_obj = place_surface_object(obj, spec.room_size_m, placed, placed_by_name, rng)
        if placed_obj is None:
            if obj.required:
                raise RuntimeError(f"Failed to place required office surface object: {obj.name}")
            continue
        placed.append(placed_obj)
        placed_by_name[placed_obj.name] = placed_obj

    return placed


def sample_floor_pose(
    obj_template: ObjectTemplate,
    relation: Optional[str],
    room_size_m: Tuple[float, float, float],
    placed_by_name: Dict[str, PlacedObject],
    rng: random.Random,
) -> Tuple[float, float, float, float]:
    relation_type, relation_target = relation_kind(relation)
    yaw = rng.choice([0.0, math.pi / 2.0, math.pi, 3.0 * math.pi / 2.0])
    hx, hy, hz = oriented_half_extents(obj_template.size_m, yaw)
    room_x, room_y, _ = room_size_m
    limit_x = room_x / 2.0 - hx - 0.08
    limit_y = room_y / 2.0 - hy - 0.08

    if limit_x <= 0 or limit_y <= 0:
        raise ValueError("Object does not fit into the room dimensions")

    target_obj = placed_by_name.get(relation_target) if relation_target else None

    if relation_type == "center":
        x = rng.uniform(-min(0.7, limit_x), min(0.7, limit_x))
        y = rng.uniform(-min(0.7, limit_y), min(0.7, limit_y))
    elif relation_type == "near_wall":
        side = rng.choice(["north", "south", "east", "west"])
        if side == "north":
            x = rng.uniform(-limit_x, limit_x)
            y = limit_y - rng.uniform(0.02, min(0.4, limit_y))
        elif side == "south":
            x = rng.uniform(-limit_x, limit_x)
            y = -limit_y + rng.uniform(0.02, min(0.4, limit_y))
        elif side == "east":
            x = limit_x - rng.uniform(0.02, min(0.4, limit_x))
            y = rng.uniform(-limit_y, limit_y)
        else:
            x = -limit_x + rng.uniform(0.02, min(0.4, limit_x))
            y = rng.uniform(-limit_y, limit_y)
    elif relation_type in {"near", "around"} and target_obj is not None:
        target_x, target_y, _ = target_obj.position_m
        target_hx, target_hy, _ = oriented_half_extents(target_obj.size_m, target_obj.yaw_rad)
        near_radius = max(target_hx + hx, target_hy + hy)
        if relation_type == "near":
            radius = near_radius + rng.uniform(0.20, 0.55)
        else:
            radius = near_radius + rng.uniform(0.45, 1.20)
        angle = rng.uniform(0.0, 2.0 * math.pi)
        x = target_x + radius * math.cos(angle)
        y = target_y + radius * math.sin(angle)
        x = clamp(x, -limit_x, limit_x)
        y = clamp(y, -limit_y, limit_y)
    elif relation_type == "corner":
        sign_x = rng.choice([-1.0, 1.0])
        sign_y = rng.choice([-1.0, 1.0])
        x = sign_x * (limit_x - rng.uniform(0.0, min(0.4, limit_x)))
        y = sign_y * (limit_y - rng.uniform(0.0, min(0.4, limit_y)))
    else:
        x = rng.uniform(-limit_x, limit_x)
        y = rng.uniform(-limit_y, limit_y)

    z = hz
    return x, y, z, yaw


def place_floor_object(
    planned: PlannedObject,
    room_size_m: Tuple[float, float, float],
    placed: List[PlacedObject],
    placed_by_name: Dict[str, PlacedObject],
    rng: random.Random,
) -> Optional[PlacedObject]:
    template = OBJECT_TEMPLATES[planned.template_name]
    material = MATERIAL_DB[template.material]
    mass, inertia = compute_mass_and_inertia(template)

    for _ in range(120):
        x, y, z, yaw = sample_floor_pose(template, planned.relation, room_size_m, placed_by_name, rng)
        candidate_h = oriented_half_extents(template.size_m, yaw)
        collision = False
        for other in placed:
            other_h = oriented_half_extents(other.size_m, other.yaw_rad)
            if overlap_3d((x, y, z), candidate_h, other.position_m, other_h):
                collision = True
                break
        if collision:
            continue

        return PlacedObject(
            name=planned.name,
            template_name=template.template_name,
            category=template.category,
            size_m=template.size_m,
            material=template.material,
            support=template.support,
            is_static=template.is_static,
            position_m=(x, y, z),
            yaw_rad=yaw,
            parent=None,
            mass_kg=mass,
            inertia_kgm2=inertia,
            friction=material.friction,
            rgba=material.rgba,
        )

    return None


def pick_support_candidates(
    planned: PlannedObject,
    placed_by_name: Dict[str, PlacedObject],
) -> List[PlacedObject]:
    relation_type, relation_target = relation_kind(planned.relation)
    if relation_type == "on" and relation_target in placed_by_name:
        target = placed_by_name[relation_target]
        if target.template_name in SUPPORT_SURFACES:
            return [target]

    supports: List[PlacedObject] = []
    for obj in placed_by_name.values():
        if obj.template_name in SUPPORT_SURFACES:
            supports.append(obj)
    supports.sort(key=lambda item: item.size_m[0] * item.size_m[1], reverse=True)
    return supports


def place_surface_object(
    planned: PlannedObject,
    room_size_m: Tuple[float, float, float],
    placed: List[PlacedObject],
    placed_by_name: Dict[str, PlacedObject],
    rng: random.Random,
) -> Optional[PlacedObject]:
    template = OBJECT_TEMPLATES[planned.template_name]
    material = MATERIAL_DB[template.material]
    mass, inertia = compute_mass_and_inertia(template)
    candidates = pick_support_candidates(planned, placed_by_name)

    if not candidates:
        # Fallback: place it on floor if there is no suitable support object.
        fallback = PlannedObject(
            name=planned.name,
            template_name=planned.template_name,
            required=planned.required,
            confidence=planned.confidence,
            relation="near_wall",
        )
        return place_floor_object(fallback, room_size_m, placed, placed_by_name, rng)

    for parent in candidates:
        parent_hx, parent_hy, parent_hz = oriented_half_extents(parent.size_m, parent.yaw_rad)
        for _ in range(80):
            yaw = rng.choice([0.0, math.pi / 2.0, math.pi, 3.0 * math.pi / 2.0])
            hx, hy, hz = oriented_half_extents(template.size_m, yaw)
            margin = 0.03

            if hx >= parent_hx - margin or hy >= parent_hy - margin:
                continue

            x = parent.position_m[0] + rng.uniform(-(parent_hx - hx - margin), parent_hx - hx - margin)
            y = parent.position_m[1] + rng.uniform(-(parent_hy - hy - margin), parent_hy - hy - margin)
            z = parent.position_m[2] + parent_hz + hz + 0.01

            candidate_h = (hx, hy, hz)
            collision = False
            for other in placed:
                if other.name == parent.name:
                    continue
                other_h = oriented_half_extents(other.size_m, other.yaw_rad)
                if overlap_3d((x, y, z), candidate_h, other.position_m, other_h):
                    collision = True
                    break
            if collision:
                continue

            return PlacedObject(
                name=planned.name,
                template_name=template.template_name,
                category=template.category,
                size_m=template.size_m,
                material=template.material,
                support=template.support,
                is_static=template.is_static,
                position_m=(x, y, z),
                yaw_rad=yaw,
                parent=parent.name,
                mass_kg=mass,
                inertia_kgm2=inertia,
                friction=material.friction,
                rgba=material.rgba,
            )

    return None


def place_objects(spec: SceneSpec, seed: int) -> List[PlacedObject]:
    classroom_layout = place_classroom_objects(spec, seed)
    if classroom_layout is not None:
        return classroom_layout

    restaurant_layout = place_restaurant_objects(spec, seed)
    if restaurant_layout is not None:
        return restaurant_layout

    office_layout = place_office_objects(spec, seed)
    if office_layout is not None:
        return office_layout

    rng = random.Random(seed)

    floor_objects = [obj for obj in spec.objects if OBJECT_TEMPLATES[obj.template_name].support != "surface"]
    surface_objects = [obj for obj in spec.objects if OBJECT_TEMPLATES[obj.template_name].support == "surface"]

    floor_objects.sort(key=lambda item: (not item.required, -template_volume(item.template_name)))
    surface_objects.sort(key=lambda item: (not item.required, -template_volume(item.template_name)))

    placed: List[PlacedObject] = []
    placed_by_name: Dict[str, PlacedObject] = {}

    for obj in floor_objects:
        placed_obj = place_floor_object(obj, spec.room_size_m, placed, placed_by_name, rng)
        if placed_obj is None:
            if obj.required:
                raise RuntimeError(f"Failed to place required floor object: {obj.name}")
            continue
        placed.append(placed_obj)
        placed_by_name[placed_obj.name] = placed_obj

    for obj in surface_objects:
        placed_obj = place_surface_object(obj, spec.room_size_m, placed, placed_by_name, rng)
        if placed_obj is None:
            if obj.required:
                raise RuntimeError(f"Failed to place required surface object: {obj.name}")
            continue
        placed.append(placed_obj)
        placed_by_name[placed_obj.name] = placed_obj

    return placed


def validate_scene(spec: SceneSpec, placed: Sequence[PlacedObject]) -> ValidationResult:
    issues: List[Dict[str, Any]] = []
    room_x, room_y, _ = spec.room_size_m

    placed_by_name = {obj.name: obj for obj in placed}

    for obj in placed:
        hx, hy, hz = oriented_half_extents(obj.size_m, obj.yaw_rad)
        x, y, z = obj.position_m
        if abs(x) + hx > room_x / 2.0 - 0.001 or abs(y) + hy > room_y / 2.0 - 0.001 or z - hz < -0.001:
            issues.append({"type": "boundary", "object": obj.name, "details": "Object is out of room bounds"})

        if obj.support == "surface":
            if not obj.parent:
                issues.append({"type": "support", "object": obj.name, "details": "Surface object has no parent support"})
            elif obj.parent not in placed_by_name:
                issues.append({"type": "support", "object": obj.name, "details": "Parent support not found in placed objects"})

    placed_list = list(placed)
    for idx in range(len(placed_list)):
        for jdx in range(idx + 1, len(placed_list)):
            a = placed_list[idx]
            b = placed_list[jdx]
            if a.parent == b.name or b.parent == a.name:
                continue
            ah = oriented_half_extents(a.size_m, a.yaw_rad)
            bh = oriented_half_extents(b.size_m, b.yaw_rad)
            if overlap_3d(a.position_m, ah, b.position_m, bh):
                issues.append({"type": "collision", "objects": [a.name, b.name], "details": "Objects overlap"})

    metrics = {
        "num_objects": float(len(placed)),
        "num_issues": float(len(issues)),
        "room_area_m2": float(spec.room_size_m[0] * spec.room_size_m[1]),
    }
    return ValidationResult(success=len(issues) == 0, issues=issues, metrics=metrics)


def generate_placements_with_repair(
    spec: SceneSpec,
    seed: int,
    max_attempts: int = 30,
) -> Tuple[List[PlacedObject], ValidationResult, ValidationResult, bool]:
    first_validation: Optional[ValidationResult] = None
    last_validation: Optional[ValidationResult] = None
    last_exception: Optional[Exception] = None

    for attempt in range(max_attempts):
        current_seed = seed + attempt
        try:
            placed = place_objects(spec, current_seed)
            validation = validate_scene(spec, placed)
            if attempt == 0:
                first_validation = validation
            last_validation = validation
            if validation.success:
                used_repair = attempt > 0
                return placed, first_validation or validation, validation, used_repair
        except Exception as exc:
            last_exception = exc
            if attempt == 0:
                first_validation = ValidationResult(
                    success=False,
                    issues=[{"type": "placement", "details": str(exc)}],
                    metrics={"num_objects": 0.0, "num_issues": 1.0, "room_area_m2": spec.room_size_m[0] * spec.room_size_m[1]},
                )

    if last_validation is not None:
        raise RuntimeError(f"Failed to generate collision-free scene after {max_attempts} attempts: {last_validation.issues}")
    if last_exception is not None:
        raise RuntimeError(f"Failed to place scene objects: {last_exception}")
    raise RuntimeError("Unknown scene generation failure")


def _fmt(values: Iterable[float]) -> str:
    return " ".join(f"{value:.6f}" for value in values)


def _write_xml(root: ET.Element, path: Path) -> None:
    tree = ET.ElementTree(root)
    ET.indent(tree, space="  ")
    tree.write(path, encoding="utf-8", xml_declaration=True)


def export_mjcf(spec: SceneSpec, placed: Sequence[PlacedObject], output_path: Path) -> None:
    room_x, room_y, room_z = spec.room_size_m
    root = ET.Element("mujoco", {"model": f"scene_{spec.room_type}"})
    ET.SubElement(root, "compiler", {"angle": "radian", "coordinate": "local"})
    ET.SubElement(root, "option", {"gravity": "0 0 -9.81", "timestep": "0.002"})

    asset = ET.SubElement(root, "asset")
    ET.SubElement(asset, "material", {"name": "floor_mat", "rgba": _fmt((0.80, 0.80, 0.80, 1.00))})

    unique_materials = sorted({obj.material for obj in placed})
    for material_name in unique_materials:
        material = MATERIAL_DB[material_name]
        ET.SubElement(
            asset,
            "material",
            {
                "name": f"mat_{material_name}",
                "rgba": _fmt(material.rgba),
                "specular": "0.15",
                "shininess": "0.4",
            },
        )

    worldbody = ET.SubElement(root, "worldbody")
    ET.SubElement(
        worldbody,
        "light",
        {
            "name": "key_light",
            "pos": _fmt((0.0, 0.0, room_z + 1.5)),
            "directional": "false",
            "diffuse": _fmt((1.1, 1.1, 1.1)),
            "specular": _fmt((0.2, 0.2, 0.2)),
        },
    )

    ET.SubElement(
        worldbody,
        "geom",
        {
            "name": "floor",
            "type": "plane",
            "size": _fmt((room_x / 2.0, room_y / 2.0, 0.1)),
            "material": "floor_mat",
            "friction": _fmt(MATERIAL_DB["concrete"].friction),
        },
    )

    wall_thickness = 0.05
    wall_height = room_z / 2.0
    ET.SubElement(
        worldbody,
        "geom",
        {
            "name": "wall_north",
            "type": "box",
            "pos": _fmt((0.0, room_y / 2.0, wall_height)),
            "size": _fmt((room_x / 2.0, wall_thickness, wall_height)),
            "rgba": _fmt((0.92, 0.92, 0.92, 1.0)),
            "contype": "1",
            "conaffinity": "1",
        },
    )
    ET.SubElement(
        worldbody,
        "geom",
        {
            "name": "wall_south",
            "type": "box",
            "pos": _fmt((0.0, -room_y / 2.0, wall_height)),
            "size": _fmt((room_x / 2.0, wall_thickness, wall_height)),
            "rgba": _fmt((0.92, 0.92, 0.92, 1.0)),
            "contype": "1",
            "conaffinity": "1",
        },
    )
    ET.SubElement(
        worldbody,
        "geom",
        {
            "name": "wall_east",
            "type": "box",
            "pos": _fmt((room_x / 2.0, 0.0, wall_height)),
            "size": _fmt((wall_thickness, room_y / 2.0, wall_height)),
            "rgba": _fmt((0.92, 0.92, 0.92, 1.0)),
            "contype": "1",
            "conaffinity": "1",
        },
    )
    ET.SubElement(
        worldbody,
        "geom",
        {
            "name": "wall_west",
            "type": "box",
            "pos": _fmt((-room_x / 2.0, 0.0, wall_height)),
            "size": _fmt((wall_thickness, room_y / 2.0, wall_height)),
            "rgba": _fmt((0.92, 0.92, 0.92, 1.0)),
            "contype": "1",
            "conaffinity": "1",
        },
    )

    for obj in placed:
        body = ET.SubElement(
            worldbody,
            "body",
            {
                "name": obj.name,
                "pos": _fmt(obj.position_m),
                "euler": _fmt((0.0, 0.0, obj.yaw_rad)),
            },
        )
        if not obj.is_static:
            ET.SubElement(body, "freejoint")

        ET.SubElement(
            body,
            "inertial",
            {
                "pos": _fmt((0.0, 0.0, 0.0)),
                "mass": f"{obj.mass_kg:.6f}",
                "diaginertia": _fmt(obj.inertia_kgm2),
            },
        )

        hx, hy, hz = (obj.size_m[0] / 2.0, obj.size_m[1] / 2.0, obj.size_m[2] / 2.0)
        ET.SubElement(
            body,
            "geom",
            {
                "name": f"{obj.name}_geom",
                "type": "box",
                "size": _fmt((hx, hy, hz)),
                "material": f"mat_{obj.material}",
                "friction": _fmt(obj.friction),
                "contype": "1",
                "conaffinity": "1",
            },
        )

    _write_xml(root, output_path)


def export_urdf(spec: SceneSpec, placed: Sequence[PlacedObject], output_path: Path) -> None:
    root = ET.Element("robot", {"name": f"scene_{spec.room_type}"})
    ET.SubElement(root, "link", {"name": "world"})

    for obj in placed:
        link = ET.SubElement(root, "link", {"name": obj.name})
        inertial = ET.SubElement(link, "inertial")
        ET.SubElement(inertial, "origin", {"xyz": "0 0 0", "rpy": "0 0 0"})
        ET.SubElement(inertial, "mass", {"value": f"{obj.mass_kg:.6f}"})
        ET.SubElement(
            inertial,
            "inertia",
            {
                "ixx": f"{obj.inertia_kgm2[0]:.6f}",
                "iyy": f"{obj.inertia_kgm2[1]:.6f}",
                "izz": f"{obj.inertia_kgm2[2]:.6f}",
                "ixy": "0",
                "ixz": "0",
                "iyz": "0",
            },
        )

        visual = ET.SubElement(link, "visual")
        ET.SubElement(visual, "origin", {"xyz": "0 0 0", "rpy": "0 0 0"})
        geometry_v = ET.SubElement(visual, "geometry")
        ET.SubElement(geometry_v, "box", {"size": _fmt(obj.size_m)})

        collision = ET.SubElement(link, "collision")
        ET.SubElement(collision, "origin", {"xyz": "0 0 0", "rpy": "0 0 0"})
        geometry_c = ET.SubElement(collision, "geometry")
        ET.SubElement(geometry_c, "box", {"size": _fmt(obj.size_m)})

        joint = ET.SubElement(root, "joint", {"name": f"world_to_{obj.name}", "type": "fixed"})
        ET.SubElement(joint, "parent", {"link": "world"})
        ET.SubElement(joint, "child", {"link": obj.name})
        ET.SubElement(
            joint,
            "origin",
            {
                "xyz": _fmt(obj.position_m),
                "rpy": _fmt((0.0, 0.0, obj.yaw_rad)),
            },
        )

    _write_xml(root, output_path)


def write_manifest(
    spec: SceneSpec,
    placed: Sequence[PlacedObject],
    validation_before: ValidationResult,
    validation_after: ValidationResult,
    used_repair: bool,
    output_dir: Path,
    seed: int,
    planning_source: str,
    layout_strategy: str,
) -> None:
    manifest = {
        "pipeline": {
            "planning_source": planning_source,
            "layout_strategy": layout_strategy,
        },
        "spec": {
            "prompt_ru": spec.prompt_ru,
            "room_type": spec.room_type,
            "room_size_m": list(spec.room_size_m),
            "objects": [asdict(obj) for obj in spec.objects],
        },
        "placed_objects": [asdict(obj) for obj in placed],
        "validation_before_repair": asdict(validation_before),
        "validation_after_repair": asdict(validation_after),
        "used_repair": used_repair,
        "seed": seed,
        "output_dir": str(output_dir),
        "files": {
            "mjcf": str(output_dir / "scene.xml"),
            "urdf": str(output_dir / "scene.urdf"),
        },
    }
    (output_dir / "scene_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def run_mujoco_viewer(scene_xml_path: Path, duration_s: float) -> None:
    import mujoco
    import mujoco.viewer

    model = mujoco.MjModel.from_xml_path(str(scene_xml_path))
    data = mujoco.MjData(model)

    with mujoco.viewer.launch_passive(model, data) as viewer:
        start = time.time()
        while viewer.is_running() and (duration_s <= 0.0 or (time.time() - start) < duration_s):
            step_start = time.time()
            mujoco.mj_step(model, data)
            viewer.sync()
            to_sleep = model.opt.timestep - (time.time() - step_start)
            if to_sleep > 0:
                time.sleep(to_sleep)


def save_scene(
    prompt_ru: str,
    use_llm: bool,
    output_root: Path,
    seed: int,
    verbose: bool = False,
    require_llm: bool = False,
    llm_timeout_s: float = DEFAULT_LLM_TIMEOUT_S,
    llm_max_attempts: int = DEFAULT_LLM_MAX_ATTEMPTS,
) -> Dict[str, Any]:
    if verbose:
        print("[1/4] Отправляем промпт и строим план сцены...")

    spec, planning_source = build_scene_spec_with_source(
        prompt_ru=prompt_ru,
        use_llm=use_llm,
        require_llm=require_llm,
        llm_timeout_s=llm_timeout_s,
        llm_max_attempts=llm_max_attempts,
    )
    planned_counts = summarize_template_counts(spec.objects)
    layout_strategy = layout_strategy_name(spec.room_type)

    if verbose:
        print(f"      source={planning_source}, room={spec.room_type}, size={tuple(round(v, 2) for v in spec.room_size_m)}")
        print(f"      planned_objects={len(spec.objects)} ({format_template_counts(planned_counts)})")

    if verbose:
        print("[2/4] Получаем объекты, их количество и размещение...")

    placed, before_repair, after_repair, used_repair = generate_placements_with_repair(spec, seed=seed)
    placed_counts = summarize_template_counts(placed)

    if verbose:
        print(f"      layout={layout_strategy}, placed_objects={len(placed)} ({format_template_counts(placed_counts)})")
        print(f"      repair_used={used_repair}, issues_before={len(before_repair.issues)}, issues_after={len(after_repair.issues)}")

    if verbose:
        print("[3/4] Собираем сцену в MJCF/URDF...")

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_dir = output_root / timestamp
    output_dir.mkdir(parents=True, exist_ok=False)

    scene_xml = output_dir / "scene.xml"
    scene_urdf = output_dir / "scene.urdf"

    export_mjcf(spec, placed, scene_xml)
    export_urdf(spec, placed, scene_urdf)
    write_manifest(
        spec=spec,
        placed=placed,
        validation_before=before_repair,
        validation_after=after_repair,
        used_repair=used_repair,
        output_dir=output_dir,
        seed=seed,
        planning_source=planning_source,
        layout_strategy=layout_strategy,
    )

    return {
        "spec": spec,
        "placed": placed,
        "before_repair": before_repair,
        "after_repair": after_repair,
        "used_repair": used_repair,
        "output_dir": output_dir,
        "scene_xml": scene_xml,
        "scene_urdf": scene_urdf,
        "planning_source": planning_source,
        "layout_strategy": layout_strategy,
        "planned_counts": planned_counts,
        "placed_counts": placed_counts,
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate MuJoCo scene from Russian text prompt")
    parser.add_argument("--prompt", type=str, default=None, help="Russian natural language prompt")
    parser.add_argument("--run", action="store_true", help="Run MuJoCo viewer after generation (default behavior)")
    parser.add_argument("--no-run", action="store_true", help="Do not launch MuJoCo viewer")
    parser.add_argument("--duration", type=float, default=120.0, help="Viewer duration in seconds (<=0 means unlimited)")
    parser.add_argument("--no-llm", action="store_true", help="Disable LLM planning and use heuristic parser")
    parser.add_argument(
        "--llm-timeout",
        type=float,
        default=DEFAULT_LLM_TIMEOUT_S,
        help="Timeout in seconds for one LLM planning attempt",
    )
    parser.add_argument(
        "--llm-max-attempts",
        type=int,
        default=DEFAULT_LLM_MAX_ATTEMPTS,
        help="Number of LLM planning attempts before heuristic fallback",
    )
    parser.add_argument(
        "--require-llm",
        action="store_true",
        help="Fail if LLM planning is unavailable (disable heuristic fallback)",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for deterministic placement")
    parser.add_argument("--output-root", type=str, default="outputs", help="Root folder for generated scenes")
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    if args.no_llm and args.require_llm:
        parser.error("--require-llm cannot be used with --no-llm")
    if args.llm_timeout <= 0:
        parser.error("--llm-timeout must be > 0")
    if args.llm_max_attempts < 1:
        parser.error("--llm-max-attempts must be >= 1")

    prompt_ru = args.prompt
    if not prompt_ru:
        prompt_ru = input("Введите описание сцены: ").strip()

    if not prompt_ru:
        raise ValueError("Empty prompt is not allowed")

    result = save_scene(
        prompt_ru=prompt_ru,
        use_llm=not args.no_llm,
        output_root=Path(args.output_root),
        seed=args.seed,
        verbose=True,
        require_llm=args.require_llm,
        llm_timeout_s=args.llm_timeout,
        llm_max_attempts=args.llm_max_attempts,
    )

    output_dir: Path = result["output_dir"]
    scene_xml: Path = result["scene_xml"]
    scene_urdf: Path = result["scene_urdf"]
    before: ValidationResult = result["before_repair"]
    after: ValidationResult = result["after_repair"]

    print("Scene generation completed")
    print(f"Prompt: {prompt_ru}")
    print(f"Output directory: {output_dir}")
    print(f"MJCF: {scene_xml}")
    print(f"URDF: {scene_urdf}")
    print(f"Validation before repair: success={before.success}, issues={len(before.issues)}")
    print(f"Validation after repair: success={after.success}, issues={len(after.issues)}")

    should_run = (not args.no_run) or args.run
    if should_run:
        print("[4/4] Открываем сцену в MuJoCo viewer...")
        run_mujoco_viewer(scene_xml_path=scene_xml, duration_s=args.duration)
    else:
        print("[4/4] Viewer пропущен (--no-run).")


if __name__ == "__main__":
    main()