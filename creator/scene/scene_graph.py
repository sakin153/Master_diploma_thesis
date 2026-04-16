"""SceneGraph: structured intermediate representation of a 3D scene.

More expressive than the old flat list of dicts — includes:
- Spatial relations (on_surface, wall_mounted, stacked_on, near, against_wall)
- Room specification (dimensions, type, style)
- Physics hints (static/dynamic per object)

This is the single source of truth between the LLM planning stage
and the MuJoCo assembly stage.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# Relation types
# ---------------------------------------------------------------------------

class Relation(str, Enum):
    ON_FLOOR       = "on_floor"       # free-standing on the floor
    ON_SURFACE     = "on_surface"     # resting on top of another object
    AGAINST_WALL   = "against_wall"   # floor object pushed against a wall
    WALL_MOUNTED   = "wall_mounted"   # physically attached to a wall (shelf, painting)
    STACKED_ON     = "stacked_on"     # stacked on another object (books in stack)
    HANGING        = "hanging"        # hanging from ceiling (chandelier)
    NEAR           = "near"           # soft proximity constraint
    IN_FRONT_OF    = "in_front_of"    # soft directional constraint
    BEHIND         = "behind"
    LEFT_OF        = "left_of"
    RIGHT_OF       = "right_of"
    FACING         = "facing"         # object faces another


# ---------------------------------------------------------------------------
# SceneObject
# ---------------------------------------------------------------------------

@dataclass
class SceneObject:
    """A single object in the scene graph."""
    name: str                          # Human name, e.g. "School Desk"
    category: str = ""                 # Category, e.g. "furniture/desk"
    quantity: int = 1                  # Number of instances

    # Size in scene coordinates (metres): [width, height, depth]
    # width = scene X, height = scene Z (mesh Y), depth = scene Y (mesh Z)
    size: Optional[List[float]] = None

    # Spatial placement
    relation: Relation = Relation.ON_FLOOR
    relation_target: str = ""          # name of target object for relational placement
    wall_side: str = ""                # "north"|"south"|"east"|"west" for wall placement
    height_on_wall: float = 1.4        # metres above floor for wall-mounted objects
    near_distance: List[float] = field(default_factory=lambda: [0.3, 1.5])

    # Physics
    is_static: Optional[bool] = None   # None = auto-detect from category

    # Notes from LLM
    notes: str = ""

    def is_wall_attached(self) -> bool:
        return self.relation in (Relation.WALL_MOUNTED, Relation.AGAINST_WALL)

    def as_constraint_dict(self) -> Dict[str, Any]:
        """Convert to old-style constraint dict for backward compatibility."""
        c: Dict[str, Any] = {}
        if self.relation == Relation.ON_SURFACE and self.relation_target:
            c = {"type": "on", "target": self.relation_target, "hard": True}
        elif self.relation == Relation.NEAR and self.relation_target:
            c = {"type": "near", "target": self.relation_target,
                 "distance": self.near_distance, "hard": False, "weight": 0.8}
        elif self.relation == Relation.AGAINST_WALL:
            c = {"type": "region", "value": "edge", "hard": False, "weight": 0.6}
        elif self.relation == Relation.WALL_MOUNTED:
            c = {"type": "wall_mount", "hard": True}
        elif self.relation in (
            Relation.IN_FRONT_OF, Relation.BEHIND,
            Relation.LEFT_OF, Relation.RIGHT_OF,
        ) and self.relation_target:
            c = {"type": self.relation.value, "target": self.relation_target,
                 "hard": False, "weight": 0.7}
        return c


# ---------------------------------------------------------------------------
# Room spec
# ---------------------------------------------------------------------------

@dataclass
class RoomSpec:
    room_type: str = "other"
    room_style: str = "modern"
    half_size: float = 2.5       # metres
    wall_height: float = 3.0     # metres
    has_windows: bool = True
    has_door: bool = True
    floor_material: str = "wood"  # wood|tile|carpet|concrete


# ---------------------------------------------------------------------------
# SceneGraph
# ---------------------------------------------------------------------------

@dataclass
class SceneGraph:
    """Full structured description of a scene."""
    query: str = ""
    room: RoomSpec = field(default_factory=RoomSpec)
    objects: List[SceneObject] = field(default_factory=list)

    def get_wall_mounted(self) -> List[SceneObject]:
        return [o for o in self.objects if o.relation == Relation.WALL_MOUNTED]

    def get_floor_objects(self) -> List[SceneObject]:
        return [o for o in self.objects
                if o.relation in (Relation.ON_FLOOR, Relation.AGAINST_WALL)]

    def get_surface_objects(self) -> List[SceneObject]:
        return [o for o in self.objects if o.relation == Relation.ON_SURFACE]

    def to_semantic_plan(self) -> Dict[str, Any]:
        """Convert to the old-style semantic_plan format for backward compat."""
        plan_objects = []
        for obj in self.objects:
            c = obj.as_constraint_dict()
            plan_obj: Dict[str, Any] = {"Model": obj.name}
            if c:
                plan_obj["constraints"] = [c]
            else:
                plan_obj["constraints"] = []
            plan_objects.append(plan_obj)

        return {
            "objects": plan_objects,
            "room": {
                "type": self.room.room_type,
                "half_size": self.room.half_size,
                "wall_height": self.room.wall_height,
            },
        }

    def to_flat_model_list(self) -> List[Dict[str, Any]]:
        """Expand quantities → flat list of model dicts (like old chosen_models)."""
        result = []
        for obj in self.objects:
            for _ in range(obj.quantity):
                m: Dict[str, Any] = {"Model": obj.name}
                if obj.size:
                    m["size"] = obj.size
                result.append(m)
        return result


# ---------------------------------------------------------------------------
# LLM prompt for SceneGraph generation
# ---------------------------------------------------------------------------

SCENE_GRAPH_PROMPT = """\
You are a 3D scene planner for a MuJoCo robotics simulator.

Scene description: "{description}"
Room type: {room_type}

Available models (from the database):
{catalog_str}

Task: Generate a complete SceneGraph for this scene.

For each object specify:
- How it relates to the space (on_floor / against_wall / wall_mounted / on_surface)
- Which wall it belongs to (for against_wall / wall_mounted): north, south, east, west
- What it rests on (for on_surface): the name of the supporting object
- Approximate height on wall in metres (for wall_mounted objects like shelves, paintings)

Object relations:
- on_floor: free-standing furniture (table, sofa, bed, wardrobe)
- against_wall: floor object pushed to a wall (bookcase, TV stand, cabinet)
- wall_mounted: PHYSICALLY ATTACHED to wall — no floor contact (shelf, painting, wall lamp, mirror, whiteboard)
- on_surface: resting on TOP of another object (book on shelf, laptop on desk, cup on table)
- near: soft proximity (chair near table)

IMPORTANT for wall_mounted objects:
- A bookshelf mounted to the wall → relation: wall_mounted, wall_side: north/south/east/west
- Books on that shelf → relation: on_surface, relation_target: "Bookshelf"
- A painting → relation: wall_mounted, height_on_wall: 1.6

Room sizing rule: estimate the room half-size based on total floor footprint of objects.
For {room_type} with these objects use: {room_half_hint}m half-size.

Output JSON only:
```json
{{
  "room": {{
    "room_type": "{room_type}",
    "room_style": "modern",
    "half_size": {room_half_hint},
    "wall_height": 3.0,
    "floor_material": "wood"
  }},
  "objects": [
    {{
      "name": "Object Name (must match catalog)",
      "quantity": 1,
      "relation": "on_floor",
      "relation_target": "",
      "wall_side": "",
      "height_on_wall": 0.0,
      "near_distance": [0.3, 1.5],
      "notes": ""
    }}
  ]
}}
```

Use ONLY model names that exist in the catalog. Return valid JSON inside ```json ... ```.
"""


def generate_scene_graph(
    description: str,
    room_type: str,
    room_half_hint: float,
    catalog: List[Dict[str, Any]],
    prompt_model_fn: Any,
    llm_model: str = "",
    verbose: bool = True,
) -> SceneGraph:
    """Call LLM to generate a SceneGraph for the scene."""
    catalog_lines = []
    for m in catalog[:60]:  # limit to avoid token overflow
        name = m.get("name") or m.get("Model") or ""
        cats = ", ".join((m.get("categories") or [])[:2])
        catalog_lines.append(f"- {name}" + (f" ({cats})" if cats else ""))
    catalog_str = "\n".join(catalog_lines)

    prompt = SCENE_GRAPH_PROMPT.format(
        description=description,
        room_type=room_type,
        catalog_str=catalog_str,
        room_half_hint=room_half_hint,
    )

    try:
        raw = prompt_model_fn(prompt, description, llm_model)
        graph = _parse_scene_graph(raw, description, room_type, room_half_hint)
        if verbose:
            wm = len(graph.get_wall_mounted())
            fl = len(graph.get_floor_objects())
            print(f"[scene_graph] Generated: {len(graph.objects)} objects "
                  f"({fl} floor, {wm} wall-mounted), room={graph.room.half_size*2:.0f}m²")
        return graph
    except Exception as exc:
        if verbose:
            print(f"[scene_graph] LLM generation failed ({exc}), returning empty graph")
        sg = SceneGraph(query=description)
        sg.room = RoomSpec(room_type=room_type, half_size=room_half_hint)
        return sg


def _parse_scene_graph(
    raw: Any,
    query: str,
    room_type: str,
    default_half: float,
) -> SceneGraph:
    text = str(raw) if not isinstance(raw, str) else raw
    match = re.search(r"```json\s*(.*?)\s*```", text, re.DOTALL)
    if not match:
        match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise ValueError("No JSON found in LLM output")

    data = json.loads(match.group(1) if "```" in text else match.group(0))

    # Room
    room_data = data.get("room", {})
    room = RoomSpec(
        room_type=str(room_data.get("room_type", room_type)),
        room_style=str(room_data.get("room_style", "modern")),
        half_size=float(room_data.get("half_size", default_half)),
        wall_height=float(room_data.get("wall_height", 3.0)),
        floor_material=str(room_data.get("floor_material", "wood")),
    )

    # Objects
    objects: List[SceneObject] = []
    for o in data.get("objects", []):
        if not isinstance(o, dict):
            continue
        name = str(o.get("name", "")).strip()
        if not name:
            continue

        rel_str = str(o.get("relation", "on_floor")).lower().replace("-", "_")
        try:
            rel = Relation(rel_str)
        except ValueError:
            rel = Relation.ON_FLOOR

        obj = SceneObject(
            name=name,
            quantity=max(1, int(o.get("quantity", 1))),
            relation=rel,
            relation_target=str(o.get("relation_target", "")),
            wall_side=str(o.get("wall_side", "")).lower(),
            height_on_wall=float(o.get("height_on_wall", 1.4)),
            near_distance=list(o.get("near_distance", [0.3, 1.5])),
            notes=str(o.get("notes", "")),
        )
        objects.append(obj)

    graph = SceneGraph(query=query, room=room, objects=objects)
    return graph
