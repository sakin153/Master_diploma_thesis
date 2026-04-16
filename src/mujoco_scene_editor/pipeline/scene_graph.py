from __future__ import annotations

from enum import Enum
from typing import Annotated, Any, Literal  # noqa: F401 (Literal used in SceneObject + SceneRelation)

from pydantic import BaseModel, ConfigDict, Field


class RelationType(str, Enum):
    on = "on"
    ontop = "ontop"
    on_floor = "on_floor"
    against_wall = "against_wall"
    wall_mounted = "wall_mounted"   # physically attached to wall (shelf, painting, lamp)
    inside = "inside"
    next_to = "next_to"
    aligned_with = "aligned_with"
    reachable_by_robot = "reachable_by_robot"


class SceneObject(BaseModel):
    """Single object instance in a scene graph.

    Notes:
    - JSON uses key `class` (reserved word in Python). We expose it via
      `class_name` with an alias.
    - `size_m` is full extents in meters (X, Y, Z). For primitives, we
      approximate collision with that.
    """

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    id: str = Field(..., description="Unique id, stable across relations")
    class_name: Annotated[str, Field(alias="class", min_length=1)]
    name: str | None = None
    affordances: list[str] = Field(default_factory=list)

    movable: bool = True
    material: Literal[
        "plastic",
        "wood",
        "metal",
        "glass",
        "rubber",
        "paper",
        "generic",
    ] = "generic"

    # Optional geometry hints.
    primitive: Literal["box", "cylinder", "sphere"] | None = None
    size_m: list[float] | None = Field(
        default=None,
        description="Full extents [x,y,z] in meters (for box-like collision).",
        min_length=3,
        max_length=3,
    )

    # Optional explicit mesh selection.
    mesh_file: str | None = None
    mesh_scale: float | None = Field(
        default=None, description="Uniform mesh scale multiplier (meters-based)."
    )


class SceneRelation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: RelationType
    subject: str = Field(..., description="SceneObject.id")
    object: str = Field(default="", description="SceneObject.id (empty for wall_mounted)")

    # Wall-mounting hints (only for type=wall_mounted).
    wall_side: Literal["north", "south", "east", "west"] | None = Field(
        default=None,
        description="Which wall to mount on (north=+Y, south=-Y, east=+X, west=-X).",
    )
    height_m: float | None = Field(
        default=None,
        ge=0.0,
        description="Height above floor in metres for wall-mounted objects.",
    )

    # Optional numeric hints.
    distance_m: float | None = Field(default=None, ge=0.0)


class GlobalConstraints(BaseModel):
    model_config = ConfigDict(extra="forbid")

    # Kept intentionally small; add as needed.
    keep_clear_radius_m: float | None = Field(
        default=None,
        description="Radius around origin that must remain free (walkway / safe zone).",
        ge=0.0,
    )
    workspace_height_m: float | None = Field(
        default=None,
        description="Preferred working height (e.g., table top height).",
        ge=0.0,
    )
    room_half_size_m: float | None = Field(
        default=None,
        description="Half-size of the square room (metres). Auto-computed if null.",
        ge=1.0,
    )
    wall_height_m: float | None = Field(
        default=None,
        description="Room wall height in metres (default 3.0).",
        ge=1.5,
    )


class SceneGraph(BaseModel):
    """Validated intermediate representation between prompt and MJCF."""

    model_config = ConfigDict(extra="forbid")

    scene_type: str = Field(..., min_length=1)
    objects: list[SceneObject] = Field(default_factory=list)
    relations: list[SceneRelation] = Field(default_factory=list)
    constraints: GlobalConstraints = Field(default_factory=GlobalConstraints)

    # Optional debugging/provenance.
    llm_notes: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


def index_objects(graph: SceneGraph) -> dict[str, SceneObject]:
    by_id: dict[str, SceneObject] = {}
    for obj in graph.objects:
        if obj.id in by_id:
            raise ValueError(f"Duplicate object id: {obj.id}")
        by_id[obj.id] = obj
    return by_id
