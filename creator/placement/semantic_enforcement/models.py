"""Core data models for Semantic Plan Enforcement.

This module defines the data structures used throughout the semantic plan
enforcement pipeline. All models use dataclasses for simplicity and type safety.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class Position:
    """Object position (absolute or relative).
    
    Exactly one of `absolute` or `relative` must be specified.
    
    Attributes:
        absolute: Absolute position as {x, y, z} in meters
        relative: Relative position specification with fields:
            - relative_to: target object ID
            - direction: "front"|"back"|"left"|"right"|"front_left"|"front_right"|
                        "back_left"|"back_right"|"above"|"below"
            - distance: distance in meters
            - reference_point: "center"|"front_edge"|"back_edge"|"left_edge"|
                             "right_edge"|"top_surface"
    """
    absolute: Optional[Dict[str, float]] = None  # {x, y, z}
    relative: Optional[Dict[str, Any]] = None  # {relative_to, direction, distance, reference_point}


@dataclass
class Orientation:
    """Object orientation (absolute or relative).
    
    Exactly one of `absolute` or `relative` must be specified.
    
    Attributes:
        absolute: Absolute orientation as {yaw_deg, pitch_deg, roll_deg}
        relative: Relative orientation specification with fields:
            - facing: target object ID to face
            - facing_direction: "front"|"back"|"left_side"|"right_side"
            - facing_away: boolean, if True face away from target
    """
    absolute: Optional[Dict[str, float]] = None  # {yaw_deg, pitch_deg, roll_deg}
    relative: Optional[Dict[str, Any]] = None  # {facing, facing_direction, facing_away}


@dataclass
class SemanticObject:
    """Object in semantic plan with position and orientation specifications.
    
    Attributes:
        id: Unique object identifier
        Model: Model name from catalog
        type: Object type ("furniture" or "small_object")
        size: Object dimensions as {width, length, height} in meters
        is_static: Whether object is static (True) or dynamic (False)
        model_loc: Path to 3D model file
        position: Position specification (absolute or relative)
        orientation: Orientation specification (absolute or relative)
        metadata: Additional metadata from original plan
    """
    id: str
    Model: str
    type: str
    size: Dict[str, float]  # {width, length, height}
    is_static: bool
    model_loc: str
    position: Position
    orientation: Orientation
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SemanticPlan:
    """Complete semantic plan from LLM.
    
    Attributes:
        schema_version: Schema version for backward compatibility
        room_size: Room dimensions as {width, length, height} in meters
        objects: List of semantic objects to place
        metadata: Additional metadata (e.g., room type, style)
    """
    schema_version: str
    room_size: Dict[str, float]  # {width, length, height}
    objects: List[SemanticObject]
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PlacedObject:
    """Object with resolved absolute position and orientation.
    
    All positions and orientations are absolute (no relative specifications).
    
    Attributes:
        id: Unique object identifier (matches SemanticObject.id)
        Model: Model name from catalog
        type: Object type
        size: Object dimensions as {width, length, height} in meters
        is_static: Whether object is static
        model_loc: Path to 3D model file
        position: Absolute position as {x, y, z} in meters
        orientation: Absolute orientation as {yaw_deg, pitch_deg, roll_deg}
        metadata: Original semantic plan data for traceability
    """
    id: str
    Model: str
    type: str
    size: Dict[str, float]
    is_static: bool
    model_loc: str
    position: Dict[str, float]  # {x, y, z} - always absolute
    orientation: Dict[str, float]  # {yaw_deg, pitch_deg, roll_deg} - always absolute
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PlacementSolution:
    """Complete placement solution with all objects positioned.
    
    Attributes:
        objects: List of placed objects with absolute coordinates
        room_size: Room dimensions
        metadata: Additional metadata (e.g., pipeline timing, statistics)
    """
    objects: List[PlacedObject]
    room_size: Dict[str, float]
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ValidationReport:
    """Validation report comparing semantic plan with final scene.
    
    Attributes:
        total_objects: Total number of objects in semantic plan
        missing_objects: List of object IDs missing from final scene
        position_errors: List of position mismatches with details:
            {id, planned, actual, error_m}
        orientation_errors: List of orientation mismatches with details:
            {id, planned, actual, error_deg}
        compliance_score: Overall compliance score (0.0 to 1.0)
        summary: Human-readable summary of validation results
    """
    total_objects: int
    missing_objects: List[str]
    position_errors: List[Dict[str, Any]]
    orientation_errors: List[Dict[str, Any]]
    compliance_score: float
    summary: str
