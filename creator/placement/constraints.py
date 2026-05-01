"""Constraint classes for spatial relationships in scene placement.

This module defines various types of spatial constraints that can be applied
to objects in a 3D scene. Constraints are used by the Layout Solver to determine
valid positions and orientations for objects.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Optional


class ConstraintType(Enum):
    """Types of spatial constraints."""
    DISTANCE = "distance"
    ALIGNMENT = "alignment"
    FACING = "facing"
    CONTAINMENT = "containment"
    COLLISION_AVOIDANCE = "collision_avoidance"
    SUPPORT = "support"
    ACCESSIBILITY = "accessibility"
    CLEARANCE = "clearance"


class DistanceMeasurement(Enum):
    """Methods for measuring distance between objects."""
    CENTER_TO_CENTER = "center_to_center"
    EDGE_TO_EDGE = "edge_to_edge"
    SURFACE_TO_SURFACE = "surface_to_surface"
    CLOSEST_POINT = "closest_point"


class Axis(Enum):
    """Coordinate axes for alignment."""
    X = "x"
    Y = "y"
    Z = "z"


class AlignmentType(Enum):
    """Types of alignment between objects."""
    CENTER = "center"
    EDGE = "edge"
    FACE = "face"
    MIN = "min"
    MAX = "max"


class FacingType(Enum):
    """Types of facing/orientation relationships."""
    TOWARDS = "towards"
    AWAY_FROM = "away_from"
    PARALLEL = "parallel"
    PERPENDICULAR = "perpendicular"


@dataclass
class Constraint:
    """Base class for spatial constraints.

    Attributes:
        id: Unique identifier for this constraint
        constraint_type: Type of constraint (distance, alignment, etc.)
        source_object: ID of the object being constrained
        target_object: ID of the target object (None for absolute constraints)
        parameters: Additional constraint-specific parameters
        priority: Priority value for conflict resolution (higher = more important)
        is_hard: Whether this is a hard constraint (must be satisfied) or soft (preferred)
    """

    id: str
    constraint_type: ConstraintType
    source_object: str
    target_object: Optional[str] = None
    parameters: Dict[str, Any] = field(default_factory=dict)
    priority: float = 1.0
    is_hard: bool = True

    def __post_init__(self):
        """Validate constraint parameters."""
        if self.priority < 0:
            raise ValueError(f"Priority must be non-negative, got {self.priority}")

    def to_dict(self) -> Dict[str, Any]:
        """Convert constraint to dictionary representation.

        Returns:
            Dictionary containing all constraint data
        """
        return {
            "id": self.id,
            "constraint_type": self.constraint_type.value,
            "source_object": self.source_object,
            "target_object": self.target_object,
            "parameters": self.parameters,
            "priority": self.priority,
            "is_hard": self.is_hard
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Constraint':
        """Create constraint from dictionary representation.

        Args:
            data: Dictionary containing constraint data

        Returns:
            Constraint instance
        """
        constraint_type = ConstraintType(data["constraint_type"])
        return cls(
            id=data["id"],
            constraint_type=constraint_type,
            source_object=data["source_object"],
            target_object=data.get("target_object"),
            parameters=data.get("parameters", {}),
            priority=data.get("priority", 1.0),
            is_hard=data.get("is_hard", True)
        )


@dataclass
class DistanceConstraint(Constraint):
    """Constraint on distance between two objects.

    Attributes:
        min_distance: Minimum allowed distance (in meters)
        max_distance: Maximum allowed distance (in meters)
        measurement_type: How to measure distance between objects
    """

    min_distance: float = 0.0
    max_distance: float = float('inf')
    measurement_type: DistanceMeasurement = DistanceMeasurement.CENTER_TO_CENTER

    def __post_init__(self):
        """Initialize and validate distance constraint."""
        # Set constraint type
        self.constraint_type = ConstraintType.DISTANCE

        # Validate distance parameters
        if self.min_distance < 0:
            raise ValueError(f"min_distance must be non-negative, got {self.min_distance}")
        if self.max_distance < self.min_distance:
            raise ValueError(
                f"max_distance ({self.max_distance}) must be >= min_distance ({self.min_distance})"
            )

        # Store in parameters dict for base class compatibility
        self.parameters.update({
            "min_distance": self.min_distance,
            "max_distance": self.max_distance,
            "measurement_type": self.measurement_type.value
        })

        # Call parent validation
        super().__post_init__()

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        result = super().to_dict()
        result.update({
            "min_distance": self.min_distance,
            "max_distance": self.max_distance,
            "measurement_type": self.measurement_type.value
        })
        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'DistanceConstraint':
        """Create distance constraint from dictionary."""
        measurement_type = DistanceMeasurement(
            data.get("measurement_type", DistanceMeasurement.CENTER_TO_CENTER.value)
        )
        return cls(
            id=data["id"],
            constraint_type=ConstraintType.DISTANCE,
            source_object=data["source_object"],
            target_object=data.get("target_object"),
            parameters=data.get("parameters", {}),
            priority=data.get("priority", 1.0),
            is_hard=data.get("is_hard", True),
            min_distance=data.get("min_distance", 0.0),
            max_distance=data.get("max_distance", float('inf')),
            measurement_type=measurement_type
        )


@dataclass
class AlignmentConstraint(Constraint):
    """Constraint on alignment between objects along an axis.

    Attributes:
        alignment_axis: Axis along which to align (X, Y, or Z)
        alignment_type: Type of alignment (center, edge, face, etc.)
        tolerance: Allowed deviation from perfect alignment (in meters)
    """

    alignment_axis: Axis = Axis.X
    alignment_type: AlignmentType = AlignmentType.CENTER
    tolerance: float = 0.01

    def __post_init__(self):
        """Initialize and validate alignment constraint."""
        # Set constraint type
        self.constraint_type = ConstraintType.ALIGNMENT

        # Validate tolerance
        if self.tolerance < 0:
            raise ValueError(f"tolerance must be non-negative, got {self.tolerance}")

        # Store in parameters dict
        self.parameters.update({
            "alignment_axis": self.alignment_axis.value,
            "alignment_type": self.alignment_type.value,
            "tolerance": self.tolerance
        })

        # Call parent validation
        super().__post_init__()

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        result = super().to_dict()
        result.update({
            "alignment_axis": self.alignment_axis.value,
            "alignment_type": self.alignment_type.value,
            "tolerance": self.tolerance
        })
        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AlignmentConstraint':
        """Create alignment constraint from dictionary."""
        alignment_axis = Axis(data.get("alignment_axis", Axis.X.value))
        alignment_type = AlignmentType(data.get("alignment_type", AlignmentType.CENTER.value))
        return cls(
            id=data["id"],
            constraint_type=ConstraintType.ALIGNMENT,
            source_object=data["source_object"],
            target_object=data.get("target_object"),
            parameters=data.get("parameters", {}),
            priority=data.get("priority", 1.0),
            is_hard=data.get("is_hard", True),
            alignment_axis=alignment_axis,
            alignment_type=alignment_type,
            tolerance=data.get("tolerance", 0.01)
        )


@dataclass
class FacingConstraint(Constraint):
    """Constraint on orientation/facing direction of an object.

    Attributes:
        facing_direction: Target direction vector as (x, y, z) tuple
        angle_tolerance: Allowed angular deviation in degrees
        facing_type: Type of facing relationship (towards, away_from, parallel, etc.)
    """

    facing_direction: tuple[float, float, float] = (1.0, 0.0, 0.0)
    angle_tolerance: float = 10.0
    facing_type: FacingType = FacingType.TOWARDS

    def __post_init__(self):
        """Initialize and validate facing constraint."""
        # Set constraint type
        self.constraint_type = ConstraintType.FACING

        # Validate angle tolerance
        if self.angle_tolerance < 0 or self.angle_tolerance > 180:
            raise ValueError(
                f"angle_tolerance must be between 0 and 180 degrees, got {self.angle_tolerance}"
            )

        # Validate facing direction is not zero vector
        dx, dy, dz = self.facing_direction
        magnitude = (dx**2 + dy**2 + dz**2) ** 0.5
        if magnitude < 1e-6:
            raise ValueError("facing_direction cannot be a zero vector")

        # Store in parameters dict
        self.parameters.update({
            "facing_direction": list(self.facing_direction),
            "angle_tolerance": self.angle_tolerance,
            "facing_type": self.facing_type.value
        })

        # Call parent validation
        super().__post_init__()

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        result = super().to_dict()
        result.update({
            "facing_direction": list(self.facing_direction),
            "angle_tolerance": self.angle_tolerance,
            "facing_type": self.facing_type.value
        })
        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'FacingConstraint':
        """Create facing constraint from dictionary."""
        facing_direction = tuple(data.get("facing_direction", [1.0, 0.0, 0.0]))
        facing_type = FacingType(data.get("facing_type", FacingType.TOWARDS.value))
        return cls(
            id=data["id"],
            constraint_type=ConstraintType.FACING,
            source_object=data["source_object"],
            target_object=data.get("target_object"),
            parameters=data.get("parameters", {}),
            priority=data.get("priority", 1.0),
            is_hard=data.get("is_hard", True),
            facing_direction=facing_direction,
            angle_tolerance=data.get("angle_tolerance", 10.0),
            facing_type=facing_type
        )


@dataclass
class ContainmentConstraint(Constraint):
    """Constraint for placing objects within boundaries.

    This constraint ensures that an object is placed within the boundaries
    of a target region or container. Can be used for room boundaries,
    container objects, or defined regions.

    Attributes:
        boundary_type: Type of boundary ("box", "sphere", "polygon", "room")
        boundary_params: Parameters defining the boundary (depends on boundary_type)
        margin: Minimum distance from boundary edges (in meters)
        allow_partial: Whether partial containment is allowed
    """

    boundary_type: str = "box"
    boundary_params: Dict[str, Any] = field(default_factory=dict)
    margin: float = 0.0
    allow_partial: bool = False

    def __post_init__(self):
        """Initialize and validate containment constraint."""
        # Set constraint type
        self.constraint_type = ConstraintType.CONTAINMENT

        # Validate margin
        if self.margin < 0:
            raise ValueError(f"margin must be non-negative, got {self.margin}")

        # Validate boundary type
        valid_types = {"box", "sphere", "polygon", "room", "cylinder"}
        if self.boundary_type not in valid_types:
            raise ValueError(
                f"boundary_type must be one of {valid_types}, got '{self.boundary_type}'"
            )

        # Store in parameters dict
        self.parameters.update({
            "boundary_type": self.boundary_type,
            "boundary_params": self.boundary_params,
            "margin": self.margin,
            "allow_partial": self.allow_partial
        })

        # Call parent validation
        super().__post_init__()

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        result = super().to_dict()
        result.update({
            "boundary_type": self.boundary_type,
            "boundary_params": self.boundary_params,
            "margin": self.margin,
            "allow_partial": self.allow_partial
        })
        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ContainmentConstraint':
        """Create containment constraint from dictionary."""
        return cls(
            id=data["id"],
            constraint_type=ConstraintType.CONTAINMENT,
            source_object=data["source_object"],
            target_object=data.get("target_object"),
            parameters=data.get("parameters", {}),
            priority=data.get("priority", 1.0),
            is_hard=data.get("is_hard", True),
            boundary_type=data.get("boundary_type", "box"),
            boundary_params=data.get("boundary_params", {}),
            margin=data.get("margin", 0.0),
            allow_partial=data.get("allow_partial", False)
        )


# Factory function for creating constraints from dictionaries
def create_constraint(data: Dict[str, Any]) -> Constraint:
    """Factory function to create appropriate constraint type from dictionary.

    Args:
        data: Dictionary containing constraint data with 'constraint_type' key

    Returns:
        Appropriate Constraint subclass instance

    Raises:
        ValueError: If constraint_type is unknown
    """
    constraint_type_str = data.get("constraint_type", "")

    # Map constraint types to classes
    constraint_classes = {
        ConstraintType.DISTANCE.value: DistanceConstraint,
        ConstraintType.ALIGNMENT.value: AlignmentConstraint,
        ConstraintType.FACING.value: FacingConstraint,
        ConstraintType.CONTAINMENT.value: ContainmentConstraint,
    }

    # Get the appropriate class
    constraint_class = constraint_classes.get(constraint_type_str)

    if constraint_class is None:
        # Fall back to base Constraint class for unknown types
        return Constraint.from_dict(data)

    return constraint_class.from_dict(data)
