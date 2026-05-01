"""Basic unit tests for constraint classes.

Tests the creation, validation, and serialization of constraint objects.
"""

import pytest
from creator.placement.constraints import (
    Constraint,
    DistanceConstraint,
    AlignmentConstraint,
    FacingConstraint,
    ContainmentConstraint,
    ConstraintType,
    DistanceMeasurement,
    Axis,
    AlignmentType,
    FacingType,
    create_constraint,
)


class TestBaseConstraint:
    """Tests for the base Constraint class."""

    def test_create_basic_constraint(self):
        """Test creating a basic constraint with required fields."""
        constraint = Constraint(
            id="c1",
            constraint_type=ConstraintType.DISTANCE,
            source_object="table",
            target_object="chair"
        )
        assert constraint.id == "c1"
        assert constraint.constraint_type == ConstraintType.DISTANCE
        assert constraint.source_object == "table"
        assert constraint.target_object == "chair"
        assert constraint.priority == 1.0
        assert constraint.is_hard is True

    def test_constraint_with_custom_priority(self):
        """Test creating a constraint with custom priority."""
        constraint = Constraint(
            id="c2",
            constraint_type=ConstraintType.ALIGNMENT,
            source_object="obj1",
            priority=2.5,
            is_hard=False
        )
        assert constraint.priority == 2.5
        assert constraint.is_hard is False

    def test_constraint_negative_priority_raises_error(self):
        """Test that negative priority raises ValueError."""
        with pytest.raises(ValueError, match="Priority must be non-negative"):
            Constraint(
                id="c3",
                constraint_type=ConstraintType.DISTANCE,
                source_object="obj1",
                priority=-1.0
            )

    def test_constraint_to_dict(self):
        """Test converting constraint to dictionary."""
        constraint = Constraint(
            id="c4",
            constraint_type=ConstraintType.FACING,
            source_object="obj1",
            target_object="obj2",
            parameters={"key": "value"},
            priority=1.5,
            is_hard=False
        )
        result = constraint.to_dict()
        assert result["id"] == "c4"
        assert result["constraint_type"] == "facing"
        assert result["source_object"] == "obj1"
        assert result["target_object"] == "obj2"
        assert result["parameters"] == {"key": "value"}
        assert result["priority"] == 1.5
        assert result["is_hard"] is False

    def test_constraint_from_dict(self):
        """Test creating constraint from dictionary."""
        data = {
            "id": "c5",
            "constraint_type": "distance",
            "source_object": "obj1",
            "target_object": "obj2",
            "priority": 2.0,
            "is_hard": True
        }
        constraint = Constraint.from_dict(data)
        assert constraint.id == "c5"
        assert constraint.constraint_type == ConstraintType.DISTANCE
        assert constraint.source_object == "obj1"
        assert constraint.target_object == "obj2"
        assert constraint.priority == 2.0
        assert constraint.is_hard is True


class TestDistanceConstraint:
    """Tests for DistanceConstraint class."""

    def test_create_distance_constraint(self):
        """Test creating a distance constraint."""
        constraint = DistanceConstraint(
            id="d1",
            source_object="chair",
            target_object="table",
            min_distance=0.5,
            max_distance=1.5,
            measurement_type=DistanceMeasurement.EDGE_TO_EDGE
        )
        assert constraint.id == "d1"
        assert constraint.constraint_type == ConstraintType.DISTANCE
        assert constraint.min_distance == 0.5
        assert constraint.max_distance == 1.5
        assert constraint.measurement_type == DistanceMeasurement.EDGE_TO_EDGE

    def test_distance_constraint_default_values(self):
        """Test distance constraint with default values."""
        constraint = DistanceConstraint(
            id="d2",
            source_object="obj1",
            target_object="obj2"
        )
        assert constraint.min_distance == 0.0
        assert constraint.max_distance == float('inf')
        assert constraint.measurement_type == DistanceMeasurement.CENTER_TO_CENTER

    def test_distance_constraint_negative_min_raises_error(self):
        """Test that negative min_distance raises ValueError."""
        with pytest.raises(ValueError, match="min_distance must be non-negative"):
            DistanceConstraint(
                id="d3",
                source_object="obj1",
                target_object="obj2",
                min_distance=-1.0
            )

    def test_distance_constraint_max_less_than_min_raises_error(self):
        """Test that max_distance < min_distance raises ValueError."""
        with pytest.raises(ValueError, match="max_distance.*must be >= min_distance"):
            DistanceConstraint(
                id="d4",
                source_object="obj1",
                target_object="obj2",
                min_distance=2.0,
                max_distance=1.0
            )

    def test_distance_constraint_to_dict(self):
        """Test converting distance constraint to dictionary."""
        constraint = DistanceConstraint(
            id="d5",
            source_object="obj1",
            target_object="obj2",
            min_distance=0.3,
            max_distance=1.0,
            measurement_type=DistanceMeasurement.CLOSEST_POINT
        )
        result = constraint.to_dict()
        assert result["min_distance"] == 0.3
        assert result["max_distance"] == 1.0
        assert result["measurement_type"] == "closest_point"

    def test_distance_constraint_from_dict(self):
        """Test creating distance constraint from dictionary."""
        data = {
            "id": "d6",
            "source_object": "obj1",
            "target_object": "obj2",
            "min_distance": 0.5,
            "max_distance": 2.0,
            "measurement_type": "edge_to_edge"
        }
        constraint = DistanceConstraint.from_dict(data)
        assert constraint.min_distance == 0.5
        assert constraint.max_distance == 2.0
        assert constraint.measurement_type == DistanceMeasurement.EDGE_TO_EDGE


class TestAlignmentConstraint:
    """Tests for AlignmentConstraint class."""

    def test_create_alignment_constraint(self):
        """Test creating an alignment constraint."""
        constraint = AlignmentConstraint(
            id="a1",
            source_object="chair1",
            target_object="chair2",
            alignment_axis=Axis.Y,
            alignment_type=AlignmentType.CENTER,
            tolerance=0.05
        )
        assert constraint.id == "a1"
        assert constraint.constraint_type == ConstraintType.ALIGNMENT
        assert constraint.alignment_axis == Axis.Y
        assert constraint.alignment_type == AlignmentType.CENTER
        assert constraint.tolerance == 0.05

    def test_alignment_constraint_default_values(self):
        """Test alignment constraint with default values."""
        constraint = AlignmentConstraint(
            id="a2",
            source_object="obj1",
            target_object="obj2"
        )
        assert constraint.alignment_axis == Axis.X
        assert constraint.alignment_type == AlignmentType.CENTER
        assert constraint.tolerance == 0.01

    def test_alignment_constraint_negative_tolerance_raises_error(self):
        """Test that negative tolerance raises ValueError."""
        with pytest.raises(ValueError, match="tolerance must be non-negative"):
            AlignmentConstraint(
                id="a3",
                source_object="obj1",
                target_object="obj2",
                tolerance=-0.1
            )

    def test_alignment_constraint_to_dict(self):
        """Test converting alignment constraint to dictionary."""
        constraint = AlignmentConstraint(
            id="a4",
            source_object="obj1",
            target_object="obj2",
            alignment_axis=Axis.Z,
            alignment_type=AlignmentType.EDGE,
            tolerance=0.02
        )
        result = constraint.to_dict()
        assert result["alignment_axis"] == "z"
        assert result["alignment_type"] == "edge"
        assert result["tolerance"] == 0.02

    def test_alignment_constraint_from_dict(self):
        """Test creating alignment constraint from dictionary."""
        data = {
            "id": "a5",
            "source_object": "obj1",
            "target_object": "obj2",
            "alignment_axis": "y",
            "alignment_type": "face",
            "tolerance": 0.03
        }
        constraint = AlignmentConstraint.from_dict(data)
        assert constraint.alignment_axis == Axis.Y
        assert constraint.alignment_type == AlignmentType.FACE
        assert constraint.tolerance == 0.03


class TestFacingConstraint:
    """Tests for FacingConstraint class."""

    def test_create_facing_constraint(self):
        """Test creating a facing constraint."""
        constraint = FacingConstraint(
            id="f1",
            source_object="chair",
            target_object="table",
            facing_direction=(0.0, 1.0, 0.0),
            angle_tolerance=15.0,
            facing_type=FacingType.TOWARDS
        )
        assert constraint.id == "f1"
        assert constraint.constraint_type == ConstraintType.FACING
        assert constraint.facing_direction == (0.0, 1.0, 0.0)
        assert constraint.angle_tolerance == 15.0
        assert constraint.facing_type == FacingType.TOWARDS

    def test_facing_constraint_default_values(self):
        """Test facing constraint with default values."""
        constraint = FacingConstraint(
            id="f2",
            source_object="obj1",
            target_object="obj2"
        )
        assert constraint.facing_direction == (1.0, 0.0, 0.0)
        assert constraint.angle_tolerance == 10.0
        assert constraint.facing_type == FacingType.TOWARDS

    def test_facing_constraint_invalid_angle_raises_error(self):
        """Test that invalid angle tolerance raises ValueError."""
        with pytest.raises(ValueError, match="angle_tolerance must be between 0 and 180"):
            FacingConstraint(
                id="f3",
                source_object="obj1",
                target_object="obj2",
                angle_tolerance=200.0
            )

    def test_facing_constraint_zero_vector_raises_error(self):
        """Test that zero facing direction raises ValueError."""
        with pytest.raises(ValueError, match="facing_direction cannot be a zero vector"):
            FacingConstraint(
                id="f4",
                source_object="obj1",
                target_object="obj2",
                facing_direction=(0.0, 0.0, 0.0)
            )

    def test_facing_constraint_to_dict(self):
        """Test converting facing constraint to dictionary."""
        constraint = FacingConstraint(
            id="f5",
            source_object="obj1",
            target_object="obj2",
            facing_direction=(1.0, 0.0, 0.0),
            angle_tolerance=20.0,
            facing_type=FacingType.PARALLEL
        )
        result = constraint.to_dict()
        assert result["facing_direction"] == [1.0, 0.0, 0.0]
        assert result["angle_tolerance"] == 20.0
        assert result["facing_type"] == "parallel"

    def test_facing_constraint_from_dict(self):
        """Test creating facing constraint from dictionary."""
        data = {
            "id": "f6",
            "source_object": "obj1",
            "target_object": "obj2",
            "facing_direction": [0.0, 0.0, 1.0],
            "angle_tolerance": 5.0,
            "facing_type": "away_from"
        }
        constraint = FacingConstraint.from_dict(data)
        assert constraint.facing_direction == (0.0, 0.0, 1.0)
        assert constraint.angle_tolerance == 5.0
        assert constraint.facing_type == FacingType.AWAY_FROM


class TestContainmentConstraint:
    """Tests for ContainmentConstraint class."""

    def test_create_containment_constraint(self):
        """Test creating a containment constraint."""
        constraint = ContainmentConstraint(
            id="c1",
            source_object="apple",
            target_object="box",
            boundary_type="box",
            boundary_params={"width": 1.0, "height": 1.0, "depth": 1.0},
            margin=0.05,
            allow_partial=False
        )
        assert constraint.id == "c1"
        assert constraint.constraint_type == ConstraintType.CONTAINMENT
        assert constraint.boundary_type == "box"
        assert constraint.boundary_params == {"width": 1.0, "height": 1.0, "depth": 1.0}
        assert constraint.margin == 0.05
        assert constraint.allow_partial is False

    def test_containment_constraint_default_values(self):
        """Test containment constraint with default values."""
        constraint = ContainmentConstraint(
            id="c2",
            source_object="obj1",
            target_object="room"
        )
        assert constraint.boundary_type == "box"
        assert constraint.boundary_params == {}
        assert constraint.margin == 0.0
        assert constraint.allow_partial is False

    def test_containment_constraint_negative_margin_raises_error(self):
        """Test that negative margin raises ValueError."""
        with pytest.raises(ValueError, match="margin must be non-negative"):
            ContainmentConstraint(
                id="c3",
                source_object="obj1",
                target_object="room",
                margin=-0.1
            )

    def test_containment_constraint_invalid_boundary_type_raises_error(self):
        """Test that invalid boundary type raises ValueError."""
        with pytest.raises(ValueError, match="boundary_type must be one of"):
            ContainmentConstraint(
                id="c4",
                source_object="obj1",
                target_object="room",
                boundary_type="invalid_type"
            )

    def test_containment_constraint_to_dict(self):
        """Test converting containment constraint to dictionary."""
        constraint = ContainmentConstraint(
            id="c5",
            source_object="obj1",
            target_object="room",
            boundary_type="sphere",
            boundary_params={"radius": 2.0},
            margin=0.1,
            allow_partial=True
        )
        result = constraint.to_dict()
        assert result["boundary_type"] == "sphere"
        assert result["boundary_params"] == {"radius": 2.0}
        assert result["margin"] == 0.1
        assert result["allow_partial"] is True

    def test_containment_constraint_from_dict(self):
        """Test creating containment constraint from dictionary."""
        data = {
            "id": "c6",
            "source_object": "obj1",
            "target_object": "room",
            "boundary_type": "polygon",
            "boundary_params": {"vertices": [[0, 0], [1, 0], [1, 1], [0, 1]]},
            "margin": 0.2,
            "allow_partial": False
        }
        constraint = ContainmentConstraint.from_dict(data)
        assert constraint.boundary_type == "polygon"
        assert constraint.boundary_params == {"vertices": [[0, 0], [1, 0], [1, 1], [0, 1]]}
        assert constraint.margin == 0.2
        assert constraint.allow_partial is False


class TestConstraintFactory:
    """Tests for the create_constraint factory function."""

    def test_create_distance_constraint_from_dict(self):
        """Test creating distance constraint via factory."""
        data = {
            "id": "d1",
            "constraint_type": "distance",
            "source_object": "obj1",
            "target_object": "obj2",
            "min_distance": 0.5,
            "max_distance": 1.5
        }
        constraint = create_constraint(data)
        assert isinstance(constraint, DistanceConstraint)
        assert constraint.min_distance == 0.5
        assert constraint.max_distance == 1.5

    def test_create_alignment_constraint_from_dict(self):
        """Test creating alignment constraint via factory."""
        data = {
            "id": "a1",
            "constraint_type": "alignment",
            "source_object": "obj1",
            "target_object": "obj2",
            "alignment_axis": "y"
        }
        constraint = create_constraint(data)
        assert isinstance(constraint, AlignmentConstraint)
        assert constraint.alignment_axis == Axis.Y

    def test_create_facing_constraint_from_dict(self):
        """Test creating facing constraint via factory."""
        data = {
            "id": "f1",
            "constraint_type": "facing",
            "source_object": "obj1",
            "target_object": "obj2",
            "facing_type": "parallel"
        }
        constraint = create_constraint(data)
        assert isinstance(constraint, FacingConstraint)
        assert constraint.facing_type == FacingType.PARALLEL

    def test_create_containment_constraint_from_dict(self):
        """Test creating containment constraint via factory."""
        data = {
            "id": "c1",
            "constraint_type": "containment",
            "source_object": "obj1",
            "target_object": "room",
            "boundary_type": "sphere"
        }
        constraint = create_constraint(data)
        assert isinstance(constraint, ContainmentConstraint)
        assert constraint.boundary_type == "sphere"

    def test_create_unknown_constraint_type_returns_base_constraint(self):
        """Test that unknown constraint type returns base Constraint."""
        data = {
            "id": "u1",
            "constraint_type": "unknown_type",
            "source_object": "obj1"
        }
        constraint = create_constraint(data)
        assert isinstance(constraint, Constraint)
        assert not isinstance(constraint, (DistanceConstraint, AlignmentConstraint,
                                          FacingConstraint, ContainmentConstraint))
