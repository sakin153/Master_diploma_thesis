"""Basic verification tests for Anchor System implementation.

These tests verify that Task 2 (Anchor System) is correctly implemented.
"""

import math
import pytest

from creator.placement.anchor_system import (
    AnchorPoint,
    AnchorType,
    GlobalAnchor,
    LocalAnchor,
    AnchorSystem,
)
from creator.placement.geometry import Vec2, OBB


class TestAnchorClasses:
    """Test anchor dataclasses and enums (Task 2.1)."""

    def test_anchor_type_enum(self):
        """Verify AnchorType enum has required types."""
        assert AnchorType.WALL.value == "wall"
        assert AnchorType.CORNER.value == "corner"
        assert AnchorType.CENTER.value == "center"
        assert AnchorType.SURFACE.value == "surface"
        assert AnchorType.EDGE.value == "edge"

    def test_anchor_point_dataclass(self):
        """Verify AnchorPoint has required fields."""
        anchor = AnchorPoint(
            id="test_anchor",
            position=Vec2(1.0, 2.0),
            orientation=0.5,
            anchor_type=AnchorType.WALL,
            availability=True,
            constraints=["constraint1"],
            metadata={"key": "value"}
        )

        assert anchor.id == "test_anchor"
        assert anchor.position.x == 1.0
        assert anchor.position.y == 2.0
        assert anchor.orientation == 0.5
        assert anchor.anchor_type == AnchorType.WALL
        assert anchor.availability is True
        assert anchor.constraints == ["constraint1"]
        assert anchor.metadata == {"key": "value"}

    def test_anchor_point_availability_methods(self):
        """Verify AnchorPoint availability management."""
        anchor = AnchorPoint(
            id="test",
            position=Vec2(0, 0),
            orientation=0,
            anchor_type=AnchorType.WALL
        )

        assert anchor.is_available() is True

        anchor.mark_used()
        assert anchor.is_available() is False

        anchor.mark_available()
        assert anchor.is_available() is True

    def test_global_anchor_creation(self):
        """Verify GlobalAnchor can be created."""
        anchor = GlobalAnchor(
            id="north_wall",
            position=Vec2(0, 5),
            orientation=math.pi,
            anchor_type=AnchorType.WALL
        )

        assert isinstance(anchor, AnchorPoint)
        assert anchor.anchor_type == AnchorType.WALL

    def test_global_anchor_validation(self):
        """Verify GlobalAnchor validates anchor type."""
        # Valid types should work
        for anchor_type in [AnchorType.WALL, AnchorType.CORNER,
                           AnchorType.CENTER, AnchorType.DOOR,
                           AnchorType.WINDOW]:
            anchor = GlobalAnchor(
                id="test",
                position=Vec2(0, 0),
                orientation=0,
                anchor_type=anchor_type
            )
            assert anchor.anchor_type == anchor_type

        # Invalid types should raise error
        with pytest.raises(ValueError, match="GlobalAnchor must have type"):
            GlobalAnchor(
                id="test",
                position=Vec2(0, 0),
                orientation=0,
                anchor_type=AnchorType.SURFACE  # Invalid for GlobalAnchor
            )

    def test_local_anchor_creation(self):
        """Verify LocalAnchor can be created."""
        anchor = LocalAnchor(
            id="table_surface_top",
            position=Vec2(1, 1),
            orientation=0,
            anchor_type=AnchorType.SURFACE,
            source_object_id="table_1"
        )

        assert isinstance(anchor, AnchorPoint)
        assert anchor.anchor_type == AnchorType.SURFACE
        assert anchor.source_object_id == "table_1"

    def test_local_anchor_validation(self):
        """Verify LocalAnchor validates anchor type and source."""
        # Valid types should work
        for anchor_type in [AnchorType.SURFACE, AnchorType.EDGE]:
            anchor = LocalAnchor(
                id="test",
                position=Vec2(0, 0),
                orientation=0,
                anchor_type=anchor_type,
                source_object_id="obj_1"
            )
            assert anchor.anchor_type == anchor_type

        # Invalid type should raise error
        with pytest.raises(ValueError, match="LocalAnchor must have type"):
            LocalAnchor(
                id="test",
                position=Vec2(0, 0),
                orientation=0,
                anchor_type=AnchorType.WALL,  # Invalid for LocalAnchor
                source_object_id="obj_1"
            )

        # Missing source_object_id should raise error
        with pytest.raises(ValueError, match="must have a source_object_id"):
            LocalAnchor(
                id="test",
                position=Vec2(0, 0),
                orientation=0,
                anchor_type=AnchorType.SURFACE,
                source_object_id=""  # Empty source
            )


class TestAnchorSystem:
    """Test AnchorSystem methods (Task 2.2)."""

    def test_anchor_system_initialization(self):
        """Verify AnchorSystem initializes correctly."""
        system = AnchorSystem()

        assert system.global_anchors == []
        assert system.local_anchors == []
        assert system._anchor_index == {}

    def test_create_global_anchors_rectangular(self):
        """Verify create_global_anchors() for rectangular room."""
        system = AnchorSystem()

        # Create anchors for 10x10 room (half_size = 5)
        anchors = system.create_global_anchors(
            room_boundary=[(5, 5), (-5, 5), (-5, -5), (5, -5)],
            room_half_size=5.0
        )

        # Should create 4 walls + 4 corners + 1 center = 9 anchors
        assert len(anchors) == 9

        # Check wall anchors
        wall_anchors = [a for a in anchors if a.anchor_type == AnchorType.WALL]
        assert len(wall_anchors) == 4

        # Check corner anchors
        corner_anchors = [a for a in anchors if a.anchor_type == AnchorType.CORNER]
        assert len(corner_anchors) == 4

        # Check center anchor
        center_anchors = [a for a in anchors if a.anchor_type == AnchorType.CENTER]
        assert len(center_anchors) == 1
        assert center_anchors[0].position.x == 0
        assert center_anchors[0].position.y == 0

        # Verify anchors are added to system
        assert len(system.global_anchors) == 9
        assert len(system._anchor_index) == 9

    def test_create_global_anchors_polygonal(self):
        """Verify create_global_anchors() for polygonal room."""
        system = AnchorSystem()

        # Create anchors for triangular room
        triangle = [(0, 0), (4, 0), (2, 3)]
        anchors = system.create_global_anchors(room_boundary=triangle)

        # Should create 3 walls + 3 corners + 1 center = 7 anchors
        assert len(anchors) == 7

        # Check wall anchors
        wall_anchors = [a for a in anchors if a.anchor_type == AnchorType.WALL]
        assert len(wall_anchors) == 3

        # Check corner anchors
        corner_anchors = [a for a in anchors if a.anchor_type == AnchorType.CORNER]
        assert len(corner_anchors) == 3

        # Check center anchor (centroid of triangle)
        center_anchors = [a for a in anchors if a.anchor_type == AnchorType.CENTER]
        assert len(center_anchors) == 1
        # Centroid of triangle: ((0+4+2)/3, (0+0+3)/3) = (2, 1)
        assert abs(center_anchors[0].position.x - 2.0) < 0.01
        assert abs(center_anchors[0].position.y - 1.0) < 0.01

    def test_create_local_anchors(self):
        """Verify create_local_anchors() creates anchors from object."""
        system = AnchorSystem()

        # Create OBB for a table
        table_obb = OBB(
            center=Vec2(2.0, 3.0),
            half_extents=Vec2(1.0, 0.5),
            yaw_rad=0.0
        )

        anchors = system.create_local_anchors(
            object_id="table_1",
            obb=table_obb,
            object_type="table"
        )

        # Should create 5 surface anchors + 4 edge anchors = 9 anchors
        assert len(anchors) == 9

        # Check surface anchors
        surface_anchors = [a for a in anchors if a.anchor_type == AnchorType.SURFACE]
        assert len(surface_anchors) == 5

        # Check edge anchors
        edge_anchors = [a for a in anchors if a.anchor_type == AnchorType.EDGE]
        assert len(edge_anchors) == 4

        # Verify all anchors have correct source_object_id
        for anchor in anchors:
            assert anchor.source_object_id == "table_1"

        # Verify anchors are added to system
        assert len(system.local_anchors) == 9
        assert len(system._anchor_index) == 9

    def test_find_best_anchor_by_type(self):
        """Verify find_best_anchor() filters by type."""
        system = AnchorSystem()

        # Create some anchors
        system.create_global_anchors(
            room_boundary=[(5, 5), (-5, 5), (-5, -5), (5, -5)],
            room_half_size=5.0
        )

        # Find wall anchor
        wall_anchor = system.find_best_anchor(anchor_type=AnchorType.WALL)
        assert wall_anchor is not None
        assert wall_anchor.anchor_type == AnchorType.WALL

        # Find corner anchor
        corner_anchor = system.find_best_anchor(anchor_type=AnchorType.CORNER)
        assert corner_anchor is not None
        assert corner_anchor.anchor_type == AnchorType.CORNER

        # Find center anchor
        center_anchor = system.find_best_anchor(anchor_type=AnchorType.CENTER)
        assert center_anchor is not None
        assert center_anchor.anchor_type == AnchorType.CENTER

    def test_find_best_anchor_by_position(self):
        """Verify find_best_anchor() finds closest anchor to position."""
        system = AnchorSystem()

        # Create rectangular room anchors
        system.create_global_anchors(
            room_boundary=[(5, 5), (-5, 5), (-5, -5), (5, -5)],
            room_half_size=5.0
        )

        # Find anchor closest to north wall
        position_hint = Vec2(0, 4)
        anchor = system.find_best_anchor(
            anchor_type=AnchorType.WALL,
            position_hint=position_hint
        )

        assert anchor is not None
        assert anchor.id == "north_wall"

    def test_find_best_anchor_with_max_distance(self):
        """Verify find_best_anchor() respects max_distance."""
        system = AnchorSystem()

        # Create rectangular room anchors
        system.create_global_anchors(
            room_boundary=[(5, 5), (-5, 5), (-5, -5), (5, -5)],
            room_half_size=5.0
        )

        # Try to find anchor near (0, 0) with very small max_distance
        # Should not find wall anchors (they're 5 units away)
        anchor = system.find_best_anchor(
            anchor_type=AnchorType.WALL,
            position_hint=Vec2(0, 0),
            max_distance=1.0
        )

        assert anchor is None  # No wall within 1 unit of center

        # But should find center anchor
        anchor = system.find_best_anchor(
            anchor_type=AnchorType.CENTER,
            position_hint=Vec2(0, 0),
            max_distance=1.0
        )

        assert anchor is not None
        assert anchor.anchor_type == AnchorType.CENTER

    def test_cascade_anchors(self):
        """Verify cascade_anchors() creates anchors for new objects."""
        system = AnchorSystem()

        # Create initial global anchors
        system.create_global_anchors(
            room_boundary=[(5, 5), (-5, 5), (-5, -5), (5, -5)],
            room_half_size=5.0
        )

        initial_anchor_count = len(system._anchor_index)

        # Place some objects
        placed_objects = [
            {
                "id": "table_1",
                "obb": OBB(center=Vec2(0, 0), half_extents=Vec2(1, 0.5), yaw_rad=0),
                "type": "table"
            },
            {
                "id": "chair_1",
                "obb": OBB(center=Vec2(2, 0), half_extents=Vec2(0.4, 0.4), yaw_rad=0),
                "type": "chair"
            }
        ]

        # Cascade anchors
        system.cascade_anchors(placed_objects)

        # Should have created local anchors for both objects
        # Each object creates 9 anchors (5 surfaces + 4 edges)
        assert len(system.local_anchors) == 18
        assert len(system._anchor_index) == initial_anchor_count + 18

        # Verify anchors are created for correct objects
        table_anchors = [a for a in system.local_anchors
                        if a.source_object_id == "table_1"]
        assert len(table_anchors) == 9

        chair_anchors = [a for a in system.local_anchors
                        if a.source_object_id == "chair_1"]
        assert len(chair_anchors) == 9

    def test_get_anchor(self):
        """Verify get_anchor() retrieves anchor by ID."""
        system = AnchorSystem()

        system.create_global_anchors(
            room_boundary=[(5, 5), (-5, 5), (-5, -5), (5, -5)],
            room_half_size=5.0
        )

        # Get existing anchor
        anchor = system.get_anchor("north_wall")
        assert anchor is not None
        assert anchor.id == "north_wall"

        # Get non-existent anchor
        anchor = system.get_anchor("nonexistent")
        assert anchor is None

    def test_get_available_anchors(self):
        """Verify get_available_anchors() returns only available anchors."""
        system = AnchorSystem()

        system.create_global_anchors(
            room_boundary=[(5, 5), (-5, 5), (-5, -5), (5, -5)],
            room_half_size=5.0
        )

        # All anchors should be available initially
        available = system.get_available_anchors()
        assert len(available) == 9

        # Mark one anchor as used
        north_wall = system.get_anchor("north_wall")
        north_wall.mark_used()

        # Should now have 8 available anchors
        available = system.get_available_anchors()
        assert len(available) == 8

        # Filter by type
        available_walls = system.get_available_anchors(anchor_type=AnchorType.WALL)
        assert len(available_walls) == 3  # 4 walls - 1 used = 3

    def test_remove_anchors_for_object(self):
        """Verify remove_anchors_for_object() removes local anchors."""
        system = AnchorSystem()

        # Create local anchors for an object
        table_obb = OBB(center=Vec2(0, 0), half_extents=Vec2(1, 0.5), yaw_rad=0)
        system.create_local_anchors("table_1", table_obb, "table")

        assert len(system.local_anchors) == 9

        # Remove anchors for the object
        removed_count = system.remove_anchors_for_object("table_1")

        assert removed_count == 9
        assert len(system.local_anchors) == 0
        assert len(system._anchor_index) == 0

        # Try to remove again (should return 0)
        removed_count = system.remove_anchors_for_object("table_1")
        assert removed_count == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
