"""Anchor System for object positioning.

This module implements a hierarchical anchor system for positioning objects
in 3D scenes. Anchors provide reference points for object placement, including
global anchors (walls, corners, room center) and local anchors (object surfaces,
edges, functional points).
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from creator.placement.geometry import Vec2, OBB


class AnchorType(Enum):
    """Types of anchor points for object positioning."""

    # Global anchor types (room-level)
    WALL = "wall"
    CORNER = "corner"
    CENTER = "center"

    # Local anchor types (object-level)
    SURFACE = "surface"
    EDGE = "edge"

    # Architectural elements
    DOOR = "door"
    WINDOW = "window"


@dataclass
class AnchorPoint:
    """Base class for anchor points.

    Attributes:
        id: Unique identifier for this anchor
        position: 2D position in the scene (XY plane)
        orientation: Orientation angle in radians (yaw)
        anchor_type: Type of anchor (wall, corner, surface, etc.)
        availability: Whether this anchor is currently available for use
        constraints: List of constraint IDs that must be satisfied
        metadata: Additional anchor-specific data
    """

    id: str
    position: Vec2
    orientation: float  # yaw in radians
    anchor_type: AnchorType
    availability: bool = True
    constraints: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def is_available(self) -> bool:
        """Check if this anchor is available for use."""
        return self.availability

    def mark_used(self) -> None:
        """Mark this anchor as used (unavailable)."""
        self.availability = False

    def mark_available(self) -> None:
        """Mark this anchor as available."""
        self.availability = True


@dataclass
class GlobalAnchor(AnchorPoint):
    """Global anchor point (room-level).

    Global anchors are created from room geometry and include walls, corners,
    and the room center. They remain fixed throughout the placement process.

    Additional attributes:
        wall_direction: For wall anchors, the direction vector of the wall
        corner_type: For corner anchors, the type (NW, NE, SW, SE)
    """

    def __post_init__(self):
        """Validate global anchor configuration."""
        valid_types = {
            AnchorType.WALL, AnchorType.CORNER,
            AnchorType.CENTER, AnchorType.DOOR,
            AnchorType.WINDOW
        }
        if self.anchor_type not in valid_types:
            raise ValueError(
                f"GlobalAnchor must have type WALL, CORNER, CENTER, "
                f"DOOR, or WINDOW, got {self.anchor_type}"
            )


@dataclass
class LocalAnchor(AnchorPoint):
    """Local anchor point (object-level).

    Local anchors are created from placed objects and include surfaces, edges,
    and functional points. They are dynamically created as objects are placed.

    Additional attributes:
        source_object_id: ID of the object that provides this anchor
        surface_normal: For surface anchors, the normal vector
        edge_direction: For edge anchors, the direction along the edge
    """

    source_object_id: str = ""

    def __post_init__(self):
        """Validate local anchor configuration."""
        if self.anchor_type not in {AnchorType.SURFACE, AnchorType.EDGE}:
            raise ValueError(
                f"LocalAnchor must have type SURFACE or EDGE, "
                f"got {self.anchor_type}"
            )
        if not self.source_object_id:
            raise ValueError("LocalAnchor must have a source_object_id")


class AnchorSystem:
    """System for managing global and local anchors.

    The anchor system maintains a collection of anchor points and provides
    methods for creating, querying, and updating anchors as objects are placed.
    """

    def __init__(self):
        """Initialize the anchor system."""
        self.global_anchors: List[GlobalAnchor] = []
        self.local_anchors: List[LocalAnchor] = []
        self._anchor_index: Dict[str, AnchorPoint] = {}

    def create_global_anchors(
        self,
        room_boundary: List[Tuple[float, float]],
        room_half_size: Optional[float] = None
    ) -> List[GlobalAnchor]:
        """Create global anchors from room geometry.

        For rectangular rooms, creates:
        - 4 wall anchors (north, south, east, west)
        - 4 corner anchors (NW, NE, SW, SE)
        - 1 center anchor

        For arbitrary polygonal rooms, creates:
        - Wall anchors at the midpoint of each edge
        - Corner anchors at each vertex
        - 1 center anchor at the centroid

        Args:
            room_boundary: List of (x, y) coordinates defining the room boundary
            room_half_size: For rectangular rooms, the half-size (optional)

        Returns:
            List of created GlobalAnchor objects
        """
        anchors = []

        # Detect if room is rectangular
        if room_half_size is not None:
            # Rectangular room - create standard wall and corner anchors
            anchors.extend(self._create_rectangular_anchors(room_half_size))
        elif room_boundary and len(room_boundary) >= 3:
            # Arbitrary polygonal room
            anchors.extend(self._create_polygonal_anchors(room_boundary))
        else:
            # Default: create simple rectangular room with default size
            anchors.extend(self._create_rectangular_anchors(5.0))

        # Add anchors to the system
        for anchor in anchors:
            self.global_anchors.append(anchor)
            self._anchor_index[anchor.id] = anchor

        return anchors

    def _create_rectangular_anchors(self, room_half_size: float) -> List[GlobalAnchor]:
        """Create anchors for a rectangular room."""
        import math

        anchors = []

        # Wall anchors (at midpoint of each wall)
        walls = [
            ("north_wall", Vec2(0, room_half_size), math.pi),  # facing inward
            ("south_wall", Vec2(0, -room_half_size), 0.0),
            ("east_wall", Vec2(room_half_size, 0), -math.pi/2),
            ("west_wall", Vec2(-room_half_size, 0), math.pi/2),
        ]

        for wall_id, position, orientation in walls:
            anchors.append(GlobalAnchor(
                id=wall_id,
                position=position,
                orientation=orientation,
                anchor_type=AnchorType.WALL,
                metadata={"wall_name": wall_id}
            ))

        # Corner anchors
        corners = [
            ("corner_nw", Vec2(-room_half_size, room_half_size), 3*math.pi/4),
            ("corner_ne", Vec2(room_half_size, room_half_size), -3*math.pi/4),
            ("corner_sw", Vec2(-room_half_size, -room_half_size), math.pi/4),
            ("corner_se", Vec2(room_half_size, -room_half_size), -math.pi/4),
        ]

        for corner_id, position, orientation in corners:
            anchors.append(GlobalAnchor(
                id=corner_id,
                position=position,
                orientation=orientation,
                anchor_type=AnchorType.CORNER,
                metadata={"corner_name": corner_id}
            ))

        # Center anchor
        anchors.append(GlobalAnchor(
            id="room_center",
            position=Vec2(0, 0),
            orientation=0.0,
            anchor_type=AnchorType.CENTER,
            metadata={"is_center": True}
        ))

        return anchors

    def _create_polygonal_anchors(
        self,
        room_boundary: List[Tuple[float, float]]
    ) -> List[GlobalAnchor]:
        """Create anchors for an arbitrary polygonal room."""
        import math

        anchors = []
        n = len(room_boundary)

        if n < 3:
            raise ValueError("Room boundary must have at least 3 vertices")

        # Calculate centroid for center anchor
        cx = sum(p[0] for p in room_boundary) / n
        cy = sum(p[1] for p in room_boundary) / n

        # Create corner and wall anchors
        for i in range(n):
            p1 = room_boundary[i]
            p2 = room_boundary[(i + 1) % n]

            # Corner anchor at vertex
            corner_id = f"corner_{i}"
            corner_pos = Vec2(p1[0], p1[1])

            # Calculate corner orientation (bisector of adjacent edges)
            prev_p = room_boundary[(i - 1) % n]
            next_p = p2

            # Vector from corner to previous and next vertices
            v_prev = Vec2(prev_p[0] - p1[0], prev_p[1] - p1[1]).normalized()
            v_next = Vec2(next_p[0] - p1[0], next_p[1] - p1[1]).normalized()

            # Bisector direction (average of normalized vectors)
            bisector = (v_prev + v_next).normalized()
            corner_orientation = math.atan2(bisector.y, bisector.x)

            anchors.append(GlobalAnchor(
                id=corner_id,
                position=corner_pos,
                orientation=corner_orientation,
                anchor_type=AnchorType.CORNER,
                metadata={"vertex_index": i}
            ))

            # Wall anchor at edge midpoint
            wall_id = f"wall_{i}"
            wall_pos = Vec2((p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2)

            # Wall orientation (perpendicular to edge, facing inward)
            edge_vec = Vec2(p2[0] - p1[0], p2[1] - p1[1])
            # Perpendicular vector (rotate 90 degrees)
            perp = Vec2(-edge_vec.y, edge_vec.x).normalized()

            # Check if perpendicular points inward (toward centroid)
            to_center = Vec2(cx - wall_pos.x, cy - wall_pos.y)
            if perp.dot(to_center) < 0:
                perp = perp * (-1.0)

            wall_orientation = math.atan2(perp.y, perp.x)

            anchors.append(GlobalAnchor(
                id=wall_id,
                position=wall_pos,
                orientation=wall_orientation,
                anchor_type=AnchorType.WALL,
                metadata={"edge_index": i}
            ))

        # Center anchor at centroid
        anchors.append(GlobalAnchor(
            id="room_center",
            position=Vec2(cx, cy),
            orientation=0.0,
            anchor_type=AnchorType.CENTER,
            metadata={"is_center": True}
        ))

        return anchors

    def create_local_anchors(
        self,
        object_id: str,
        obb: OBB,
        object_type: str = "generic"
    ) -> List[LocalAnchor]:
        """Create local anchors from a placed object.

        Creates anchors at:
        - Object surfaces (top, front, back, left, right)
        - Object edges (front edge, back edge, side edges)

        Args:
            object_id: Unique identifier of the placed object
            obb: Oriented bounding box of the object
            object_type: Type of object (used to determine anchor types)

        Returns:
            List of created LocalAnchor objects
        """
        import math

        anchors = []

        # Get object axes
        c = math.cos(obb.yaw_rad)
        s = math.sin(obb.yaw_rad)
        axis_x = Vec2(c, s)  # Forward direction
        axis_y = Vec2(-s, c)  # Left direction

        # Surface anchors (at center of each face)
        surfaces = [
            ("top", obb.center, obb.yaw_rad),  # Top surface (for placing objects on)
            ("front", obb.center + axis_x * obb.half_extents.x, obb.yaw_rad),
            ("back", obb.center - axis_x * obb.half_extents.x, obb.yaw_rad + math.pi),
            ("left", obb.center + axis_y * obb.half_extents.y, obb.yaw_rad + math.pi/2),
            ("right", obb.center - axis_y * obb.half_extents.y, obb.yaw_rad - math.pi/2),
        ]

        for surface_name, position, orientation in surfaces:
            anchor_id = f"{object_id}_surface_{surface_name}"
            anchors.append(LocalAnchor(
                id=anchor_id,
                position=position,
                orientation=orientation,
                anchor_type=AnchorType.SURFACE,
                source_object_id=object_id,
                metadata={
                    "surface_name": surface_name,
                    "object_type": object_type
                }
            ))

        # Edge anchors (at midpoint of edges)
        edges = [
            ("front_left",
             obb.center + axis_x * obb.half_extents.x +
             axis_y * obb.half_extents.y,
             obb.yaw_rad),
            ("front_right",
             obb.center + axis_x * obb.half_extents.x -
             axis_y * obb.half_extents.y,
             obb.yaw_rad),
            ("back_left",
             obb.center - axis_x * obb.half_extents.x +
             axis_y * obb.half_extents.y,
             obb.yaw_rad + math.pi),
            ("back_right",
             obb.center - axis_x * obb.half_extents.x -
             axis_y * obb.half_extents.y,
             obb.yaw_rad + math.pi),
        ]

        for edge_name, position, orientation in edges:
            anchor_id = f"{object_id}_edge_{edge_name}"
            anchors.append(LocalAnchor(
                id=anchor_id,
                position=position,
                orientation=orientation,
                anchor_type=AnchorType.EDGE,
                source_object_id=object_id,
                metadata={
                    "edge_name": edge_name,
                    "object_type": object_type
                }
            ))

        # Add anchors to the system
        for anchor in anchors:
            self.local_anchors.append(anchor)
            self._anchor_index[anchor.id] = anchor

        return anchors

    def find_best_anchor(
        self,
        anchor_type: Optional[AnchorType] = None,
        position_hint: Optional[Vec2] = None,
        constraints: Optional[List[str]] = None,
        max_distance: Optional[float] = None
    ) -> Optional[AnchorPoint]:
        """Find the best anchor point matching the given criteria.

        Args:
            anchor_type: Filter by anchor type (None = any type)
            position_hint: Prefer anchors near this position
            constraints: List of constraint IDs that must be satisfied
            max_distance: Maximum distance from position_hint

        Returns:
            The best matching AnchorPoint, or None if no suitable anchor found
        """
        candidates = []

        # Collect all available anchors
        all_anchors = self.global_anchors + self.local_anchors

        for anchor in all_anchors:
            # Skip unavailable anchors
            if not anchor.is_available():
                continue

            # Filter by type
            if anchor_type is not None and anchor.anchor_type != anchor_type:
                continue

            # Filter by constraints
            if constraints is not None:
                if not all(c in anchor.constraints for c in constraints):
                    continue

            # Filter by distance
            if position_hint is not None and max_distance is not None:
                dist = (anchor.position - position_hint).length()
                if dist > max_distance:
                    continue

            candidates.append(anchor)

        if not candidates:
            return None

        # If no position hint, return first candidate
        if position_hint is None:
            return candidates[0]

        # Return anchor closest to position hint
        best_anchor = min(
            candidates,
            key=lambda a: (a.position - position_hint).length()
        )

        return best_anchor

    def cascade_anchors(self, placed_objects: List[Dict[str, Any]]) -> None:
        """Update anchor system after objects are placed.

        This method implements the cascade system where placed objects
        automatically become anchors for subsequent objects. It:
        1. Creates local anchors for newly placed objects
        2. Updates availability of anchors based on occlusion
        3. Removes anchors for deleted objects

        Args:
            placed_objects: List of placed object dictionaries with 'id', 'obb', 'type'
        """
        # Track which objects already have anchors
        existing_sources = {a.source_object_id for a in self.local_anchors}

        # Create anchors for new objects
        for obj in placed_objects:
            obj_id = obj.get("id", "")
            if not obj_id or obj_id in existing_sources:
                continue

            # Get OBB from object
            obb = obj.get("obb")
            if obb is None:
                continue

            obj_type = obj.get("type", "generic")
            self.create_local_anchors(obj_id, obb, obj_type)

        # Update anchor availability based on occlusion
        # (Simple implementation: all anchors remain available)
        # TODO: Implement occlusion detection for more sophisticated availability

    def get_anchor(self, anchor_id: str) -> Optional[AnchorPoint]:
        """Get an anchor by its ID.

        Args:
            anchor_id: The ID of the anchor to retrieve

        Returns:
            The AnchorPoint if found, None otherwise
        """
        return self._anchor_index.get(anchor_id)

    def get_available_anchors(
        self,
        anchor_type: Optional[AnchorType] = None
    ) -> List[AnchorPoint]:
        """Get all available anchors, optionally filtered by type.

        Args:
            anchor_type: Filter by anchor type (None = all types)

        Returns:
            List of available AnchorPoint objects
        """
        all_anchors = self.global_anchors + self.local_anchors

        available = [a for a in all_anchors if a.is_available()]

        if anchor_type is not None:
            available = [a for a in available if a.anchor_type == anchor_type]

        return available

    def remove_anchors_for_object(self, object_id: str) -> int:
        """Remove all local anchors associated with an object.

        Args:
            object_id: ID of the object whose anchors should be removed

        Returns:
            Number of anchors removed
        """
        # Find anchors to remove
        to_remove = [
            a for a in self.local_anchors
            if a.source_object_id == object_id
        ]

        # Remove from lists and index
        for anchor in to_remove:
            self.local_anchors.remove(anchor)
            if anchor.id in self._anchor_index:
                del self._anchor_index[anchor.id]

        return len(to_remove)
