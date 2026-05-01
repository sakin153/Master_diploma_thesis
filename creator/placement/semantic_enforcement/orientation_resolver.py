"""Orientation resolver for semantic plan enforcement.

This module converts relative orientations to absolute yaw angles by:
1. Calculating the angle from source to target position
2. Adjusting based on target's orientation and facing_direction
3. Supporting facing_away modifier
4. Normalizing angles to [0, 360) range
"""

import math
from typing import Any, Dict, Tuple


class OrientationResolver:
    """Resolves relative orientations to absolute yaw angles.

    The resolver handles:
    - Calculating facing angle using atan2 from source to target position
    - Adjusting angle based on target's orientation and facing_direction
    - Supporting facing_away modifier (add 180° to calculated angle)
    - Normalizing angles to [0, 360) range
    """

    def resolve_orientations(
        self,
        semantic_plan: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Resolve all relative orientations to absolute yaw angles.

        Process:
        1. For each object with relative orientation:
           - Get source object's absolute position (must be resolved first)
           - Get target object's absolute position and orientation
           - Calculate angle to face target based on facing_direction
           - Apply facing_away modifier if specified
           - Set absolute yaw_deg
           - Normalize to [0, 360) range

        Args:
            semantic_plan: Semantic plan with relative orientations

        Returns:
            Semantic plan with all orientations as absolute angles
        """
        # Create a working copy
        resolved_plan = {
            "schema_version": semantic_plan["schema_version"],
            "room_size": semantic_plan["room_size"],
            "objects": [dict(obj) for obj in semantic_plan["objects"]],
            "metadata": semantic_plan.get("metadata", {}),
        }

        objects = resolved_plan["objects"]

        # Resolve orientations for each object
        for obj in objects:
            orientation = obj.get("orientation", {})

            # Skip if already absolute
            if orientation.get("absolute") is not None:
                continue

            # Resolve relative orientation
            relative = orientation.get("relative")
            if relative is None:
                continue

            target_id = relative["facing"]
            target_obj = self._find_object_by_id(objects, target_id)
            if target_obj is None:
                raise ValueError(
                    f"Object '{obj['id']}' references non-existent "
                    f"target '{target_id}' for orientation"
                )

            # Get source position (must be absolute by now)
            source_pos = obj["position"].get("absolute")
            if source_pos is None:
                raise ValueError(
                    f"Object '{obj['id']}' does not have absolute position yet. "
                    f"Distance resolution must be performed before orientation resolution."
                )

            # Get target position (must be absolute)
            target_pos = target_obj["position"].get("absolute")
            if target_pos is None:
                raise ValueError(
                    f"Target object '{target_id}' does not have absolute position yet"
                )

            # Get target orientation (must be absolute)
            target_orientation = target_obj["orientation"].get("absolute")
            if target_orientation is None:
                raise ValueError(
                    f"Target object '{target_id}' does not have absolute orientation yet"
                )

            # Calculate facing angle
            facing_direction = relative.get("facing_direction", "front")
            facing_away = relative.get("facing_away", False)

            yaw_deg = self._calculate_facing_angle(
                source_pos=(source_pos["x"], source_pos["y"], source_pos["z"]),
                target_pos=(target_pos["x"], target_pos["y"], target_pos["z"]),
                target_orientation=target_orientation["yaw_deg"],
                facing_direction=facing_direction,
                facing_away=facing_away,
            )

            # Update object with absolute orientation
            # Preserve pitch and roll as 0 if not specified
            obj["orientation"] = {
                "absolute": {
                    "yaw_deg": yaw_deg,
                    "pitch_deg": 0.0,
                    "roll_deg": 0.0,
                },
                "relative": None,
            }

        return resolved_plan

    def _calculate_facing_angle(
        self,
        source_pos: Tuple[float, float, float],
        target_pos: Tuple[float, float, float],
        target_orientation: float,
        facing_direction: str,
        facing_away: bool = False,
    ) -> float:
        """Calculate yaw angle to face target.

        The calculation:
        1. Determine which side of the target to face based on facing_direction
        2. Calculate the point on that side of the target to look at
        3. Calculate the angle from source to that point using atan2
        4. Apply facing_away modifier (add 180°)
        5. Normalize to [0, 360) range

        Coordinate system:
        - +X is right
        - +Y is forward
        - yaw=0° means facing +Y direction
        - yaw=90° means facing -X direction (turned left)
        - yaw=180° means facing -Y direction (turned around)
        - yaw=270° means facing +X direction (turned right)

        Args:
            source_pos: Source object position (x, y, z)
            target_pos: Target object position (x, y, z)
            target_orientation: Target object's yaw angle in degrees
            facing_direction: Which side of target to face
                - "front": face the front of target
                - "back": face the back of target
                - "left_side": face the left side of target
                - "right_side": face the right side of target
            facing_away: If True, face away from target instead of towards

        Returns:
            Yaw angle in degrees, normalized to [0, 360)
        """
        # Determine which side of the target to face
        # Each side is defined by the target's orientation:
        # - front: target_orientation + 0° (the direction target is facing)
        # - back: target_orientation + 180° (opposite of front)
        # - left_side: target_orientation + 90° (perpendicular left)
        # - right_side: target_orientation - 90° (perpendicular right)
        
        if facing_direction == "front":
            # Face the front of the target
            # The front face is perpendicular to the target's forward direction
            # We want to look at the front face, which means looking towards
            # the direction the target is facing
            side_angle_offset = 0.0
            
        elif facing_direction == "back":
            # Face the back of the target
            # The back is opposite to the front
            side_angle_offset = 180.0
            
        elif facing_direction == "left_side":
            # Face the left side of the target
            # Left is 90° counterclockwise from front
            side_angle_offset = 90.0
            
        elif facing_direction == "right_side":
            # Face the right side of the target
            # Right is 90° clockwise from front (or -90°)
            side_angle_offset = -90.0
            
        else:
            raise ValueError(f"Unknown facing_direction: {facing_direction}")
        
        # Calculate the direction of the side we want to face
        # This is the target's orientation plus the side offset
        side_direction_deg = target_orientation + side_angle_offset
        
        # Calculate a point on that side of the target
        # We'll use a point 1 meter away from the target center in the side direction
        side_direction_rad = math.radians(side_direction_deg)
        side_point_x = target_pos[0] + math.sin(side_direction_rad)
        side_point_y = target_pos[1] + math.cos(side_direction_rad)
        
        # Calculate the angle from source to the side point
        # This is the angle the source needs to face to look at that side
        dx = side_point_x - source_pos[0]
        dy = side_point_y - source_pos[1]
        
        # atan2(dy, dx) gives angle where 0° is +X (right)
        # But our yaw convention is 0° = +Y (forward)
        # So we need to convert: yaw = 90° - atan2_angle
        # Or equivalently: yaw = atan2(dx, dy) where we swap dx and dy
        yaw_deg = math.degrees(math.atan2(dx, dy))
        
        # Apply facing_away modifier
        if facing_away:
            yaw_deg += 180.0
        
        # Normalize to [0, 360) range
        yaw_deg = self._normalize_angle(yaw_deg)
        
        return yaw_deg

    def _normalize_angle(self, angle_deg: float) -> float:
        """Normalize angle to [0, 360) range.

        Args:
            angle_deg: Angle in degrees

        Returns:
            Normalized angle in [0, 360) range
        """
        # Use modulo to bring angle into [0, 360) range
        normalized = angle_deg % 360.0

        # Handle negative angles
        if normalized < 0:
            normalized += 360.0

        return normalized

    def _find_object_by_id(
        self,
        objects: list[Dict[str, Any]],
        obj_id: str,
    ) -> Dict[str, Any] | None:
        """Find object by ID in list.

        Args:
            objects: List of objects
            obj_id: Object ID to find

        Returns:
            Object dictionary or None if not found
        """
        for obj in objects:
            if obj["id"] == obj_id:
                return obj
        return None
