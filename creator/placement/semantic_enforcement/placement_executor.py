"""Placement Executor for Semantic Plan Enforcement.

This module implements the PlacementExecutor class, which is a "dumb executor"
that faithfully places objects exactly as specified in the semantic plan without
any modifications or "smart" adjustments.

The PlacementExecutor's role is to:
1. Take a semantic plan with all absolute positions and orientations
2. Place ALL objects from the plan without omissions
3. Use exact positions and orientations (no collision avoidance or adjustments)
4. Preserve all metadata from SemanticPlan to PlacementSolution
5. Log warnings for potential issues but still place objects as instructed
"""

import logging
from typing import Any, Dict, List

from creator.placement.semantic_enforcement.models import (
    PlacedObject,
    PlacementSolution,
    SemanticPlan,
)

logger = logging.getLogger(__name__)


class PlacementExecutor:
    """Dumb executor that places objects exactly as specified in semantic plan.
    
    This class does NOT:
    - Modify positions to avoid collisions
    - Adjust orientations for aesthetics
    - Apply automatic spacing
    - Optimize layout
    - Make any autonomous decisions
    
    It ONLY:
    - Reads positions and orientations from plan
    - Places objects at those exact coordinates
    - Preserves all metadata
    - Logs warnings (but doesn't act on them)
    
    **Validates: Requirements 4.1, 4.2, 4.3, 4.4, 4.5, 4.6, 4.7, 4.8**
    """
    
    def __init__(self, collision_check_enabled: bool = True):
        """Initialize PlacementExecutor.
        
        Args:
            collision_check_enabled: Whether to check for potential collisions
                                    (only logs warnings, doesn't prevent placement)
        """
        self.collision_check_enabled = collision_check_enabled
    
    def execute_placement(
        self,
        semantic_plan: SemanticPlan,
    ) -> PlacementSolution:
        """Execute placement exactly as specified in semantic plan.
        
        This method places ALL objects from the semantic plan without any
        modifications. It uses exact positions and orientations from the plan,
        preserves all metadata, and logs warnings for potential issues.
        
        Args:
            semantic_plan: Semantic plan with all absolute positions and orientations
        
        Returns:
            PlacementSolution with all objects placed exactly as specified
        
        Raises:
            ValueError: If semantic plan contains relative positions or orientations
                       (must be resolved before calling this method)
        
        **Validates: Requirements 4.1, 4.2, 4.3, 4.4, 4.5, 4.6, 4.7, 4.8**
        """
        logger.info(
            f"[PlacementExecutor] Starting placement execution for "
            f"{len(semantic_plan.objects)} objects"
        )
        
        # Validate that all positions and orientations are absolute
        self._validate_all_absolute(semantic_plan)
        
        # Place all objects without modifications
        placed_objects: List[PlacedObject] = []
        
        for semantic_obj in semantic_plan.objects:
            # Extract absolute position (guaranteed to exist by validation)
            position = semantic_obj.position.absolute
            if position is None:
                # This should never happen due to validation, but be defensive
                raise ValueError(
                    f"Object {semantic_obj.id} has no absolute position. "
                    f"All positions must be resolved before placement execution."
                )
            
            # Extract absolute orientation (guaranteed to exist by validation)
            orientation = semantic_obj.orientation.absolute
            if orientation is None:
                # This should never happen due to validation, but be defensive
                raise ValueError(
                    f"Object {semantic_obj.id} has no absolute orientation. "
                    f"All orientations must be resolved before placement execution."
                )
            
            # Create placed object with exact coordinates from plan
            placed_obj = PlacedObject(
                id=semantic_obj.id,
                Model=semantic_obj.Model,
                type=semantic_obj.type,
                size=semantic_obj.size.copy(),  # Copy to avoid mutation
                is_static=semantic_obj.is_static,
                model_loc=semantic_obj.model_loc,
                position=position.copy(),  # Copy to avoid mutation
                orientation=orientation.copy(),  # Copy to avoid mutation
                metadata=semantic_obj.metadata.copy(),  # Preserve original metadata
            )
            
            placed_objects.append(placed_obj)
            
            logger.debug(
                f"[PlacementExecutor] Placed object {semantic_obj.id} at "
                f"position ({position['x']:.3f}, {position['y']:.3f}, {position['z']:.3f}), "
                f"orientation (yaw={orientation['yaw_deg']:.1f}°)"
            )
        
        # Verify object count matches
        if len(placed_objects) != len(semantic_plan.objects):
            logger.error(
                f"[PlacementExecutor] Object count mismatch: "
                f"semantic plan has {len(semantic_plan.objects)} objects, "
                f"but only {len(placed_objects)} were placed"
            )
            raise RuntimeError(
                f"Object count mismatch: expected {len(semantic_plan.objects)}, "
                f"got {len(placed_objects)}"
            )
        
        # Verify all object IDs match
        semantic_ids = {obj.id for obj in semantic_plan.objects}
        placed_ids = {obj.id for obj in placed_objects}
        
        if semantic_ids != placed_ids:
            missing_ids = semantic_ids - placed_ids
            extra_ids = placed_ids - semantic_ids
            
            error_msg = "Object ID mismatch: "
            if missing_ids:
                error_msg += f"missing {missing_ids}, "
            if extra_ids:
                error_msg += f"extra {extra_ids}"
            
            logger.error(f"[PlacementExecutor] {error_msg}")
            raise RuntimeError(error_msg)
        
        # Check for potential collisions (log warnings only)
        if self.collision_check_enabled:
            self._check_collisions(placed_objects)
        
        # Check for objects outside room boundaries (log warnings only)
        self._check_room_boundaries(placed_objects, semantic_plan.room_size)
        
        logger.info(
            f"[PlacementExecutor] Placement execution completed successfully. "
            f"Placed {len(placed_objects)} objects."
        )
        
        # Create placement solution
        solution = PlacementSolution(
            objects=placed_objects,
            room_size=semantic_plan.room_size.copy(),
            metadata={
                "schema_version": semantic_plan.schema_version,
                "executor": "PlacementExecutor",
                "collision_check_enabled": self.collision_check_enabled,
                **semantic_plan.metadata,
            },
        )
        
        return solution
    
    def _validate_all_absolute(self, semantic_plan: SemanticPlan) -> None:
        """Validate that all positions and orientations are absolute.
        
        Args:
            semantic_plan: Semantic plan to validate
        
        Raises:
            ValueError: If any object has relative position or orientation
        """
        for obj in semantic_plan.objects:
            # Check position
            if obj.position.absolute is None:
                raise ValueError(
                    f"Object {obj.id} has relative position. "
                    f"All positions must be resolved to absolute coordinates "
                    f"before calling execute_placement()."
                )
            
            if obj.position.relative is not None:
                logger.warning(
                    f"Object {obj.id} has both absolute and relative position. "
                    f"Using absolute position only."
                )
            
            # Check orientation
            if obj.orientation.absolute is None:
                raise ValueError(
                    f"Object {obj.id} has relative orientation. "
                    f"All orientations must be resolved to absolute angles "
                    f"before calling execute_placement()."
                )
            
            if obj.orientation.relative is not None:
                logger.warning(
                    f"Object {obj.id} has both absolute and relative orientation. "
                    f"Using absolute orientation only."
                )
    
    def _check_collisions(self, placed_objects: List[PlacedObject]) -> None:
        """Check for potential collisions between objects.
        
        This method only logs warnings - it does NOT prevent placement or
        modify positions. The LLM is responsible for avoiding collisions.
        
        Args:
            placed_objects: List of placed objects to check
        """
        # Simple AABB (Axis-Aligned Bounding Box) collision detection
        for i, obj1 in enumerate(placed_objects):
            for obj2 in placed_objects[i + 1:]:
                if self._objects_overlap(obj1, obj2):
                    logger.warning(
                        f"[PlacementExecutor] Potential collision detected between "
                        f"{obj1.id} and {obj2.id}. Objects will be placed as specified."
                    )
    
    def _objects_overlap(self, obj1: PlacedObject, obj2: PlacedObject) -> bool:
        """Check if two objects' bounding boxes overlap.
        
        Uses simple AABB (Axis-Aligned Bounding Box) collision detection.
        Note: This doesn't account for rotation, so it's conservative.
        
        Args:
            obj1: First object
            obj2: Second object
        
        Returns:
            True if bounding boxes overlap, False otherwise
        """
        # Get object centers
        x1, y1, z1 = obj1.position['x'], obj1.position['y'], obj1.position['z']
        x2, y2, z2 = obj2.position['x'], obj2.position['y'], obj2.position['z']
        
        # Get object half-sizes
        hw1, hl1, hh1 = (
            obj1.size['width'] / 2,
            obj1.size['length'] / 2,
            obj1.size['height'] / 2,
        )
        hw2, hl2, hh2 = (
            obj2.size['width'] / 2,
            obj2.size['length'] / 2,
            obj2.size['height'] / 2,
        )
        
        # Check overlap in each axis
        x_overlap = abs(x1 - x2) < (hw1 + hw2)
        y_overlap = abs(y1 - y2) < (hl1 + hl2)
        z_overlap = abs(z1 - z2) < (hh1 + hh2)
        
        return x_overlap and y_overlap and z_overlap
    
    def _check_room_boundaries(
        self,
        placed_objects: List[PlacedObject],
        room_size: Dict[str, float],
    ) -> None:
        """Check if any objects are outside room boundaries.
        
        This method only logs warnings - it does NOT prevent placement or
        modify positions. The LLM is responsible for keeping objects in bounds.
        
        Args:
            placed_objects: List of placed objects to check
            room_size: Room dimensions (width, length, height)
        """
        room_width = room_size['width']
        room_length = room_size['length']
        room_height = room_size['height']
        
        for obj in placed_objects:
            x, y, z = obj.position['x'], obj.position['y'], obj.position['z']
            hw, hl, hh = (
                obj.size['width'] / 2,
                obj.size['length'] / 2,
                obj.size['height'] / 2,
            )
            
            # Check if object extends outside room boundaries
            out_of_bounds = False
            reasons = []
            
            if x - hw < -room_width / 2 or x + hw > room_width / 2:
                out_of_bounds = True
                reasons.append(f"x={x:.2f} (room width: {room_width})")
            
            if y - hl < -room_length / 2 or y + hl > room_length / 2:
                out_of_bounds = True
                reasons.append(f"y={y:.2f} (room length: {room_length})")
            
            if z - hh < 0 or z + hh > room_height:
                out_of_bounds = True
                reasons.append(f"z={z:.2f} (room height: {room_height})")
            
            if out_of_bounds:
                logger.warning(
                    f"[PlacementExecutor] Object {obj.id} is outside room boundaries: "
                    f"{', '.join(reasons)}. Object will be placed as specified."
                )
