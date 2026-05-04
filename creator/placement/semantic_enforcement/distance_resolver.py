"""Distance resolver for semantic plan enforcement.

This module converts relative positions to absolute coordinates by:
1. Building a dependency graph of object relationships
2. Performing topological sort to determine resolution order
3. Calculating absolute positions based on reference objects
4. Detecting circular dependencies
"""

import math
from typing import Any, Dict, List, Set, Tuple


class CircularDependencyError(Exception):
    """Raised when circular dependencies are detected in relative positioning."""
    pass


class DistanceResolver:
    """Resolves relative positions to absolute coordinates.
    
    The resolver handles:
    - Dependency graph building (which objects depend on which)
    - Topological sort for correct resolution order
    - Direction vector calculation for all direction types
    - Reference point calculation for all reference point types
    - Position calculation: reference_point + (direction_vector * distance)
    - Z-coordinate preservation based on object type
    """
    
    # Direction vectors (unit vectors in XY plane)
    # These are base directions before accounting for target orientation
    DIRECTION_VECTORS = {
        "front": (0.0, 1.0),
        "back": (0.0, -1.0),
        "left": (-1.0, 0.0),
        "right": (1.0, 0.0),
        "front_left": (-0.707, 0.707),
        "front_right": (0.707, 0.707),
        "back_left": (-0.707, -0.707),
        "back_right": (0.707, -0.707),
        "above": (0.0, 0.0),  # Special case: vertical only
        "below": (0.0, 0.0),  # Special case: vertical only
    }
    
    def resolve_positions(
        self,
        semantic_plan: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Resolve all relative positions to absolute coordinates.
        
        Process:
        1. Build dependency graph (which objects depend on which)
        2. Topological sort to determine resolution order
        3. Resolve positions in order (absolute first, then relative)
        4. For each relative position:
           - Get target object's position and size
           - Calculate direction vector based on target's orientation
           - Apply distance along direction
           - Set reference point (center, edge, etc.)
           - Preserve z-coordinate based on object type
        
        Args:
            semantic_plan: Semantic plan with relative positions
            
        Returns:
            Semantic plan with all positions as absolute coordinates
            
        Raises:
            CircularDependencyError: If circular dependencies detected
        """
        # Create a working copy
        resolved_plan = {
            "schema_version": semantic_plan["schema_version"],
            "room_size": semantic_plan["room_size"],
            "objects": [dict(obj) for obj in semantic_plan["objects"]],
            "metadata": semantic_plan.get("metadata", {}),
        }
        
        objects = resolved_plan["objects"]
        
        # Build dependency graph
        dependencies = self._build_dependency_graph(objects)
        
        # Detect circular dependencies
        self._detect_circular_dependencies(dependencies)
        
        # Topological sort to get resolution order
        resolution_order = self._topological_sort(objects, dependencies)
        
        # Resolve positions in order
        for obj_id in resolution_order:
            obj = self._find_object_by_id(objects, obj_id)
            if obj is None:
                continue
            
            position = obj["position"]
            
            # Skip if already absolute
            if position.get("absolute") is not None:
                continue
            
            # Resolve relative position
            relative = position.get("relative")
            if relative is None:
                continue
            
            target_id = relative["relative_to"]
            target_obj = self._find_object_by_id(objects, target_id)
            if target_obj is None:
                raise ValueError(
                    f"Object '{obj_id}' references non-existent "
                    f"target '{target_id}'"
                )
            
            # Calculate absolute position
            absolute_pos = self._calculate_absolute_position(
                obj, target_obj, relative
            )
            
            # Update object with absolute position
            obj["position"] = {"absolute": absolute_pos, "relative": None}
        
        return resolved_plan
    
    def _build_dependency_graph(
        self,
        objects: List[Dict[str, Any]],
    ) -> Dict[str, Set[str]]:
        """Build dependency graph: obj_id -> set of objects it depends on.
        
        Args:
            objects: List of semantic objects
            
        Returns:
            Dictionary mapping object ID to set of dependency IDs
        """
        dependencies: Dict[str, Set[str]] = {}
        
        for obj in objects:
            obj_id = obj["id"]
            dependencies[obj_id] = set()
            
            position = obj.get("position", {})
            relative = position.get("relative")
            
            if relative is not None:
                target_id = relative.get("relative_to")
                if target_id:
                    dependencies[obj_id].add(target_id)
        
        return dependencies
    
    def _detect_circular_dependencies(
        self,
        dependencies: Dict[str, Set[str]],
    ) -> None:
        """Detect circular dependencies using DFS.
        
        Args:
            dependencies: Dependency graph
            
        Raises:
            CircularDependencyError: If circular dependencies detected
        """
        visited: Set[str] = set()
        rec_stack: Set[str] = set()
        
        def dfs(node: str, path: List[str]) -> None:
            visited.add(node)
            rec_stack.add(node)
            path.append(node)
            
            for neighbor in dependencies.get(node, set()):
                if neighbor not in visited:
                    dfs(neighbor, path)
                elif neighbor in rec_stack:
                    # Found cycle
                    cycle_start = path.index(neighbor)
                    cycle = path[cycle_start:] + [neighbor]
                    raise CircularDependencyError(
                        f"Circular dependency detected: {' -> '.join(cycle)}"
                    )
            
            rec_stack.remove(node)
            path.pop()
        
        for node in dependencies:
            if node not in visited:
                dfs(node, [])
    
    def _topological_sort(
        self,
        objects: List[Dict[str, Any]],
        dependencies: Dict[str, Set[str]],
    ) -> List[str]:
        """Perform topological sort to determine resolution order.
        
        Objects with absolute positions come first, then objects with
        relative positions in dependency order.
        
        Args:
            objects: List of semantic objects
            dependencies: Dependency graph
            
        Returns:
            List of object IDs in resolution order
        """
        # Separate absolute and relative objects
        absolute_ids = []
        relative_ids = []
        
        for obj in objects:
            obj_id = obj["id"]
            position = obj.get("position", {})
            
            if position.get("absolute") is not None:
                absolute_ids.append(obj_id)
            else:
                relative_ids.append(obj_id)
        
        # Kahn's algorithm for topological sort
        in_degree = {obj_id: 0 for obj_id in relative_ids}
        
        for obj_id in relative_ids:
            for dep in dependencies.get(obj_id, set()):
                if dep in in_degree:
                    in_degree[obj_id] += 1
        
        queue = [obj_id for obj_id in relative_ids if in_degree[obj_id] == 0]
        sorted_relative = []
        
        while queue:
            current = queue.pop(0)
            sorted_relative.append(current)
            
            # Update in-degrees of dependent objects
            for obj_id in relative_ids:
                if current in dependencies.get(obj_id, set()):
                    in_degree[obj_id] -= 1
                    if in_degree[obj_id] == 0:
                        queue.append(obj_id)
        
        # Absolute positions first, then relative in dependency order
        return absolute_ids + sorted_relative
    
    def _calculate_absolute_position(
        self,
        obj: Dict[str, Any],
        target_obj: Dict[str, Any],
        relative: Dict[str, Any],
    ) -> Dict[str, float]:
        """Calculate absolute position from relative specification.
        
        Args:
            obj: Object being positioned
            target_obj: Target object (reference)
            relative: Relative position specification
            
        Returns:
            Absolute position as {x, y, z}
        """
        # Get target's absolute position
        target_pos = target_obj["position"].get("absolute")
        if target_pos is None:
            raise ValueError(
                f"Target object '{target_obj['id']}' does not have "
                f"absolute position yet"
            )
        
        # Get reference point on target
        reference_point = self._get_reference_point(
            target_obj,
            relative.get("reference_point", "center"),
        )
        
        # Get direction vector
        direction = relative["direction"]
        distance = relative["distance"]
        
        direction_vector = self._calculate_direction_vector(
            direction,
            target_obj,
        )
        
        # Calculate position: reference_point + (direction_vector * distance)
        x = reference_point[0] + direction_vector[0] * distance
        y = reference_point[1] + direction_vector[1] * distance
        
        # Determine z-coordinate based on object type and direction
        z = self._calculate_z_coordinate(
            obj, target_obj, direction, reference_point[2]
        )
        
        return {"x": x, "y": y, "z": z}
    
    def _calculate_direction_vector(
        self,
        direction: str,
        target_obj: Dict[str, Any],
    ) -> Tuple[float, float]:
        """Calculate unit direction vector accounting for target orientation.
        
        Directions:
        - front: along target's forward direction
        - back: opposite of forward
        - left: perpendicular left
        - right: perpendicular right
        - front_left: 45° between front and left
        - front_right: 45° between front and right
        - back_left: 45° between back and left
        - back_right: 45° between back and right
        - above/below: (0, 0) - handled separately in z-coordinate
        
        Args:
            direction: Direction string
            target_obj: Target object with orientation
            
        Returns:
            Unit direction vector (dx, dy)
        """
        if direction not in self.DIRECTION_VECTORS:
            raise ValueError(f"Unknown direction: {direction}")
        
        # Get base direction vector
        base_dx, base_dy = self.DIRECTION_VECTORS[direction]
        
        # For above/below, return (0, 0) as these are vertical only
        if direction in ("above", "below"):
            return (0.0, 0.0)
        
        # Get target's orientation (yaw angle in degrees)
        orientation = target_obj.get("orientation", {})
        absolute_orient = orientation.get("absolute")
        
        if absolute_orient is None:
            # Target doesn't have absolute orientation yet
            # This shouldn't happen if topological sort is correct
            raise ValueError(
                f"Target object '{target_obj['id']}' does not have "
                f"absolute orientation yet"
            )
        
        yaw_deg = absolute_orient.get("yaw_deg", 0.0)
        yaw_rad = math.radians(yaw_deg)
        
        # Rotate base direction by target's yaw
        cos_yaw = math.cos(yaw_rad)
        sin_yaw = math.sin(yaw_rad)
        
        dx = base_dx * cos_yaw - base_dy * sin_yaw
        dy = base_dx * sin_yaw + base_dy * cos_yaw
        
        return (dx, dy)
    
    def _get_reference_point(
        self,
        target_obj: Dict[str, Any],
        reference_point: str,
    ) -> Tuple[float, float, float]:
        """Get coordinates of reference point on target object.
        
        Reference points:
        - center: geometric center
        - front_edge: center of front face
        - back_edge: center of back face
        - left_edge: center of left face
        - right_edge: center of right face
        - top_surface: center of top face
        
        Args:
            target_obj: Target object
            reference_point: Reference point type
            
        Returns:
            Reference point coordinates (x, y, z)
        """
        # Get target's absolute position (center)
        target_pos = target_obj["position"]["absolute"]
        cx = target_pos["x"]
        cy = target_pos["y"]
        cz = target_pos["z"]
        
        # Get target's size
        size = target_obj["size"]
        width = size["width"]
        length = size["length"]
        height = size["height"]
        
        # Get target's orientation (use default if not present)
        orientation = target_obj.get("orientation", {}).get("absolute", {})
        yaw_deg = orientation.get("yaw_deg", 0.0)
        yaw_rad = math.radians(yaw_deg)
        
        cos_yaw = math.cos(yaw_rad)
        sin_yaw = math.sin(yaw_rad)
        
        # Calculate reference point offset from center
        if reference_point == "center":
            return (cx, cy, cz)
        
        elif reference_point == "front_edge":
            # Front is along +Y in local coordinates
            offset_x = 0.0
            offset_y = length / 2.0
            
        elif reference_point == "back_edge":
            # Back is along -Y in local coordinates
            offset_x = 0.0
            offset_y = -length / 2.0
            
        elif reference_point == "left_edge":
            # Left is along -X in local coordinates
            offset_x = -width / 2.0
            offset_y = 0.0
            
        elif reference_point == "right_edge":
            # Right is along +X in local coordinates
            offset_x = width / 2.0
            offset_y = 0.0
            
        elif reference_point == "top_surface":
            # Top surface center
            return (cx, cy, cz + height / 2.0)
        
        else:
            raise ValueError(f"Unknown reference point: {reference_point}")
        
        # Rotate offset by target's yaw
        rotated_x = offset_x * cos_yaw - offset_y * sin_yaw
        rotated_y = offset_x * sin_yaw + offset_y * cos_yaw
        
        return (cx + rotated_x, cy + rotated_y, cz)
    
    def _calculate_z_coordinate(
        self,
        obj: Dict[str, Any],
        target_obj: Dict[str, Any],
        direction: str,
        reference_z: float,
    ) -> float:
        """Calculate z-coordinate based on object type and direction.
        
        Rules:
        - Floor objects (furniture): z = half_height (bottom at z=0)
        - Table-top objects: z = table_surface_height + half_height
        - "above" direction: z = reference_z + target_height/2 + obj_height/2
        - "below" direction: z = reference_z - target_height/2 - obj_height/2
        - Other directions: preserve object type default
        
        Args:
            obj: Object being positioned
            target_obj: Target object
            direction: Direction string
            reference_z: Z-coordinate of reference point
            
        Returns:
            Z-coordinate for object
        """
        obj_height = obj["size"]["height"]
        obj_type = obj.get("type", "furniture")
        
        # Handle vertical directions
        if direction == "above":
            target_height = target_obj["size"]["height"]
            # Place object on top of target
            return reference_z + target_height / 2.0 + obj_height / 2.0
        
        elif direction == "below":
            target_height = target_obj["size"]["height"]
            # Place object below target
            return reference_z - target_height / 2.0 - obj_height / 2.0
        
        # For horizontal directions, determine z based on object type
        if obj_type == "furniture":
            # Floor objects: bottom at z=0
            return obj_height / 2.0
        
        else:
            # Small objects: check if target is a surface
            target_type = target_obj.get("type", "furniture")
            
            if target_type == "furniture":
                # Place on top of furniture surface
                target_pos = target_obj["position"]["absolute"]
                target_height = target_obj["size"]["height"]
                target_z = target_pos["z"]
                surface_z = target_z + target_height / 2.0
                return surface_z + obj_height / 2.0
            else:
                # Default: floor level
                return obj_height / 2.0
    
    def _find_object_by_id(
        self,
        objects: List[Dict[str, Any]],
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
