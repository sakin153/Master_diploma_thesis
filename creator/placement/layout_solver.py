"""Layout Solver for computing precise object positions based on constraints.

This module implements the LayoutSolver class that computes exact coordinates
for objects based on semantic plans and spatial constraints. It adaptively
chooses between DFS+Beam Search for simple scenes and MILP for complex scenes.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

from creator.placement.constraint_engine import (
    ConstraintEngine,
    ConstraintSolution,
)
from creator.placement.constraints import Constraint
from creator.placement.floor_solver import solve_floor_placements
from creator.placement.geometry import (
    Vec2,
    model_to_obb,
    obb_overlap,
)
from creator.placement.scene_graph import SceneGraph


class PlacementAlgorithm(Enum):
    """Available placement algorithms."""
    DFS_BEAM_SEARCH = "dfs_beam_search"
    MILP = "milp"
    HYBRID = "hybrid"


@dataclass
class SceneComplexity:
    """Metrics for scene complexity analysis.

    Attributes:
        num_objects: Total number of objects to place
        num_constraints: Total number of constraints
        num_hard_constraints: Number of hard constraints
        num_soft_constraints: Number of soft constraints
        max_constraint_degree: Maximum number of constraints per object
        has_stacking: Whether scene includes stacking
        has_arbitrary_geometry: Whether room has non-rectangular geometry
        complexity_score: Overall complexity score (0.0 to 1.0)
    """

    num_objects: int = 0
    num_constraints: int = 0
    num_hard_constraints: int = 0
    num_soft_constraints: int = 0
    max_constraint_degree: int = 0
    has_stacking: bool = False
    has_arbitrary_geometry: bool = False
    complexity_score: float = 0.0


@dataclass
class PlacementSolution:
    """Solution to a placement problem.

    Attributes:
        placed_objects: List of objects with computed positions
        constraint_solution: Solution from constraint engine
        algorithm_used: Algorithm that was used for placement
        success: Whether placement was successful
        execution_time: Time taken to compute solution (seconds)
        quality_score: Overall quality score of the solution
        alternatives: List of alternative placements if primary failed
        failure_reason: Reason for failure if success is False
    """

    placed_objects: List[Dict[str, Any]] = field(default_factory=list)
    constraint_solution: Optional[ConstraintSolution] = None
    algorithm_used: PlacementAlgorithm = PlacementAlgorithm.DFS_BEAM_SEARCH
    success: bool = False
    execution_time: float = 0.0
    quality_score: float = 0.0
    alternatives: List[List[Dict[str, Any]]] = field(default_factory=list)
    failure_reason: Optional[str] = None


@dataclass
class RoomGeometry:
    """Geometry specification for a room.

    Attributes:
        boundary_polygon: Outer boundary as list of 2D points
        holes: List of interior holes (each is a list of 2D points)
        wall_height: Height of walls in meters
        architectural_elements: List of architectural features (doors, windows)
        is_rectangular: Whether the room is a simple rectangle
        half_size: Half-size for rectangular rooms (for backward compatibility)
    """

    boundary_polygon: List[Vec2] = field(default_factory=list)
    holes: List[List[Vec2]] = field(default_factory=list)
    wall_height: float = 2.5
    architectural_elements: List[Dict[str, Any]] = field(default_factory=list)
    is_rectangular: bool = True
    half_size: float = 5.0


@dataclass
class StackingLevel:
    """Represents a level in a stacking hierarchy.

    Attributes:
        level: Level number (0 = floor, 1 = first level, etc.)
        supporting_object: Object providing support (None for floor)
        supported_objects: Objects placed on this level
        total_weight: Total weight of objects on this level
        support_capacity: Maximum weight this level can support
        is_stable: Whether this level is physically stable
    """

    level: int = 0
    supporting_object: Optional[Dict[str, Any]] = None
    supported_objects: List[Dict[str, Any]] = field(default_factory=list)
    total_weight: float = 0.0
    support_capacity: float = float('inf')
    is_stable: bool = True


class LayoutSolver:
    """Solver for computing precise object positions.

    The LayoutSolver takes a semantic plan with constraints and computes
    exact 3D coordinates for all objects. It adaptively chooses between
    DFS+Beam Search for simple scenes and MILP for complex scenes.

    Attributes:
        constraint_engine: Engine for managing and solving constraints
        algorithm_preference: Preferred algorithm (None for adaptive)
        dfs_threshold: Max objects for DFS (above this, use MILP)
        constraint_threshold: Max constraints for DFS
    """

    def __init__(
        self,
        constraint_engine: Optional[ConstraintEngine] = None,
        algorithm_preference: Optional[PlacementAlgorithm] = None,
        dfs_threshold: int = 20,
        constraint_threshold: int = 50,
    ):
        """Initialize the layout solver.

        Args:
            constraint_engine: Constraint engine to use (creates new if None)
            algorithm_preference: Preferred algorithm (None for adaptive)
            dfs_threshold: Max objects for DFS algorithm
            constraint_threshold: Max constraints for DFS algorithm
        """
        self.constraint_engine = constraint_engine or ConstraintEngine()
        self.algorithm_preference = algorithm_preference
        self.dfs_threshold = dfs_threshold
        self.constraint_threshold = constraint_threshold

    def choose_algorithm(
        self, scene_complexity: SceneComplexity
    ) -> PlacementAlgorithm:
        """Choose optimal algorithm based on scene complexity.

        Args:
            scene_complexity: Complexity metrics for the scene

        Returns:
            The chosen placement algorithm
        """
        # If user specified preference, use it
        if self.algorithm_preference is not None:
            return self.algorithm_preference

        # Adaptive selection based on complexity
        if scene_complexity.num_objects < self.dfs_threshold:
            if scene_complexity.num_constraints < self.constraint_threshold:
                # Simple scene: use DFS + Beam Search
                return PlacementAlgorithm.DFS_BEAM_SEARCH

        # Complex scene: use MILP
        # Note: MILP implementation is a placeholder for now
        # In practice, we fall back to DFS for MVP
        return PlacementAlgorithm.DFS_BEAM_SEARCH

    def analyze_complexity(
        self,
        scene_graph: SceneGraph,
        constraints: List[Constraint],
        room_geometry: RoomGeometry,
    ) -> SceneComplexity:
        """Analyze scene complexity to guide algorithm selection.

        Args:
            scene_graph: Scene graph with objects
            constraints: List of spatial constraints
            room_geometry: Room geometry specification

        Returns:
            SceneComplexity metrics
        """
        complexity = SceneComplexity()

        # Count objects (excluding root)
        complexity.num_objects = len(scene_graph.get_all_objects())

        # Count constraints
        complexity.num_constraints = len(constraints)
        complexity.num_hard_constraints = sum(
            1 for c in constraints if c.is_hard
        )
        complexity.num_soft_constraints = (
            complexity.num_constraints - complexity.num_hard_constraints
        )

        # Compute max constraint degree
        constraint_counts: Dict[str, int] = {}
        for constraint in constraints:
            constraint_counts[constraint.source_object] = (
                constraint_counts.get(constraint.source_object, 0) + 1
            )
            if constraint.target_object:
                constraint_counts[constraint.target_object] = (
                    constraint_counts.get(constraint.target_object, 0) + 1
                )

        complexity.max_constraint_degree = (
            max(constraint_counts.values()) if constraint_counts else 0
        )

        # Check for stacking (objects with parent-child relationships)
        for node in scene_graph.get_all_objects():
            if node.parent and node.parent != scene_graph.root:
                complexity.has_stacking = True
                break

        # Check for arbitrary geometry
        complexity.has_arbitrary_geometry = not room_geometry.is_rectangular

        # Compute overall complexity score (0.0 to 1.0)
        # Weighted combination of factors
        obj_score = min(1.0, complexity.num_objects / 50.0)
        constraint_score = min(1.0, complexity.num_constraints / 100.0)
        degree_score = min(1.0, complexity.max_constraint_degree / 10.0)
        stacking_score = 0.3 if complexity.has_stacking else 0.0
        geometry_score = 0.2 if complexity.has_arbitrary_geometry else 0.0

        complexity.complexity_score = (
            0.3 * obj_score +
            0.3 * constraint_score +
            0.2 * degree_score +
            0.1 * stacking_score +
            0.1 * geometry_score
        )

        return complexity

    def solve_placement(
        self,
        scene_graph: SceneGraph,
        room_geometry: Optional[RoomGeometry] = None,
        semantic_plan: Optional[Dict[str, Any]] = None,
        seed: int = 42,
    ) -> PlacementSolution:
        """Solve the placement problem for all objects in the scene.

        Args:
            scene_graph: Scene graph with objects and relationships
            room_geometry: Room geometry specification
            semantic_plan: Optional semantic plan from LLM
            seed: Random seed for reproducibility

        Returns:
            PlacementSolution with computed positions
        """
        import time

        start_time = time.time()

        # Default room geometry
        if room_geometry is None:
            room_geometry = RoomGeometry()

        # Extract constraints from scene graph
        constraints = self._extract_constraints_from_graph(scene_graph)

        # Analyze complexity
        complexity = self.analyze_complexity(
            scene_graph, constraints, room_geometry
        )

        # Choose algorithm
        algorithm = self.choose_algorithm(complexity)

        # Solve based on chosen algorithm
        if algorithm == PlacementAlgorithm.DFS_BEAM_SEARCH:
            solution = self._solve_with_dfs(
                scene_graph, room_geometry, semantic_plan, seed
            )
        elif algorithm == PlacementAlgorithm.MILP:
            solution = self._solve_with_milp(
                scene_graph, room_geometry, constraints
            )
        else:
            # Hybrid approach (future work)
            solution = self._solve_with_dfs(
                scene_graph, room_geometry, semantic_plan, seed
            )

        solution.algorithm_used = algorithm
        solution.execution_time = time.time() - start_time

        return solution

    def validate_stacking(
        self, placed_objects: List[Dict[str, Any]]
    ) -> List[StackingLevel]:
        """Validate multi-level stacking and check load-bearing capacity.

        Args:
            placed_objects: List of placed objects with positions

        Returns:
            List of stacking levels with stability information
        """
        levels: List[StackingLevel] = []

        # Group objects by height (z-coordinate)
        height_groups: Dict[float, List[Dict[str, Any]]] = {}
        for obj in placed_objects:
            pose = obj.get("Pose", {})
            z = float(pose.get("z", 0.0))
            # Round to nearest 0.1m to group objects at similar heights
            z_rounded = round(z * 10) / 10
            height_groups.setdefault(z_rounded, []).append(obj)

        # Sort by height
        sorted_heights = sorted(height_groups.keys())

        # Create stacking levels
        for level_idx, height in enumerate(sorted_heights):
            level = StackingLevel(level=level_idx)
            level.supported_objects = height_groups[height]

            # Compute total weight
            for obj in level.supported_objects:
                # Default weight based on size if not specified
                size = obj.get("size", [1.0, 1.0, 1.0])
                volume = size[0] * size[1] * size[2]
                # Assume average density of 100 kg/m³
                weight = obj.get("weight", volume * 100.0)
                level.total_weight += weight

            # Find supporting object (if not floor level)
            if level_idx > 0:
                # Find object below that provides support
                for obj in level.supported_objects:
                    supporting = self._find_supporting_object(
                        obj, placed_objects
                    )
                    if supporting:
                        level.supporting_object = supporting
                        # Get support capacity from supporting object
                        level.support_capacity = supporting.get(
                            "support_capacity", 1000.0
                        )
                        break

            # Check stability
            if level.supporting_object:
                level.is_stable = (
                    level.total_weight <= level.support_capacity
                )
            else:
                # Floor level is always stable
                level.is_stable = level_idx == 0

            levels.append(level)

        return levels

    def _extract_constraints_from_graph(
        self, scene_graph: SceneGraph
    ) -> List[Constraint]:
        """Extract constraints from scene graph nodes.

        Args:
            scene_graph: Scene graph with objects

        Returns:
            List of Constraint objects
        """
        constraints = []

        for node in scene_graph.get_all_objects():
            for constraint_dict in node.constraints:
                # Convert dict to Constraint object
                # This is a simplified version - in practice, would use
                # constraint factory from constraints.py
                constraint = Constraint(
                    id=constraint_dict.get("id", f"c_{node.id}_{len(constraints)}"),
                    constraint_type=constraint_dict.get("type"),
                    source_object=node.id,
                    target_object=constraint_dict.get("target"),
                    parameters=constraint_dict.get("parameters", {}),
                    priority=constraint_dict.get("priority", 1.0),
                    is_hard=constraint_dict.get("is_hard", True),
                )
                constraints.append(constraint)

        return constraints

    def _solve_with_dfs(
        self,
        scene_graph: SceneGraph,
        room_geometry: RoomGeometry,
        semantic_plan: Optional[Dict[str, Any]],
        seed: int,
    ) -> PlacementSolution:
        """Solve placement using DFS + Beam Search algorithm.

        This integrates the existing floor_solver.py implementation.

        Args:
            scene_graph: Scene graph with objects
            room_geometry: Room geometry specification
            semantic_plan: Semantic plan from LLM
            seed: Random seed

        Returns:
            PlacementSolution with computed positions
        """
        solution = PlacementSolution()

        # Convert scene graph to format expected by floor_solver
        objects_to_place = []
        for node in scene_graph.get_all_objects():
            obj_dict = dict(node.metadata)
            obj_dict["Model"] = node.object_type
            obj_dict["id"] = node.id

            # CRITICAL FIX: Preserve size and is_static from metadata
            # These fields are essential for floor_solver to work correctly
            if "size" not in obj_dict:
                # Fallback to default size if not in metadata
                obj_dict["size"] = [1.0, 1.0, 1.0]
                print(f"[layout_solver] WARNING: {node.id} missing size, using default")
            else:
                print(f"[layout_solver] {node.id} has size: {obj_dict['size']}")
            
            if "is_static" not in obj_dict:
                # Fallback to default is_static if not in metadata
                obj_dict["is_static"] = True
                print(f"[layout_solver] WARNING: {node.id} missing is_static, using default")
            else:
                print(f"[layout_solver] {node.id} has is_static: {obj_dict['is_static']}")

            # Add constraints
            if node.constraints:
                obj_dict["constraints"] = node.constraints

            objects_to_place.append(obj_dict)

        # If no semantic plan provided, create minimal one
        if semantic_plan is None:
            semantic_plan = {
                "objects": objects_to_place,
                "room": {
                    "polygon": room_geometry.boundary_polygon,
                    "forbidden_zones": [],
                },
            }

        try:
            # Call existing floor solver
            placed = solve_floor_placements(
                full_placed_models=objects_to_place,
                semantic_plan=semantic_plan,
                room_half_size=room_geometry.half_size,
                seed=seed,
            )

            solution.placed_objects = placed
            solution.success = True
            solution.quality_score = self._compute_quality_score(placed)

        except Exception as e:
            solution.success = False
            solution.failure_reason = f"DFS placement failed: {str(e)}"

        return solution

    def _solve_with_milp(
        self,
        scene_graph: SceneGraph,
        room_geometry: RoomGeometry,
        constraints: List[Constraint],
    ) -> PlacementSolution:
        """Solve placement using MILP (Mixed Integer Linear Programming).

        This is a placeholder for future MILP implementation.
        For now, falls back to DFS.

        Args:
            scene_graph: Scene graph with objects
            room_geometry: Room geometry specification
            constraints: List of constraints

        Returns:
            PlacementSolution with computed positions
        """
        # TODO: Implement MILP solver using PuLP or similar
        # For MVP, fall back to DFS
        return self._solve_with_dfs(
            scene_graph, room_geometry, None, seed=42
        )

    def _find_supporting_object(
        self, obj: Dict[str, Any], all_objects: List[Dict[str, Any]]
    ) -> Optional[Dict[str, Any]]:
        """Find the object that supports the given object.

        Args:
            obj: Object to find support for
            all_objects: List of all placed objects

        Returns:
            Supporting object if found, None otherwise
        """
        obj_pose = obj.get("Pose", {})
        obj_x = float(obj_pose.get("x", 0.0))
        obj_y = float(obj_pose.get("y", 0.0))
        obj_z = float(obj_pose.get("z", 0.0))

        # Find objects below this one
        candidates = []
        for other in all_objects:
            if other == obj:
                continue

            other_pose = other.get("Pose", {})
            other_z = float(other_pose.get("z", 0.0))

            # Must be below
            if other_z >= obj_z:
                continue

            # Check if horizontally aligned (within footprint)
            other_x = float(other_pose.get("x", 0.0))
            other_y = float(other_pose.get("y", 0.0))

            # Get sizes
            obj_size = obj.get("size", [1.0, 1.0, 1.0])
            other_size = other.get("size", [1.0, 1.0, 1.0])

            # Check overlap in XY plane
            dx = abs(obj_x - other_x)
            dy = abs(obj_y - other_y)

            obj_hx = obj_size[0] / 2.0
            obj_hy = obj_size[2] / 2.0 if len(obj_size) > 2 else obj_hx
            other_hx = other_size[0] / 2.0
            other_hy = (
                other_size[2] / 2.0 if len(other_size) > 2 else other_hx
            )

            if dx < (obj_hx + other_hx) and dy < (obj_hy + other_hy):
                # Overlaps in XY, could be supporting
                candidates.append((other_z, other))

        # Return the highest object below
        if candidates:
            candidates.sort(key=lambda x: x[0], reverse=True)
            return candidates[0][1]

        return None

    def _compute_quality_score(
        self, placed_objects: List[Dict[str, Any]]
    ) -> float:
        """Compute overall quality score for a placement solution.

        Args:
            placed_objects: List of placed objects

        Returns:
            Quality score between 0.0 and 1.0
        """
        if not placed_objects:
            return 0.0

        # Check for collisions
        collision_penalty = 0.0
        obbs = [model_to_obb(obj) for obj in placed_objects]

        for i, obb_a in enumerate(obbs):
            for obb_b in obbs[i + 1:]:
                if obb_overlap(obb_a, obb_b):
                    collision_penalty += 0.1

        # Check for objects outside room (simplified)
        out_of_bounds_penalty = 0.0
        for obj in placed_objects:
            pose = obj.get("Pose", {})
            x = abs(float(pose.get("x", 0.0)))
            y = abs(float(pose.get("y", 0.0)))
            # Simplified check for rectangular room
            if x > 5.0 or y > 5.0:
                out_of_bounds_penalty += 0.1

        # Compute score (1.0 - penalties)
        score = max(0.0, 1.0 - collision_penalty - out_of_bounds_penalty)

        return score

    def generate_alternatives(
        self,
        scene_graph: SceneGraph,
        room_geometry: RoomGeometry,
        max_alternatives: int = 5,
    ) -> List[PlacementSolution]:
        """Generate alternative placement solutions.

        Args:
            scene_graph: Scene graph with objects
            room_geometry: Room geometry specification
            max_alternatives: Maximum number of alternatives to generate

        Returns:
            List of alternative PlacementSolution objects
        """
        alternatives = []

        # Try different random seeds
        for i in range(max_alternatives):
            seed = 42 + i * 1000
            solution = self.solve_placement(
                scene_graph, room_geometry, seed=seed
            )
            if solution.success:
                alternatives.append(solution)

        # Sort by quality score
        alternatives.sort(key=lambda s: s.quality_score, reverse=True)

        return alternatives

    def supports_arbitrary_geometry(self) -> bool:
        """Check if solver supports arbitrary room geometry.

        Returns:
            True if arbitrary geometry is supported
        """
        # DFS solver supports arbitrary geometry via polygon boundaries
        return True

    def supports_stacking(self) -> bool:
        """Check if solver supports multi-level stacking.

        Returns:
            True if stacking is supported
        """
        # Stacking is supported via validate_stacking method
        return True
