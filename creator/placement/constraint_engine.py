"""Constraint engine for solving spatial constraints in scene placement.

This module implements the ConstraintEngine class that manages and solves
spatial constraints for object placement. It supports hard and soft constraints
with priority-based conflict resolution.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple

from creator.placement.constraints import Constraint, ConstraintType


class ConflictResolutionStrategy(Enum):
    """Strategies for resolving constraint conflicts."""
    PRIORITY_BASED = "priority_based"
    WEIGHTED_SUM = "weighted_sum"
    LEXICOGRAPHIC = "lexicographic"


@dataclass
class ConstraintSolution:
    """Solution to a constraint satisfaction problem.

    Attributes:
        satisfied_constraints: List of constraint IDs that are satisfied
        violated_constraints: List of constraint IDs that are violated
        object_positions: Dictionary mapping object IDs to their positions
        constraint_scores: Dictionary mapping constraint IDs to scores
        is_valid: Whether the solution satisfies all hard constraints
        total_score: Overall quality score of the solution
    """

    satisfied_constraints: List[str] = field(default_factory=list)
    violated_constraints: List[str] = field(default_factory=list)
    object_positions: Dict[str, Tuple[float, float, float]] = field(
        default_factory=dict
    )
    constraint_scores: Dict[str, float] = field(default_factory=dict)
    is_valid: bool = False
    total_score: float = 0.0


@dataclass
class ConstraintConflict:
    """Represents a conflict between constraints.

    Attributes:
        constraint_ids: List of conflicting constraint IDs
        conflict_type: Type of conflict (e.g., "incompatible", "overconstrained")
        description: Human-readable description of the conflict
        severity: Severity level (0.0 to 1.0)
    """

    constraint_ids: List[str]
    conflict_type: str
    description: str
    severity: float = 1.0


@dataclass
class ValidationResult:
    """Result of constraint solution validation.

    Attributes:
        is_valid: Whether the solution is valid
        violations: List of constraint violations
        warnings: List of warnings (soft constraint violations)
        conflicts: List of detected conflicts
    """

    is_valid: bool
    violations: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    conflicts: List[ConstraintConflict] = field(default_factory=list)


@dataclass
class Resolution:
    """Result of conflict resolution.

    Attributes:
        resolved_constraints: List of constraints after resolution
        removed_constraints: List of constraint IDs that were removed
        relaxed_constraints: List of constraint IDs that were relaxed
        strategy_used: The resolution strategy that was applied
        success: Whether the conflict was successfully resolved
    """

    resolved_constraints: List[Constraint]
    removed_constraints: List[str] = field(default_factory=list)
    relaxed_constraints: List[str] = field(default_factory=list)
    strategy_used: str = ""
    success: bool = False


class ConstraintEngine:
    """Engine for managing and solving spatial constraints.

    The ConstraintEngine maintains a collection of constraints and provides
    methods to solve them, validate solutions, and resolve conflicts.
    It implements a priority-based system where hard constraints must be
    satisfied and soft constraints are optimized with weighted coefficients.

    Attributes:
        constraints: Dictionary mapping constraint IDs to Constraint objects
        conflict_resolution_strategy: Strategy for resolving conflicts
    """

    def __init__(
        self,
        conflict_resolution_strategy: ConflictResolutionStrategy = (
            ConflictResolutionStrategy.PRIORITY_BASED
        ),
    ):
        """Initialize the constraint engine.

        Args:
            conflict_resolution_strategy: Strategy for resolving conflicts
        """
        self.constraints: Dict[str, Constraint] = {}
        self.conflict_resolution_strategy = conflict_resolution_strategy

    def add_constraint(self, constraint: Constraint) -> None:
        """Add a constraint to the system.

        Args:
            constraint: The constraint to add

        Raises:
            ValueError: If a constraint with the same ID already exists
        """
        if constraint.id in self.constraints:
            raise ValueError(
                f"Constraint with ID '{constraint.id}' already exists"
            )

        self.constraints[constraint.id] = constraint

    def remove_constraint(self, constraint_id: str) -> bool:
        """Remove a constraint from the system.

        Args:
            constraint_id: ID of the constraint to remove

        Returns:
            True if the constraint was removed, False if it didn't exist
        """
        if constraint_id in self.constraints:
            del self.constraints[constraint_id]
            return True
        return False

    def get_constraint(self, constraint_id: str) -> Optional[Constraint]:
        """Get a constraint by ID.

        Args:
            constraint_id: ID of the constraint to retrieve

        Returns:
            The constraint if found, None otherwise
        """
        return self.constraints.get(constraint_id)

    def get_constraints_for_object(self, object_id: str) -> List[Constraint]:
        """Get all constraints involving a specific object.

        Args:
            object_id: ID of the object

        Returns:
            List of constraints where the object is source or target
        """
        result = []
        for constraint in self.constraints.values():
            if (
                constraint.source_object == object_id
                or constraint.target_object == object_id
            ):
                result.append(constraint)
        return result

    def solve_constraints(
        self, objects: List[Dict[str, Any]]
    ) -> ConstraintSolution:
        """Solve the system of constraints.

        This method attempts to find positions for objects that satisfy
        all hard constraints and optimize soft constraints based on their
        priorities and weights.

        Args:
            objects: List of scene objects to place

        Returns:
            ConstraintSolution containing the solution and metadata
        """
        solution = ConstraintSolution()

        # Separate hard and soft constraints
        hard_constraints = [
            c for c in self.constraints.values() if c.is_hard
        ]
        soft_constraints = [
            c for c in self.constraints.values() if not c.is_hard
        ]

        # Sort constraints by priority (higher priority first)
        hard_constraints.sort(key=lambda c: c.priority, reverse=True)
        soft_constraints.sort(key=lambda c: c.priority, reverse=True)

        # Initialize object positions (placeholder - actual solving logic
        # would be implemented based on specific constraint types)
        for obj in objects:
            obj_id = obj.get("id") or obj.get("Model") or obj.get("name", "")
            if obj_id:
                pose = obj.get("Pose", {})
                position = (
                    float(pose.get("x", 0.0)),
                    float(pose.get("y", 0.0)),
                    float(pose.get("z", 0.0)),
                )
                solution.object_positions[obj_id] = position

        # Evaluate hard constraints
        for constraint in hard_constraints:
            score = self._evaluate_constraint(constraint, solution)
            solution.constraint_scores[constraint.id] = score

            if score >= 0.99:  # Constraint is satisfied
                solution.satisfied_constraints.append(constraint.id)
            else:
                solution.violated_constraints.append(constraint.id)

        # Check if all hard constraints are satisfied
        solution.is_valid = len(solution.violated_constraints) == 0

        # Evaluate soft constraints
        soft_score = 0.0
        for constraint in soft_constraints:
            score = self._evaluate_constraint(constraint, solution)
            solution.constraint_scores[constraint.id] = score

            # Weight the score by priority
            weighted_score = score * constraint.priority
            soft_score += weighted_score

            if score >= 0.99:
                solution.satisfied_constraints.append(constraint.id)
            else:
                solution.violated_constraints.append(constraint.id)

        # Calculate total score
        if solution.is_valid:
            # Normalize soft score by total possible score
            max_soft_score = sum(c.priority for c in soft_constraints)
            if max_soft_score > 0:
                solution.total_score = soft_score / max_soft_score
            else:
                solution.total_score = 1.0
        else:
            solution.total_score = 0.0

        return solution

    def validate_solution(
        self, solution: ConstraintSolution
    ) -> ValidationResult:
        """Validate the correctness of a constraint solution.

        Args:
            solution: The solution to validate

        Returns:
            ValidationResult containing validation status and details
        """
        result = ValidationResult(is_valid=True)

        # Check hard constraints
        for constraint_id, constraint in self.constraints.items():
            if not constraint.is_hard:
                continue

            score = solution.constraint_scores.get(constraint_id, 0.0)

            if score < 0.99:  # Hard constraint violated
                result.is_valid = False
                result.violations.append(
                    f"Hard constraint '{constraint_id}' violated "
                    f"(score: {score:.2f})"
                )

        # Check soft constraints
        for constraint_id, constraint in self.constraints.items():
            if constraint.is_hard:
                continue

            score = solution.constraint_scores.get(constraint_id, 0.0)

            if score < 0.99:  # Soft constraint violated
                result.warnings.append(
                    f"Soft constraint '{constraint_id}' not fully satisfied "
                    f"(score: {score:.2f})"
                )

        # Detect conflicts
        conflicts = self._detect_conflicts()
        result.conflicts = conflicts

        return result

    def resolve_conflicts(
        self, conflicts: List[ConstraintConflict]
    ) -> Resolution:
        """Resolve conflicts between constraints.

        This method applies the configured conflict resolution strategy
        to resolve conflicts. Hard constraints always take priority over
        soft constraints.

        Args:
            conflicts: List of conflicts to resolve

        Returns:
            Resolution containing the resolved constraints
        """
        resolution = Resolution(
            resolved_constraints=list(self.constraints.values()),
            strategy_used=self.conflict_resolution_strategy.value,
        )

        if not conflicts:
            resolution.success = True
            return resolution

        # Apply resolution strategy
        if (
            self.conflict_resolution_strategy
            == ConflictResolutionStrategy.PRIORITY_BASED
        ):
            resolution = self._resolve_by_priority(conflicts)
        elif (
            self.conflict_resolution_strategy
            == ConflictResolutionStrategy.WEIGHTED_SUM
        ):
            resolution = self._resolve_by_weighted_sum(conflicts)
        elif (
            self.conflict_resolution_strategy
            == ConflictResolutionStrategy.LEXICOGRAPHIC
        ):
            resolution = self._resolve_lexicographically(conflicts)

        return resolution

    def _evaluate_constraint(
        self, constraint: Constraint, solution: ConstraintSolution
    ) -> float:
        """Evaluate how well a constraint is satisfied.

        Args:
            constraint: The constraint to evaluate
            solution: The current solution

        Returns:
            Score between 0.0 (not satisfied) and 1.0 (fully satisfied)
        """
        # Placeholder implementation - actual evaluation would depend on
        # constraint type and would check geometric properties
        # For now, return 1.0 (satisfied) as a stub
        return 1.0

    def _detect_conflicts(self) -> List[ConstraintConflict]:
        """Detect conflicts between constraints.

        Returns:
            List of detected conflicts
        """
        conflicts = []

        # Check for conflicting constraints on the same object
        object_constraints: Dict[str, List[Constraint]] = {}
        for constraint in self.constraints.values():
            obj_id = constraint.source_object
            if obj_id not in object_constraints:
                object_constraints[obj_id] = []
            object_constraints[obj_id].append(constraint)

        # Look for incompatible constraints
        for obj_id, constraints_list in object_constraints.items():
            # Check for conflicting distance constraints
            distance_constraints = [
                c
                for c in constraints_list
                if c.constraint_type == ConstraintType.DISTANCE
            ]

            for i, c1 in enumerate(distance_constraints):
                for c2 in distance_constraints[i + 1 :]:
                    if c1.target_object == c2.target_object:
                        # Check if distance ranges are incompatible
                        min1 = c1.parameters.get("min_distance", 0.0)
                        max1 = c1.parameters.get("max_distance", float("inf"))
                        min2 = c2.parameters.get("min_distance", 0.0)
                        max2 = c2.parameters.get("max_distance", float("inf"))

                        # Check for non-overlapping ranges
                        if max1 < min2 or max2 < min1:
                            conflicts.append(
                                ConstraintConflict(
                                    constraint_ids=[c1.id, c2.id],
                                    conflict_type="incompatible_distance",
                                    description=(
                                        f"Distance constraints on {obj_id} "
                                        f"to {c1.target_object} have "
                                        f"non-overlapping ranges"
                                    ),
                                    severity=1.0
                                    if (c1.is_hard and c2.is_hard)
                                    else 0.5,
                                )
                            )

        return conflicts

    def _resolve_by_priority(
        self, conflicts: List[ConstraintConflict]
    ) -> Resolution:
        """Resolve conflicts using priority-based strategy.

        Hard constraints always win over soft constraints.
        Among constraints of the same type, higher priority wins.

        Args:
            conflicts: List of conflicts to resolve

        Returns:
            Resolution with conflicts resolved
        """
        resolution = Resolution(
            strategy_used="priority_based", success=True
        )

        removed_ids: Set[str] = set()

        for conflict in conflicts:
            # Get the constraints involved in the conflict
            conflict_constraints = [
                self.constraints[cid]
                for cid in conflict.constraint_ids
                if cid in self.constraints
            ]

            if not conflict_constraints:
                continue

            # Sort by: hard constraints first, then by priority
            conflict_constraints.sort(
                key=lambda c: (not c.is_hard, -c.priority)
            )

            # Keep the highest priority constraint, remove others
            keep_constraint = conflict_constraints[0]
            for constraint in conflict_constraints[1:]:
                removed_ids.add(constraint.id)

        # Build resolved constraint list
        resolution.resolved_constraints = [
            c for c in self.constraints.values() if c.id not in removed_ids
        ]
        resolution.removed_constraints = list(removed_ids)

        return resolution

    def _resolve_by_weighted_sum(
        self, conflicts: List[ConstraintConflict]
    ) -> Resolution:
        """Resolve conflicts using weighted sum strategy.

        Attempts to satisfy all constraints proportionally to their weights.

        Args:
            conflicts: List of conflicts to resolve

        Returns:
            Resolution with conflicts resolved
        """
        resolution = Resolution(
            strategy_used="weighted_sum", success=True
        )

        relaxed_ids: Set[str] = set()

        for conflict in conflicts:
            # For weighted sum, we relax soft constraints involved in conflicts
            for constraint_id in conflict.constraint_ids:
                constraint = self.constraints.get(constraint_id)
                if constraint and not constraint.is_hard:
                    relaxed_ids.add(constraint_id)

        resolution.resolved_constraints = list(self.constraints.values())
        resolution.relaxed_constraints = list(relaxed_ids)

        return resolution

    def _resolve_lexicographically(
        self, conflicts: List[ConstraintConflict]
    ) -> Resolution:
        """Resolve conflicts using lexicographic strategy.

        Satisfies constraints in strict priority order.

        Args:
            conflicts: List of conflicts to resolve

        Returns:
            Resolution with conflicts resolved
        """
        resolution = Resolution(
            strategy_used="lexicographic", success=True
        )

        # Sort all constraints by priority
        sorted_constraints = sorted(
            self.constraints.values(),
            key=lambda c: (not c.is_hard, -c.priority),
        )

        # For lexicographic ordering, we keep constraints in priority order
        # and mark lower-priority conflicting constraints as removed
        removed_ids: Set[str] = set()

        for conflict in conflicts:
            conflict_constraints = [
                self.constraints[cid]
                for cid in conflict.constraint_ids
                if cid in self.constraints
            ]

            if not conflict_constraints:
                continue

            # Sort by priority
            conflict_constraints.sort(
                key=lambda c: (not c.is_hard, -c.priority)
            )

            # Remove all but the highest priority
            for constraint in conflict_constraints[1:]:
                removed_ids.add(constraint.id)

        resolution.resolved_constraints = [
            c for c in sorted_constraints if c.id not in removed_ids
        ]
        resolution.removed_constraints = list(removed_ids)

        return resolution

    def clear_constraints(self) -> None:
        """Remove all constraints from the engine."""
        self.constraints.clear()

    def get_constraint_count(self) -> int:
        """Get the total number of constraints.

        Returns:
            Number of constraints in the system
        """
        return len(self.constraints)

    def get_hard_constraint_count(self) -> int:
        """Get the number of hard constraints.

        Returns:
            Number of hard constraints
        """
        return sum(1 for c in self.constraints.values() if c.is_hard)

    def get_soft_constraint_count(self) -> int:
        """Get the number of soft constraints.

        Returns:
            Number of soft constraints
        """
        return sum(1 for c in self.constraints.values() if not c.is_hard)
