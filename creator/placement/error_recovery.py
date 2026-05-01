"""Error recovery system for placement failures.

This module implements a multi-level error recovery strategy for handling
placement failures in the universal scene placement system. It provides
mechanisms for generating alternatives, relaxing constraints, and rolling
back to stable states.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from creator.placement.constraints import Constraint
from creator.placement.geometry import OBB, Vec2, model_to_obb, obb_overlap
from creator.placement.scene_graph import SceneGraph, SceneNode


class FailureReason(Enum):
    """Reasons for placement failure."""
    CONSTRAINT_CONFLICT = "constraint_conflict"
    NO_VALID_POSITION = "no_valid_position"
    COLLISION = "collision"
    OUT_OF_BOUNDS = "out_of_bounds"
    INSUFFICIENT_SPACE = "insufficient_space"
    PHYSICS_UNSTABLE = "physics_unstable"
    UNSUPPORTED_CONSTRAINT = "unsupported_constraint"


class RecoveryAction(Enum):
    """Recovery actions to take after failure."""
    GENERATE_ALTERNATIVES = "generate_alternatives"
    RELAX_CONSTRAINTS = "relax_constraints"
    ROLLBACK = "rollback"
    ESCALATE_TO_PLANNER = "escalate_to_planner"
    FAIL = "fail"


class RelaxationStrategy(Enum):
    """Strategies for relaxing constraints."""
    REDUCE_WEIGHTS = "reduce_weights"
    SOFTEN_HARD_CONSTRAINTS = "soften_hard_constraints"
    REMOVE_LOWEST_PRIORITY = "remove_lowest_priority"
    INCREASE_TOLERANCES = "increase_tolerances"


@dataclass
class PlacementLog:
    """Log entry for a placement attempt.
    
    Attributes:
        timestamp: When the placement was attempted
        object_id: ID of the object being placed
        attempted_positions: List of positions tried
        constraint_scores: Scores for each constraint
        failure_reason: Reason for failure (if failed)
        recovery_action: Action taken to recover (if any)
        final_position: Final position if successful
    """
    timestamp: datetime
    object_id: str
    attempted_positions: List[Tuple[float, float, float]] = field(default_factory=list)
    constraint_scores: Dict[str, float] = field(default_factory=dict)
    failure_reason: Optional[str] = None
    recovery_action: Optional[str] = None
    final_position: Optional[Tuple[float, float, float]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert log entry to dictionary."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "object_id": self.object_id,
            "attempted_positions": self.attempted_positions,
            "constraint_scores": self.constraint_scores,
            "failure_reason": self.failure_reason,
            "recovery_action": self.recovery_action,
            "final_position": self.final_position,
        }


@dataclass
class PlacementContext:
    """Context information for placement operations.
    
    Attributes:
        scene_graph: Current scene graph
        placed_objects: List of already placed objects
        room_bounds: Room boundary information
        available_anchors: List of available anchor points
        clearance_zones: List of clearance zones to avoid
    """
    scene_graph: SceneGraph
    placed_objects: List[Dict[str, Any]] = field(default_factory=list)
    room_bounds: Dict[str, float] = field(default_factory=dict)
    available_anchors: List[Dict[str, Any]] = field(default_factory=list)
    clearance_zones: List[Any] = field(default_factory=list)


@dataclass
class AlternativePlacement:
    """Alternative placement option.
    
    Attributes:
        position: Proposed position (x, y, z)
        orientation: Proposed orientation (yaw in radians)
        score: Quality score (higher is better)
        satisfied_constraints: List of satisfied constraint IDs
        violated_constraints: List of violated constraint IDs
        reason: Explanation for this alternative
    """
    position: Tuple[float, float, float]
    orientation: float
    score: float
    satisfied_constraints: List[str] = field(default_factory=list)
    violated_constraints: List[str] = field(default_factory=list)
    reason: str = ""


@dataclass
class StateCheckpoint:
    """Checkpoint of scene state for rollback.
    
    Attributes:
        timestamp: When checkpoint was created
        scene_graph: Copy of scene graph at checkpoint
        placed_objects: Copy of placed objects at checkpoint
        constraints: Copy of active constraints at checkpoint
        metadata: Additional checkpoint metadata
    """
    timestamp: datetime
    scene_graph: SceneGraph
    placed_objects: List[Dict[str, Any]]
    constraints: List[Constraint]
    metadata: Dict[str, Any] = field(default_factory=dict)


class ErrorRecoverySystem:
    """System for recovering from placement errors.
    
    This class implements a multi-level error recovery strategy:
    1. Generate alternative positions with relaxed constraints
    2. Relax soft constraints to find feasible solutions
    3. Rollback to previous stable state if necessary
    4. Escalate to LLM planner for plan revision
    
    Attributes:
        logs: List of placement logs for analysis
        checkpoints: Stack of state checkpoints for rollback
        max_alternatives: Maximum number of alternatives to generate
        max_relaxation_iterations: Maximum constraint relaxation attempts
    """
    
    def __init__(
        self,
        max_alternatives: int = 5,
        max_relaxation_iterations: int = 3,
    ):
        """Initialize error recovery system.
        
        Args:
            max_alternatives: Maximum alternatives to generate per failure
            max_relaxation_iterations: Maximum relaxation attempts
        """
        self.logs: List[PlacementLog] = []
        self.checkpoints: List[StateCheckpoint] = []
        self.max_alternatives = max_alternatives
        self.max_relaxation_iterations = max_relaxation_iterations
    
    def handle_placement_failure(
        self,
        failed_object: Dict[str, Any],
        reason: FailureReason,
        context: PlacementContext,
    ) -> RecoveryAction:
        """Determine recovery strategy for a placement failure.
        
        This method analyzes the failure reason and context to determine
        the most appropriate recovery action. It follows a hierarchy:
        1. Try generating alternatives for simple failures
        2. Relax constraints for constraint conflicts
        3. Rollback for physics instability
        4. Escalate to planner for fundamental issues
        
        Args:
            failed_object: Object that failed to place
            reason: Reason for placement failure
            context: Current placement context
            
        Returns:
            Recommended recovery action
        """
        object_id = failed_object.get("Model", "unknown")
        
        # Log the failure
        log_entry = PlacementLog(
            timestamp=datetime.now(),
            object_id=object_id,
            failure_reason=reason.value,
        )
        self.logs.append(log_entry)
        
        # Determine recovery action based on failure reason
        if reason == FailureReason.CONSTRAINT_CONFLICT:
            # Try relaxing constraints first
            return RecoveryAction.RELAX_CONSTRAINTS
        
        elif reason == FailureReason.NO_VALID_POSITION:
            # Generate alternatives with different anchor points
            return RecoveryAction.GENERATE_ALTERNATIVES
        
        elif reason == FailureReason.COLLISION:
            # Try alternatives first, then relax if needed
            if len(context.placed_objects) < 10:
                return RecoveryAction.GENERATE_ALTERNATIVES
            else:
                return RecoveryAction.RELAX_CONSTRAINTS
        
        elif reason == FailureReason.OUT_OF_BOUNDS:
            # Generate alternatives closer to room center
            return RecoveryAction.GENERATE_ALTERNATIVES
        
        elif reason == FailureReason.INSUFFICIENT_SPACE:
            # This is a fundamental issue - escalate to planner
            return RecoveryAction.ESCALATE_TO_PLANNER
        
        elif reason == FailureReason.PHYSICS_UNSTABLE:
            # Rollback to stable state
            return RecoveryAction.ROLLBACK
        
        elif reason == FailureReason.UNSUPPORTED_CONSTRAINT:
            # Escalate to planner for plan revision
            return RecoveryAction.ESCALATE_TO_PLANNER
        
        else:
            # Unknown failure - try alternatives as default
            return RecoveryAction.GENERATE_ALTERNATIVES
    
    def generate_alternatives(
        self,
        object: Dict[str, Any],
        constraints: List[Constraint],
        context: PlacementContext,
        max_alternatives: Optional[int] = None,
    ) -> List[AlternativePlacement]:
        """Generate alternative placement positions for an object.
        
        This method generates multiple alternative positions by:
        1. Trying different anchor points
        2. Varying distances from constraints
        3. Exploring different orientations
        4. Relaxing soft constraints progressively
        
        Args:
            object: Object to place
            constraints: List of constraints to satisfy
            context: Current placement context
            max_alternatives: Maximum alternatives to generate (uses default if None)
            
        Returns:
            List of alternative placements sorted by score (best first)
        """
        if max_alternatives is None:
            max_alternatives = self.max_alternatives
        
        alternatives: List[AlternativePlacement] = []
        object_id = object.get("Model", "unknown")
        
        # Get object dimensions
        size = object.get("size", [1.0, 1.0, 1.0])
        half_width = float(size[0]) / 2.0 if len(size) > 0 else 0.5
        half_depth = float(size[2]) / 2.0 if len(size) > 2 else 0.5
        half_height = float(size[1]) / 2.0 if len(size) > 1 else 0.5
        
        # Get room bounds
        room_half_size = context.room_bounds.get("half_size", 5.0)
        
        # Strategy 1: Try different anchor points
        for anchor in context.available_anchors[:max_alternatives * 2]:
            anchor_pos = anchor.get("position", (0.0, 0.0, 0.0))
            
            # Try positions around the anchor
            for offset_x in [-1.0, 0.0, 1.0]:
                for offset_y in [-1.0, 0.0, 1.0]:
                    if len(alternatives) >= max_alternatives:
                        break
                    
                    # Calculate candidate position
                    x = anchor_pos[0] + offset_x
                    y = anchor_pos[1] + offset_y
                    z = half_height  # Start on floor
                    
                    # Check if position is valid
                    if abs(x) > room_half_size - half_width:
                        continue
                    if abs(y) > room_half_size - half_depth:
                        continue
                    
                    # Try different orientations
                    for yaw in [0.0, 1.5708, 3.14159, 4.71239]:  # 0°, 90°, 180°, 270°
                        if len(alternatives) >= max_alternatives:
                            break
                        
                        # Create candidate placement
                        candidate_obj = dict(object)
                        candidate_obj["Pose"] = {"x": x, "y": y, "z": z}
                        candidate_obj["yaw_deg"] = yaw * 57.2958  # Convert to degrees
                        
                        # Check for collisions
                        candidate_obb = model_to_obb(candidate_obj)
                        has_collision = False
                        
                        for placed in context.placed_objects:
                            placed_obb = model_to_obb(placed)
                            if obb_overlap(candidate_obb, placed_obb, margin=0.05):
                                has_collision = True
                                break
                        
                        if has_collision:
                            continue
                        
                        # Evaluate constraints
                        satisfied, violated, score = self._evaluate_constraints(
                            candidate_obj, constraints, context
                        )
                        
                        # Add alternative
                        alternatives.append(AlternativePlacement(
                            position=(x, y, z),
                            orientation=yaw,
                            score=score,
                            satisfied_constraints=satisfied,
                            violated_constraints=violated,
                            reason=f"Near anchor {anchor.get('id', 'unknown')}",
                        ))
        
        # Strategy 2: Try room regions if not enough alternatives
        if len(alternatives) < max_alternatives:
            regions = [
                ("center", 0.0, 0.0),
                ("north", 0.0, room_half_size * 0.5),
                ("south", 0.0, -room_half_size * 0.5),
                ("east", room_half_size * 0.5, 0.0),
                ("west", -room_half_size * 0.5, 0.0),
            ]
            
            for region_name, rx, ry in regions:
                if len(alternatives) >= max_alternatives:
                    break
                
                # Add some randomness around region center
                for dx in [-0.5, 0.0, 0.5]:
                    for dy in [-0.5, 0.0, 0.5]:
                        if len(alternatives) >= max_alternatives:
                            break
                        
                        x = rx + dx
                        y = ry + dy
                        z = half_height
                        
                        # Check bounds
                        if abs(x) > room_half_size - half_width:
                            continue
                        if abs(y) > room_half_size - half_depth:
                            continue
                        
                        # Create candidate
                        candidate_obj = dict(object)
                        candidate_obj["Pose"] = {"x": x, "y": y, "z": z}
                        candidate_obj["yaw_deg"] = 0.0
                        
                        # Check collisions
                        candidate_obb = model_to_obb(candidate_obj)
                        has_collision = False
                        
                        for placed in context.placed_objects:
                            placed_obb = model_to_obb(placed)
                            if obb_overlap(candidate_obb, placed_obb, margin=0.05):
                                has_collision = True
                                break
                        
                        if has_collision:
                            continue
                        
                        # Evaluate constraints
                        satisfied, violated, score = self._evaluate_constraints(
                            candidate_obj, constraints, context
                        )
                        
                        alternatives.append(AlternativePlacement(
                            position=(x, y, z),
                            orientation=0.0,
                            score=score,
                            satisfied_constraints=satisfied,
                            violated_constraints=violated,
                            reason=f"In {region_name} region",
                        ))
        
        # Sort by score (best first)
        alternatives.sort(key=lambda a: a.score, reverse=True)
        
        # Log the alternatives
        if self.logs:
            self.logs[-1].attempted_positions = [
                alt.position for alt in alternatives
            ]
            self.logs[-1].recovery_action = RecoveryAction.GENERATE_ALTERNATIVES.value
        
        return alternatives[:max_alternatives]
    
    def relax_constraints(
        self,
        constraints: List[Constraint],
        relaxation_strategy: RelaxationStrategy,
    ) -> List[Constraint]:
        """Relax constraints to find feasible solutions.
        
        This method applies various relaxation strategies to make
        constraints easier to satisfy:
        - Reduce weights of soft constraints
        - Convert hard constraints to soft
        - Remove lowest priority constraints
        - Increase tolerance values
        
        Args:
            constraints: List of constraints to relax
            relaxation_strategy: Strategy to use for relaxation
            
        Returns:
            List of relaxed constraints
        """
        relaxed = []
        
        for constraint in constraints:
            relaxed_constraint = Constraint(
                id=constraint.id,
                constraint_type=constraint.constraint_type,
                source_object=constraint.source_object,
                target_object=constraint.target_object,
                parameters=dict(constraint.parameters),
                priority=constraint.priority,
                is_hard=constraint.is_hard,
            )
            
            if relaxation_strategy == RelaxationStrategy.REDUCE_WEIGHTS:
                # Reduce priority/weight by 50%
                relaxed_constraint.priority = constraint.priority * 0.5
                
            elif relaxation_strategy == RelaxationStrategy.SOFTEN_HARD_CONSTRAINTS:
                # Convert hard constraints to soft (except critical ones)
                if constraint.is_hard and constraint.priority < 10.0:
                    relaxed_constraint.is_hard = False
                    relaxed_constraint.priority = max(1.0, constraint.priority * 0.8)
                    
            elif relaxation_strategy == RelaxationStrategy.REMOVE_LOWEST_PRIORITY:
                # Skip constraints with priority < 1.0
                if constraint.priority < 1.0:
                    continue
                    
            elif relaxation_strategy == RelaxationStrategy.INCREASE_TOLERANCES:
                # Increase tolerance values in parameters
                if "tolerance" in relaxed_constraint.parameters:
                    old_tol = relaxed_constraint.parameters["tolerance"]
                    relaxed_constraint.parameters["tolerance"] = old_tol * 1.5
                
                if "min_distance" in relaxed_constraint.parameters:
                    old_min = relaxed_constraint.parameters["min_distance"]
                    relaxed_constraint.parameters["min_distance"] = old_min * 0.8
                
                if "max_distance" in relaxed_constraint.parameters:
                    old_max = relaxed_constraint.parameters["max_distance"]
                    relaxed_constraint.parameters["max_distance"] = old_max * 1.2
            
            relaxed.append(relaxed_constraint)
        
        # Log the relaxation
        if self.logs:
            self.logs[-1].recovery_action = RecoveryAction.RELAX_CONSTRAINTS.value
        
        return relaxed
    
    def rollback_to_stable_state(
        self,
        scene_graph: SceneGraph,
        checkpoint: Optional[StateCheckpoint] = None,
    ) -> SceneGraph:
        """Rollback scene graph to a previous stable state.
        
        This method restores the scene graph to a previous checkpoint,
        effectively undoing recent placement operations that led to
        instability or failure.
        
        Args:
            scene_graph: Current (unstable) scene graph
            checkpoint: Specific checkpoint to restore (uses latest if None)
            
        Returns:
            Restored scene graph from checkpoint
        """
        if checkpoint is None:
            if not self.checkpoints:
                # No checkpoints available - return current graph
                return scene_graph
            checkpoint = self.checkpoints[-1]
        
        # Create a new scene graph from checkpoint
        restored_graph = SceneGraph()
        
        # Restore root node
        restored_graph.root = checkpoint.scene_graph.root
        
        # Restore all nodes
        restored_graph.nodes = dict(checkpoint.scene_graph.nodes)
        
        # Restore global anchors
        restored_graph.global_anchors = list(checkpoint.scene_graph.global_anchors)
        
        # Log the rollback
        if self.logs:
            self.logs[-1].recovery_action = RecoveryAction.ROLLBACK.value
        
        return restored_graph
    
    def create_checkpoint(
        self,
        scene_graph: SceneGraph,
        placed_objects: List[Dict[str, Any]],
        constraints: List[Constraint],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> StateCheckpoint:
        """Create a checkpoint of current scene state.
        
        Args:
            scene_graph: Current scene graph
            placed_objects: Currently placed objects
            constraints: Active constraints
            metadata: Optional metadata to store with checkpoint
            
        Returns:
            Created checkpoint
        """
        checkpoint = StateCheckpoint(
            timestamp=datetime.now(),
            scene_graph=scene_graph,
            placed_objects=list(placed_objects),
            constraints=list(constraints),
            metadata=metadata or {},
        )
        
        self.checkpoints.append(checkpoint)
        return checkpoint
    
    def get_failure_statistics(self) -> Dict[str, Any]:
        """Get statistics about placement failures.
        
        Returns:
            Dictionary with failure statistics
        """
        total_attempts = len(self.logs)
        failures = [log for log in self.logs if log.failure_reason is not None]
        successes = total_attempts - len(failures)
        
        # Count failures by reason
        failure_counts: Dict[str, int] = {}
        for log in failures:
            reason = log.failure_reason or "unknown"
            failure_counts[reason] = failure_counts.get(reason, 0) + 1
        
        # Count recovery actions
        recovery_counts: Dict[str, int] = {}
        for log in self.logs:
            if log.recovery_action:
                recovery_counts[log.recovery_action] = \
                    recovery_counts.get(log.recovery_action, 0) + 1
        
        return {
            "total_attempts": total_attempts,
            "successes": successes,
            "failures": len(failures),
            "success_rate": successes / total_attempts if total_attempts > 0 else 0.0,
            "failure_by_reason": failure_counts,
            "recovery_actions": recovery_counts,
            "checkpoints_created": len(self.checkpoints),
        }
    
    def _evaluate_constraints(
        self,
        candidate_obj: Dict[str, Any],
        constraints: List[Constraint],
        context: PlacementContext,
    ) -> Tuple[List[str], List[str], float]:
        """Evaluate how well a candidate satisfies constraints.
        
        Args:
            candidate_obj: Candidate object placement
            constraints: List of constraints to evaluate
            context: Placement context
            
        Returns:
            Tuple of (satisfied_ids, violated_ids, total_score)
        """
        satisfied = []
        violated = []
        total_score = 0.0
        
        for constraint in constraints:
            # Simple heuristic evaluation
            # In a real implementation, this would use the ConstraintEngine
            
            constraint_satisfied = True
            constraint_score = constraint.priority
            
            # Check basic feasibility
            if constraint.is_hard:
                constraint_score *= 2.0
            
            if constraint_satisfied:
                satisfied.append(constraint.id)
                total_score += constraint_score
            else:
                violated.append(constraint.id)
                if constraint.is_hard:
                    total_score -= constraint_score * 2.0
                else:
                    total_score -= constraint_score * 0.5
        
        return satisfied, violated, total_score
