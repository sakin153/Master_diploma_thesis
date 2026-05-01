"""Main pipeline orchestrator for Semantic Plan Enforcement.

This module implements the SemanticEnforcementPipeline class, which orchestrates
all components of the semantic plan enforcement system. It takes a semantic plan
from the LLM and runs it through the entire pipeline to produce a MuJoCo XML scene.

Pipeline flow:
1. SemanticPlan (from LLM)
2. SchemaValidator - validates structure and references
3. DistanceResolver - converts relative positions to absolute coordinates
4. OrientationResolver - converts relative orientations to absolute angles
5. PlacementExecutor - places objects exactly as specified
6. SceneAssembly - creates MuJoCo XML from placement solution

The PipelineTracker logs objects at each stage for debugging and validation.
"""

import logging
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from creator.placement.semantic_enforcement.distance_resolver import (
    DistanceResolver,
    CircularDependencyError,
)
from creator.placement.semantic_enforcement.models import (
    PlacementSolution,
    SemanticPlan,
)
from creator.placement.semantic_enforcement.orientation_resolver import (
    OrientationResolver,
)
from creator.placement.semantic_enforcement.placement_executor import (
    PlacementExecutor,
)
from creator.placement.semantic_enforcement.pipeline_tracker import (
    PipelineTracker,
)
from creator.placement.semantic_enforcement.scene_assembly import (
    SceneAssembly,
)
from creator.placement.semantic_enforcement.schema_validator import (
    SchemaValidator,
)

logger = logging.getLogger(__name__)


class PipelineError(Exception):
    """Base exception for pipeline errors."""
    pass


class SchemaValidationError(PipelineError):
    """Raised when semantic plan fails schema validation."""
    pass


class ResolutionError(PipelineError):
    """Raised when position or orientation resolution fails."""
    pass


class PlacementError(PipelineError):
    """Raised when placement execution fails."""
    pass


class AssemblyError(PipelineError):
    """Raised when scene assembly fails."""
    pass


class SemanticEnforcementPipeline:
    """Main pipeline orchestrator for semantic plan enforcement.

    This class coordinates all components of the semantic enforcement pipeline:
    - SchemaValidator: validates semantic plan structure
    - DistanceResolver: resolves relative positions to absolute coordinates
    - OrientationResolver: resolves relative orientations to absolute angles
    - PlacementExecutor: places objects exactly as specified
    - SceneAssembly: creates MuJoCo XML from placement solution
    - PipelineTracker: tracks objects through all stages

    The pipeline follows a fail-fast strategy: critical errors stop execution
    immediately with clear error messages. Warnings are logged but don't stop
    the pipeline.

    Example:
        pipeline = SemanticEnforcementPipeline(
            workspace_root=".",
            output_dir="output/scenes",
            enable_tracking=True,
        )

        xml, solution, trace = pipeline.execute_pipeline(semantic_plan_dict)

    **Validates: Requirements 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 2.1, 3.1, 4.1, 5.1, 7.1, 8.1, 10.1**
    """

    def __init__(
        self,
        workspace_root: str = ".",
        output_dir: Optional[str] = None,
        enable_tracking: bool = True,
        enable_collision_check: bool = True,
    ):
        """Initialize pipeline with configuration.

        Args:
            workspace_root: Root directory for resolving model_loc paths
            output_dir: Directory for output files (XML, traces). If None, uses workspace_root.
            enable_tracking: Whether to enable pipeline tracking
            enable_collision_check: Whether to check for collisions (logs warnings only)
        """
        self.workspace_root = Path(workspace_root)
        self.output_dir = Path(output_dir) if output_dir else self.workspace_root
        self.enable_tracking = enable_tracking

        # Initialize components
        self.schema_validator = SchemaValidator()
        self.distance_resolver = DistanceResolver()
        self.orientation_resolver = OrientationResolver()
        self.placement_executor = PlacementExecutor(
            collision_check_enabled=enable_collision_check
        )
        self.scene_assembly = SceneAssembly(workspace_root=str(self.workspace_root))

        # Initialize tracker if enabled
        self.tracker: Optional[PipelineTracker] = None
        if enable_tracking:
            self.tracker = PipelineTracker(
                output_dir=str(self.output_dir / "analysis_logs")
            )

        logger.info(
            f"[Pipeline] Initialized SemanticEnforcementPipeline "
            f"(tracking={enable_tracking}, collision_check={enable_collision_check})"
        )

    def execute_pipeline(
        self,
        semantic_plan_dict: Dict[str, Any],
        output_filename: Optional[str] = None,
    ) -> Tuple[str, PlacementSolution, Optional[Dict[str, Any]]]:
        """Execute the complete semantic enforcement pipeline.

        This method orchestrates all pipeline stages:
        1. Schema validation
        2. Distance resolution (relative → absolute positions)
        3. Orientation resolution (relative → absolute angles)
        4. Placement execution
        5. Scene assembly (MuJoCo XML generation)

        Objects are tracked at each stage if tracking is enabled.

        Args:
            semantic_plan_dict: Semantic plan dictionary from LLM
            output_filename: Optional filename for XML output (without extension)

        Returns:
            Tuple of (mujoco_xml, placement_solution, pipeline_trace):
            - mujoco_xml: MuJoCo XML string
            - placement_solution: PlacementSolution with all placed objects
            - pipeline_trace: Pipeline trace dictionary (None if tracking disabled)

        Raises:
            SchemaValidationError: If semantic plan fails schema validation
            ResolutionError: If position/orientation resolution fails
            PlacementError: If placement execution fails
            AssemblyError: If scene assembly fails

        **Validates: Requirements 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 2.1, 3.1, 4.1, 5.1, 7.1**
        """
        logger.info("[Pipeline] Starting semantic enforcement pipeline")

        try:
            # Stage 1: Schema Validation
            logger.info("[Pipeline] Stage 1: Schema Validation")
            validated_plan = self._validate_schema(semantic_plan_dict)

            # Track initial semantic plan
            if self.tracker:
                self.tracker.log_stage("semantic_plan", validated_plan["objects"])

            # Stage 2: Distance Resolution
            logger.info("[Pipeline] Stage 2: Distance Resolution")
            resolved_positions = self._resolve_distances(validated_plan)

            # Track after distance resolution
            if self.tracker:
                self.tracker.log_stage(
                    "distance_resolver", resolved_positions["objects"]
                )

            # Stage 3: Orientation Resolution
            logger.info("[Pipeline] Stage 3: Orientation Resolution")
            resolved_orientations = self._resolve_orientations(resolved_positions)

            # Track after orientation resolution
            if self.tracker:
                self.tracker.log_stage(
                    "orientation_resolver", resolved_orientations["objects"]
                )

            # Stage 4: Placement Execution
            logger.info("[Pipeline] Stage 4: Placement Execution")
            placement_solution = self._execute_placement(resolved_orientations)

            # Track after placement execution
            if self.tracker:
                self.tracker.log_stage(
                    "placement_executor", placement_solution.objects
                )

            # Stage 5: Scene Assembly
            logger.info("[Pipeline] Stage 5: Scene Assembly")
            mujoco_xml = self._assemble_scene(
                placement_solution, output_filename
            )

            # Track after scene assembly (extract objects from XML would be complex,
            # so we just log the placement solution objects again)
            if self.tracker:
                self.tracker.log_stage("scene_assembly", placement_solution.objects)

            # Generate pipeline trace
            pipeline_trace = None
            if self.tracker:
                pipeline_trace = self.tracker.generate_trace()
                self.tracker.save_trace(pipeline_trace)

            logger.info(
                f"[Pipeline] Pipeline completed successfully. "
                f"Placed {len(placement_solution.objects)} objects."
            )

            return mujoco_xml, placement_solution, pipeline_trace

        except Exception as e:
            logger.error(f"[Pipeline] Pipeline failed: {e}")
            raise

    def _validate_schema(self, semantic_plan_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Validate semantic plan schema.

        Args:
            semantic_plan_dict: Semantic plan dictionary

        Returns:
            Validated semantic plan dictionary

        Raises:
            SchemaValidationError: If validation fails
        """
        try:
            validation_result = self.schema_validator.validate_plan(
                semantic_plan_dict
            )

            if not validation_result.success:
                error_msg = "Schema validation failed:\n" + "\n".join(
                    validation_result.errors
                )
                logger.error(f"[Pipeline] {error_msg}")
                raise SchemaValidationError(error_msg)

            logger.info(
                f"[Pipeline] Schema validation passed. "
                f"Plan contains {len(semantic_plan_dict['objects'])} objects."
            )

            return semantic_plan_dict

        except SchemaValidationError:
            raise
        except Exception as e:
            raise SchemaValidationError(f"Schema validation error: {e}") from e

    def _resolve_distances(
        self, semantic_plan: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Resolve relative positions to absolute coordinates.

        Args:
            semantic_plan: Semantic plan with relative positions

        Returns:
            Semantic plan with all positions as absolute coordinates

        Raises:
            ResolutionError: If distance resolution fails
        """
        try:
            resolved_plan = self.distance_resolver.resolve_positions(semantic_plan)

            logger.info(
                f"[Pipeline] Distance resolution completed. "
                f"Resolved {len(resolved_plan['objects'])} objects."
            )

            return resolved_plan

        except CircularDependencyError as e:
            error_msg = f"Circular dependency detected in relative positioning: {e}"
            logger.error(f"[Pipeline] {error_msg}")
            raise ResolutionError(error_msg) from e
        except Exception as e:
            error_msg = f"Distance resolution failed: {e}"
            logger.error(f"[Pipeline] {error_msg}")
            raise ResolutionError(error_msg) from e

    def _resolve_orientations(
        self, semantic_plan: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Resolve relative orientations to absolute angles.

        Args:
            semantic_plan: Semantic plan with relative orientations

        Returns:
            Semantic plan with all orientations as absolute angles

        Raises:
            ResolutionError: If orientation resolution fails
        """
        try:
            resolved_plan = self.orientation_resolver.resolve_orientations(
                semantic_plan
            )

            logger.info(
                f"[Pipeline] Orientation resolution completed. "
                f"Resolved {len(resolved_plan['objects'])} objects."
            )

            return resolved_plan

        except Exception as e:
            error_msg = f"Orientation resolution failed: {e}"
            logger.error(f"[Pipeline] {error_msg}")
            raise ResolutionError(error_msg) from e

    def _execute_placement(
        self, semantic_plan: Dict[str, Any]
    ) -> PlacementSolution:
        """Execute placement exactly as specified in semantic plan.

        Args:
            semantic_plan: Semantic plan with all absolute positions and orientations

        Returns:
            PlacementSolution with all placed objects

        Raises:
            PlacementError: If placement execution fails
        """
        try:
            # Convert dict to SemanticPlan dataclass
            # For now, we'll pass the dict directly since PlacementExecutor
            # expects a SemanticPlan object. We need to convert it.
            from creator.placement.semantic_enforcement.models import (
                Position,
                Orientation,
                SemanticObject,
                SemanticPlan,
            )

            # Convert objects
            semantic_objects = []
            for obj_dict in semantic_plan["objects"]:
                # Convert position
                position = Position(
                    absolute=obj_dict["position"].get("absolute"),
                    relative=obj_dict["position"].get("relative"),
                )

                # Convert orientation
                orientation = Orientation(
                    absolute=obj_dict["orientation"].get("absolute"),
                    relative=obj_dict["orientation"].get("relative"),
                )

                # Create SemanticObject
                semantic_obj = SemanticObject(
                    id=obj_dict["id"],
                    Model=obj_dict["Model"],
                    type=obj_dict["type"],
                    size=obj_dict["size"],
                    is_static=obj_dict["is_static"],
                    model_loc=obj_dict["model_loc"],
                    position=position,
                    orientation=orientation,
                    metadata=obj_dict.get("metadata", {}),
                )
                semantic_objects.append(semantic_obj)

            # Create SemanticPlan
            semantic_plan_obj = SemanticPlan(
                schema_version=semantic_plan["schema_version"],
                room_size=semantic_plan["room_size"],
                objects=semantic_objects,
                metadata=semantic_plan.get("metadata", {}),
            )

            # Execute placement
            placement_solution = self.placement_executor.execute_placement(
                semantic_plan_obj
            )

            logger.info(
                f"[Pipeline] Placement execution completed. "
                f"Placed {len(placement_solution.objects)} objects."
            )

            return placement_solution

        except Exception as e:
            error_msg = f"Placement execution failed: {e}"
            logger.error(f"[Pipeline] {error_msg}")
            raise PlacementError(error_msg) from e

    def _assemble_scene(
        self,
        placement_solution: PlacementSolution,
        output_filename: Optional[str] = None,
    ) -> str:
        """Assemble MuJoCo XML from placement solution.

        Args:
            placement_solution: Placement solution with all placed objects
            output_filename: Optional filename for XML output (without extension)

        Returns:
            MuJoCo XML string

        Raises:
            AssemblyError: If scene assembly fails
        """
        try:
            # Determine output path
            output_path = None
            if output_filename:
                output_path = str(self.output_dir / f"{output_filename}.xml")

            # Assemble scene
            mujoco_xml = self.scene_assembly.assemble_scene(
                placement_solution, output_path=output_path
            )

            logger.info(
                f"[Pipeline] Scene assembly completed. "
                f"Generated MuJoCo XML with {len(placement_solution.objects)} objects."
            )

            return mujoco_xml

        except Exception as e:
            error_msg = f"Scene assembly failed: {e}"
            logger.error(f"[Pipeline] {error_msg}")
            raise AssemblyError(error_msg) from e
