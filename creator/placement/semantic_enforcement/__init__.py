"""Semantic Plan Enforcement Module.

This module enforces LLM-generated semantic plans through a strict pipeline:
Schema Validation → Distance Resolution → Orientation Resolution →
Placement Execution → Scene Assembly → Validation.

The core principle: LLM is the main composer and architect of the scene,
and the system is a "dumb executor" that faithfully implements LLM instructions
without modifications.
"""

from creator.placement.semantic_enforcement.models import (
    Position,
    Orientation,
    SemanticObject,
    SemanticPlan,
    PlacedObject,
    PlacementSolution,
    ValidationReport,
)
from creator.placement.semantic_enforcement.distance_resolver import (
    DistanceResolver,
    CircularDependencyError,
)
from creator.placement.semantic_enforcement.orientation_resolver import (
    OrientationResolver,
)
from creator.placement.semantic_enforcement.placement_executor import (
    PlacementExecutor,
)
from creator.placement.semantic_enforcement.scene_assembly import (
    SceneAssembly,
)
from creator.placement.semantic_enforcement.pipeline_tracker import (
    PipelineTracker,
    StageLog,
    StageChange,
)
from creator.placement.semantic_enforcement.pipeline import (
    SemanticEnforcementPipeline,
    PipelineError,
    SchemaValidationError,
    ResolutionError,
    PlacementError,
    AssemblyError,
)

__all__ = [
    "Position",
    "Orientation",
    "SemanticObject",
    "SemanticPlan",
    "PlacedObject",
    "PlacementSolution",
    "ValidationReport",
    "DistanceResolver",
    "CircularDependencyError",
    "OrientationResolver",
    "PlacementExecutor",
    "SceneAssembly",
    "PipelineTracker",
    "StageLog",
    "StageChange",
    "SemanticEnforcementPipeline",
    "PipelineError",
    "SchemaValidationError",
    "ResolutionError",
    "PlacementError",
    "AssemblyError",
]

__version__ = "1.0.0"
