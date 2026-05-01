"""Universal Placement System for 3D scene generation.

This module provides the main UniversalPlacementSystem class that integrates
all components of the placement system: CommandInterpreter, LLM Planner,
SceneGraph, AnchorSystem, ConstraintEngine, LayoutSolver, and PhysicsValidator.
"""

import json
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from creator.placement.anchor_system import AnchorSystem
from creator.placement.command_interpreter import CommandInterpreter
from creator.placement.constraint_engine import ConstraintEngine
from creator.placement.formatter import ResultFormatter, PlacementResult, PlacedObject
from creator.placement.layout_solver import (
    LayoutSolver,
    PlacementSolution,
    RoomGeometry,
)
from creator.placement.parser import SemanticPlanParser
from creator.placement.physics import (
    check_collisions,
    check_support_capacity,
    validate_and_repair_layout,
    validate_stacking,
)
from creator.placement.plan import build_semantic_plan
from creator.placement.scene_graph import SceneGraph
from creator.llm.model import prompt_model


class ExportFormat(Enum):
    """Supported export formats for scene generation."""
    MUJOCO_XML = "mujoco_xml"
    USD = "usd"
    GLTF = "gltf"
    JSON = "json"


@dataclass
class GenerationConfig:
    """Configuration for scene generation.

    Attributes:
        room_half_size: Half-size of the room (for rectangular rooms)
        room_boundary: Boundary polygon for arbitrary room shapes
        room_height: Height of the room walls
        seed: Random seed for reproducibility
        max_placement_attempts: Maximum attempts for object placement
        enable_physics_validation: Whether to validate physics
        enable_error_recovery: Whether to attempt error recovery
        export_formats: List of formats to export the scene to
        include_metadata: Whether to include metadata in exports
    """
    room_half_size: float = 5.0
    room_boundary: Optional[List[Tuple[float, float]]] = None
    room_height: float = 3.0
    seed: int = 42
    max_placement_attempts: int = 3
    enable_physics_validation: bool = True
    enable_error_recovery: bool = True
    export_formats: List[ExportFormat] = field(
        default_factory=lambda: [ExportFormat.MUJOCO_XML]
    )
    include_metadata: bool = True


@dataclass
class GenerationResult:
    """Result of scene generation.

    Attributes:
        success: Whether generation was successful
        scene_graph: Generated scene graph
        placement_solution: Placement solution from LayoutSolver
        placed_objects: List of placed objects with final positions
        exports: Dictionary mapping format names to exported content
        metadata: Generation metadata (timing, attempts, etc.)
        errors: List of errors encountered during generation
        warnings: List of warnings from validation
    """
    success: bool
    scene_graph: Optional[SceneGraph] = None
    placement_solution: Optional[PlacementSolution] = None
    placed_objects: List[Dict[str, Any]] = field(default_factory=list)
    exports: Dict[str, str] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


class UniversalPlacementSystem:
    """Universal system for 3D scene placement.

    This class integrates all components of the placement system to provide
    end-to-end scene generation from natural language descriptions to
    exportable 3D scene formats.

    The system follows a multi-stage pipeline:
    1. Command Interpretation: Parse natural language for precise constraints
    2. Semantic Planning (LLM): Generate semantic plan with relationships
    3. Scene Graph Construction: Build hierarchical object representation
    4. Anchor System: Create global and local positioning anchors
    5. Constraint Resolution: Solve spatial constraints
    6. Layout Solving: Compute precise object positions
    7. Physics Validation: Validate and repair physical stability
    8. Export: Format results for target platforms

    Attributes:
        command_interpreter: Interpreter for spatial commands
        parser: Parser for semantic plans
        anchor_system: System for managing anchors
        constraint_engine: Engine for constraint solving
        layout_solver: Solver for computing positions
        formatter: Formatter for exporting results
    """

    def __init__(
        self,
        command_interpreter: Optional[CommandInterpreter] = None,
        parser: Optional[SemanticPlanParser] = None,
        anchor_system: Optional[AnchorSystem] = None,
        constraint_engine: Optional[ConstraintEngine] = None,
        layout_solver: Optional[LayoutSolver] = None,
        formatter: Optional[ResultFormatter] = None,
    ):
        """Initialize the universal placement system.

        Args:
            command_interpreter: Command interpreter (creates new if None)
            parser: Semantic plan parser (creates new if None)
            anchor_system: Anchor system (creates new if None)
            constraint_engine: Constraint engine (creates new if None)
            layout_solver: Layout solver (creates new if None)
            formatter: Result formatter (creates new if None)
        """
        self.command_interpreter = command_interpreter or CommandInterpreter()
        self.parser = parser or SemanticPlanParser()
        self.anchor_system = anchor_system or AnchorSystem()
        self.constraint_engine = constraint_engine or ConstraintEngine()
        self.layout_solver = layout_solver or LayoutSolver(
            constraint_engine=self.constraint_engine
        )
        self.formatter = formatter or ResultFormatter()

    def generate_scene(
        self,
        description: str,
        chosen_models: List[Dict[str, Any]],
        config: Optional[GenerationConfig] = None,
        prompt_model: Optional[Any] = None,
        prompt_template: Optional[str] = None,
        semantic_plan: Optional[Dict[str, Any]] = None,  # NEW: Accept pre-built semantic plan
    ) -> GenerationResult:
        """Generate a complete 3D scene from natural language description.

        This is the main entry point for scene generation. It orchestrates
        all components to produce a complete, validated scene.

        Args:
            description: Natural language description of the scene
            chosen_models: List of model dictionaries to place in the scene
            config: Generation configuration (uses defaults if None)
            prompt_model: LLM model for semantic planning (optional)
            prompt_template: Template for LLM prompts (optional)
            semantic_plan: Pre-built semantic plan (optional, will generate if None)

        Returns:
            GenerationResult with scene data and exports

        Example:
            >>> system = UniversalPlacementSystem()
            >>> models = [
            ...     {"Model": "table", "size": [1.5, 0.8, 0.75]},
            ...     {"Model": "chair", "size": [0.5, 0.5, 1.0]},
            ... ]
            >>> result = system.generate_scene(
            ...     "Place a table in the center with a chair beside it",
            ...     models
            ... )
            >>> if result.success:
            ...     print(result.exports["mujoco_xml"])
        """
        if config is None:
            config = GenerationConfig()

        result = GenerationResult(success=False)
        start_time = time.time()

        try:
            # Stage 1: Command Interpretation
            print("[UniversalPlacementSystem] Stage 1: Command Interpretation")
            spatial_command = self.command_interpreter.parse_spatial_command(
                description
            )
            result.metadata["spatial_command"] = {
                "positioning_type": spatial_command.positioning_type.value,
                "precise_constraints": len(spatial_command.precise_constraints),
                "relative_positions": spatial_command.relative_positions,
            }

            # Stage 2: Semantic Planning (LLM)
            print("[UniversalPlacementSystem] Stage 2: Semantic Planning")
            
            # CRITICAL FIX: Use pre-built semantic_plan if provided
            if semantic_plan is not None:
                print("[UniversalPlacementSystem] Using pre-built semantic plan from runner")
            else:
                print("[UniversalPlacementSystem] Generating new semantic plan")
                semantic_plan = self._generate_semantic_plan(
                    description=description,
                    chosen_models=chosen_models,
                    spatial_command=spatial_command,
                    prompt_model=prompt_model,
                    prompt_template=prompt_template,
                )
            
            # Ensure schema_version is present
            if isinstance(semantic_plan, dict) and "schema_version" not in semantic_plan:
                semantic_plan["schema_version"] = "1.0"

            # Validate semantic plan
            parsed_plan = self.parser.parse(semantic_plan)
            validation_warnings = self.parser.validate_plan(parsed_plan)
            if validation_warnings:
                result.warnings.extend(validation_warnings)

            # Stage 3: Scene Graph Construction
            print("[UniversalPlacementSystem] Stage 3: Scene Graph Construction")
            scene_graph = self._build_scene_graph(parsed_plan)
            result.scene_graph = scene_graph

            # Stage 4: Anchor System Setup
            print("[UniversalPlacementSystem] Stage 4: Anchor System Setup")
            room_geometry = self._create_room_geometry(config)
            self._setup_anchors(room_geometry)

            # Stage 5: Constraint Resolution
            print("[UniversalPlacementSystem] Stage 5: Constraint Resolution")
            self._resolve_constraints(parsed_plan)

            # Stage 6: Layout Solving
            print("[UniversalPlacementSystem] Stage 6: Layout Solving")
            placement_solution = self._solve_layout(
                scene_graph=scene_graph,
                room_geometry=room_geometry,
                semantic_plan=semantic_plan,
                config=config,
            )

            if not placement_solution.success:
                # Attempt error recovery if enabled
                if config.enable_error_recovery:
                    print("[UniversalPlacementSystem] Attempting error recovery...")
                    placement_solution = self._attempt_error_recovery(
                        scene_graph=scene_graph,
                        room_geometry=room_geometry,
                        semantic_plan=semantic_plan,
                        config=config,
                        failure_reason=placement_solution.failure_reason,
                    )

                if not placement_solution.success:
                    result.errors.append(
                        f"Layout solving failed: {placement_solution.failure_reason}"
                    )
                    result.metadata["placement_attempts"] = config.max_placement_attempts
                    return result

            result.placement_solution = placement_solution
            result.placed_objects = placement_solution.placed_objects

            # Stage 7: Physics Validation
            if config.enable_physics_validation:
                print("[UniversalPlacementSystem] Stage 7: Physics Validation")
                validation_result = self._validate_physics(
                    placement_solution.placed_objects,
                    room_geometry,
                    semantic_plan,
                )

                if validation_result["has_issues"]:
                    result.warnings.extend(validation_result["warnings"])

                    # Repair if needed
                    if validation_result["needs_repair"]:
                        print("[UniversalPlacementSystem] Repairing physics issues...")
                        result.placed_objects = validate_and_repair_layout(
                            result.placed_objects,
                            room_half_size=config.room_half_size,
                            semantic_plan=semantic_plan,
                        )

            # Stage 8: Export
            print("[UniversalPlacementSystem] Stage 8: Export")
            result.exports = self._export_scene(
                placed_objects=result.placed_objects,
                scene_graph=scene_graph,
                room_geometry=room_geometry,
                config=config,
            )

            # Success!
            result.success = True
            result.metadata["generation_time"] = time.time() - start_time
            result.metadata["num_objects"] = len(result.placed_objects)
            result.metadata["algorithm_used"] = placement_solution.algorithm_used.value
            result.metadata["quality_score"] = placement_solution.quality_score

            print(f"[UniversalPlacementSystem] Scene generation completed successfully "
                  f"in {result.metadata['generation_time']:.2f}s")

        except Exception as e:
            result.success = False
            result.errors.append(f"Scene generation failed: {str(e)}")
            result.metadata["generation_time"] = time.time() - start_time
            print(f"[UniversalPlacementSystem] Error: {e}")

        return result

    def _generate_semantic_plan(
        self,
        description: str,
        chosen_models: List[Dict[str, Any]],
        spatial_command: Any,
        prompt_model: Optional[Any],
        prompt_template: Optional[str],
    ) -> Dict[str, Any]:
        """Generate semantic plan using LLM.

        Args:
            description: Natural language description
            chosen_models: List of models to place
            spatial_command: Parsed spatial command
            prompt_model: LLM model for planning
            prompt_template: Template for prompts

        Returns:
            Semantic plan dictionary
        """
        if prompt_model is None or prompt_template is None:
            # Fallback: create simple plan without LLM
            return self._create_fallback_plan(chosen_models, spatial_command)

        # Use existing build_semantic_plan from plan.py
        semantic_plan = build_semantic_plan(
            prompt_model=prompt_model,
            prompt_template=prompt_template,
            query=description,
            chosen_model="",  # Not used in new system
            chosen_models=chosen_models,
            context_models=[],
            use_scene_graph=True,
            command_interpreter=self.command_interpreter,
        )

        # Fix: Add 'id' and 'type' fields to objects if missing
        # The existing build_semantic_plan returns objects with 'Model' but not 'id'
        # We need to ensure IDs are unique by adding index
        if isinstance(semantic_plan, dict) and "objects" in semantic_plan:
            print(f"[universal_system] Processing {len(semantic_plan['objects'])} objects from semantic_plan")
            
            # Create mapping from Model name to size (from chosen_models)
            # Use case-insensitive matching and handle partial matches
            model_sizes = {}
            print(f"[universal_system] Building model_sizes from {len(chosen_models)} chosen_models:")
            for model in chosen_models:
                model_name = model.get("Model", "")
                if model_name and "size" in model:
                    # Store with original name and lowercase version
                    model_sizes[model_name] = model.get("size")
                    model_sizes[model_name.lower()] = model.get("size")
                    print(f"[universal_system]   {model_name}: {model.get('size')}")
            
            # First pass: create unique IDs and mapping
            id_mapping = {}
            
            for i, obj in enumerate(semantic_plan["objects"]):
                model_name = obj.get("Model", f"object_{i}")
                print(f"[universal_system] Processing object {i}: Model='{model_name}'")
                
                # Clean model name for use in ID (remove spaces, special chars)
                clean_name = model_name.replace(" ", "_").replace("-", "_")
                new_id = f"{clean_name}_{i}"
                
                # Create mapping from Model name to new ID (for constraint targets)
                id_mapping[model_name] = new_id
                
                # Set new unique ID (always overwrite to ensure uniqueness)
                obj["id"] = new_id
                
                # CRITICAL FIX: Preserve size from chosen_models
                # Try exact match first, then case-insensitive, then fuzzy match
                size = None
                if model_name in model_sizes:
                    size = model_sizes[model_name]
                    print(f"[universal_system]   ✓ Exact match found")
                elif model_name.lower() in model_sizes:
                    size = model_sizes[model_name.lower()]
                    print(f"[universal_system]   ✓ Case-insensitive match found")
                else:
                    # Fuzzy match: check if model_name contains any key from model_sizes
                    model_lower = model_name.lower()
                    for key, val in model_sizes.items():
                        if isinstance(key, str) and key.lower() in model_lower:
                            size = val
                            print(f"[universal_system]   ✓ Fuzzy match found: '{key}' in '{model_name}'")
                            break
                    if not size:
                        print(f"[universal_system]   ✗ No match found for '{model_name}'")
                
                if size:
                    obj["size"] = size
                    print(f"[universal_system] ✓ Size preserved for {model_name}: {size}")
                else:
                    # Fallback: use default size
                    obj["size"] = [1.0, 1.0, 1.0]
                    print(f"[universal_system] ⚠ WARNING: No size found for {model_name}, using default")
                    print(f"[universal_system]   Available keys in model_sizes: {list(model_sizes.keys())[:10]}")
                
                if "type" not in obj:
                    # Infer type from is_static
                    if obj.get("is_static", True):
                        obj["type"] = "furniture"
                    else:
                        obj["type"] = "small_object"
            
            # NEW: Determine object dynamics using LLM
            print("[universal_system] Determining object dynamics...")
            semantic_plan = self._determine_object_dynamics(
                semantic_plan=semantic_plan,
                user_prompt=description,
                prompt_model=prompt_model,
            )
            
            # Second pass: update all target references in constraints
            for obj in semantic_plan["objects"]:
                if "constraints" in obj and isinstance(obj["constraints"], list):
                    for constraint in obj["constraints"]:
                        if "target" in constraint:
                            old_target = constraint["target"]
                            # Try to map to new ID
                            if old_target in id_mapping:
                                constraint["target"] = id_mapping[old_target]

        return semantic_plan

    def _determine_object_dynamics(
        self,
        semantic_plan: Dict[str, Any],
        user_prompt: str,
        prompt_model: Optional[Any],
    ) -> Dict[str, Any]:
        """Determine object dynamics using LLM as robot training environment designer.
        
        This method performs a separate LLM query to determine which objects should be
        dynamic (can move/fall) vs static (fixed in place). The LLM acts as a "robot
        training environment designer" and makes informed decisions based on:
        - Original user prompt (scene description)
        - List of objects with their sizes and types
        - Physical properties (volume, dimensions)
        - Scene purpose (robot training, manipulation, etc.)
        
        Args:
            semantic_plan: Semantic plan with objects
            user_prompt: Original user prompt describing the scene
            prompt_model: LLM model for dynamics determination
            
        Returns:
            Updated semantic plan with is_static flags set by LLM
        """
        if prompt_model is None:
            # Fallback: use heuristic
            print("[universal_system] No LLM available, using fallback heuristic for dynamics")
            return self._apply_heuristic_dynamics(semantic_plan)
        
        # Build context for LLM
        objects_info = []
        for obj in semantic_plan.get("objects", []):
            model_name = obj.get("Model", "")
            size = obj.get("size", [1.0, 1.0, 1.0])
            volume = size[0] * size[1] * size[2]
            
            objects_info.append({
                "name": model_name,
                "size": size,
                "volume": volume,
                "current_is_static": obj.get("is_static", None),
            })
        
        # System prompt: LLM as robot training environment designer
        system_prompt = """You are a robot training environment designer. Your task is to determine 
which objects in a scene should be dynamic (can move/fall) vs static (fixed in place).

Consider:
- Small objects (fruits, tools, small containers) should typically be dynamic for manipulation training
- Large furniture (tables, chairs, cabinets) should typically be static
- Containers (boxes, baskets, bowls) should be dynamic if they are meant to be manipulated
- The scene purpose: training robots for manipulation, grasping, placing, etc.

Return a JSON object with format:
{
  "objects": [
    {"name": "object_name", "is_static": true/false, "reason": "explanation"},
    ...
  ]
}
"""
        
        # User prompt with scene context
        user_context = f"""Original scene request: "{user_prompt}"

Objects in scene:
{json.dumps(objects_info, indent=2)}

For each object, determine if it should be static (fixed) or dynamic (can move/fall).
Consider the scene purpose and realistic robot training scenarios."""
        
        try:
            # Call LLM
            print("[universal_system] Calling LLM for dynamics determination...")
            response = prompt_model(system_prompt + "\n\n" + user_context, user_prompt)
            
            if isinstance(response, dict) and "objects" in response:
                # Apply LLM decisions to semantic plan
                name_to_decision = {
                    obj["name"]: obj["is_static"] 
                    for obj in response["objects"]
                }
                
                for obj in semantic_plan.get("objects", []):
                    model_name = obj.get("Model", "")
                    if model_name in name_to_decision:
                        obj["is_static"] = name_to_decision[model_name]
                        print(f"[universal_system] LLM decision: {model_name} is_static={obj['is_static']}")
                    else:
                        print(f"[universal_system] WARNING: No LLM decision for {model_name}, keeping current value")
            else:
                print("[universal_system] LLM response invalid, using fallback heuristic")
                return self._apply_heuristic_dynamics(semantic_plan)
                
        except Exception as e:
            print(f"[universal_system] LLM dynamics determination failed: {e}, using fallback")
            return self._apply_heuristic_dynamics(semantic_plan)
        
        return semantic_plan

    def _apply_heuristic_dynamics(
        self,
        semantic_plan: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Fallback: apply heuristic-based dynamics determination.
        
        This method uses simple heuristics to determine object dynamics when LLM is
        unavailable or fails:
        - Furniture objects (table, chair, sofa, etc.) are static
        - Container objects (box, basket, etc.) are dynamic
        - Objects with volume < 0.3 m³ are dynamic (unless furniture)
        - Large objects (volume >= 0.3 m³) are static
        
        Args:
            semantic_plan: Semantic plan with objects
            
        Returns:
            Updated semantic plan with is_static flags set by heuristic
        """
        print("[universal_system] Applying heuristic-based dynamics determination")
        
        for obj in semantic_plan.get("objects", []):
            model_name = obj.get("Model", "")
            size = obj.get("size", [1.0, 1.0, 1.0])
            volume = size[0] * size[1] * size[2]
            
            # Check if object is furniture (should be static regardless of size)
            is_furniture = any(keyword in model_name.lower() 
                              for keyword in ["table", "chair", "sofa", "desk", "cabinet", 
                                            "shelf", "bed", "couch", "bench", "stool",
                                            "dresser", "wardrobe", "bookcase"])
            
            # Check if object is a container (should be dynamic)
            is_container = any(keyword in model_name.lower() 
                               for keyword in ["box", "basket", "container", "crate", 
                                              "bin", "bowl", "cup"])
            
            # Apply heuristic with priority order:
            # 1. Furniture is always static
            # 2. Containers are always dynamic
            # 3. Small objects (volume < 0.3) are dynamic
            # 4. Large objects (volume >= 0.3) are static
            if is_furniture:
                obj["is_static"] = True
                print(f"[universal_system] Heuristic: {model_name} is static (furniture)")
            elif is_container:
                obj["is_static"] = False
                print(f"[universal_system] Heuristic: {model_name} is dynamic (container)")
            elif volume < 0.3:
                obj["is_static"] = False
                print(f"[universal_system] Heuristic: {model_name} is dynamic (volume={volume:.4f} m³)")
            elif "is_static" not in obj:
                obj["is_static"] = True
                print(f"[universal_system] Heuristic: {model_name} is static (volume={volume:.4f} m³)")
            else:
                # Keep existing value
                print(f"[universal_system] Heuristic: {model_name} keeping existing is_static={obj.get('is_static')}")
        
        return semantic_plan

    def _create_fallback_plan(
        self,
        chosen_models: List[Dict[str, Any]],
        spatial_command: Any,
    ) -> Dict[str, Any]:
        """Create a simple fallback plan without LLM.

        Args:
            chosen_models: List of models to place
            spatial_command: Parsed spatial command

        Returns:
            Simple semantic plan
        """
        objects = []

        for i, model in enumerate(chosen_models):
            model_name = model.get("Model") or model.get("name", f"object_{i}")
            model_id = model.get("id", f"{model_name}_{i}")
            
            # Preserve size from chosen_models
            size = model.get("size", [1.0, 1.0, 1.0])

            # Create basic constraints
            constraints = []

            # First object goes to center
            if i == 0:
                constraints.append({
                    "type": "region",
                    "value": "middle",
                    "hard": False,
                    "weight": 1.0,
                })
            else:
                # Other objects near the first one
                constraints.append({
                    "type": "near",
                    "target": objects[0]["id"],
                    "distance": [0.5, 2.0],
                    "hard": False,
                    "weight": 0.8,
                })

            objects.append({
                "id": model_id,
                "Model": model_name,
                "size": size,
                "type": "furniture",
                "constraints": constraints,
            })

        semantic_plan = {"objects": objects}
        
        # Apply dynamics determination (fallback heuristic since no LLM)
        semantic_plan = self._apply_heuristic_dynamics(semantic_plan)
        
        return semantic_plan

    def _build_scene_graph(self, parsed_plan: Any) -> SceneGraph:
        """Build scene graph from parsed semantic plan.

        Args:
            parsed_plan: Parsed semantic plan

        Returns:
            SceneGraph with hierarchical object structure
        """
        scene_graph = SceneGraph()

        # Track added objects
        added_objects = {}

        # First pass: add objects without parent relationships
        for obj_dict in parsed_plan.objects:
            obj_id = obj_dict.get("id", "")
            obj_type = obj_dict.get("type", "generic")

            # Check if this object has a parent relationship
            has_parent = False
            for constraint in parsed_plan.constraints:
                if constraint.source_object == obj_id:
                    if constraint.constraint_type.value in ["support", "containment"]:
                        has_parent = True
                        break

            if not has_parent:
                # Add as root-level object
                node = scene_graph.add_object(
                    object_id=obj_id,
                    object_type=obj_type,
                    parent=None,
                    constraints=[c.__dict__ for c in parsed_plan.constraints
                                if c.source_object == obj_id],
                    metadata=obj_dict,
                )
                added_objects[obj_id] = node

        # Second pass: add objects with parent relationships
        for obj_dict in parsed_plan.objects:
            obj_id = obj_dict.get("id", "")

            if obj_id in added_objects:
                continue

            obj_type = obj_dict.get("type", "generic")

            # Find parent
            parent_node = None
            for constraint in parsed_plan.constraints:
                if constraint.source_object == obj_id:
                    if constraint.constraint_type.value in ["support", "containment"]:
                        parent_id = constraint.target_object
                        if parent_id in added_objects:
                            parent_node = added_objects[parent_id]
                            break

            node = scene_graph.add_object(
                object_id=obj_id,
                object_type=obj_type,
                parent=parent_node,
                constraints=[c.__dict__ for c in parsed_plan.constraints
                            if c.source_object == obj_id],
                metadata=obj_dict,
            )
            added_objects[obj_id] = node

        return scene_graph

    def _create_room_geometry(self, config: GenerationConfig) -> RoomGeometry:
        """Create room geometry from configuration.

        Args:
            config: Generation configuration

        Returns:
            RoomGeometry object
        """
        if config.room_boundary is not None:
            # Arbitrary polygonal room
            from creator.placement.geometry import Vec2
            boundary = [Vec2(x, y) for x, y in config.room_boundary]
            return RoomGeometry(
                boundary_polygon=boundary,
                wall_height=config.room_height,
                is_rectangular=False,
            )
        else:
            # Rectangular room
            return RoomGeometry(
                half_size=config.room_half_size,
                wall_height=config.room_height,
                is_rectangular=True,
            )

    def _setup_anchors(self, room_geometry: RoomGeometry) -> None:
        """Setup anchor system with global anchors.

        Args:
            room_geometry: Room geometry specification
        """
        if room_geometry.is_rectangular:
            # Create anchors for rectangular room
            self.anchor_system.create_global_anchors(
                room_boundary=[],
                room_half_size=room_geometry.half_size,
            )
        else:
            # Create anchors for arbitrary polygonal room
            boundary_tuples = [
                (p.x, p.y) for p in room_geometry.boundary_polygon
            ]
            self.anchor_system.create_global_anchors(
                room_boundary=boundary_tuples,
                room_half_size=None,
            )

    def _resolve_constraints(self, parsed_plan: Any) -> None:
        """Resolve constraints and detect conflicts.

        Args:
            parsed_plan: Parsed semantic plan with constraints
        """
        # Add all constraints to the engine
        for constraint in parsed_plan.constraints:
            self.constraint_engine.add_constraint(constraint)

        # Detect conflicts
        from creator.placement.constraint_engine import ConstraintConflict
        conflicts = self.constraint_engine._detect_conflicts()

        if conflicts:
            print(f"[UniversalPlacementSystem] Detected {len(conflicts)} constraint conflicts")
            # Resolve conflicts
            resolution = self.constraint_engine.resolve_conflicts(conflicts)
            if resolution.success:
                print(f"[UniversalPlacementSystem] Resolved conflicts: "
                      f"removed {len(resolution.removed_constraints)}, "
                      f"relaxed {len(resolution.relaxed_constraints)}")

    def _solve_layout(
        self,
        scene_graph: SceneGraph,
        room_geometry: RoomGeometry,
        semantic_plan: Dict[str, Any],
        config: GenerationConfig,
    ) -> PlacementSolution:
        """Solve layout to compute object positions.

        Args:
            scene_graph: Scene graph with objects
            room_geometry: Room geometry
            semantic_plan: Semantic plan from LLM
            config: Generation configuration

        Returns:
            PlacementSolution with computed positions
        """
        solution = self.layout_solver.solve_placement(
            scene_graph=scene_graph,
            room_geometry=room_geometry,
            semantic_plan=semantic_plan,
            seed=config.seed,
        )

        return solution

    def _attempt_error_recovery(
        self,
        scene_graph: SceneGraph,
        room_geometry: RoomGeometry,
        semantic_plan: Dict[str, Any],
        config: GenerationConfig,
        failure_reason: Optional[str],
    ) -> PlacementSolution:
        """Attempt to recover from placement failure.

        Args:
            scene_graph: Scene graph with objects
            room_geometry: Room geometry
            semantic_plan: Semantic plan
            config: Generation configuration
            failure_reason: Reason for initial failure

        Returns:
            PlacementSolution from recovery attempt
        """
        print(f"[UniversalPlacementSystem] Recovery reason: {failure_reason}")

        # Try with different random seeds
        for attempt in range(config.max_placement_attempts - 1):
            seed = config.seed + (attempt + 1) * 1000
            print(f"[UniversalPlacementSystem] Recovery attempt {attempt + 1} "
                  f"with seed {seed}")

            solution = self.layout_solver.solve_placement(
                scene_graph=scene_graph,
                room_geometry=room_geometry,
                semantic_plan=semantic_plan,
                seed=seed,
            )

            if solution.success:
                print(f"[UniversalPlacementSystem] Recovery successful!")
                return solution

        # All attempts failed
        print(f"[UniversalPlacementSystem] Recovery failed after "
              f"{config.max_placement_attempts} attempts")
        return PlacementSolution(
            success=False,
            failure_reason=f"Failed after {config.max_placement_attempts} attempts",
        )

    def _validate_physics(
        self,
        placed_objects: List[Dict[str, Any]],
        room_geometry: RoomGeometry,
        semantic_plan: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Validate physics of placed objects.

        Args:
            placed_objects: List of placed objects
            room_geometry: Room geometry
            semantic_plan: Semantic plan

        Returns:
            Dictionary with validation results
        """
        issues = []
        warnings = []

        # Check collisions
        collisions = check_collisions(placed_objects, collision_margin=0.01)
        if collisions:
            issues.append(f"Found {len(collisions)} collisions")
            warnings.extend([
                f"Collision between {c['object_a']} and {c['object_b']}"
                for c in collisions
            ])

        # Validate stacking
        stacking_result = validate_stacking(
            placed_objects,
            stability_threshold=0.7,
        )
        if not stacking_result["is_stable"]:
            issues.append(f"Found {len(stacking_result['unstable_objects'])} unstable objects")
            warnings.extend([
                f"Unstable: {issue['object']} - {issue['description']}"
                for issue in stacking_result["stability_issues"]
            ])

        # Check support capacity
        capacity_result = check_support_capacity(
            placed_objects,
            default_capacity=50.0,
        )
        if not capacity_result["capacity_ok"]:
            issues.append(f"Found {len(capacity_result['overloaded_objects'])} overloaded objects")
            warnings.extend([
                f"Overloaded: {issue['supporter']} - {issue['description']}"
                for issue in capacity_result["capacity_issues"]
            ])

        return {
            "has_issues": len(issues) > 0,
            "needs_repair": len(collisions) > 0 or not stacking_result["is_stable"],
            "issues": issues,
            "warnings": warnings,
        }

    def _export_scene(
        self,
        placed_objects: List[Dict[str, Any]],
        scene_graph: SceneGraph,
        room_geometry: RoomGeometry,
        config: GenerationConfig,
    ) -> Dict[str, str]:
        """Export scene to requested formats.

        Args:
            placed_objects: List of placed objects
            scene_graph: Scene graph
            room_geometry: Room geometry
            config: Generation configuration

        Returns:
            Dictionary mapping format names to exported content
        """
        exports = {}

        # Convert to PlacementResult format
        placement_result = PlacementResult(
            objects=[
                PlacedObject(
                    id=obj.get("id", obj.get("Model", "")),
                    model_name=obj.get("Model", ""),
                    position=(
                        float(obj.get("Pose", {}).get("x", 0.0)),
                        float(obj.get("Pose", {}).get("y", 0.0)),
                        float(obj.get("Pose", {}).get("z", 0.0)),
                    ),
                    orientation=(
                        float(obj.get("Pose", {}).get("yaw", 0.0)),
                        0.0,
                        0.0,
                    ),
                    metadata=obj,
                )
                for obj in placed_objects
            ],
            scene_graph=scene_graph,
            room_geometry={
                "half_size": room_geometry.half_size,
                "height": room_geometry.wall_height,
            },
            metadata={
                "include_metadata": config.include_metadata,
            },
        )

        # Export to requested formats
        for export_format in config.export_formats:
            if export_format == ExportFormat.MUJOCO_XML:
                exports["mujoco_xml"] = self.formatter.to_mujoco_xml(
                    placement_result,
                    include_metadata=config.include_metadata,
                )
            elif export_format == ExportFormat.USD:
                exports["usd"] = self.formatter.to_usd(
                    placement_result,
                    include_metadata=config.include_metadata,
                )
            elif export_format == ExportFormat.GLTF:
                exports["gltf"] = self.formatter.to_gltf(
                    placement_result,
                    include_metadata=config.include_metadata,
                )
            elif export_format == ExportFormat.JSON:
                exports["json"] = self.formatter.serialize(
                    placement_result,
                    format="json",
                )

        return exports



def llm_replan_layout(
    placed_models: List[Dict[str, Any]],
    semantic_plan: Dict[str, Any],
    room_half_size: float,
    collision_details: List[Dict[str, Any]],
    max_retries: int = 3,
) -> Tuple[List[Dict[str, Any]], bool]:
    """LLM fallback for replanning object layout when gradient resolution fails.
    
    This function is invoked when algorithmic collision resolution cannot converge.
    It uses the LLM to determine optimal object positions that eliminate overlaps
    while preserving semantic constraints and room boundaries.
    
    Args:
        placed_models: Current placed objects with positions
        semantic_plan: Semantic plan with constraints
        room_half_size: Half-size of the room (for rectangular rooms)
        collision_details: List of collision dictionaries from check_collisions()
        max_retries: Maximum number of LLM retry attempts (default: 3)
        
    Returns:
        Tuple of (replanned_models, success):
        - replanned_models: Updated placed objects with new positions
        - success: Whether LLM replanning succeeded
    """
    print(f"[llm_replan_layout] Invoking LLM fallback for {len(placed_models)} objects")
    print(f"[llm_replan_layout] Reason: {len(collision_details)} collision(s) after gradient resolution")
    
    # Extract object information for LLM prompt
    objects_info = []
    for i, obj in enumerate(placed_models):
        model_name = obj.get("Model", f"object_{i}")
        size = obj.get("size", [1.0, 1.0, 1.0])
        pose = obj.get("Pose", {})
        current_pos = (
            float(pose.get("x", 0.0)),
            float(pose.get("y", 0.0)),
            float(pose.get("z", 0.0)),
        )
        
        objects_info.append({
            "index": i,
            "name": model_name,
            "size": size,
            "current_position": current_pos,
        })
    
    # Extract semantic constraints
    constraints_info = []
    objects_list = semantic_plan.get("objects", [])
    for obj_dict in objects_list:
        obj_name = obj_dict.get("Model", "")
        constraints = obj_dict.get("constraints", [])
        for constraint in constraints:
            if isinstance(constraint, dict):
                constraints_info.append({
                    "object": obj_name,
                    "type": constraint.get("type", ""),
                    "target": constraint.get("target", ""),
                    "value": constraint.get("value", ""),
                })
    
    # Format collision information
    collision_summary = []
    for collision in collision_details:
        collision_summary.append({
            "object_a": collision["object_a"],
            "object_b": collision["object_b"],
            "overlap_depth": collision["overlap_depth"],
        })
    
    # Build LLM prompt
    system_prompt = """You are a 3D scene layout optimizer. Your task is to reposition objects to eliminate overlaps while preserving semantic constraints and room boundaries.

You will receive:
1. List of objects with their current positions and sizes
2. Semantic constraints (near, on_top_of, region, etc.)
3. Room boundaries
4. Current collision information

Your goal:
- Eliminate ALL overlaps between objects
- Preserve semantic constraints as much as possible
- Keep all objects within room boundaries
- Maintain reasonable spacing between objects (minimum 0.05m clearance)

Return a JSON object with format:
{
  "positions": [
    {"index": 0, "x": 1.5, "y": 0.0, "z": 0.4},
    {"index": 1, "x": -1.2, "y": 0.5, "z": 0.4},
    ...
  ],
  "reasoning": "Brief explanation of positioning strategy"
}

IMPORTANT:
- Only modify X and Y positions for objects at the same Z-level (same surface)
- Do NOT change Z positions (height) - objects must stay on their current surfaces
- Ensure minimum clearance of 0.05m between object bounding boxes
- Respect room boundaries: -room_half_size <= x,y <= room_half_size
"""
    
    user_prompt = f"""Room boundaries: [-{room_half_size}, {room_half_size}] in both X and Y

Objects to reposition:
{json.dumps(objects_info, indent=2)}

Semantic constraints:
{json.dumps(constraints_info, indent=2)}

Current collisions (MUST be eliminated):
{json.dumps(collision_summary, indent=2)}

Please provide new X,Y positions for all objects that eliminate overlaps while preserving constraints."""
    
    # Attempt LLM replanning with retries
    for attempt in range(max_retries):
        try:
            print(f"[llm_replan_layout] LLM attempt {attempt + 1}/{max_retries}...")
            
            # Call LLM
            response = prompt_model(system_prompt, user_prompt)
            
            if not isinstance(response, dict) or "positions" not in response:
                print(f"[llm_replan_layout] Invalid response format (attempt {attempt + 1})")
                if attempt < max_retries - 1:
                    # Modify prompt for retry
                    user_prompt += "\n\nPrevious attempt failed. Please ensure you return valid JSON with 'positions' array."
                    continue
                else:
                    print("[llm_replan_layout] All retry attempts exhausted")
                    return placed_models, False
            
            # Extract positions from response
            new_positions = response["positions"]
            reasoning = response.get("reasoning", "No reasoning provided")
            
            print(f"[llm_replan_layout] LLM reasoning: {reasoning}")
            
            # Validate response
            validation_result = _validate_llm_positions(
                placed_models=placed_models,
                new_positions=new_positions,
                room_half_size=room_half_size,
                semantic_plan=semantic_plan,
            )
            
            if not validation_result["valid"]:
                print(f"[llm_replan_layout] Validation failed: {validation_result['reason']}")
                if attempt < max_retries - 1:
                    # Modify prompt for retry
                    user_prompt += f"\n\nPrevious attempt failed validation: {validation_result['reason']}. Please fix these issues."
                    continue
                else:
                    print("[llm_replan_layout] All retry attempts exhausted")
                    return placed_models, False
            
            # Apply new positions
            replanned_models = []
            for obj in placed_models:
                obj_copy = dict(obj)
                replanned_models.append(obj_copy)
            
            for pos_update in new_positions:
                idx = pos_update.get("index")
                if idx is None or idx < 0 or idx >= len(replanned_models):
                    continue
                
                pose = dict(replanned_models[idx].get("Pose", {}))
                pose["x"] = float(pos_update.get("x", pose.get("x", 0.0)))
                pose["y"] = float(pos_update.get("y", pose.get("y", 0.0)))
                # Keep Z unchanged (objects stay on their surfaces)
                replanned_models[idx]["Pose"] = pose
            
            # Final collision check
            final_collisions = check_collisions(replanned_models, collision_margin=0.01)
            
            if final_collisions:
                print(f"[llm_replan_layout] LLM solution still has {len(final_collisions)} collision(s)")
                if attempt < max_retries - 1:
                    # Provide feedback for retry
                    remaining_collisions = [
                        f"{c['object_a']} <-> {c['object_b']} (depth={c['overlap_depth']:.3f}m)"
                        for c in final_collisions
                    ]
                    user_prompt += f"\n\nPrevious solution still had collisions: {remaining_collisions}. Please increase spacing."
                    continue
                else:
                    print("[llm_replan_layout] WARNING: LLM solution has collisions, but using it anyway (best effort)")
                    return replanned_models, True  # Partial success
            
            print(f"[llm_replan_layout] SUCCESS: LLM replanning eliminated all collisions")
            return replanned_models, True
            
        except Exception as e:
            print(f"[llm_replan_layout] Error during attempt {attempt + 1}: {e}")
            if attempt < max_retries - 1:
                continue
            else:
                print("[llm_replan_layout] All retry attempts exhausted due to errors")
                return placed_models, False
    
    # Should not reach here, but return failure as fallback
    return placed_models, False


def _validate_llm_positions(
    placed_models: List[Dict[str, Any]],
    new_positions: List[Dict[str, Any]],
    room_half_size: float,
    semantic_plan: Dict[str, Any],
) -> Dict[str, Any]:
    """Validate LLM-proposed positions for feasibility.
    
    Args:
        placed_models: Original placed objects
        new_positions: LLM-proposed positions
        room_half_size: Room boundary
        semantic_plan: Semantic plan with constraints
        
    Returns:
        Dictionary with keys:
        - valid: Whether positions are valid (bool)
        - reason: Reason for validation failure (str)
    """
    # Check that all objects have positions
    if len(new_positions) != len(placed_models):
        return {
            "valid": False,
            "reason": f"Position count mismatch: expected {len(placed_models)}, got {len(new_positions)}",
        }
    
    # Check that all positions are within room boundaries
    for pos in new_positions:
        x = pos.get("x", 0.0)
        y = pos.get("y", 0.0)
        
        if abs(x) > room_half_size or abs(y) > room_half_size:
            return {
                "valid": False,
                "reason": f"Position ({x}, {y}) outside room boundaries [-{room_half_size}, {room_half_size}]",
            }
    
    # Check that indices are valid
    for pos in new_positions:
        idx = pos.get("index")
        if idx is None or idx < 0 or idx >= len(placed_models):
            return {
                "valid": False,
                "reason": f"Invalid object index: {idx}",
            }
    
    # All checks passed
    return {"valid": True, "reason": ""}
