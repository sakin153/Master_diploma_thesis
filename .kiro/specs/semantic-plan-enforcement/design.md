# Design Document: Semantic Plan Enforcement

## Overview

This design document specifies the architecture for enforcing semantic plans in the World Creator 3D scene generation system. The core principle is that **LLM is the main composer and architect** of the scene, creating detailed placement instructions that the system executes faithfully without modifications.

### Current Problem

The existing system has a disconnect between the LLM's semantic plan and the final scene assembly. The LLM creates detailed instructions (e.g., "4 tables with 4 chairs each, chairs facing tables"), but these instructions are not strictly enforced through the pipeline. Objects get lost, positions drift, and orientations change during processing.

### Solution Approach

We introduce a **strict enforcement pipeline** where:

1. **LLM_Scene_Composer**: LLM creates a detailed Semantic_Plan with explicit instructions for every object
2. **Distance_Resolver**: Converts relative positions ("0.5m in front of table") to absolute (x, y, z) coordinates
3. **Orientation_Resolver**: Converts relative orientations ("facing table") to absolute yaw angles
4. **Placement_Executor**: "Dumb executor" that places objects exactly as instructed without modifications
5. **Constraint_Validator**: Validates that the final scene matches the Semantic_Plan
6. **Pipeline_Tracker**: Tracks objects through all stages to detect losses or modifications

### Key Design Principles

- **LLM Authority**: LLM has complete control over scene composition
- **Faithful Execution**: System executes instructions without "smart" modifications
- **Explicit Over Implicit**: All positions and orientations are explicitly specified
- **Traceability**: Every object is tracked from plan to final XML
- **Validation**: Final scene is validated against original plan

## Architecture

### System Components

```
┌─────────────────────────────────────────────────────────────────┐
│                      LLM_Scene_Composer                          │
│  (Creates detailed Semantic_Plan with all placement instructions)│
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
                    ┌────────────────┐
                    │ Semantic_Plan  │
                    │  (JSON Schema) │
                    └────────┬───────┘
                             │
                             ▼
                  ┌──────────────────────┐
                  │  Schema_Validator    │
                  │ (Validates structure)│
                  └──────────┬───────────┘
                             │
                             ▼
                  ┌──────────────────────┐
                  │ Distance_Resolver    │
                  │ (Relative → Absolute)│
                  └──────────┬───────────┘
                             │
                             ▼
                  ┌──────────────────────┐
                  │ Orientation_Resolver │
                  │ (Relative → Absolute)│
                  └──────────┬───────────┘
                             │
                             ▼
                  ┌──────────────────────┐
                  │ Placement_Executor   │
                  │ ("Dumb" executor)    │
                  └──────────┬───────────┘
                             │
                             ▼
                  ┌──────────────────────┐
                  │  Scene_Assembly      │
                  │ (Creates MuJoCo XML) │
                  └──────────┬───────────┘
                             │
                             ▼
                  ┌──────────────────────┐
                  │ Constraint_Validator │
                  │ (Validates result)   │
                  └──────────────────────┘

         ┌────────────────────────────────┐
         │     Pipeline_Tracker           │
         │ (Tracks objects at each stage) │
         └────────────────────────────────┘
```

### Data Flow

1. **User Query** → LLM_Scene_Composer
2. **Semantic_Plan** (with relative coordinates) → Schema_Validator
3. **Validated Plan** → Distance_Resolver
4. **Plan with Absolute Positions** → Orientation_Resolver
5. **Plan with Absolute Orientations** → Placement_Executor
6. **Placement_Solution** → Scene_Assembly
7. **MuJoCo XML** → Constraint_Validator
8. **Validation Report** → User

At each stage, Pipeline_Tracker logs object counts and metadata.

## Components and Interfaces

### 1. LLM_Scene_Composer

**Purpose**: Generate detailed semantic plan with explicit placement instructions for every object.

**Input**:
- User query (string)
- Chosen models (list of available 3D models)
- Room size (width, length, height)

**Output**: Semantic_Plan (JSON)

**Interface**:
```python
class LLMSceneComposer:
    def generate_semantic_plan(
        self,
        user_query: str,
        chosen_models: List[Dict[str, Any]],
        room_size: Dict[str, float],
    ) -> Dict[str, Any]:
        """Generate semantic plan from user query.
        
        Returns:
            Semantic plan with schema_version, room_size, and objects list.
            Each object has: id, Model, type, size, is_static, model_loc,
            position (absolute or relative), orientation (absolute or relative).
        """
        pass
```

**Prompt Engineering**:
- System prompt defines LLM role as "scene composer and architect"
- Provides JSON schema with examples
- Emphasizes explicit instructions for every object
- Includes examples of relative positioning and orientation for various object combinations
- LLM has full control over placement logic through explicit instructions

### 2. Schema_Validator

**Purpose**: Validate that Semantic_Plan follows the defined JSON schema.

**Input**: Semantic_Plan (JSON)

**Output**: Validation result (success/failure with error messages)

**Interface**:
```python
class SchemaValidator:
    def validate_plan(self, semantic_plan: Dict[str, Any]) -> ValidationResult:
        """Validate semantic plan against schema.
        
        Checks:
        - Required fields present (schema_version, objects, room_size)
        - Each object has required fields (id, Model, type, size, etc.)
        - Position is either absolute OR relative (not both)
        - Orientation is either absolute OR relative (not both)
        - Relative references point to existing object IDs
        
        Returns:
            ValidationResult with success flag and error messages.
        """
        pass
```

**Schema Definition**:
```json
{
  "schema_version": "1.0",
  "room_size": {"width": float, "length": float, "height": float},
  "objects": [
    {
      "id": "unique_id",
      "Model": "model_name",
      "type": "furniture|small_object",
      "size": {"width": float, "length": float, "height": float},
      "is_static": boolean,
      "model_loc": "path/to/model.xml",
      "position": {
        "absolute": {"x": float, "y": float, "z": float}
        OR
        "relative": {
          "relative_to": "object_id",
          "direction": "front|back|left|right|...",
          "distance": float,
          "reference_point": "center|front_edge|..."
        }
      },
      "orientation": {
        "absolute": {"yaw_deg": float, "pitch_deg": float, "roll_deg": float}
        OR
        "relative": {
          "facing": "object_id",
          "facing_direction": "front|back|left_side|right_side",
          "facing_away": boolean
        }
      }
    }
  ]
}
```

### 3. Distance_Resolver

**Purpose**: Convert relative positions to absolute (x, y, z) coordinates.

**Input**: Semantic_Plan with relative positions

**Output**: Semantic_Plan with all positions as absolute coordinates

**Interface**:
```python
class DistanceResolver:
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
        
        Returns:
            Semantic plan with all positions as absolute coordinates.
        """
        pass
    
    def _calculate_direction_vector(
        self,
        direction: str,
        target_orientation: float,
    ) -> Tuple[float, float]:
        """Calculate unit direction vector.
        
        Directions:
        - front: along target's forward direction
        - back: opposite of forward
        - left: perpendicular left
        - right: perpendicular right
        - front_left: 45° between front and left
        - etc.
        """
        pass
    
    def _get_reference_point(
        self,
        target_object: Dict[str, Any],
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
        """
        pass
```

**Algorithm**:
1. **Dependency Resolution**: Build directed graph where edge A→B means "A's position depends on B"
2. **Cycle Detection**: Detect circular dependencies and report error
3. **Topological Sort**: Determine order to resolve positions
4. **Position Calculation**:
   - For absolute positions: keep as-is
   - For relative positions:
     - Get target object's absolute position
     - Get target object's size and orientation
     - Calculate reference point on target
     - Calculate direction vector (accounting for target's orientation)
     - Apply distance along direction
     - Set z-coordinate based on object type (floor vs table-top)

### 4. Orientation_Resolver

**Purpose**: Convert relative orientations to absolute yaw angles.

**Input**: Semantic_Plan with relative orientations

**Output**: Semantic_Plan with all orientations as absolute angles

**Interface**:
```python
class OrientationResolver:
    def resolve_orientations(
        self,
        semantic_plan: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Resolve all relative orientations to absolute yaw angles.
        
        Process:
        1. For each object with relative orientation:
           - Get target object's position and orientation
           - Calculate angle to face target based on facing_direction
           - Apply facing_away modifier if specified
           - Set absolute yaw_deg
        
        Returns:
            Semantic plan with all orientations as absolute angles.
        """
        pass
    
    def _calculate_facing_angle(
        self,
        source_pos: Tuple[float, float, float],
        target_pos: Tuple[float, float, float],
        target_orientation: float,
        facing_direction: str,
        facing_away: bool = False,
    ) -> float:
        """Calculate yaw angle to face target.
        
        facing_direction:
        - front: face the front of target
        - back: face the back of target
        - left_side: face the left side of target
        - right_side: face the right side of target
        
        facing_away: if True, face away from target instead of towards
        """
        pass
```

**Algorithm**:
1. **For each object with relative orientation**:
   - Get source object's absolute position (must be resolved first)
   - Get target object's absolute position and orientation
   - Calculate vector from source to target
   - Adjust for facing_direction (front/back/left/right of target)
   - Calculate angle using atan2
   - Apply facing_away modifier (add 180°)
   - Normalize to [0, 360) range

### 5. Placement_Executor

**Purpose**: "Dumb executor" that places objects exactly as specified without modifications.

**Input**: Semantic_Plan with all absolute positions and orientations

**Output**: Placement_Solution with placed objects

**Interface**:
```python
class PlacementExecutor:
    def execute_placement(
        self,
        semantic_plan: Dict[str, Any],
    ) -> PlacementSolution:
        """Execute placement exactly as specified in semantic plan.
        
        Rules:
        - Place ALL objects from semantic plan
        - Use exact positions from plan (no collision avoidance)
        - Use exact orientations from plan (no automatic adjustments)
        - Preserve all metadata (uuid, model_loc, size, is_static)
        - Log warnings for potential collisions but still place
        
        Returns:
            PlacementSolution with all objects placed.
        """
        pass
```

**Key Principle**: This component does NOT:
- Modify positions to avoid collisions
- Adjust orientations for aesthetics
- Apply automatic spacing
- Optimize layout
- Make any decisions

It ONLY:
- Reads positions and orientations from plan
- Places objects at those exact coordinates
- Preserves all metadata
- Logs warnings (but doesn't act on them)

### 6. Scene_Assembly

**Purpose**: Create MuJoCo XML from Placement_Solution.

**Input**: Placement_Solution with placed objects

**Output**: MuJoCo XML string

**Interface**:
```python
class SceneAssembly:
    def assemble_scene(
        self,
        placement_solution: PlacementSolution,
        room_size: Dict[str, float],
    ) -> str:
        """Assemble MuJoCo XML from placement solution.
        
        Process:
        1. Create XML root with worldbody
        2. Add floor and walls
        3. For each object in placement_solution:
           - Verify model_loc file exists
           - Create body element with position and orientation
           - Include 3D model via <include>
           - Set freejoint for dynamic objects
           - Preserve object id as name attribute
        
        Returns:
            MuJoCo XML string.
        """
        pass
```

**XML Structure**:
```xml
<mujoco model="scene">
  <worldbody>
    <light pos="0 0 3" dir="0 0 -1"/>
    <geom name="floor" type="plane" size="10 10 0.1"/>
    
    <body name="table_1" pos="0 0 0.4">
      <freejoint/>  <!-- if not is_static -->
      <include file="models/table.xml"/>
    </body>
    
    <body name="chair_1" pos="0.5 0 0.4" euler="0 0 90">
      <freejoint/>
      <include file="models/chair.xml"/>
    </body>
  </worldbody>
</mujoco>
```

### 7. Constraint_Validator

**Purpose**: Validate that final scene matches Semantic_Plan.

**Input**: 
- Original Semantic_Plan
- Final MuJoCo XML

**Output**: Validation report

**Interface**:
```python
class ConstraintValidator:
    def validate_scene(
        self,
        semantic_plan: Dict[str, Any],
        mujoco_xml: str,
    ) -> ValidationReport:
        """Validate final scene against semantic plan.
        
        Checks:
        - Object count matches
        - All planned objects present in XML
        - Position errors (distance between planned and actual)
        - Orientation errors (angular difference)
        
        Returns:
            ValidationReport with compliance score and detailed errors.
        """
        pass
```

**Validation Report**:
```python
@dataclass
class ValidationReport:
    total_objects: int
    missing_objects: List[str]
    position_errors: List[Dict[str, Any]]  # {id, planned, actual, error_m}
    orientation_errors: List[Dict[str, Any]]  # {id, planned, actual, error_deg}
    compliance_score: float  # 0.0 to 1.0
    summary: str
```

**Thresholds**:
- Position error > 0.1m: marked as mismatch
- Orientation error > 5°: marked as mismatch
- Compliance score = (correct_objects / total_objects) * 100%

### 8. Pipeline_Tracker

**Purpose**: Track objects through all pipeline stages.

**Input**: Object data at each stage

**Output**: Pipeline trace file (JSON)

**Interface**:
```python
class PipelineTracker:
    def log_stage(
        self,
        stage_name: str,
        objects: List[Dict[str, Any]],
    ) -> None:
        """Log object data at a pipeline stage."""
        pass
    
    def generate_trace(self) -> Dict[str, Any]:
        """Generate complete pipeline trace.
        
        Returns:
            Trace with stages, object counts, and full object data.
        """
        pass
    
    def detect_changes(
        self,
        stage_a: str,
        stage_b: str,
    ) -> Dict[str, Any]:
        """Detect changes between two stages.
        
        Returns:
            Dict with added, removed, and modified objects.
        """
        pass
```

**Trace Format**:
```json
{
  "pipeline_trace": {
    "timestamp": "2024-01-15T10:30:00Z",
    "stages": [
      {
        "name": "semantic_plan",
        "timestamp": "2024-01-15T10:30:01Z",
        "object_count": 20,
        "objects": [...]
      },
      {
        "name": "distance_resolver",
        "timestamp": "2024-01-15T10:30:02Z",
        "object_count": 20,
        "objects": [...]
      },
      ...
    ],
    "changes": [
      {
        "from_stage": "semantic_plan",
        "to_stage": "distance_resolver",
        "added": [],
        "removed": [],
        "modified": [{"id": "chair_1", "field": "position", "old": {...}, "new": {...}}]
      }
    ]
  }
}
```

## Data Models

### Semantic_Plan Schema

```python
@dataclass
class Position:
    """Object position (absolute or relative)."""
    absolute: Optional[Dict[str, float]] = None  # {x, y, z}
    relative: Optional[Dict[str, Any]] = None  # {relative_to, direction, distance, reference_point}

@dataclass
class Orientation:
    """Object orientation (absolute or relative)."""
    absolute: Optional[Dict[str, float]] = None  # {yaw_deg, pitch_deg, roll_deg}
    relative: Optional[Dict[str, Any]] = None  # {facing, facing_direction, facing_away}

@dataclass
class SemanticObject:
    """Object in semantic plan."""
    id: str
    Model: str
    type: str  # "furniture" or "small_object"
    size: Dict[str, float]  # {width, length, height}
    is_static: bool
    model_loc: str
    position: Position
    orientation: Orientation

@dataclass
class SemanticPlan:
    """Complete semantic plan."""
    schema_version: str
    room_size: Dict[str, float]  # {width, length, height}
    objects: List[SemanticObject]
```

### Placement_Solution Schema

```python
@dataclass
class PlacedObject:
    """Object with resolved absolute position and orientation."""
    id: str
    Model: str
    type: str
    size: Dict[str, float]
    is_static: bool
    model_loc: str
    position: Dict[str, float]  # {x, y, z} - always absolute
    orientation: Dict[str, float]  # {yaw_deg, pitch_deg, roll_deg} - always absolute
    metadata: Dict[str, Any]  # Original semantic plan data

@dataclass
class PlacementSolution:
    """Complete placement solution."""
    objects: List[PlacedObject]
    room_size: Dict[str, float]
    metadata: Dict[str, Any]
```

## Error Handling

### Error Types

1. **Schema Validation Errors**:
   - Missing required fields
   - Invalid field types
   - Circular dependencies in relative positioning
   - References to non-existent objects

2. **Resolution Errors**:
   - Cannot resolve relative position (target not found)
   - Cannot resolve relative orientation (target not found)
   - Circular dependency detected

3. **Assembly Errors**:
   - Model file not found (model_loc invalid)
   - Invalid XML structure

4. **Validation Errors**:
   - Object count mismatch
   - Missing objects in final scene
   - Position/orientation errors exceed thresholds

### Error Handling Strategy

**Fail Fast**: System should fail immediately on critical errors rather than attempting recovery.

**Critical Errors** (stop execution):
- Schema validation failure
- Circular dependency in positioning
- Missing model files

**Warnings** (log but continue):
- Potential collisions detected
- Objects outside room boundaries
- Position/orientation errors within tolerance

**Error Messages**: Should include:
- Error type and severity
- Object ID(s) involved
- Expected vs actual values
- Suggested fix

## Testing Strategy

### Unit Tests

1. **Schema_Validator Tests**:
   - Valid plans pass validation
   - Invalid plans fail with correct error messages
   - Edge cases (empty objects list, missing fields)

2. **Distance_Resolver Tests**:
   - Absolute positions unchanged
   - Relative positions correctly calculated
   - All direction types supported
   - Reference points correctly calculated
   - Dependency resolution works
   - Circular dependencies detected

3. **Orientation_Resolver Tests**:
   - Absolute orientations unchanged
   - Relative orientations correctly calculated
   - All facing_direction types supported
   - facing_away modifier works

4. **Placement_Executor Tests**:
   - All objects placed
   - Positions exactly match plan
   - Orientations exactly match plan
   - Metadata preserved
   - No modifications applied

5. **Scene_Assembly Tests**:
   - Valid MuJoCo XML generated
   - All objects included
   - Positions and orientations correct in XML
   - Model includes present

6. **Constraint_Validator Tests**:
   - Detects missing objects
   - Calculates position errors correctly
   - Calculates orientation errors correctly
   - Compliance score accurate

### Integration Tests

1. **End-to-End Pipeline**:
   - Simple scene (2 objects with relative positioning)
   - Complex scene (20+ objects with mixed absolute/relative positioning)
   - Multi-level dependencies (A depends on B, B depends on C)
   - All objects present in final XML
   - Positions within tolerance
   - Orientations within tolerance

2. **Universal Relative Placement**:
   - Any object type relative to any other object type
   - All direction types (front, back, left, right, front_left, etc.)
   - All reference points (center, edges, corners, top_surface)
   - All facing directions (front, back, left_side, right_side)
   - Facing_away modifier

3. **Pipeline Tracking**:
   - Objects tracked at all stages
   - No objects lost between stages
   - Changes correctly detected

### Property-Based Tests

Property-based testing is NOT applicable to this feature because:

1. **Infrastructure-like Nature**: The system is primarily about data transformation and XML generation, similar to IaC
2. **Deterministic Transformations**: Given a semantic plan, the output is deterministic (no randomness)
3. **External Dependencies**: Relies on LLM output and file system (model files)
4. **Configuration Validation**: Most testing is about validating structure and format

**Alternative Testing Approach**:
- **Schema validation tests**: Verify JSON schema compliance
- **Snapshot tests**: Compare generated XML against known-good examples
- **Integration tests**: End-to-end pipeline with concrete examples
- **Regression tests**: Ensure specific bug fixes remain fixed

## Implementation Plan

### Phase 1: Core Components (Week 1)

1. Define Semantic_Plan JSON schema
2. Implement Schema_Validator
3. Implement Distance_Resolver
4. Implement Orientation_Resolver
5. Unit tests for above components

### Phase 2: Execution and Assembly (Week 2)

1. Implement Placement_Executor
2. Implement Scene_Assembly
3. Implement Pipeline_Tracker
4. Unit tests for above components

### Phase 3: Validation and Integration (Week 3)

1. Implement Constraint_Validator
2. Integration tests for full pipeline
3. Universal relative placement testing with various object combinations
4. End-to-end testing

### Phase 4: LLM Integration (Week 4)

1. Update LLM prompts for Semantic_Plan generation
2. Add examples to prompts
3. Test LLM output quality
4. Iterate on prompt engineering

### Phase 5: Polish and Documentation (Week 5)

1. Error message improvements
2. Logging and debugging tools
3. User documentation
4. Performance optimization

## Dependencies

### External Libraries

- **Python 3.10+**: Core language
- **pydantic**: Schema validation
- **lxml**: XML generation and parsing
- **numpy**: Vector math for position/orientation calculations
- **pytest**: Testing framework

### Internal Dependencies

- **creator.llm.model**: LLM integration
- **creator.sim_interfaces**: MuJoCo interface
- **creator.placement.geometry**: Vector math utilities

### File System Dependencies

- Model files (model_loc paths must exist)
- Output directories for trace files and reports

## Performance Considerations

### Computational Complexity

- **Distance Resolution**: O(n) where n = number of objects (with topological sort)
- **Orientation Resolution**: O(n) where n = number of objects
- **Placement Execution**: O(n) where n = number of objects
- **Scene Assembly**: O(n) where n = number of objects
- **Validation**: O(n) where n = number of objects

**Overall**: O(n) linear complexity, suitable for scenes with 100+ objects.

### Memory Usage

- Semantic_Plan: ~1KB per object
- Placement_Solution: ~1KB per object
- Pipeline Trace: ~5KB per object (includes all stages)

**Estimate**: 100 objects = ~700KB total memory usage (negligible)

### Optimization Opportunities

1. **Lazy Resolution**: Only resolve positions/orientations when needed
2. **Caching**: Cache calculated reference points and direction vectors
3. **Parallel Processing**: Resolve independent objects in parallel
4. **Incremental Validation**: Validate during assembly rather than after

## Security Considerations

### Input Validation

- **Semantic_Plan**: Validate all numeric values are finite (no NaN, Inf)
- **File Paths**: Validate model_loc paths don't escape allowed directories
- **Object IDs**: Sanitize IDs to prevent injection attacks

### File System Access

- **Model Files**: Only allow reading from approved directories
- **Output Files**: Write trace files and reports to designated output directory
- **Path Traversal**: Prevent "../" in model_loc paths

### LLM Output

- **Schema Validation**: Always validate LLM output against schema
- **Sanitization**: Sanitize all string fields from LLM
- **Limits**: Enforce maximum object count (e.g., 1000 objects per scene)

## Monitoring and Debugging

### Logging Levels

- **DEBUG**: Detailed position/orientation calculations
- **INFO**: Pipeline stage transitions, object counts
- **WARNING**: Potential collisions, objects outside bounds
- **ERROR**: Schema validation failures, missing files

### Debug Tools

1. **Pipeline Visualizer**: HTML visualization of pipeline trace
2. **Position Debugger**: 2D plot showing object positions at each stage
3. **Validation Report Viewer**: HTML report with detailed errors

### Metrics

- **Pipeline Success Rate**: % of scenes that complete without errors
- **Compliance Score Distribution**: Histogram of validation scores
- **Object Loss Rate**: % of objects lost between stages
- **Average Position Error**: Mean position error across all objects
- **Average Orientation Error**: Mean orientation error across all objects

## Future Enhancements

### Phase 2 Features

1. **Multi-Level Dependency Resolution**: Support objects positioned relative to objects that are themselves relative
2. **Constraint Relaxation**: Automatically relax constraints when conflicts detected
3. **Collision Repair**: Optional post-processing to fix collisions while preserving intent
4. **VLM Validation**: Use vision-language model to validate scene matches user intent

### Advanced Patterns

1. **Multi-Level Dependencies**: Support objects positioned relative to objects that are themselves relative (already supported by topological sort)
2. **Constraint Relaxation**: Automatically relax constraints when conflicts detected
3. **Collision Repair**: Optional post-processing to fix collisions while preserving intent
4. **VLM Validation**: Use vision-language model to validate scene matches user intent

### Export Formats

1. **USD Export**: Export to Universal Scene Description
2. **URDF Export**: Export to Unified Robot Description Format
3. **Gazebo Export**: Export to Gazebo simulation format

## Conclusion

This design provides a robust architecture for enforcing semantic plans in the World Creator system. By treating the LLM as the authoritative composer and the system as a faithful executor, we ensure that user intent is preserved from natural language description to final 3D scene.

The key innovation is the **strict enforcement pipeline** where each component has a single, well-defined responsibility and no component makes autonomous decisions that could deviate from the LLM's plan. This architectural principle ensures traceability, debuggability, and predictability.

The system is designed to be:
- **Modular**: Each component can be tested and replaced independently
- **Traceable**: Every object is tracked through all stages
- **Validatable**: Final scene is validated against original plan
- **Extensible**: New features can be added without breaking existing functionality

Implementation will proceed in phases, with core components first, followed by integration, validation, and LLM prompt engineering. The testing strategy emphasizes schema validation, snapshot testing, and integration tests rather than property-based testing, which is not applicable to this infrastructure-like system.
