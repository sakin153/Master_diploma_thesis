# Object Placement Orientation and Physics Fix Design

## Overview

This design addresses two critical bugs in the Universal Placement System that affect scene realism:

1. **Orientation Bug**: Objects with `region:edge` constraints are placed with yaw=0° instead of facing the room center, causing furniture like sofas to face walls rather than the interior space.

2. **Physics Bug**: Small objects (volume < 0.1 m³) lose their `is_static=false` flag during the data flow from semantic planning through scene graph to layout solving, causing them to float in mid-air instead of responding to physics.

The root cause is a **data loss problem** in the pipeline: object size and physics metadata are not preserved when converting from semantic_plan → scene_graph → layout_solver → floor_solver. Without size information, the floor_solver cannot calculate proper orientations or determine physics properties.

The fix strategy involves preserving size and is_static metadata through the entire pipeline and ensuring the orientation calculation logic in floor_solver is properly invoked.

## Glossary

- **Bug_Condition (C)**: The condition that triggers the bugs - when size/physics metadata is lost during pipeline data flow
- **Property (P)**: The desired behavior - objects should have correct orientations and physics properties based on their size and constraints
- **Preservation**: Existing placement logic for non-edge objects and large furniture that must remain unchanged
- **scene_graph**: Data structure in `universal_system.py` that represents object relationships and metadata
- **semantic_plan**: LLM-generated plan with object specifications including size and is_static flags
- **objects_to_place**: List passed to floor_solver from layout_solver containing object metadata
- **_get_yaw_candidates_for_position()**: Function in `floor_solver.py` that calculates face-center yaw for edge objects
- **_compute_face_center_yaw()**: Function that computes the yaw angle to face room center (0, 0)
- **region:edge constraint**: Constraint indicating object should be placed near room walls/edges
- **is_static flag**: Boolean property determining if object responds to physics (false = dynamic)

## Bug Details

### Bug Condition

The bugs manifest when the Universal Placement System processes objects through its pipeline stages. The system loses critical metadata (size, is_static) when converting between data structures, resulting in:

1. **Orientation Bug**: Objects placed with yaw=0° because floor_solver lacks size information needed to invoke `_get_yaw_candidates_for_position()`
2. **Physics Bug**: Small objects marked as static because is_static flag is not preserved through scene_graph

**Formal Specification:**
```
FUNCTION isBugCondition(input)
  INPUT: input of type PlacementPipelineData
  OUTPUT: boolean
  
  RETURN (input.semantic_plan.objects[i].size EXISTS)
         AND (input.scene_graph.nodes[i].metadata.size NOT EXISTS)
         OR (input.semantic_plan.objects[i].is_static EXISTS)
         AND (input.objects_to_place[i].is_static NOT EXISTS)
         OR (input.objects_to_place[i].size NOT EXISTS)
END FUNCTION
```

**Data Flow Problem:**
```
runner.py: models_for_universal (has size) 
  → UniversalSystem.generate_scene(chosen_models)
    → _generate_semantic_plan() (creates semantic_plan with size in model_sizes dict)
      → _build_scene_graph(parsed_plan) (metadata from parsed_plan)
        → layout_solver._solve_with_dfs(scene_graph)
          → objects_to_place = [dict(node.metadata)] ← SIZE LOST HERE!
            → solve_floor_placements(objects_to_place) ← No size info!
```

### Examples

**Orientation Bug:**
- **Input**: Living room scene ("гостиная") with sofas having `region:edge` constraint
- **Expected**: Sofas placed near walls with yaw angles facing center (e.g., yaw=180° for sofa at y=-2.0)
- **Actual**: All objects placed with yaw=0° (from logs: `[placed] furniture → x=-2.00 y=-0.75 yaw=0°`)
- **Cause**: `_get_yaw_candidates_for_position()` exists but is never called because size info is missing

**Physics Bug:**
- **Input**: Scene with apples (volume ~0.001 m³) in boxes on table
- **Expected**: Apples marked as `is_static=false` to enable physics simulation
- **Actual**: Apples marked as `is_static=true`, causing them to float in mid-air
- **Cause**: `_generate_semantic_plan()` correctly sets `is_static=False` but this is lost in scene_graph conversion

**Edge Case:**
- **Input**: Large furniture (volume ≥ 0.1 m³) without region:edge constraint
- **Expected**: Static object with default yaw candidates (0°, 90°, 180°, 270°)
- **Actual**: Should continue working correctly (preservation requirement)

## Expected Behavior

### Preservation Requirements

**Unchanged Behaviors:**
- Objects without `region:edge` constraint must continue to use default yaw candidates
- Large objects (volume ≥ 0.1 m³) must continue to be marked as static
- Existing constraint processing (near, on, inside, beside) must remain unchanged
- Scene generation pipeline stages must continue in the same order
- MuJoCo XML export format must remain unchanged
- Gradient-based overlap resolution must continue to work
- DFS + beam search placement algorithm must remain unchanged

**Scope:**
All inputs that do NOT involve edge-placed objects or small dynamic objects should be completely unaffected by this fix. This includes:
- Interior-placed furniture with default orientations
- Large static furniture (tables, chairs, sofas without edge constraints)
- Constraint resolution for non-edge constraints
- Room geometry calculations and boundary handling
- Collision detection and overlap scoring

## Hypothesized Root Cause

Based on the bug description and code analysis, the root causes are:

1. **Missing Size in Scene Graph**: The `_build_scene_graph()` function in `universal_system.py` does not copy size information from semantic_plan objects into scene_graph node metadata
   - semantic_plan has size in model_sizes dict
   - scene_graph nodes are created from parsed_plan metadata
   - Size is not included in node.metadata

2. **Missing Size in Layout Solver**: The `_solve_with_dfs()` function in `layout_solver.py` creates objects_to_place from scene_graph node.metadata without size
   - Line: `obj_dict = dict(node.metadata)`
   - This dict does not contain size field
   - floor_solver receives objects without size information

3. **Missing is_static in Scene Graph**: The is_static flag set in `_generate_semantic_plan()` is not preserved in scene_graph node metadata
   - semantic_plan correctly sets is_static based on volume
   - scene_graph construction does not copy this flag
   - Layout solver and floor_solver never receive is_static information

4. **Orientation Logic Not Triggered**: Without size information, floor_solver cannot determine if an object should use face-center yaw
   - `_get_yaw_candidates_for_position()` exists and works correctly
   - But it's never called because the code path requires size metadata
   - Default yaw=0° is used instead

## Correctness Properties

Property 1: Bug Condition - Size and Physics Metadata Preservation

_For any_ object in the placement pipeline where size and is_static are defined in semantic_plan, the fixed system SHALL preserve this metadata through scene_graph construction and layout solving, ensuring floor_solver receives complete object information including size dimensions and physics properties.

**Validates: Requirements 2.1, 2.2, 2.3, 2.4, 2.5, 2.6**

Property 2: Preservation - Non-Edge Object Behavior

_For any_ object that does NOT have a region:edge constraint or has volume ≥ 0.1 m³, the fixed system SHALL produce exactly the same placement behavior as the original system, preserving default yaw candidates for non-edge objects and static physics for large furniture.

**Validates: Requirements 3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 3.7, 3.8**

## Fix Implementation

### Changes Required

Assuming our root cause analysis is correct:

**File**: `creator/placement/universal_system.py`

**Function**: `_generate_semantic_plan()` and `_build_scene_graph()`

**Specific Changes**:
1. **Preserve Size in Scene Graph**: Modify `_build_scene_graph()` to include size in node.metadata
   - After creating scene_graph nodes, look up size from model_sizes dict
   - Add size to node.metadata: `node.metadata["size"] = model_sizes.get(model_name, [1.0, 1.0, 1.0])`
   - Ensure size is available for layout_solver

2. **Preserve is_static in Scene Graph**: Ensure is_static flag flows from semantic_plan to scene_graph
   - When creating nodes, copy is_static from semantic_plan objects
   - Add to node.metadata: `node.metadata["is_static"] = obj.get("is_static", True)`
   - Verify flag is preserved through pipeline

**File**: `creator/placement/layout_solver.py`

**Function**: `_solve_with_dfs()`

**Specific Changes**:
3. **Include Size in objects_to_place**: Ensure size is copied from node.metadata to objects_to_place
   - Verify that `obj_dict = dict(node.metadata)` includes size field
   - If size is missing, log warning and use default [1.0, 1.0, 1.0]
   - Pass complete metadata to floor_solver

4. **Include is_static in objects_to_place**: Ensure is_static flag is passed to floor_solver
   - Copy is_static from node.metadata to obj_dict
   - Ensure floor_solver receives physics information

**File**: `creator/placement/floor_solver.py`

**Function**: `solve_floor_placements()`

**Specific Changes**:
5. **Verify Orientation Logic Invocation**: Ensure `_get_yaw_candidates_for_position()` is called when size is available
   - Check that size information is present in objects_to_place
   - Verify edge objects trigger face-center yaw calculation
   - Log yaw candidates for debugging

## Testing Strategy

### Validation Approach

The testing strategy follows a two-phase approach: first, surface counterexamples that demonstrate the bugs on unfixed code, then verify the fix works correctly and preserves existing behavior.

### Exploratory Bug Condition Checking

**Goal**: Surface counterexamples that demonstrate the bugs BEFORE implementing the fix. Confirm or refute the root cause analysis. If we refute, we will need to re-hypothesize.

**Test Plan**: Write tests that trace data flow through the pipeline and assert that size and is_static metadata are preserved at each stage. Run these tests on the UNFIXED code to observe where data is lost.

**Test Cases**:
1. **Metadata Loss Test**: Generate semantic_plan with size, trace through scene_graph, verify size is missing in objects_to_place (will fail on unfixed code)
2. **Orientation Test**: Generate living room with edge sofas, verify all objects have yaw=0° in logs (will fail on unfixed code)
3. **Physics Test**: Generate scene with small objects, verify they are marked as static in output XML (will fail on unfixed code)
4. **Pipeline Trace Test**: Add logging at each pipeline stage to show where size/is_static are lost (diagnostic test)

**Expected Counterexamples**:
- scene_graph nodes missing size field in metadata
- objects_to_place list missing size and is_static fields
- floor_solver logs showing yaw=0° for all edge objects
- MuJoCo XML showing is_static="true" for small objects
- Possible causes: metadata not copied in _build_scene_graph(), dict(node.metadata) missing fields

### Fix Checking

**Goal**: Verify that for all inputs where the bug condition holds, the fixed function produces the expected behavior.

**Pseudocode:**
```
FOR ALL input WHERE isBugCondition(input) DO
  result := generate_scene_fixed(input)
  ASSERT result.scene_graph.nodes[i].metadata.size EXISTS
  ASSERT result.objects_to_place[i].size EXISTS
  ASSERT result.objects_to_place[i].is_static EXISTS
  ASSERT result.placed_objects[i].yaw_deg != 0 FOR edge objects
  ASSERT result.placed_objects[i].is_static == False FOR small objects
END FOR
```

### Preservation Checking

**Goal**: Verify that for all inputs where the bug condition does NOT hold, the fixed function produces the same result as the original function.

**Pseudocode:**
```
FOR ALL input WHERE NOT isBugCondition(input) DO
  ASSERT generate_scene_original(input).placed_objects = generate_scene_fixed(input).placed_objects
END FOR
```

**Testing Approach**: Property-based testing is recommended for preservation checking because:
- It generates many test cases automatically across the input domain
- It catches edge cases that manual unit tests might miss
- It provides strong guarantees that behavior is unchanged for all non-buggy inputs

**Test Plan**: Observe behavior on UNFIXED code first for non-edge objects and large furniture, then write property-based tests capturing that behavior.

**Test Cases**:
1. **Non-Edge Object Preservation**: Generate scenes with objects without region:edge constraint, verify yaw candidates remain [0°, 90°, 180°, 270°]
2. **Large Furniture Preservation**: Generate scenes with large furniture (volume ≥ 0.1 m³), verify they remain static
3. **Constraint Preservation**: Generate scenes with various constraints (near, on, beside), verify constraint resolution unchanged
4. **Export Format Preservation**: Generate scenes and export to MuJoCo XML, verify format structure unchanged

### Unit Tests

- Test `_build_scene_graph()` preserves size from model_sizes dict to node.metadata
- Test `_build_scene_graph()` preserves is_static from semantic_plan to node.metadata
- Test `_solve_with_dfs()` includes size in objects_to_place dict
- Test `_solve_with_dfs()` includes is_static in objects_to_place dict
- Test `_get_yaw_candidates_for_position()` returns face-center yaw for edge objects near walls
- Test `_compute_face_center_yaw()` calculates correct angles for various positions
- Test edge cases: missing size (use default), missing is_static (use default true)

### Property-Based Tests

- Generate random semantic_plans with varying object sizes and verify size is preserved through pipeline
- Generate random object volumes and verify is_static is correctly set based on 0.1 m³ threshold
- Generate random room configurations with edge objects and verify face-center yaw is calculated
- Generate random constraint combinations and verify preservation of non-edge placement logic
- Test that all large objects (volume ≥ 0.1 m³) remain static across many scenarios

### Integration Tests

- Test full pipeline: "гостиная" scene → verify sofas have non-zero yaw facing center
- Test full pipeline: "Стол и две коробки на нем в каждой коробке по 3 яблока" → verify apples are dynamic
- Test saved scene XML contains correct yaw values for edge objects
- Test saved scene XML contains is_static="false" for small objects
- Test generation logs show face-center yaw candidates being used
- Test physics simulation: small objects fall when placed in air, large objects remain fixed
