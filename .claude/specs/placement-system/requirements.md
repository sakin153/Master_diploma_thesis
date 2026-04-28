# Requirements Document

## Introduction

The **placement system** is the core algorithmic subsystem of `world-creator` responsible for converting a list of scaled 3D objects and a set of semantic constraints (produced by the LLM in Stage 4) into a collision-free, physically plausible 3D layout inside a bounded MuJoCo room. Given the room polygon, per-object oriented bounding boxes (OBBs), and a constraint graph, the placement system selects a 6-DoF pose for every object such that (a) no two objects interpenetrate, (b) every object lies inside the room, (c) semantic constraints (`on_floor`, `against_wall`, `wall_mounted`, `near`, `far`, directional relations, surface containment) are satisfied to the maximum feasible extent, and (d) the resulting layout survives a short MuJoCo settling simulation without large displacements.

This spec defines the requirements for the placement subsystem as a coherent unit, covering input contracts, the staged solver (floor → wall → surface → repair), constraint semantics, fallback behavior when constraints conflict, observability, and integration with the surrounding pipeline (`runner.py`, `engine_refine.py`, `vlm_validator.py`).

## Requirements

### Requirement 1: Input contract and object classification

**User Story:** As the pipeline orchestrator (`runner.py`), I want to hand the placement system a well-defined bundle of room geometry, scaled object models, and a constraint graph, so that placement is decoupled from upstream LLM and model-loading stages and can be unit-tested in isolation.

#### Acceptance Criteria

1. WHEN the placement system is invoked THEN it SHALL accept a `PlacementInput` containing: room half-extents `(hx, hy, hz)`, a list of `PlacedObjectSpec` entries (each with stable `id`, category label, OBB dimensions, mass, and `is_static` flag), and a `SceneGraph` describing semantic constraints between object IDs.
2. IF any `PlacedObjectSpec.obb` has a non-positive dimension THEN the system SHALL reject the input with a descriptive error and SHALL NOT enter the solver.
3. WHEN classifying objects THEN the system SHALL assign each object exactly one of `{floor, wall_mounted, surface_small}` based on (a) explicit constraints in the scene graph (e.g. `wall_mounted`, `on_surface`) and (b) a volume threshold (default 0.06 m³) for `surface_small` fallback.
4. IF an object has both `on_floor` and `wall_mounted` constraints THEN the system SHALL prefer the explicit `wall_mounted` annotation and SHALL log the conflict to the constraint report.
5. WHEN the input contains zero objects THEN the system SHALL return an empty layout without error.

### Requirement 2: Floor placement solver

**User Story:** As a scene author, I want large objects (sofas, desks, beds, wardrobes) to be placed on the floor inside the room polygon without overlapping each other or the walls, so that the resulting MuJoCo scene is physically valid.

#### Acceptance Criteria

1. WHEN the floor solver runs THEN it SHALL place objects in descending order of footprint area (largest first).
2. WHEN evaluating a candidate pose THEN the system SHALL check 2D OBB collisions against all already-placed floor objects using SAT, with an AABB pre-filter for performance.
3. WHEN a candidate pose places any part of an object's footprint outside the room polygon THEN that candidate SHALL be rejected.
4. WHEN searching for a pose THEN the system SHALL use beam search with a configurable `beam_width` (default 12) over a jittered grid of candidate positions and a discrete set of yaw rotations.
5. IF an object cannot be placed anywhere with zero overlap THEN the system SHALL retain the lowest-overlap pose, mark the object as `placement_compromised=true` in the report, and continue placing remaining objects.
6. WHEN scoring candidates THEN the system SHALL combine, with documented weights, the terms: overlap penalty, out-of-room penalty, clearance reward, region-preference (inbound) reward, and per-constraint satisfaction term.
7. WHEN no feasible layout is found within the beam search budget THEN the system SHALL emit a structured failure object naming the offending objects and the dominant unsatisfied terms.

### Requirement 3: Wall and surface placement

**User Story:** As a scene author, I want shelves, paintings, books, and cups to attach to walls or rest on surfaces of larger objects, so that scenes look semantically correct rather than just collision-free.

#### Acceptance Criteria

1. WHEN an object is classified as `wall_mounted` THEN the system SHALL attach it to one of the four room walls with its back face flush against the wall and SHALL respect any height range specified by the constraint (defaulting to a category-based height if unspecified).
2. WHEN an object is classified as `surface_small` THEN the system SHALL place it on the top face of an explicit support object if a constraint identifies one, otherwise on the top face of the nearest semantically compatible support (e.g. cup → desk/table; book → shelf/desk).
3. WHEN placing on a surface THEN the system SHALL ensure the small object's footprint lies entirely within the support's top face polygon, and SHALL avoid collisions with other objects already placed on that surface.
4. IF no compatible support exists for a `surface_small` object THEN the system SHALL fall back to floor placement and SHALL log the demotion to the constraint report.
5. WHEN multiple wall-mounted objects target the same wall THEN the system SHALL distribute them along the wall without horizontal overlap and SHALL respect any `left_of` / `right_of` constraints between them.

### Requirement 4: Semantic constraint satisfaction and repair

**User Story:** As an LLM constraint planner, I want the placement system to honor the spatial relations I emit (`near`, `far`, `left_of`, `right_of`, `in_front_of`, `behind`, `face_to`, `center_aligned`), so that the rendered scene matches the user's intent.

#### Acceptance Criteria

1. WHEN evaluating a layout THEN the system SHALL compute, for every constraint in the scene graph, a numeric satisfaction score in `[0, 1]` and a boolean `satisfied` flag using documented thresholds.
2. WHEN the scoring pass finishes THEN the system SHALL run an iterative repair loop that perturbs poses of low-satisfaction objects (translation jitter and yaw resampling) for up to `max_repair_iters` (default 8) iterations, accepting only changes that improve the global constraint score without introducing collisions.
3. IF `near(A, B)` is requested THEN `satisfied` SHALL require `distance(A.center, B.center) ≤ near_threshold` (default 1.0 m, configurable per-constraint).
4. IF `far(A, B)` is requested THEN `satisfied` SHALL require `distance(A.center, B.center) ≥ far_threshold` (default 2.5 m).
5. IF `face_to(A, B)` is requested THEN `satisfied` SHALL require the angle between `A`'s forward axis and the vector from `A` to `B` to be `≤ 25°`.
6. WHEN a directional constraint (`left_of`, `right_of`, `in_front_of`, `behind`) is evaluated THEN it SHALL be measured in the local frame of the reference object's forward axis, not in world coordinates.
7. WHEN repair cannot satisfy a constraint THEN the system SHALL leave the highest-scoring pose in place and record the residual violation in `scene_constraint_report_latest.json`.

### Requirement 5: Physics-aware refinement

**User Story:** As a downstream MuJoCo consumer, I want the geometric layout to be lightly settled by physics so that residual sub-millimeter penetrations and floating objects are eliminated before rendering or VLM validation.

#### Acceptance Criteria

1. WHEN geometric placement completes THEN the system SHALL hand the layout to `engine_refine.py` which builds a proxy world and runs MuJoCo for a configurable settle duration (default 0.5–2.0 s).
2. WHEN reading back settled poses THEN the system SHALL reject any object whose displacement exceeds `max_settle_drift` (default 0.15 m horizontal, 0.10 m vertical) and SHALL roll that object back to its pre-settle pose with a logged warning.
3. IF an object's `is_static` flag is true THEN the settle pass SHALL freeze that object's pose (e.g. via `mocap` or zero-mass weld) so it does not drift.
4. WHEN settling completes THEN the system SHALL re-run constraint validation on the post-physics poses and SHALL append the post-physics report to `scene_constraint_report_latest.json` under a `post_physics` key.

### Requirement 6: Failure modes, fallbacks, and determinism

**User Story:** As a developer debugging a failing scene, I want the placement system to fail loudly and reproducibly rather than silently produce garbage, so that I can diagnose issues without re-running the full pipeline.

#### Acceptance Criteria

1. WHEN the solver is invoked with the same input and the same RNG seed THEN it SHALL produce byte-identical output layouts across runs.
2. IF the room is too small to contain the union of object footprints (with circulation multiplier applied) THEN the system SHALL raise `RoomTooSmallError` carrying the required vs. available area and SHALL NOT silently overlap objects.
3. WHEN any object is `placement_compromised` THEN the system SHALL surface this fact in the return value so `runner.py` can decide whether to retry with relaxed constraints or larger room.
4. WHEN the solver exceeds its global wall-clock budget (default 60 s) THEN it SHALL return the best layout found so far flagged as `partial=true`, rather than running unboundedly.

### Requirement 7: Observability and reporting

**User Story:** As a maintainer, I want every placement run to leave behind a structured report describing what was placed, what was compromised, and which constraints were violated, so that regressions are visible without re-instrumenting the code.

#### Acceptance Criteria

1. WHEN a placement run finishes THEN the system SHALL write `scene_constraint_report_latest.json` containing per-object pose, per-constraint satisfaction score, repair iteration count, and any compromise flags.
2. WHEN a placement run finishes THEN the system SHALL write/refresh `scene_graph_latest.json` reflecting the final scene graph including any constraints the solver added or relaxed.
3. WHEN logging at INFO level THEN the system SHALL emit one line per major stage (`floor`, `wall`, `surface`, `repair`, `physics`) with object count, elapsed time, and a one-line outcome summary.
4. IF DEBUG logging is enabled THEN the system SHALL additionally emit per-candidate scores for the top-K candidates considered during beam search, gated behind a verbosity flag to avoid log spam by default.

### Requirement 8: Integration with the broader pipeline

**User Story:** As the orchestrator (`runner.py`) and as the optional VLM validator (`vlm_validator.py`), I want a stable Python API for the placement subsystem so that I can call it, inspect its output, and request a re-solve with modified constraints without reaching into private internals.

#### Acceptance Criteria

1. WHEN `runner.py` calls `solve_placement(input)` THEN it SHALL receive a `PlacementResult` containing the final pose for every object, the constraint report, the compromise flags, and a serializable `SceneGraph`.
2. WHEN `vlm_validator.py` requests a re-solve with modified constraints THEN the placement system SHALL accept a `previous_result` argument and SHALL warm-start from the prior layout to reduce churn on objects whose constraints did not change.
3. WHEN the placement system mutates the scene graph (e.g. adds an inferred `on_surface` edge) THEN those mutations SHALL be returned in the result rather than written back into the caller's input object.
4. WHEN imported THEN the placement subsystem SHALL NOT trigger Ollama calls, model downloads, or MuJoCo compilation as a side-effect of import — only when its public solve function is called.