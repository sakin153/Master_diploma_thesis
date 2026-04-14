# MuJoCo Scene Placement Principle

## Goal
Build a robust scene generation pipeline:

Plan (LLM) -> Candidate poses (geometry) -> Constraint scoring/optimization -> Collision-safe placement -> Physics-aware post-pass.

This document is implementation-ready and can be used as a task spec for an AI coding agent.

## 1) System Architecture

### Stage A: Semantic planning (LLM)
Input:
- user prompt
- room polygon and structural metadata (doors, windows, passages)
- model catalog metadata (name, bbox, categories, tags)

Output JSON plan:
- required objects
- object-level constraints:
  - region preference: edge | middle
  - relative constraints: left_of, right_of, in_front_of, behind, near, far
  - orientation constraints: face_to, face_same_as
  - alignment constraints: center_aligned
  - optional hard/soft flag and weight

### Stage B: Deterministic geometric solver (core)
For floor objects:
1. Generate feasible pose candidates from room polygon.
2. Remove forbidden zones (doors/windows/passages).
3. Enumerate candidate yaw angles (0, 90, 180, 270, optionally finer).
4. Hard checks:
   - inside room polygon
   - no overlap with already placed objects
   - clearance constraints (walkable margins)
5. Soft scoring:
   - edge/middle preference
   - near/far distances
   - relative directional relations
   - face/alignment terms
6. Search strategy:
   - primary: DFS + beam pruning / branch-and-bound
   - fallback: MILP for dense scenes

For wall objects (2.5D):
- place along wall segments
- validate wall occupancy vs openings
- check 3D AABB collisions against existing objects

For small objects:
- choose receptacles (table/shelf/counter)
- sample spawn poses on support surfaces
- run contact/collision filter
- optional physics spawn test and retry

### Stage C: Physics validation (MuJoCo)
Short rollout (0.5-2.0 s):
- resolve residual penetrations
- reject unstable placements (tipping/sliding beyond threshold)
- repair failed objects iteratively

## 2) Constraint DSL (JSON)

Use this schema for LLM -> solver contract:

```json
{
  "room": {
    "polygon": [[0, 0], [6, 0], [6, 5], [0, 5]],
    "forbidden_zones": [
      {"type": "door", "polygon": [[2.4, 0], [3.2, 0], [3.2, 0.8], [2.4, 0.8]]}
    ]
  },
  "objects": [
    {
      "id": "table_1",
      "class": "table",
      "bbox": [1.2, 0.8, 0.75],
      "yaw_candidates_deg": [0, 90, 180, 270],
      "constraints": [
        {"type": "region", "value": "middle", "hard": false, "weight": 1.2}
      ]
    },
    {
      "id": "chair_1",
      "class": "chair",
      "bbox": [0.45, 0.45, 0.9],
      "constraints": [
        {"type": "near", "target": "table_1", "distance": [0.2, 0.6], "hard": false, "weight": 1.0},
        {"type": "face_to", "target": "table_1", "hard": false, "weight": 0.7}
      ]
    },
    {
      "id": "sofa_1",
      "class": "sofa",
      "bbox": [2.0, 0.9, 1.0],
      "constraints": [
        {"type": "region", "value": "edge", "hard": true},
        {"type": "in_front_of", "target": "tv_1", "hard": false, "weight": 0.8}
      ]
    }
  ]
}
```

## 3) Scoring Model

Total score for candidate placement P:

S(P) = w_region * s_region + w_dist * s_distance + w_rel * s_relative + w_face * s_face + w_align * s_align - w_penalty * penalties

Where:
- hard constraints must pass before scoring
- each soft sub-score normalized to [0, 1]
- penalties include clearance violations, near-collision margins, poor support quality

## 4) Hard Constraint Checks

At minimum:
- inside-room polygon for object footprint
- no overlap between oriented 2D footprints (SAT / polygon intersection)
- clearance from forbidden zones and walk paths
- support validity for small objects (contact area threshold)

Recommended numerics:
- epsilon for geometry: 1e-6
- min clearance margin: 0.03-0.08 m (configurable)
- collision inflation for safety: +0.01 m on XY footprint

## 5) Solver Strategy

### Default: DFS + Beam
- object ordering: largest footprint first, then high-degree constraint graph nodes
- branching: top-K candidates per object (K configurable)
- pruning:
  - infeasible hard constraints
  - optimistic upper-bound score below current best

### Alternative: MILP
Use when:
- many objects in tight spaces
- global relation consistency is hard for DFS

Variables:
- continuous center (x, y)
- binary orientation selectors
- binary non-overlap disjunctions

Objective:
- maximize weighted soft constraints

## 6) MuJoCo Validation Pass

After geometric solution:
1. Instantiate objects in MuJoCo XML.
2. Run short simulation rollout.
3. Measure:
   - max linear displacement
   - max angular displacement
   - contact penetration indicators (if available)
4. Fail and repair if thresholds exceeded.

Repair policy:
- local jitter search around failed object
- re-solve neighborhood subset
- fallback to next-best global candidate

## 7) Integration Plan (for this repository)

1. Add planner output schema and parser module.
2. Implement floor solver module:
   - candidate generation
   - hard-check geometry
   - soft scoring
   - DFS/beam search
3. Implement optional MILP module behind feature flag.
4. Add wall-object solver module.
5. Add small-object receptacle placement module.
6. Add MuJoCo post-validation and repair loop.
7. Expose config in one place (weights, thresholds, search limits).

## 8) Config Defaults

```yaml
solver:
  method: dfs_beam
  beam_width: 12
  max_candidates_per_object: 80
  yaw_candidates_deg: [0, 90, 180, 270]
  collision_inflation_xy_m: 0.01
  clearance_m: 0.05
weights:
  region: 1.2
  distance: 1.0
  relative: 0.9
  face: 0.7
  alignment: 0.6
  penalty: 2.0
physics_validation:
  enabled: true
  rollout_seconds: 1.0
  max_linear_drift_m: 0.03
  max_angular_drift_deg: 8.0
  repair_attempts: 4
```

## 9) Acceptance Criteria

A generated scene is accepted if:
- all hard constraints pass
- no object overlaps after geometric stage
- MuJoCo validation passes drift/stability thresholds
- final score >= configurable minimum

## 10) Prompt for AI Coding Agent

Use this exact prompt to delegate implementation:

"Implement a MuJoCo scene placement pipeline with three stages: (1) LLM semantic plan JSON, (2) deterministic geometric solver (DFS+beam, optional MILP), (3) MuJoCo physics validation and repair. Follow the schema and constraints in MUJOCO_SCENE_PLACEMENT_PRINCIPLE.md. Keep hard constraints strict (inside room, no overlap, forbidden zones), and optimize soft constraints (edge/middle, near/far, relative direction, face, center alignment). Add clear module boundaries, unit tests for geometry checks and scoring, and configuration-driven weights/thresholds. Ensure deterministic behavior under fixed seeds and provide a debug mode that logs candidate scoring and prune reasons."