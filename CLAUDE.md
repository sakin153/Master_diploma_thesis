# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Project Does

**world-creator** generates physically-valid 3D MuJoCo scenes from natural language queries (e.g., `"sofa with book"`, `"classroom with 10 desks"`). It uses an LLM (Deepseek via local Ollama) to plan scenes and a geometric solver to place objects without collisions, then validates/settles placements with a short MuJoCo physics simulation.

## Running the Project

Edit `main.py` to set your query, then run:

```bash
python main.py
```

Key variables at the top of `main.py`:
- `QUERY` — natural language scene description
- `SIMULATOR` — currently only `"mujoco"` is supported
- `CACHE_DIR` — defaults to `/var/tmp/ciare/` if `None`

The pipeline writes `scene_latest.xml` (MuJoCo world) under `$CACHE_DIR/worlds/`.

**VLM validation** (optional, off by default):
```python
generate_world(simulator="mujoco", query="...", vlm_validation=True, max_vlm_iters=2)
```

## Prerequisites

- **Ollama** must be running at `http://localhost:11434` with the `deepseek-v3.1:671b-cloud` model loaded
- Python 3.8.1–3.10
- `obj2mjcf` binary (for converting OBJ/GLB → MJCF)

```bash
pip install -r requirements-main.txt
# or: poetry install
```

## Linting / Formatting

```bash
black creator/ main.py
isort creator/ main.py
pytest
```

No test suite exists yet — `pytest` will find nothing to run.

## Architecture: 7-Stage Pipeline

All stages are orchestrated in `creator/runner.py:generate_world()`:

| Stage | What happens | Key file(s) |
|-------|-------------|-------------|
| 0 — Prompt expansion | Short query → rich `SceneSpec` (room type, style, estimated dims) | `creator/scene/prompt_expander.py` |
| 1 — Object extraction | LLM picks object names from Objaverse catalog | `creator/runner.py`, `creator/contexts_prompts/objects.py` |
| 2 — Model loading & scale | Download GLB/OBJ, compute OBBs, normalize to real-world scale | `creator/sim_interfaces/mujoco.py` |
| 3 — Room sizing | Derive room half-size from object footprints × circulation multiplier | `creator/scene/room_planner.py` |
| 4 — Semantic plan | LLM emits constraint JSON (on_floor, against_wall, near, etc.) + scene graph | `creator/placement/plan.py`, `creator/contexts_prompts/constraints.py` |
| 5 — Layout solving | Beam-search floor placement + wall placement + surface placement + constraint repair | `creator/placement/` |
| 6 — Physics refinement | Proxy MuJoCo settle (0.5–2.0 s) resolves residual penetrations | `creator/postprocess/engine_refine.py` |
| 7 — VLM validation | Optional: vision-LLM checks rendered layout, iterative repair | `creator/scene/vlm_validator.py` |

## Key Module Map

```
creator/
  runner.py               # Top-level pipeline (generate_world)
  llm/model.py            # Ollama wrapper (prompt_model) with SQLite cache
  model_databases/
    objaverse.py          # Loads Objaverse catalog; caches to /var/tmp/ciare/
  scene/
    prompt_expander.py    # Stage 0: query → SceneSpec
    room_planner.py       # Stage 3: room half-size from footprints
    scene_graph.py        # SceneGraph data structure + spatial relation types
    vlm_validator.py      # Stage 7: VLM repair loop
  placement/
    plan.py               # Stage 4: build_semantic_plan() — calls LLM for constraints
    floor_solver.py       # Stage 5a: DFS + beam search, OBB/SAT collision
    wall_solver.py        # Stage 5b: wall-mounted objects (shelves, paintings)
    small_objects.py      # Stage 5c: surface placement (book on shelf, cup on desk)
    arranger.py           # Grid/row/cluster layouts for repeated objects
    geometry.py           # OBB, SAT, AABB primitives
    constraint_validation.py  # Evaluates + iteratively repairs constraint violations
    physics.py            # Post-physics semantic constraint validation
  sim_interfaces/
    mujoco.py             # MujocoSimInterface: model loading, scale normalization,
                          #   XML assembly, per-category height/size tables
  postprocess/
    engine_refine.py      # Builds proxy world, runs MuJoCo, reads back settled poses
  contexts_prompts/       # LLM prompt templates (objects, constraints, model selection)
  xml/worlds.py           # MuJoCo XML helpers
  utils/cache.py          # Cache directory management
```

## Placement Solver Details

The solver in `creator/placement/` is the core algorithmic contribution. Key design decisions documented in `MUJOCO_SCENE_PLACEMENT_PRINCIPLE.md`:

- **Object ordering**: largest footprint first (greedy, improves feasibility)
- **Candidates**: regular grid over room polygon, jittered, plus near-constraint-target samples
- **Beam search**: keeps top `beam_width` (default 12) partial layouts per step
- **Collision check**: SAT on 2D OBB projections; fast AABB pre-filter
- **Multi-term loss**: inbound (region preference) + overlap + clearance + constraint satisfaction
- **Constraint types**: `on_floor`, `against_wall`, `wall_mounted`, `near`, `far`, `left_of`, `right_of`, `in_front_of`, `behind`, `face_to`, `center_aligned`
- **Small-object threshold**: volume < 0.06 m³ → placed on surfaces, not floor

## LLM Integration

`creator/llm/model.py` wraps Ollama. Responses are cached in SQLite at `$CACHE_DIR/.ollama_cache.sqlite3` keyed by `(prompt, context, model)` — **delete the cache file to force fresh LLM calls**.

Model name used throughout: `"deepseek-v3.1:671b-cloud"` (set as `chosen_model` in `runner.py`).

## Scale Normalization

`MujocoSimInterface.normalize_models_to_realistic_scale()` in `creator/sim_interfaces/mujoco.py` uses hard-coded per-category height targets (e.g., `cup=0.11m`, `desk=0.75m`, `wardrobe=2.0m`) and max horizontal extents. Check and extend these tables when adding new object categories.

## Output Files

| File | Contents |
|------|----------|
| `$CACHE_DIR/worlds/scene_latest.xml` | Generated MuJoCo world |
| `scene_graph_latest.json` | Constraint graph from Stage 4 |
| `scene_constraint_report_latest.json` | Constraint validation report |
| `$CACHE_DIR/worlds/world_db.json` | TinyDB log of all generated worlds |
| `MUJOCO_LOG.TXT` | MuJoCo physics engine output |
