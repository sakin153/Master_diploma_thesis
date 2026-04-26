# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this project is

**EmbodiedGen** — a generative engine for creating 3D assets ready for embodied-AI simulators. The thesis work adds a custom REST API (`api/`) on top of the upstream library to serve text/image → 3D generation as an async job queue.

Full pipeline: text prompt → text-to-image (SD/SDXL/Kolors) → background removal → 3D mesh (Hunyuan3D-2) → URDF → MuJoCo MJCF.

## Environment setup

```bash
conda create -n embodiedgen python=3.10.13 -y
conda activate embodiedgen
pip install torch==2.4.0 torchvision==0.19.0 xformers==0.0.27.post2 \
    --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt --use-deprecated=legacy-resolver
pip install plyfile tables diffusers==0.30.3
# Git deps
pip install --no-deps "utils3d @ git+https://github.com/EasternJournalist/utils3d.git@9a4eb15"
pip install "clip @ git+https://github.com/openai/CLIP.git"
pip install --no-deps "kolors @ git+https://github.com/HochCC/Kolors.git"
pip install --no-deps "MoGe @ git+https://github.com/microsoft/MoGe.git@a8c3734"
pip install gsplat==1.5.3
pip install "nvdiffrast @ git+https://github.com/NVlabs/nvdiffrast.git@729261d"
# Hunyuan3D-2 (submodule)
cd Hunyuan3D-2 && pip install -e . && cd ..
git submodule update --init --recursive
pip install --no-deps -e .
```

For dev tools: `pip install -e .[dev] && pre-commit install`

## Common commands

```bash
# Run API server
uvicorn api.main:app --host 0.0.0.0 --port 8000 --workers 1

# Direct generation (CLI)
python generate.py "a wooden chair" --name chair
python generate.py "iron dumbbell" --model sdxl-turbo --skip_mjcf

# Tests (unit only by default, see pytest.ini)
python -m pytest
python -m pytest tests/test_unit/test_gpt_client.py  # single file
python -m pytest tests/test_examples/              # integration tests

# Lint / format
bash scripts/autoformat.sh ./

# Docker
docker build -f Dockerfile.base -t embodiedgen-base:latest .  # once
docker compose up --build
HUNYUAN3D_TEXTURE=0 docker compose up --build  # for 8 GB GPU
```

## Key environment variables

| Variable | Default | Purpose |
|---|---|---|
| `TEXT_MODEL` | `sd15` | Text-to-image model: `sd15`, `sdxl-turbo`, `kolors`, `sd35`, `flux` |
| `HUNYUAN3D_TEXTURE` | `0` | `1` = apply Hunyuan3D-Paint-Turbo texture after shape gen |
| `OUTPUT_ROOT` | `outputs/jobs` | Where job outputs are saved |
| `SKIP_OPEN3D_RENDER` | — | Set to `1` in headless/Docker; switches to matplotlib renderer (Open3D segfaults without EGL) |
| `HF_HOME` | — | HuggingFace model cache |
| `TORCH_HOME` | — | Torch model cache (DINOv2 goes here) |

> **Critical:** `TEXT_MODEL` is read at module import time in `asset_gen/scripts/textto3d.py`. Set it via `os.environ["TEXT_MODEL"]` **before** importing `text_to_3d`, or export it before the process starts.

## Architecture

### `api/` — REST API (thesis addition)
- `main.py` — FastAPI app with endpoints: `POST /api/generate`, `GET /api/jobs/{id}`, `GET /api/jobs/{id}/logs`, download endpoints per format (obj/glb/urdf/mjcf/zip)
- `job_manager.py` — `JobManager` singleton: single-GPU sequential queue backed by a background thread. Jobs persist to `outputs/jobs/{job_id}/job.json` and survive restarts (in-flight jobs are marked failed on reload). During execution `_TeeStream` patches `sys.stdout/stderr` and a `logging.Handler` are installed so all output (including `print()` from deep libs) lands in `job.logs`.
- `models.py` — Pydantic models for requests/responses. `ModelChoice` enum is the canonical list of valid text-to-image model names.

### `asset_gen/` — Core generation library
- `scripts/textto3d.py` — Main pipeline entry point (`text_to_3d()` / `GenerateItem`). **Three-phase VRAM strategy**: Phase 1 — load text2img, generate all images, release. Phase 2 — load Hunyuan3D shape, generate all meshes, release. Phase 3 — (if `enable_texture`) load paint model, texture all meshes, release. Models never coexist in VRAM.
- `scripts/imageto3d.py` — Image → 3D step. Contains two lazy pipeline globals: `_PIPELINE` (shape) and `_TEXTURE_PIPELINE` (texture). Hardcodes two rotation matrices (`rot_matrix`, `mesh_add_rot`) to align Hunyuan3D's coordinate system to MuJoCo — these must be recalibrated if switching 3D backends.
- `models/model_image_runtime.py` — Unified text-to-image pipeline factory. `PIPELINE_REGISTRY` dict maps model name → `(LoaderClass, RunnerClass)`. **Must be kept in sync with `ModelChoice` enum in `api/models.py`.**
- `models/hunyuan3d.py` — Two classes: `Hunyuan3DInference` (shape, mini-turbo 0.6B, ~4 GB VRAM) and `Hunyuan3DTexture` (paint-turbo 1.3B, ~6 GB VRAM). They must never be loaded simultaneously. Requires compiled C++ extensions in `Hunyuan3D-2/hy3dgen/texgen/`.
- `utils/inference.py` — `image3d_model_infer()` has a hard `isinstance(pipe, Hunyuan3DInference)` check — update this first when switching 3D backends.
- `data/asset_converter.py` — `cvt_asset_gen_asset_to_anysim()` converts URDF → MJCF (or other simulator formats via `AssetType` enum).
- `validators/quality_checkers.py` — LLM-based quality gates (semantic consistency, segmentation, text-gen alignment). All use `GPT_CLIENT` singleton.
- `utils/gpt_clients.py` — LLM client wrapping OpenAI-compatible APIs. Config is read from `asset_gen/utils/gpt_config.yaml`. Supports `ollama`, `qwen2.5-vl`, `gpt-4o`.
- `utils/vram_utils.py` — `free_vram()` / `log_vram()` helpers called around every model load/unload.

### `generate.py` — Standalone CLI
Single-object wrapper around `text_to_3d()` + `cvt_asset_gen_asset_to_anysim()`. Sets `TEXT_MODEL` env var before importing `textto3d` to satisfy the module-level read.

### `from_zero/` — Reference / scratch
Prototype files (mirrors of `asset_gen/` scripts) written from scratch during thesis development. Not imported by the main pipeline; kept as reference.

### `Hunyuan3D-2/` — Git submodule
Tencent's 3D generation model. Installed as a separate editable package (`hy3dgen`). `hunyuan3d.py` adds its directory to `sys.path` at import time. Required for `IMAGE3D_MODEL = "HUNYUAN3D"`.

## LLM config (`asset_gen/utils/gpt_config.yaml`)

This file is **bind-mounted in Docker** — edit it directly without rebuilding the image. Switch backends:

```yaml
agent_type: "ollama"   # or "qwen2.5-vl" or "gpt-4o"

ollama:
  endpoint: http://localhost:11434/v1
  api_key: ollama
  model_name: qwen2.5vl:7b
```

In Docker, Ollama runs on the host and is accessed via `http://host.docker.internal:11434/v1`.

## Switching the 3D backend

`CLAUDE.md` notes three backends by VRAM cost:
- `"HUNYUAN3D"` (default, ~4–6 GB split across shape/texture)
- `"SAM3D"` (submodule, 10–14 GB)
- `"TRELLIS"` (20+ GB) — exceeds RTX 2070 Super capacity

To switch backends, change `IMAGE3D_MODEL` constant in `imageto3d.py` (~line 53) **and** update:
1. `utils/inference.py` — remove `isinstance(pipe, Hunyuan3DInference)` guard
2. `imageto3d.py` — recalibrate `rot_matrix`/`mesh_add_rot` for the new model's coordinate system
3. The output dict from the new model must include `"trimesh": [trimesh.Trimesh]` for downstream mesh export

## Output structure

```
outputs/jobs/{job_id}/
├── job.json          ← persisted JobStatus (Pydantic)
├── request.json      ← original GenerateRequest
└── {name}/result/
    ├── mesh/{name}.obj, .glb, .mtl
    ├── {name}.urdf
    └── mjcf/{name}.xml
```
