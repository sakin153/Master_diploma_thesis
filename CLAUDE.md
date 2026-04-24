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
| `HF_HOME` | — | HuggingFace model cache |
| `TORCH_HOME` | — | Torch model cache (DINOv2 goes here) |

> **Critical:** `TEXT_MODEL` is read at module import time in `asset_gen/scripts/textto3d.py`. Set it via `os.environ["TEXT_MODEL"]` **before** importing `text_to_3d`, or export it before the process starts.

## Architecture

### `api/` — REST API (thesis addition)
- `main.py` — FastAPI app with endpoints: `POST /api/generate`, `GET /api/jobs/{id}`, `GET /api/jobs/{id}/logs`, download endpoints per format (obj/glb/urdf/mjcf/zip)
- `job_manager.py` — `JobManager` singleton: single-GPU sequential queue backed by a background thread. Jobs persist to `outputs/jobs/{job_id}/job.json` and survive restarts (in-flight jobs are marked failed on reload).
- `models.py` — Pydantic models for requests/responses. `ModelChoice` enum is the canonical list of valid text-to-image model names.

### `asset_gen/` — Core generation library
- `scripts/textto3d.py` — Main pipeline entry point (`text_to_3d()` / `GenerateItem`). Batch-optimises VRAM: loads text2img for **all** items, releases it, then loads Hunyuan3D for **all** items.
- `scripts/imageto3d.py` — Image → 3D step. Contains two lazy pipeline globals: `_PIPELINE` (shape) and `_TEXTURE_PIPELINE` (texture). Shape is always released before texture loads — they never coexist in VRAM. Switch 3D backend by editing `IMAGE3D_MODEL` constant (line ~53): `"HUNYUAN3D"` (default, 6-10 GB), `"SAM3D"` (submodule, 10-14 GB), `"TRELLIS"` (20+ GB).
- `models/hunyuan3d.py` — Two classes: `Hunyuan3DInference` (shape, mini-turbo 0.6B) and `Hunyuan3DTexture` (paint-turbo 1.3B). Both require that the C++ extensions in `Hunyuan3D-2/hy3dgen/texgen/` are compiled.
- `models/image_comm_model.py` — Builds the HuggingFace text-to-image pipeline selected by `TEXT_MODEL`.
- `data/asset_converter.py` — `cvt_asset_gen_asset_to_anysim()` converts URDF → MJCF (or other simulator formats via `AssetType` enum).
- `validators/quality_checkers.py` — LLM-based quality gates (semantic consistency, segmentation, text-gen alignment). All use `GPT_CLIENT` singleton.
- `utils/gpt_clients.py` — LLM client wrapping OpenAI-compatible APIs. Config is read from `asset_gen/utils/gpt_config.yaml`. Supports `ollama`, `qwen2.5-vl`, `gpt-4o`.
- `utils/vram_utils.py` — `free_vram()` / `log_vram()` helpers called around model load/unload.

### `generate.py` — Standalone CLI
Single-object wrapper around `text_to_3d()` + `cvt_asset_gen_asset_to_anysim()`. Output layout: `outputs/generated/asset3d/{name}/result/mesh/*.obj|glb`, `*.urdf`, `mjcf/*.xml`.

### `Hunyuan3D-2/` — Git submodule
Tencent's 3D generation model. Installed as a separate editable package (`hy3dgen`). Required for `IMAGE3D_MODEL = "HUNYUAN3D"`.

## LLM config (`asset_gen/utils/gpt_config.yaml`)

This file is **not** auto-generated — edit it directly to switch backends:

```yaml
agent_type: "ollama"   # or "qwen2.5-vl" or "gpt-4o"

ollama:
  endpoint: http://localhost:11434/v1
  api_key: ollama
  model_name: qwen2.5vl:7b
```

In Docker, Ollama runs on the host and is accessed via `http://host.docker.internal:11434/v1`. The file is bind-mounted so no image rebuild is needed after editing it.

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
