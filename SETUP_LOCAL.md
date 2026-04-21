# Запуск без Docker (Windows + RTX 2070 Super)

## Требования

- Windows 10/11
- NVIDIA GPU (RTX 2070 Super или лучше, 8GB+ VRAM)
- CUDA 11.8 ([скачать](https://developer.nvidia.com/cuda-11-8-0-download-archive))
- [Miniconda](https://docs.conda.io/en/latest/miniconda.html)
- [Git](https://git-scm.com/)
- Ollama запущен локально: `ollama run qwen3.5:cloud`

---

## Установка (один раз)

Открой **Anaconda PowerShell Prompt** и выполни по порядку:

### 1. Создать окружение

```powershell
conda create -n embodiedgen python=3.11 -y
conda activate embodiedgen
cd C:\Users\slva_\Desktop\diploma\mesh_generation
```

### 2. PyTorch с CUDA 11.8

```powershell
pip install torch==2.4.0 torchvision==0.19.0 xformers==0.0.27.post2 --index-url https://download.pytorch.org/whl/cu118
```

### 3. Основные зависимости

```powershell
pip install -r requirements.txt --use-deprecated=legacy-resolver
pip install plyfile tables diffusers==0.30.3
```

### 4. Git-пакеты

```powershell
pip install --no-deps "utils3d@git+https://github.com/EasternJournalist/utils3d.git@9a4eb15"
pip install "clip@git+https://github.com/openai/CLIP.git"
pip install --no-deps "kolors@git+https://github.com/HochCC/Kolors.git"
pip install --no-deps "MoGe@git+https://github.com/microsoft/MoGe.git@a8c3734"
pip install gsplat==1.5.3
pip install "nvdiffrast@git+https://github.com/NVlabs/nvdiffrast.git@729261d"
```

### 5. Hunyuan3D-2 (3D-генерация)

```powershell
# Клонируем рядом с проектом
git clone https://github.com/Tencent-Hunyuan/Hunyuan3D-2
cd Hunyuan3D-2
pip install -e .
cd ..
```

> Если при установке ошибки на Windows — попробуй через WSL2 или Linux.

### 6. Git-сабмодули

```powershell
git submodule update --init --recursive
```

### 7. Установить проект

```powershell
pip install --no-deps -e .
```

---

## Запуск

### Переменные окружения (каждый раз перед запуском)

```powershell
$env:TEXT_MODEL         = "sdxl-turbo"
$env:GPT_AGENT_TYPE     = "ollama"
$env:OLLAMA_HOST        = "http://localhost:11434/v1"
$env:MODEL_NAME         = "qwen3.5:cloud"
$env:OUTPUT_ROOT        = "outputs/jobs"
$env:TORCH_HOME         = "weights/torch_cache"
$env:HF_HOME            = "weights/hf_cache"
$env:HUNYUAN3D_TEXTURE  = "1"   # "0" — отключить texture pipeline (экономия 6 GB)
```

### Вариант А — API сервер

```powershell
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

Отправить запрос:
```powershell
Invoke-RestMethod http://localhost:8000/api/generate `
  -Method POST `
  -ContentType "application/json" `
  -Body '{"prompt": "a wooden chair", "name": "chair"}'
```

Проверить статус:
```powershell
Invoke-RestMethod http://localhost:8000/api/jobs/<job_id>
```

Посмотреть логи выполнения:
```powershell
Invoke-RestMethod http://localhost:8000/api/jobs/<job_id>/logs
```

### Вариант Б — напрямую через скрипт

```powershell
python generate.py --prompt "a wooden chair" --name chair
```

Результат появится в `outputs/`.

---

## Переключение модели 3D-генерации

В файле [embodied_gen/scripts/imageto3d.py](embodied_gen/scripts/imageto3d.py) строка:

```python
IMAGE3D_MODEL = "HUNYUAN3D"   # активная модель
# IMAGE3D_MODEL = "SAM3D"     # вернуться на SAM3D (нужен сабмодуль)
# IMAGE3D_MODEL = "TRELLIS"   # нужно 20+ GB VRAM
```

---

## Веса моделей

При первом запуске автоматически скачаются:

| Модель | Размер | Куда |
|--------|--------|------|
| SDXL-Turbo (text→image) | ~7 GB | `weights/hf_cache/` |
| Hunyuan3D-2mini (shape) | ~4 GB | HuggingFace cache |
| Hunyuan3D-2 Paint (texture) | ~6 GB | HuggingFace cache |
| DINOv2 ViT-L/14 | ~1.1 GB | `weights/torch_cache/` |
| MoGe | ~1 GB | `weights/hf_cache/` |

> **Важно для РФ:** `dl.fbaipublicfiles.com` (DINOv2) может быть заблокирован.
> Скачай вручную через VPN и положи в
> `weights/torch_cache/hub/checkpoints/dinov2_vitl14_reg4_pretrain.pth`

> Если нужно отключить texture pipeline (экономия ~6 GB VRAM):
> ```powershell
> $env:HUNYUAN3D_TEXTURE = "0"
> ```
