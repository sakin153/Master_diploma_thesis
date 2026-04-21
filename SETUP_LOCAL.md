# Запуск EmbodiedGen без Docker

Пошаговая инструкция для запуска пайплайна **Text → Image → 3D → URDF/MJCF** локально.
Используемая 3D-модель — **Hunyuan3D-2mini** (~6–10 GB VRAM).

> Выбери платформу: [Linux](#linux) | [Windows](#windows)

---

# Linux

## Содержание

1. [Требования](#1-требования)
2. [Установка conda-окружения](#2-установка-conda-окружения)
3. [PyTorch с CUDA](#3-pytorch-с-cuda)
4. [Основные зависимости](#4-основные-зависимости)
5. [Git-пакеты](#5-git-пакеты)
6. [Hunyuan3D-2 (3D-генерация)](#6-hunyuan3d-2-3d-генерация)
7. [Git-сабмодули и установка проекта](#7-git-сабмодули-и-установка-проекта)
8. [Настройка языковой модели (LLM)](#8-настройка-языковой-модели-llm)
9. [Переменные окружения](#9-переменные-окружения)
10. [Первый запуск и загрузка весов](#10-первый-запуск-и-загрузка-весов)
11. [Варианты запуска](#11-варианты-запуска)
12. [Переключение 3D-модели](#12-переключение-3d-модели)
13. [Устранение проблем](#13-устранение-проблем)

---

## 1. Требования

| Компонент | Минимум | Рекомендуется |
|-----------|---------|---------------|
| ОС | Linux (Ubuntu 20.04+, Fedora 38+) | Ubuntu 22.04 / Fedora 40+ |
| GPU | NVIDIA 8 GB VRAM (RTX 2070+) | RTX 3080 / 4070+ |
| CUDA Toolkit | 11.8 (только если нужен nvcc) | 11.8 или 12.x |
| Python | 3.10 | 3.10 |
| RAM | 16 GB | 32 GB |
| Диск | 40 GB свободно | 60 GB |

> **Примечание:** PyTorch устанавливается в виде колеса cu118, которое содержит
> CUDA-библиотеки внутри себя — отдельный CUDA Toolkit не обязателен для запуска.
> nvcc (компилятор) нужен только если будете пересобирать C++-расширения из исходников.

---

## 2. Установка conda-окружения

Если Miniconda/Anaconda ещё не установлена:

```bash
# Скачать и установить Miniconda (Linux x86_64)
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh -b -p "$HOME/miniconda3"
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda init bash   # или zsh, если используешь zsh
# Перезапустить терминал, затем:
```

Создать окружение и перейти в папку проекта:

```bash
conda create -n embodiedgen python=3.10.13 -y
conda activate embodiedgen
cd ~/Документы/EmbodiedGen   # или путь к клонированному репозиторию
```

---

## 3. PyTorch с CUDA

```bash
pip install torch==2.4.0 torchvision==0.19.0 \
    xformers==0.0.27.post2 \
    --index-url https://download.pytorch.org/whl/cu118
```

Проверка:

```bash
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
# Ожидаемый вывод: True  NVIDIA GeForce RTX xxxx
```

---

## 4. Основные зависимости

```bash
pip install -r requirements.txt --use-deprecated=legacy-resolver
pip install plyfile tables diffusers==0.30.3
```

---

## 5. Git-пакеты

Устанавливаются напрямую из GitHub (нужен интернет или VPN для некоторых):

```bash
pip install --no-deps "utils3d @ git+https://github.com/EasternJournalist/utils3d.git@9a4eb15"
pip install "clip @ git+https://github.com/openai/CLIP.git"
pip install --no-deps "kolors @ git+https://github.com/HochCC/Kolors.git"
pip install --no-deps "MoGe @ git+https://github.com/microsoft/MoGe.git@a8c3734"
pip install gsplat==1.5.3
pip install "nvdiffrast @ git+https://github.com/NVlabs/nvdiffrast.git@729261d"
```

---

## 6. Hunyuan3D-2 (3D-генерация)

Клонировать рядом с проектом и установить:

```bash
# Находясь в ~/Документы/ или родительской папке EmbodiedGen
cd ..
git clone https://github.com/Tencent-Hunyuan/Hunyuan3D-2
cd Hunyuan3D-2
pip install -e .
cd ../EmbodiedGen   # вернуться в проект
```

Проверка:

```bash
python -c "from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline; print('OK')"
```

---

## 7. Git-сабмодули и установка проекта

```bash
git submodule update --init --recursive
pip install --no-deps -e .
```

Проверка установки проекта:

```bash
python -c "import embodied_gen; print('EmbodiedGen OK')"
```

---

## 8. Настройка языковой модели (LLM)

Пайплайн использует LLM для проверки качества (quality checkers). Поддерживаются три варианта.

### Вариант А — Ollama (рекомендуется, локально)

1. Установить Ollama:

```bash
curl -fsSL https://ollama.com/install.sh | sh
```

2. Скачать и запустить модель:

```bash
ollama pull qwen2.5vl:7b   # ~5 GB, vision-language модель
# или
ollama pull qwen3:8b        # ~5 GB, текстовая
```

3. Убедиться что Ollama запущена:

```bash
curl http://localhost:11434/api/tags   # должен вернуть JSON со списком моделей
```

4. Отредактировать [embodied_gen/utils/gpt_config.yaml](embodied_gen/utils/gpt_config.yaml):

```yaml
agent_type: "ollama"

ollama:
  endpoint: http://localhost:11434/v1
  api_key: ollama
  api_version: null
  model_name: qwen2.5vl:7b   # имя модели как в `ollama list`
```

### Вариант Б — OpenRouter (облако, бесплатный tier)

```yaml
agent_type: "qwen2.5-vl"

qwen2.5-vl:
  endpoint: https://openrouter.ai/api/v1
  api_key: sk-or-v1-ВАШ_КЛЮЧ
  api_version: null
  model_name: qwen/qwen2.5-vl-72b-instruct:free
```

### Вариант В — GPT-4o (Azure OpenAI)

```yaml
agent_type: "gpt-4o"

gpt-4o:
  endpoint: https://ВАШ_РЕСУРС.openai.azure.com
  api_key: ВАШ_КЛЮЧ
  api_version: 2025-01-01
  model_name: gpt-4o
```

---

## 9. Переменные окружения

Добавить в `~/.bashrc` (или экспортировать перед каждым запуском):

```bash
export TEXT_MODEL="sdxl-turbo"      # модель Text→Image (sdxl-turbo рекомендуется для 8 GB)
export OUTPUT_ROOT="outputs/jobs"   # куда сохранять результаты
export TORCH_HOME="weights/torch_cache"   # кэш весов torch
export HF_HOME="weights/hf_cache"         # кэш HuggingFace
export HUNYUAN3D_TEXTURE="1"        # "0" — выключить texture pipeline (~6 GB экономия)
```

Применить:

```bash
source ~/.bashrc
```

> **Совет:** При первом запуске используй `HUNYUAN3D_TEXTURE=0`, чтобы убедиться
> что pipeline работает без ошибок OOM. Texture добавляет ещё ~6 GB VRAM.

---

## 10. Первый запуск и загрузка весов

При первом запуске автоматически скачаются веса моделей:

| Модель | Размер | Куда |
|--------|--------|------|
| SDXL-Turbo (Text→Image) | ~7 GB | `weights/hf_cache/` |
| Hunyuan3D-2mini (shape) | ~4 GB | `weights/hf_cache/` |
| Hunyuan3D-2 Paint (texture) | ~6 GB | `weights/hf_cache/` |
| DINOv2 ViT-L/14 | ~1.1 GB | `weights/torch_cache/` |
| MoGe | ~1 GB | `weights/hf_cache/` |

**Итого:** ~19 GB при первом запуске.

> **Важно для РФ:** `dl.fbaipublicfiles.com` (DINOv2) может быть заблокирован.
> Скачай вручную через VPN и положи в:
> `weights/torch_cache/hub/checkpoints/dinov2_vitl14_reg4_pretrain.pth`

---

## 11. Варианты запуска

### Быстрый запуск через run.sh

```bash
conda activate embodiedgen

# Запустить API-сервер (рекомендуется)
bash run.sh server

# Сгенерировать объект напрямую
bash run.sh generate "a wooden chair" --name chair
```

---

### Вариант А — API-сервер

```bash
conda activate embodiedgen
uvicorn api.main:app --host 0.0.0.0 --port 8000 --workers 1
```

**Отправить запрос:**

```bash
curl -X POST http://localhost:8000/api/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "a wooden chair", "name": "chair"}'
# Вернёт: {"job_id": "abc123..."}
```

**Проверить статус задачи:**

```bash
curl http://localhost:8000/api/jobs/<job_id>
```

**Посмотреть логи:**

```bash
curl http://localhost:8000/api/jobs/<job_id>/logs
```

**Swagger UI** (интерактивная документация):

```
http://localhost:8000/docs
```

---

### Вариант Б — Прямой запуск generate.py

```bash
conda activate embodiedgen

# Базовый запуск
python generate.py "a wooden chair"

# С параметрами
python generate.py "a red ceramic mug" \
  --name mug \
  --output outputs/mug \
  --seed_img 42 \
  --seed_3d 0

# Только URDF, без MJCF
python generate.py "iron dumbbell" --skip_mjcf
```

Результат появится в `outputs/generated/` (или в `--output`):

```
outputs/generated/
└── asset3d/
    └── chair/
        └── result/
            ├── mesh/
            │   ├── chair.obj       ← 3D-меш
            │   ├── chair.glb       ← для просмотра в браузере
            │   └── chair.mtl
            ├── chair.urdf          ← для симуляторов (MuJoCo, Isaac)
            ├── mjcf/
            │   └── chair.xml       ← MuJoCo XML
            └── video.mp4           ← превью 3D-модели
```

---

### Вариант В — Только Image → 3D (без Text→Image)

Если у тебя уже есть изображение:

```bash
python -m embodied_gen.scripts.imageto3d \
  --image_path /path/to/your/image.png \
  --output_root outputs/my_object
```

---

## 12. Переключение 3D-модели

В файле [embodied_gen/scripts/imageto3d.py](embodied_gen/scripts/imageto3d.py) строка 53:

```python
IMAGE3D_MODEL = "HUNYUAN3D"   # активная модель (рекомендуется для 8 GB)
# IMAGE3D_MODEL = "SAM3D"     # вернуться на SAM3D (нужен сабмодуль ~10-14 GB)
# IMAGE3D_MODEL = "TRELLIS"   # Microsoft TRELLIS (нужно 20+ GB VRAM)
```

### Сравнение моделей

| Модель | VRAM | Качество | Скорость | Примечания |
|--------|------|----------|----------|------------|
| **HUNYUAN3D** | 6–10 GB | Высокое | Средняя | Рекомендуется. Встроенная текстуризация. |
| SAM3D | 10–14 GB | Хорошее | Быстрая | Требует сабмодуль `thirdparty/SAM3D` |
| TRELLIS | 20–25 GB | Отличное | Медленная | Нужна видеокарта 24+ GB |

### Отключить текстуризацию (экономия ~6 GB VRAM)

```bash
export HUNYUAN3D_TEXTURE=0
python generate.py "a wooden chair"
```

---

## 13. Устранение проблем

### CUDA out of memory (OOM)

1. Отключить texture pipeline:
   ```bash
   export HUNYUAN3D_TEXTURE=0
   ```

2. Уменьшить размер текстуры (по умолчанию 1024):
   ```bash
   python -m embodied_gen.scripts.imageto3d \
     --image_path image.png \
     --output_root output \
     --texture_size 512
   ```

3. Освободить VRAM от других приложений:
   ```bash
   # Посмотреть что занимает GPU
   nvidia-smi
   # или (если nvidia-smi нет в PATH)
   python -c "import torch; print(torch.cuda.memory_summary())"
   ```

---

### torch.cuda.is_available() возвращает False

```bash
# Проверить, видит ли Python GPU
python -c "import torch; print(torch.__version__, torch.version.cuda)"

# Проверить драйвер
python -c "import ctypes; ctypes.CDLL('libcuda.so.1')"
# Если ошибка — проблема с драйвером или LD_LIBRARY_PATH

# Добавить в ~/.bashrc если библиотеки не найдены:
export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH
```

---

### Ollama: connection refused

```bash
# Запустить Ollama-сервер
ollama serve &

# Проверить
curl http://localhost:11434/api/tags

# Убедиться, что модель скачана
ollama list
```

---

### DINOv2 не скачивается (РФ)

```bash
# Создать директорию
mkdir -p weights/torch_cache/hub/checkpoints

# Скачать через VPN или зеркало и положить файл:
# weights/torch_cache/hub/checkpoints/dinov2_vitl14_reg4_pretrain.pth
```

---

### ImportError: No module named 'hy3dgen'

Hunyuan3D-2 не установлен. Выполнить шаг 6:

```bash
cd ../Hunyuan3D-2   # папка рядом с EmbodiedGen
pip install -e .
cd ../EmbodiedGen
python -c "from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline; print('OK')"
```

---

### Ошибка при сборке C++-расширений (nvdiffrast, gsplat)

```bash
# Установить CUDA Toolkit (нужен nvcc)
# Fedora:
sudo dnf install cuda-toolkit-11-8

# Ubuntu:
wget https://developer.download.nvidia.com/compute/cuda/11.8.0/local_installers/cuda_11.8.0_520.61.05_linux.run
sudo sh cuda_11.8.0_520.61.05_linux.run --toolkit --silent

# Добавить в ~/.bashrc:
export PATH=/usr/local/cuda-11.8/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-11.8/lib64:$LD_LIBRARY_PATH
```

---

### Медленная загрузка с HuggingFace

```bash
# Использовать зеркало hf-mirror.com
export HF_ENDPOINT=https://hf-mirror.com
```

---

## Краткий чеклист первого запуска

```bash
# 1. Активировать окружение
conda activate embodiedgen

# 2. Убедиться что Ollama запущена
curl -s http://localhost:11434/api/tags | head -c 100

# 3. Проверить GPU
python -c "import torch; print('GPU:', torch.cuda.get_device_name(0))"

# 4. Первый тест (без текстуры — быстрее)
export HUNYUAN3D_TEXTURE=0
python generate.py "a simple wooden cube" --skip_mjcf

# 5. Посмотреть результат
ls outputs/generated/asset3d/simple/result/
```

---

---

# Windows

Инструкция для запуска на **Windows 10/11** с NVIDIA GPU.
Все команды выполняются в **Anaconda PowerShell Prompt** (не в обычном PowerShell и не в CMD).

## Содержание (Windows)

1. [Требования (Windows)](#1-требования-windows)
2. [Установка инструментов](#2-установка-инструментов)
3. [Conda-окружение (Windows)](#3-conda-окружение-windows)
4. [PyTorch с CUDA (Windows)](#4-pytorch-с-cuda-windows)
5. [Основные зависимости (Windows)](#5-основные-зависимости-windows)
6. [Git-пакеты (Windows)](#6-git-пакеты-windows)
7. [Hunyuan3D-2 (Windows)](#7-hunyuan3d-2-windows)
8. [Git-сабмодули и установка проекта (Windows)](#8-git-сабмодули-и-установка-проекта-windows)
9. [Настройка LLM (Windows)](#9-настройка-llm-windows)
10. [Переменные окружения (Windows)](#10-переменные-окружения-windows)
11. [Запуск (Windows)](#11-запуск-windows)
12. [Устранение проблем (Windows)](#12-устранение-проблем-windows)

---

## 1. Требования (Windows)

| Компонент | Минимум | Рекомендуется |
|-----------|---------|---------------|
| ОС | Windows 10 64-bit | Windows 11 |
| GPU | NVIDIA 8 GB VRAM (RTX 2070+) | RTX 3080 / 4070+ |
| CUDA Toolkit | 11.8 | 11.8 |
| Python | 3.10 | 3.10 |
| RAM | 16 GB | 32 GB |
| Диск | 50 GB свободно | 70 GB |

---

## 2. Установка инструментов

Установить по порядку (если ещё не установлено):

### Git
Скачать и установить с настройками по умолчанию:
`https://git-scm.com/download/win`

После установки перезапустить Anaconda PowerShell Prompt.

### CUDA Toolkit 11.8
`https://developer.nvidia.com/cuda-11-8-0-download-archive`

Выбрать: Windows → x86_64 → 10 → exe (network) или exe (local).
Установить с настройками по умолчанию.

Проверить после установки (новый терминал):
```powershell
nvcc --version
# nvcc: NVIDIA (R) Cuda compiler driver ... release 11.8
```

### Visual Studio Build Tools (нужен для C++-расширений)
Скачать **Build Tools for Visual Studio 2022**:
`https://visualstudio.microsoft.com/visual-cpp-build-tools/`

При установке выбрать компонент:
**"Desktop development with C++"** (включает MSVC, Windows SDK).

### Miniconda
Скачать установщик:
`https://repo.anaconda.com/miniconda/Miniconda3-latest-Windows-x86_64.exe`

При установке отметить **"Add Miniconda3 to PATH"** (или запускать через
**Anaconda PowerShell Prompt** из меню Пуск).

---

## 3. Conda-окружение (Windows)

Открыть **Anaconda PowerShell Prompt** из меню Пуск и выполнить:

```powershell
conda create -n embodiedgen python=3.10.13 -y
conda activate embodiedgen
cd C:\Users\ИМЯ_ПОЛЬЗОВАТЕЛЯ\Desktop\EmbodiedGen   # путь к проекту
```

---

## 4. PyTorch с CUDA (Windows)

```powershell
pip install torch==2.4.0 torchvision==0.19.0 `
    xformers==0.0.27.post2 `
    --index-url https://download.pytorch.org/whl/cu118
```

Проверка:

```powershell
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
# Ожидаемый вывод: True  NVIDIA GeForce RTX xxxx
```

---

## 5. Основные зависимости (Windows)

```powershell
pip install -r requirements.txt --use-deprecated=legacy-resolver
pip install plyfile tables diffusers==0.30.3
```

---

## 6. Git-пакеты (Windows)

```powershell
pip install --no-deps "utils3d @ git+https://github.com/EasternJournalist/utils3d.git@9a4eb15"
pip install "clip @ git+https://github.com/openai/CLIP.git"
pip install --no-deps "kolors @ git+https://github.com/HochCC/Kolors.git"
pip install --no-deps "MoGe @ git+https://github.com/microsoft/MoGe.git@a8c3734"
pip install gsplat==1.5.3
pip install "nvdiffrast @ git+https://github.com/NVlabs/nvdiffrast.git@729261d"
```

> Если при установке `gsplat` или `nvdiffrast` ошибка компилятора — убедись,
> что установлены **Visual Studio Build Tools** (шаг 2) и **CUDA Toolkit 11.8**.

---

## 7. Hunyuan3D-2 (Windows)

```powershell
# Перейти в родительскую папку проекта
cd ..
git clone https://github.com/Tencent-Hunyuan/Hunyuan3D-2
cd Hunyuan3D-2
pip install -e .
cd ..\EmbodiedGen   # вернуться в проект
```

Проверка:

```powershell
python -c "from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline; print('OK')"
```

---

## 8. Git-сабмодули и установка проекта (Windows)

```powershell
git submodule update --init --recursive
pip install --no-deps -e .
```

Проверка:

```powershell
python -c "import embodied_gen; print('EmbodiedGen OK')"
```

---

## 9. Настройка LLM (Windows)

Отредактировать файл `embodied_gen\utils\gpt_config.yaml`.

### Вариант А — Ollama (рекомендуется)

Скачать и установить Ollama для Windows:
`https://ollama.com/download/windows`

Скачать модель (в любом терминале):

```powershell
ollama pull qwen2.5vl:7b
```

Убедиться что Ollama запущена (запускается как фоновый сервис при установке):

```powershell
curl http://localhost:11434/api/tags
```

Файл `gpt_config.yaml`:

```yaml
agent_type: "ollama"

ollama:
  endpoint: http://localhost:11434/v1
  api_key: ollama
  api_version: null
  model_name: qwen2.5vl:7b
```

### Вариант Б — OpenRouter (облако)

```yaml
agent_type: "qwen2.5-vl"

qwen2.5-vl:
  endpoint: https://openrouter.ai/api/v1
  api_key: sk-or-v1-ВАШ_КЛЮЧ
  api_version: null
  model_name: qwen/qwen2.5-vl-72b-instruct:free
```

---

## 10. Переменные окружения (Windows)

В **Anaconda PowerShell Prompt** перед каждым запуском:

```powershell
$env:TEXT_MODEL        = "sdxl-turbo"
$env:OUTPUT_ROOT       = "outputs/jobs"
$env:TORCH_HOME        = "weights/torch_cache"
$env:HF_HOME           = "weights/hf_cache"
$env:HUNYUAN3D_TEXTURE = "1"    # "0" — выключить текстуры (~6 GB экономия)
```

Чтобы не вводить каждый раз, сохрани их в файл `env.ps1` в папке проекта:

```powershell
# env.ps1
$env:TEXT_MODEL        = "sdxl-turbo"
$env:OUTPUT_ROOT       = "outputs/jobs"
$env:TORCH_HOME        = "weights/torch_cache"
$env:HF_HOME           = "weights/hf_cache"
$env:HUNYUAN3D_TEXTURE = "1"
```

И запускай перед стартом:

```powershell
. .\env.ps1
```

---

## 11. Запуск (Windows)

### Вариант А — API-сервер

```powershell
conda activate embodiedgen
. .\env.ps1   # загрузить переменные окружения

uvicorn api.main:app --host 0.0.0.0 --port 8000 --workers 1
```

**Отправить запрос** (в другом Anaconda PowerShell Prompt):

```powershell
Invoke-RestMethod http://localhost:8000/api/generate `
  -Method POST `
  -ContentType "application/json" `
  -Body '{"prompt": "a wooden chair", "name": "chair"}'
# Вернёт: job_id
```

**Проверить статус:**

```powershell
Invoke-RestMethod http://localhost:8000/api/jobs/<job_id>
```

**Посмотреть логи:**

```powershell
Invoke-RestMethod http://localhost:8000/api/jobs/<job_id>/logs
```

**Swagger UI** (документация в браузере):

```
http://localhost:8000/docs
```

---

### Вариант Б — Прямой запуск generate.py

```powershell
conda activate embodiedgen
. .\env.ps1

# Базовый запуск
python generate.py "a wooden chair"

# С параметрами
python generate.py "a red ceramic mug" --name mug --output outputs/mug --seed_img 42

# Без текстуры (быстрее, меньше VRAM)
$env:HUNYUAN3D_TEXTURE = "0"
python generate.py "a wooden chair" --skip_mjcf
```

Результат появится в `outputs\generated\`:

```
outputs\generated\
└── asset3d\
    └── chair\
        └── result\
            ├── mesh\
            │   ├── chair.obj       ← 3D-меш
            │   ├── chair.glb       ← для просмотра в браузере
            │   └── chair.mtl
            ├── chair.urdf          ← для симуляторов
            ├── mjcf\
            │   └── chair.xml       ← MuJoCo XML
            └── video.mp4           ← превью 3D-модели
```

---

### Вариант В — Только Image → 3D

```powershell
python -m embodied_gen.scripts.imageto3d `
  --image_path C:\path\to\image.png `
  --output_root outputs\my_object
```

---

## 12. Устранение проблем (Windows)

### CUDA out of memory (OOM)

```powershell
# Отключить texture pipeline
$env:HUNYUAN3D_TEXTURE = "0"
python generate.py "a wooden chair"
```

---

### torch.cuda.is_available() возвращает False

```powershell
# Проверить версию torch и CUDA
python -c "import torch; print(torch.__version__, torch.version.cuda)"

# Убедиться, что CUDA Toolkit 11.8 установлен
nvcc --version

# Переустановить torch если нужно
pip uninstall torch torchvision xformers -y
pip install torch==2.4.0 torchvision==0.19.0 xformers==0.0.27.post2 `
    --index-url https://download.pytorch.org/whl/cu118
```

---

### Ошибка при сборке C++-расширений

Симптом: `error: Microsoft Visual C++ 14.0 or greater is required`

Решение: Установить **Visual Studio Build Tools** (шаг 2), выбрать
**"Desktop development with C++"**. После установки **перезапустить** Anaconda
PowerShell Prompt.

---

### nvdiffrast / gsplat не компилируются

```powershell
# Убедиться что переменные CUDA выставлены
$env:CUDA_HOME = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8"
$env:PATH = "$env:CUDA_HOME\bin;$env:PATH"

# Попробовать установить снова
pip install "nvdiffrast @ git+https://github.com/NVlabs/nvdiffrast.git@729261d"
```

Если ошибки продолжаются — попробуй установить через **WSL2** (Windows Subsystem
for Linux), там компиляция работает надёжнее. Инструкция для WSL2 совпадает
с [Linux-разделом](#linux) выше.

---

### Ollama не запускается

Ollama на Windows запускается как фоновый сервис. Если сервис не работает:

```powershell
# Запустить вручную (в отдельном PowerShell-окне)
& "C:\Users\$env:USERNAME\AppData\Local\Programs\Ollama\ollama.exe" serve

# Проверить
curl http://localhost:11434/api/tags
```

---

### DINOv2 не скачивается (РФ)

Скачать через VPN и положить файл:

```
weights\torch_cache\hub\checkpoints\dinov2_vitl14_reg4_pretrain.pth
```

```powershell
# Создать папку
New-Item -ItemType Directory -Force `
  -Path "weights\torch_cache\hub\checkpoints"
```

---

### Медленная загрузка с HuggingFace

```powershell
$env:HF_ENDPOINT = "https://hf-mirror.com"
```

---

## Краткий чеклист первого запуска (Windows)

```powershell
# 1. Открыть Anaconda PowerShell Prompt
conda activate embodiedgen
cd C:\Users\ИМЯ\Desktop\EmbodiedGen

# 2. Загрузить переменные окружения
. .\env.ps1

# 3. Убедиться что Ollama запущена
curl http://localhost:11434/api/tags

# 4. Проверить GPU
python -c "import torch; print('GPU:', torch.cuda.get_device_name(0))"

# 5. Первый тест (без текстуры — меньше VRAM)
$env:HUNYUAN3D_TEXTURE = "0"
python generate.py "a simple wooden cube" --skip_mjcf

# 6. Посмотреть результат
dir outputs\generated\asset3d\simple\result\
```
