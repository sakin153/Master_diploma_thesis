# EmbodiedGen API — Полная инструкция по развертыванию

Это полная пошаговая инструкция для развертывания, установки, настройки и запуска EmbodiedGen API на Windows.

---

## 📋 Требования к системе

### Аппаратное обеспечение
- **ОС**: Windows 10/11
- **RAM**: 16 GB минимум (рекомендуется 32 GB)
- **GPU**: NVIDIA CUDA-capable (опционально, но рекомендуется)
  - Память GPU: 8 GB минимум (для Hunyuan3D-2mini)
- **Диск**: 200+ GB свободного места
  - 50 GB для Hunyuan3D-2 моделей
  - 50 GB для Stable Diffusion моделей
  - 100 GB для результатов и кеша

### Программное обеспечение
- Python 3.10+
- Git
- CUDA Toolkit 12.0+ (если используется GPU)
- cuDNN (если используется GPU)

---

## 🚀 Шаг 1: Клонирование репозитория

```powershell
# Откройте PowerShell или Command Prompt

# Перейдите в нужную папку
cd C:\Users\[YourUsername]\Desktop\diploma\mesh_generation

# Клонируйте репозиторий
git clone https://github.com/your-org/Master_diploma_thesis.git
cd Master_diploma_thesis

# Переключитесь на нужную ветку
git checkout feature/generate3DObjecxts_hunuyan
```

---

## 🔧 Шаг 2: Создание виртуального окружения Python

```powershell
# Создайте виртуальное окружение
python -m venv hunyuan_env

# Активируйте окружение
.\hunyuan_env\Scripts\Activate.ps1

# Если у вас ошибка с политикой выполнения, выполните:
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

**Проверка активации:**
```powershell
# В начале строки должно быть (hunyuan_env)
# Пример: (hunyuan_env) PS C:\path\to\project>
```

---

## 📦 Шаг 3: Установка зависимостей

```powershell
# Убедитесь, что вы в виртуальном окружении (hunyuan_env активно)

# Обновите pip
python -m pip install --upgrade pip

# Установите PyTorch с поддержкой CUDA (если есть GPU)
# Для GPU (CUDA 12.1):
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Для CPU только:
# pip install torch torchvision torchaudio

# Установите остальные зависимости
pip install -r requirements.txt

# Если requirements.txt не существует, установите основные пакеты:
pip install fastapi uvicorn pydantic pillow numpy trimesh
pip install diffusers transformers accelerate safetensors
pip install huggingface-hub
pip install clip-by-openai
pip install rembg
pip install pytorch-lightning
```

---

## 📥 Шаг 4: Загрузка Hunyuan3D-2

Папка `Hunyuan3D-2` должна быть заполнена. Есть несколько способов:

### Способ A: Клонирование из GitHub (рекомендуется)

```powershell
# Убедитесь, что вы в корне проекта
cd C:\path\to\Master_diploma_thesis

# Удалите пустую папку если она есть
Remove-Item -Recurse -Force Hunyuan3D-2 -ErrorAction SilentlyContinue

# Клонируйте Hunyuan3D-2
git clone https://github.com/tencent/Hunyuan3D-2.git Hunyuan3D-2

# Перейдите в папку и загрузите модели
cd Hunyuan3D-2
# Модели загрузятся автоматически при первом запуске
cd ..
```

### Способ B: Скачивание с Hugging Face

Если у вас нет Git LFS или скорость медленная:

```powershell
# Модели скачаются автоматически из Hugging Face при первом запуске
# Они будут сохранены в: C:\Users\[YourUsername]\.cache\huggingface\hub\
```

---

## 💾 Шаг 5: Настройка переменных окружения

Создайте файл `.env` в корне проекта:

```powershell
# Создайте файл
New-Item -Path ".env" -ItemType File

# Отредактируйте его (откройте в текстовом редакторе) и добавьте:
```

**Содержимое `.env`:**
```
# GPU/CUDA settings
CUDA_VISIBLE_DEVICES=0

# HuggingFace cache (если у вас проблемы с местом на диске)
HF_HOME=C:\huggingface_cache

# Отключить предупреждение о symlinks на Windows
HF_HUB_DISABLE_SYMLINKS_WARNING=1

# Отключить параллелизм tokenizers
TOKENIZERS_PARALLELISM=false
```

**Загрузите переменные в PowerShell:**
```powershell
# Переход в PowerShell терминал проекта
$env:CUDA_VISIBLE_DEVICES="0"
$env:HF_HUB_DISABLE_SYMLINKS_WARNING="1"
$env:TOKENIZERS_PARALLELISM="false"
```

---

## 🧪 Шаг 6: Первоначальная загрузка моделей

Первый запуск скачает большие модели (это займет 30+ минут):

```powershell
# Активируйте окружение если еще не активировано
.\hunyuan_env\Scripts\Activate.ps1

# Скачайте модели заранее (опционально, но рекомендуется)
python -c "
from diffusers import StableDiffusionPipeline
import torch

# Это загрузит SD 1.5 модель (~4 GB)
pipe = StableDiffusionPipeline.from_pretrained(
    'runwayml/stable-diffusion-v1-5',
    torch_dtype=torch.float16,
    safety_checker=None
)
del pipe
print('Stable Diffusion 1.5 загружена')
"
```

---

## 🌐 Шаг 7: Запуск API сервера

```powershell
# Убедитесь что вы в корне проекта и окружение активировано
# (hunyuan_env) PS C:\path\to\Master_diploma_thesis>

# Запустите FastAPI сервер
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload

# После успешного запуска вы увидите:
# INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
```

**API доступен по адресу:**
- Локально: `http://localhost:8000`
- С другой машины: `http://[ваш_IP]:8000`
- Документация API: `http://localhost:8000/docs`

---

## 📡 Шаг 8: Основные API запросы

### 1. Генерация 3D объекта из текстового описания

**cURL:**
```bash
curl -X POST http://localhost:8000/api/generate \
  -H "Content-Type: application/json" \
  -d '{
    "items": [
      {
        "name": "wooden_chair",
        "prompt": "A wooden chair with four legs, brown color, comfortable design"
      }
    ],
    "model": "sd15",
    "n_image_retry": 2,
    "n_asset_retry": 2,
    "image_height": 768,
    "image_width": 768
  }'
```

**Python:**
```python
import requests
import json

url = "http://localhost:8000/api/generate"

payload = {
    "items": [
        {
            "name": "wooden_chair",
            "prompt": "A wooden chair with four legs, brown color, comfortable design"
        }
    ],
    "model": "sd15",
    "n_image_retry": 2,
    "n_asset_retry": 2,
    "image_height": 768,
    "image_width": 768
}

response = requests.post(url, json=payload)
job_id = response.json()["job_id"]
print(f"Job ID: {job_id}")
```

**JavaScript/Node.js:**
```javascript
const axios = require('axios');

const payload = {
  items: [
    {
      name: 'wooden_chair',
      prompt: 'A wooden chair with four legs, brown color, comfortable design'
    }
  ],
  model: 'sd15',
  n_image_retry: 2,
  n_asset_retry: 2,
  image_height: 768,
  image_width: 768
};

axios.post('http://localhost:8000/api/generate', payload)
  .then(response => {
    console.log('Job ID:', response.data.job_id);
  })
  .catch(error => console.error('Error:', error));
```

### 2. Проверка статуса генерации

```bash
# Получить статус job'а
curl http://localhost:8000/api/jobs/{job_id}

# Получить логи job'а
curl http://localhost:8000/api/jobs/{job_id}/logs
```

**Ответ:**
```json
{
  "job_id": "50147a40-0eb8-4a0e-aeb9-60968e69b434",
  "status": "completed",
  "item_count": 1,
  "model": "sd15",
  "created_at": "2026-04-22T21:40:58.123456",
  "started_at": "2026-04-22T21:40:58.234567",
  "finished_at": "2026-04-22T21:56:01.345678",
  "error": null,
  "results": {
    "wooden_chair": {
      "obj": "outputs/jobs/50147a40.../result/wooden_chair.obj",
      "glb": "outputs/jobs/50147a40.../result/wooden_chair.glb",
      "urdf": "outputs/jobs/50147a40.../result/wooden_chair.urdf",
      "mjcf": "outputs/jobs/50147a40.../result/wooden_chair.mjcf"
    }
  }
}
```

### 3. Список всех job'ов

```bash
curl http://localhost:8000/api/jobs
```

### 4. Скачивание результатов

```bash
# Скачать OBJ файл
curl -O http://localhost:8000/api/jobs/{job_id}/objects/{object_name}/download/obj

# Скачать GLB файл
curl -O http://localhost:8000/api/jobs/{job_id}/objects/{object_name}/download/glb

# Скачать URDF файл
curl -O http://localhost:8000/api/jobs/{job_id}/objects/{object_name}/download/urdf

# Скачать ZIP со всеми файлами
curl -O http://localhost:8000/api/jobs/{job_id}/download/all
```

---

## 🔌 Параметры API запроса

### GenerateRequest

```json
{
  "items": [
    {
      "name": "object_name",
      "prompt": "описание объекта",
      "prompt": null,
      "image_b64": null,
      "image_path": null,
      "asset_type": "optional category",
      "seed_img": 42,
      "seed_3d": 0
    }
  ],
  "model": "sd15",
  "n_image_retry": 2,
  "n_asset_retry": 2,
  "n_pipe_retry": 1,
  "img_denoise_step": 25,
  "text_guidance_scale": 7.0,
  "n_img_sample": 1,
  "image_height": 768,
  "image_width": 768
}
```

### Параметры:
- **name**: Название объекта для результатов
- **prompt**: Текстовое описание объекта (ОСНОВНОЙ СПОСОБ)
- **image_b64**: Base64 закодированное изображение (PNG/JPG)
- **asset_type**: Подсказка категории (e.g., "chair", "table", "cup")
- **seed_img**: Seed для генерации изображения
- **seed_3d**: Seed для генерации 3D модели
- **model**: Выбор модели ("sd15", "sdxl-turbo", "kolors", "sd35")
- **n_image_retry**: Кол-во попыток генерации изображения (1-5)
- **n_asset_retry**: Кол-во попыток генерации 3D (1-5)
- **n_pipe_retry**: Кол-во повторений pipeline (1-5)
- **image_height/width**: Размер генерируемого изображения (256-2048)
- **img_denoise_step**: Кол-во шагов diffusion (4-80)
- **text_guidance_scale**: Сила влияния prompt на генерацию (1.0-20.0)
- **n_img_sample**: Кол-во вариантов для выбора (1-4)

---

## 🎯 Полный пример: От запроса до скачивания

```python
import requests
import time
import json

# 1. Создаем request
url = "http://localhost:8000"

# Отправляем запрос на генерацию
generate_payload = {
    "items": [
        {
            "name": "red_apple",
            "prompt": "A shiny red apple, realistic, isolated on white background"
        }
    ],
    "model": "sd15",
    "n_image_retry": 2,
    "n_asset_retry": 2,
    "image_height": 768,
    "image_width": 768
}

print("📤 Отправляю запрос на генерацию...")
response = requests.post(f"{url}/api/generate", json=generate_payload)
job_data = response.json()
job_id = job_data["job_id"]
print(f"✓ Job ID: {job_id}")
print(f"✓ Статус: {job_data['status']}")
print(f"✓ Позиция в очереди: {job_data['queue_position']}")

# 2. Проверяем статус job'а
print("\n⏳ Ожидаю завершения...")
while True:
    response = requests.get(f"{url}/api/jobs/{job_id}")
    job_status = response.json()
    
    status = job_status["status"]
    print(f"   Статус: {status}")
    
    if status == "completed":
        print("✓ Job завершен!")
        break
    elif status == "failed":
        print(f"✗ Job ошибка: {job_status['error']}")
        break
    
    time.sleep(5)  # Проверяем каждые 5 секунд

# 3. Скачиваем результаты
if status == "completed":
    print("\n📥 Скачиваю результаты...")
    
    # Получаем информацию о файлах
    for obj_name, files in job_status["results"].items():
        print(f"\n   Объект: {obj_name}")
        print(f"   OBJ: {files['obj']}")
        print(f"   GLB: {files['glb']}")
        print(f"   URDF: {files['urdf']}")
        
        # Скачиваем ZIP со всеми файлами
        zip_response = requests.get(
            f"{url}/api/jobs/{job_id}/objects/{obj_name}/download/all"
        )
        
        with open(f"{obj_name}_result.zip", "wb") as f:
            f.write(zip_response.content)
        print(f"   ✓ Скачан: {obj_name}_result.zip")

# 4. Выводим статистику
print("\n📊 Информация о job'е:")
print(f"   Создан: {job_status['created_at']}")
print(f"   Начат: {job_status['started_at']}")
print(f"   Завершен: {job_status['finished_at']}")
print(f"   Объектов: {job_status['item_count']}")
```

---

## 🐛 Решение проблем

### Проблема: "No module named 'hy3dgen'"

**Решение:**
```powershell
# Убедитесь что Hunyuan3D-2 загружена
cd Master_diploma_thesis
ls Hunyuan3D-2\hy3dgen

# Если пусто, клонируйте:
Remove-Item -Recurse -Force Hunyuan3D-2
git clone https://github.com/tencent/Hunyuan3D-2.git Hunyuan3D-2
```

### Проблема: "Not enough free disk space"

**Решение:**
```powershell
# Переместите кеш HuggingFace на другой диск
$env:HF_HOME="D:\huggingface_cache"

# Или очистите старый кеш
Remove-Item -Recurse "$env:USERPROFILE\.cache\huggingface"
```

### Проблема: "CUDA out of memory"

**Решение:**
- Уменьшите `image_height` и `image_width` (попробуйте 512 вместо 768)
- Уменьшите `n_img_sample`
- Используйте модель "sd15" вместо "sdxl-turbo"

### Проблема: "Cannot identify image file" при загрузке base64

**Решение:**
- Используйте текстовый prompt вместо image_b64
- Если используете image_b64, убедитесь что это валидный PNG/JPG с правильным base64 кодированием
- Проверьте что строка base64 имеет длину кратную 4 (добавьте `=` padding если нужно)

### Проблема: API медленный или зависает

**Решение:**
```powershell
# Перезапустите сервер
# Нажмите Ctrl+C в терминале с uvicorn

# Проверьте память:
Get-Process | Where-Object {$_.ProcessName -like "*python*"} | Select-Object ProcessName, @{Name="Memory(MB)";Expression={[math]::Round($_.WorkingSet/1MB)}}

# Перезапустите сервер:
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

---

## 🏥 Проверка здоровья системы

```powershell
# Проверить что API работает
curl http://localhost:8000/health

# Получить список всех job'ов
curl http://localhost:8000/api/jobs

# Просмотреть последний job
curl http://localhost:8000/api/jobs | ConvertFrom-Json | Select-Object -First 1
```

---

## 📝 Логи и отладка

**Логи API сервера:**
- Выводятся в консоль где запущен `uvicorn`
- Ищите строки вида `[ERROR]`, `[WARNING]`, `[INFO]`

**Логи job'а:**
```powershell
# Получить логи последнего job'а
curl http://localhost:8000/api/jobs | ConvertFrom-Json | 
  Select-Object -First 1 | 
  ForEach-Object {curl "http://localhost:8000/api/jobs/$($_.job_id)/logs"}
```

**Папки с результатами:**
- `outputs/jobs/[job_id]/images/` — сгенерированные изображения
- `outputs/jobs/[job_id]/asset3d/` — 3D модели
- `outputs/jobs/[job_id]/result/` — финальные результаты (OBJ, GLB, URDF)

---

## 🚀 Быстрый старт (TL;DR)

```powershell
# 1. Активируйте окружение
.\hunyuan_env\Scripts\Activate.ps1

# 2. Загрузите Hunyuan3D-2 (если не загружена)
git clone https://github.com/tencent/Hunyuan3D-2.git Hunyuan3D-2

# 3. Запустите сервер
uvicorn api.main:app --host 0.0.0.0 --port 8000

# 4. В новом терминале PowerShell, выполните запрос:
$body = @{
    items = @(@{
        name = "test_object"
        prompt = "A wooden table"
    })
    model = "sd15"
} | ConvertTo-Json

Invoke-WebRequest -Uri "http://localhost:8000/api/generate" `
    -Method POST `
    -Body $body `
    -ContentType "application/json"
```

---

## 📞 Помощь и поддержка

- **Документация FastAPI**: http://localhost:8000/docs
- **Документация OpenAPI**: http://localhost:8000/openapi.json
- **Логи в консоли**: Проверьте окно терминала где запущен uvicorn
- **Проверка статуса**: Посетите http://localhost:8000/health

---

**Версия документа**: 1.0  
**Дата обновления**: 2026-04-22  
**Статус**: ✅ Полная инструкция
