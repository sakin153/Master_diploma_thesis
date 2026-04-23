#!/bin/bash
# run.sh — запуск EmbodiedGen API без Docker
# Использование:
#   bash run.sh            # запускает API-сервер на :8000
#   bash run.sh generate   # запустить generate.py (передай аргументы далее)
#   bash run.sh install    # только установка зависимостей

set -e

CONDA_ENV="embodiedgen"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# ── Активируем conda-окружение ────────────────────────────────────────────────
CONDA_BASE="$(conda info --base 2>/dev/null)" || {
    echo "[ERROR] conda не найдена. Установи Miniconda/Anaconda."
    exit 1
}
source "$CONDA_BASE/etc/profile.d/conda.sh"

if ! conda env list | grep -q "^$CONDA_ENV "; then
    echo "[INFO] Создаём conda-окружение $CONDA_ENV (Python 3.10)..."
    conda create -n "$CONDA_ENV" python=3.10.13 -y
fi

conda activate "$CONDA_ENV"

# ── Проверяем/устанавливаем зависимости ───────────────────────────────────────
if ! python -c "import torch" 2>/dev/null; then
    echo "[INFO] torch не найден — устанавливаем зависимости (~20-30 мин)..."
    bash "$SCRIPT_DIR/install.sh" basic
fi

# Убеждаемся, что пакет установлен в режиме разработки
if ! python -c "import asset_gen" 2>/dev/null; then
    echo "[INFO] Устанавливаем asset_gen..."
    pip install --no-deps -e .
fi

# Дополнительные пакеты, нужные для запуска
pip install -q plyfile tables diffusers==0.30.3 2>/dev/null || true

# ── Проверяем GPU ─────────────────────────────────────────────────────────────
python - <<'PYCHECK'
import torch
if torch.cuda.is_available():
    name = torch.cuda.get_device_name(0)
    vram = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(f"[GPU] {name}  ({vram:.1f} GB VRAM)")
    if vram < 8:
        print("[WARN] < 8 GB VRAM — возможны OOM при SAM3D inference")
else:
    print("[WARN] CUDA недоступна — пайплайн работает только на GPU")
PYCHECK

# ── Режим запуска ─────────────────────────────────────────────────────────────
MODE="${1:-server}"

case "$MODE" in
    install)
        echo "[OK] Зависимости установлены."
        ;;
    generate)
        shift
        echo "[INFO] Запускаем generate.py $*"
        python "$SCRIPT_DIR/generate.py" "$@"
        ;;
    server|"")
        echo "[INFO] Запускаем API-сервер на http://0.0.0.0:8000"
        echo "[INFO] Документация: http://localhost:8000/docs"
        exec uvicorn api.main:app \
            --host 0.0.0.0 \
            --port 8000 \
            --workers 1 \
            --log-level info
        ;;
    *)
        echo "Использование: bash run.sh [server|generate|install] [args...]"
        exit 1
        ;;
esac
