# ─────────────────────────────────────────────────────────────────────
# App-образ — быстрая пересборка (только код, без компиляции).
# Требует собранного базового образа: embodiedgen-base:latest
#
#   docker compose up --build   # пересборка только кода
# ─────────────────────────────────────────────────────────────────────
FROM embodiedgen-base:latest

WORKDIR /app

# Активная 3D-модель: HUNYUAN3D (установлена в базовом образе).
# Для SAM3D/TRELLIS раскомментировать COPY ниже и пересобрать base-образ.
# COPY .gitmodules .gitmodules
# COPY thirdparty/TRELLIS thirdparty/TRELLIS
# COPY thirdparty/sam3d   thirdparty/sam3d

# Исходный код проекта
COPY asset_gen asset_gen
COPY api          api
COPY generate.py  generate.py
COPY pyproject.toml setup.cfg MANIFEST.in ./

# Устанавливаем пакет (только метаданные, зависимости уже в base)
# diffusers==0.30.3 требует transformers<4.44 (FLAX_WEIGHTS_NAME удалён в 4.44+)
RUN pip install diffusers==0.30.3 plyfile tables "transformers==4.42.4"

RUN pip install --no-deps -e .

RUN mkdir -p outputs/jobs weights

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s \
    CMD curl -f http://localhost:8000/health || exit 1

# Базовый pytorch/pytorch-образ задаёт ENTRYPOINT=/opt/nvidia/nvidia_entrypoint.sh,
# который падает с "exec format error" на Windows/WSL2. Переопределяем явно.
ENTRYPOINT ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
