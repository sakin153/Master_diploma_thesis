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
COPY embodied_gen embodied_gen
COPY api          api
COPY generate.py  generate.py
COPY pyproject.toml setup.cfg MANIFEST.in ./

# Устанавливаем пакет (только метаданные, зависимости уже в base)
# Override версию diffusers: 0.34.0 ломает auto_pipeline из-за отсутствия GlmModel в transformers 4.42
RUN pip install diffusers==0.30.3 plyfile tables

RUN pip install --no-deps -e .

RUN mkdir -p outputs/jobs weights

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
