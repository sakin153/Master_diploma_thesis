# ─────────────────────────────────────────────────────────────────────
# App-образ — быстрая пересборка (только код, без компиляции).
# Требует собранного базового образа: embodiedgen-base:latest
#
#   docker compose up --build   # пересборка только кода
# ─────────────────────────────────────────────────────────────────────
FROM embodiedgen-base:latest

WORKDIR /app

# Submodules (только нужные)
COPY .gitmodules .gitmodules
COPY thirdparty/TRELLIS thirdparty/TRELLIS
COPY thirdparty/sam3d   thirdparty/sam3d

# Если flexicubes не был инициализирован (вложенный сабмодуль TRELLIS)
RUN if [ ! -f thirdparty/TRELLIS/trellis/representations/mesh/flexicubes/flexicubes.py ]; then \
    git clone --depth=1 https://github.com/nv-tlabs/FlexiCubes.git /tmp/fc && \
    cp /tmp/fc/flexicubes.py thirdparty/TRELLIS/trellis/representations/mesh/flexicubes/ && \
    rm -rf /tmp/fc; fi

# Исходный код проекта
COPY embodied_gen embodied_gen
COPY api          api
COPY generate.py  generate.py
COPY pyproject.toml setup.cfg MANIFEST.in ./

# Устанавливаем пакет (только метаданные, зависимости уже в base)
# Override версию diffusers: 0.34.0 ломает auto_pipeline из-за отсутствия GlmModel в transformers 4.42
RUN pip install diffusers==0.30.3 plyfile

RUN pip install --no-deps -e .

RUN mkdir -p outputs/jobs weights

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
