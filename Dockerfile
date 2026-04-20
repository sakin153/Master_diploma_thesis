FROM pytorch/pytorch:2.4.0-cuda11.8-cudnn9-devel

LABEL description="EmbodiedGen — text-to-3D generation API"

# ── System deps ──────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    bash \
    build-essential \
    ninja-build \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# ── CUDA environment ──────────────────────────────────────────────────
# sm_75 = RTX 2070 Super (Turing)
# sm_80 = A100, sm_86 = RTX 3090, sm_89 = RTX 4090
ENV CUDA_HOME=/usr/local/cuda-11.8 \
    PATH=/usr/local/cuda-11.8/bin:$PATH \
    LD_LIBRARY_PATH=/usr/local/cuda-11.8/lib64:$LD_LIBRARY_PATH \
    DEBIAN_FRONTEND=noninteractive \
    TORCH_CUDA_ARCH_LIST="7.5;8.0;8.6" \
    TCNN_CUDA_ARCHITECTURES=75,80,86 \
    TOKENIZERS_PARALLELISM=false \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

WORKDIR /app
COPY . .

# ── Python deps ───────────────────────────────────────────────────────
RUN pip install --no-cache-dir --upgrade pip==22.3.1 && \
    pip install --no-cache-dir \
        torch==2.4.0 torchvision==0.19.0 \
        --index-url https://download.pytorch.org/whl/cu118 && \
    pip install --no-cache-dir \
        xformers==0.0.27.post2 \
        --index-url https://download.pytorch.org/whl/cu118

# Install from requirements
RUN pip install --no-cache-dir -r requirements.txt --use-deprecated=legacy-resolver

# Install source-only packages
RUN pip install --no-cache-dir \
    "utils3d@git+https://github.com/EasternJournalist/utils3d.git@9a4eb15" \
    "clip@git+https://github.com/openai/CLIP.git" \
    "segment-anything@git+https://github.com/facebookresearch/segment-anything.git@dca509f" \
    "nvdiffrast@git+https://github.com/NVlabs/nvdiffrast.git@729261d" \
    "kolors@git+https://github.com/HochCC/Kolors.git" \
    "kaolin@git+https://github.com/NVIDIAGameWorks/kaolin.git@v0.16.0" \
    "git+https://github.com/nerfstudio-project/gsplat.git@v1.5.3" \
    "git+https://github.com/facebookresearch/pytorch3d.git@stable" \
    "MoGe@git+https://github.com/microsoft/MoGe.git@a8c3734"

# diff-gaussian-rasterization
RUN git clone --recursive https://github.com/autonomousvision/mip-splatting.git /tmp/mip-splatting && \
    pip install --no-cache-dir /tmp/mip-splatting/submodules/diff-gaussian-rasterization && \
    rm -rf /tmp/mip-splatting

# API dependencies
RUN pip install --no-cache-dir \
    fastapi==0.115.0 \
    "uvicorn[standard]==0.30.0"

# Install project
RUN pip install --no-cache-dir -e .

# ── Submodules ────────────────────────────────────────────────────────
RUN git submodule update --init thirdparty/TRELLIS thirdparty/sam3d

# ── Runtime dirs ─────────────────────────────────────────────────────
RUN mkdir -p outputs/jobs weights

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=120s \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
