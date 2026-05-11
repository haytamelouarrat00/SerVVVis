# SERVIS: ViSERVO container
# Build: docker/build.sh   (set TORCH_CUDA_ARCH_LIST for target GPU)
# Run:   docker/run.sh

FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    MPLBACKEND=Agg \
    PYOPENGL_PLATFORM=egl

# System deps:
#   build-essential + git: building CUDA extensions
#   libgl1, libglib2.0-0, libsm6, libxext6, libxrender1: OpenCV
#   libegl1, libgles2, libglvnd0, libnvidia-egl-wayland1: pyrender headless (EGL)
#   libosmesa6: pyrender fallback if EGL unavailable
#   ffmpeg: video export
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential git curl ca-certificates pkg-config \
        python3.10 python3.10-dev python3-pip python-is-python3 \
        libgl1 libglib2.0-0 libsm6 libxext6 libxrender1 \
        libegl1 libgles2 libglvnd0 libosmesa6 \
        ffmpeg \
    && rm -rf /var/lib/apt/lists/*

RUN python -m pip install --upgrade pip setuptools wheel

# Torch first (CUDA 12.1 wheels), pinned so kernel ABI matches when building extensions
RUN pip install --index-url https://download.pytorch.org/whl/cu121 \
        torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1

# Python deps
COPY docker/requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt

WORKDIR /app

# Vendored third-party: copy BEFORE SRC so CUDA-extension layers cache well.
# Target arch must match target GPU. Override at build:
#   --build-arg TORCH_CUDA_ARCH_LIST="8.6;8.9"
ARG TORCH_CUDA_ARCH_LIST="7.5;8.0;8.6;8.9;9.0"
ENV TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST} \
    FORCE_CUDA=1 \
    MAX_JOBS=4

COPY SRC/third_party /app/SRC/third_party

# 3DGS rasterizer + simple-knn + fused-ssim (compiled CUDA kernels)
RUN pip install /app/SRC/third_party/diff-gaussian-rasterization \
    || pip install /app/SRC/third_party/gaussian-splatting/submodules/diff-gaussian-rasterization
RUN if [ -d /app/SRC/third_party/gaussian-splatting/submodules/simple-knn ]; then \
        pip install /app/SRC/third_party/gaussian-splatting/submodules/simple-knn ; fi
RUN if [ -d /app/SRC/third_party/gaussian-splatting/submodules/fused-ssim ]; then \
        pip install /app/SRC/third_party/gaussian-splatting/submodules/fused-ssim ; fi

# MoGe (depth)
RUN if [ -f /app/SRC/third_party/MoGe/pyproject.toml ]; then \
        pip install /app/SRC/third_party/MoGe ; fi

# accelerated_features (XFeat) — used as source tree, just install its deps
RUN if [ -f /app/SRC/third_party/accelerated_features/requirements.txt ]; then \
        pip install -r /app/SRC/third_party/accelerated_features/requirements.txt ; fi

# dinov3 if present
RUN if [ -f /app/SRC/third_party/dinov3/setup.py ]; then \
        pip install /app/SRC/third_party/dinov3 || true ; fi

# Now project source (changes here don't bust CUDA-extension cache)
COPY SRC /app/SRC
COPY CONFIGS /app/CONFIGS

ENV PYTHONPATH=/app/SRC:/app/SRC/third_party/accelerated_features:${PYTHONPATH}
WORKDIR /app/SRC

CMD ["python", "main_trajectory.py", "--help"]
