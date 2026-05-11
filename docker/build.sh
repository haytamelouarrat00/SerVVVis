#!/usr/bin/env bash
# Build SERVIS container.
# Set TORCH_CUDA_ARCH_LIST for the GPU(s) of the TARGET machine.
#   RTX 20xx       = 7.5
#   A100           = 8.0
#   RTX 30xx       = 8.6
#   RTX 40xx / L4  = 8.9
#   H100           = 9.0
# Multiple archs OK (semicolon-separated), e.g. "8.6;8.9".
set -euo pipefail

cd "$(dirname "$0")/.."

ARCHS="${TORCH_CUDA_ARCH_LIST:-7.5;8.0;8.6;8.9;9.0}"
TAG="${TAG:-servis:latest}"

docker build \
    --build-arg TORCH_CUDA_ARCH_LIST="${ARCHS}" \
    -t "${TAG}" \
    -f Dockerfile \
    .

echo "Built ${TAG} for archs: ${ARCHS}"
