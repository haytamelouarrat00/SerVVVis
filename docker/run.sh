#!/usr/bin/env bash
# Run SERVIS container on the target machine.
# Requires: NVIDIA driver + nvidia-container-toolkit installed on host.
#
# Mounts DATA/, RUNS/, CONFIGS/ from the host so datasets and outputs persist.
# Pass any args after `--` to the container's python entry, e.g.:
#   docker/run.sh -- python main_trajectory.py --config ../CONFIGS/trajectory_kitchen_mesh.json
#
# Override target image with TAG=... and HF cache with HF_HOME=...
set -euo pipefail

cd "$(dirname "$0")/.."

TAG="${TAG:-servis:latest}"
HF_HOME_HOST="${HF_HOME:-${HOME}/.cache/huggingface}"
mkdir -p DATA RUNS CONFIGS "${HF_HOME_HOST}"

# Drop initial `--` if user used the suggested separator
if [[ "${1:-}" == "--" ]]; then shift; fi
CMD=("$@")
if [[ ${#CMD[@]} -eq 0 ]]; then
    CMD=(python main_trajectory.py --help)
fi

docker run --rm -it \
    --gpus all \
    --shm-size=16g \
    -v "$(pwd)/DATA:/app/DATA" \
    -v "$(pwd)/RUNS:/app/RUNS" \
    -v "$(pwd)/CONFIGS:/app/CONFIGS" \
    -v "${HF_HOME_HOST}:/root/.cache/huggingface" \
    -e HF_HOME=/root/.cache/huggingface \
    -e PYOPENGL_PLATFORM=egl \
    -e MPLBACKEND=Agg \
    -w /app/SRC \
    "${TAG}" \
    "${CMD[@]}"
