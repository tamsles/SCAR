#!/bin/bash
#PJM -N adiw_scar_s__SEED__
#PJM -L rscgrp=share
#PJM -L gpu=1
#PJM -L elapse=48:00:00
#PJM --fs /work
#PJM -j

set -Eeuo pipefail

SEED=__SEED__
REPO_DIR="__REPO_DIR__"
CONFIG_FILE="${REPO_DIR}/wisteria/config.sh"

if [[ ! -f "${CONFIG_FILE}" ]]; then
    echo "Missing Wisteria configuration: ${CONFIG_FILE}" >&2
    exit 2
fi

# shellcheck source=config.sh
source "${CONFIG_FILE}"
cd "${REPO_DIR}"
mkdir -p logs result/adiw_scar/total result/adiw_scar/detail

JOB_TAG="${PJM_JOBID:-manual_$(date +%Y%m%d_%H%M%S)}"
LOG_FILE="${REPO_DIR}/logs/adiw_scar_seed${SEED}_${JOB_TAG}.log"
exec > >(tee -a "${LOG_FILE}") 2>&1

echo "Start: $(date --iso-8601=seconds)"
echo "Host: $(hostname)"
echo "Repository: ${REPO_DIR}"
echo "Seed: ${SEED}"

module purge >/dev/null 2>&1 || true
module load aquarius
if [[ -n "${EXTRA_MODULES}" ]]; then
    read -r -a extra_module_array <<< "${EXTRA_MODULES}"
    module load "${extra_module_array[@]}"
fi
if [[ -n "${PYTORCH_MODULE}" ]]; then
    module load "${PYTORCH_MODULE}"
fi
module list

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-8}"
export TORCH_HOME="${TORCH_HOME:-${REPO_DIR}/.cache/torch}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-${REPO_DIR}/.cache}"
mkdir -p "${TORCH_HOME}" "${XDG_CACHE_HOME}"

if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
    echo "Python interpreter not found: ${PYTHON_BIN}" >&2
    exit 2
fi

"${PYTHON_BIN}" - <<'PY'
import sys

import numpy
import scipy
import torch
import torchvision

print("Python:", sys.version.replace("\n", " "))
print("NumPy:", numpy.__version__)
print("SciPy:", scipy.__version__)
print("PyTorch:", torch.__version__)
print("torchvision:", torchvision.__version__)
print("CUDA available:", torch.cuda.is_available())
if not torch.cuda.is_available():
    raise SystemExit(
        "CUDA is unavailable. Check PYTORCH_MODULE/PYTHON_BIN in "
        "wisteria/config.sh. The job is stopping instead of using CPU."
    )
print("GPU:", torch.cuda.get_device_name(0))
PY
nvidia-smi

MNIST_FILE="${REPO_DIR}/dataset/mnist/MNIST/raw/train-images-idx3-ubyte"
if [[ ! -f "${MNIST_FILE}" ]]; then
    echo "Bundled MNIST data is missing: ${MNIST_FILE}" >&2
    echo "Do not rely on outbound network access from a compute node." >&2
    exit 2
fi

if [[ "${RUN_TESTS}" == "1" ]]; then
    "${PYTHON_BIN}" -m unittest discover -s tests -v
fi

"${PYTHON_BIN}" main_adiw_scar.py \
    -ds mnist \
    -mo mlp \
    -shift rotation \
    -target_rotation 30 \
    -target_weak_size 1000 \
    -source_size 0 \
    -source_gen single \
    -target_gen single \
    -target_prior uniform \
    -eta -1 \
    -risk_correction abs \
    -adiw_steps 1 \
    -adiw_lr 1.0 \
    -adiw_max_weight 50 \
    -adiw_kernel_quantile 0.5 \
    -bs 256 \
    -target_bs 256 \
    -op adam \
    -lr 1e-3 \
    -wd 1e-5 \
    -ep 200 \
    -run_times 1 \
    -seed "${SEED}" \
    -gpu 0 \
    -workers 4

echo "Finish: $(date --iso-8601=seconds)"
echo "Log: ${LOG_FILE}"
echo "Results: ${REPO_DIR}/result/adiw_scar"
