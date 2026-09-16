#!/bin/bash
#PJM -N __JOB_NAME___s__SEED__
#PJM -L rscgrp=share
#PJM -L gpu=1
#PJM -L elapse=__ELAPSE__
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
mkdir -p logs result/domain_scarce/total result/domain_scarce/detail

JOB_TAG="${PJM_JOBID:-manual_$(date +%Y%m%d_%H%M%S)}"
LOG_FILE="${REPO_DIR}/logs/domain_scarce_seed${SEED}_${JOB_TAG}.log"
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
import torch
import torchvision

print("PyTorch:", torch.__version__)
print("torchvision:", torchvision.__version__)
print("CUDA available:", torch.cuda.is_available())
if not torch.cuda.is_available():
    raise SystemExit("CUDA is unavailable; stopping before training.")
print("GPU:", torch.cuda.get_device_name(0))
PY
nvidia-smi

MNIST_FILE="${REPO_DIR}/dataset/mnist/MNIST/raw/train-images-idx3-ubyte"
if [[ ! -f "${MNIST_FILE}" ]]; then
    echo "Bundled MNIST data is missing: ${MNIST_FILE}" >&2
    exit 2
fi

if [[ "${RUN_TESTS}" == "1" ]]; then
    "${PYTHON_BIN}" -m unittest discover -s tests -v
fi

for DOMAIN in __DOMAINS__; do
    echo "Starting independent SCARCE model for ${DOMAIN} domain"
    "${PYTHON_BIN}" main_domain_scarce.py \
        -domain "${DOMAIN}" \
        -ds mnist \
        -mo mlp \
        -shift rotation \
        -target_rotation 30 \
        -target_weak_size __TARGET_WEAK_SIZE__ \
        -source_size __SOURCE_SIZE__ \
        -source_gen single \
        -target_gen single \
        -class_prior uniform \
        -risk_correction abs \
        -bs 256 \
        -op adam \
        -lr 1e-3 \
        -wd 1e-5 \
        -ep __EPOCHS__ \
        -run_times 1 \
        -seed "${SEED}" \
        -gpu 0 \
        -workers __WORKERS__
done

echo "Finish: $(date --iso-8601=seconds)"
echo "Log: ${LOG_FILE}"
echo "Results: ${REPO_DIR}/result/domain_scarce"
