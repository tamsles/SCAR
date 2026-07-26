#!/bin/bash
#PJM -N label_ratio
#PJM -L rscgrp=share
#PJM -L gpu=1
#PJM -L elapse=02:00:00
#PJM --fs /work
#PJM -j

set -Eeuo pipefail

REPO_DIR="${LABEL_RATIO_REPO_DIR:-${HOME}/label_conditioned_ratio}"
source "${REPO_DIR}/wisteria/config.sh"
cd "${REPO_DIR}"
mkdir -p logs outputs/wisteria
JOB_TAG="${PJM_JOBID:-manual_$(date +%Y%m%d_%H%M%S)}"
LOG_FILE="${REPO_DIR}/logs/ratio_${JOB_TAG}.log"
exec > >(tee -a "${LOG_FILE}") 2>&1

echo "Start: $(date --iso-8601=seconds)"
echo "Host: $(hostname)"
echo "Repository: ${REPO_DIR}"
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

"${PYTHON_BIN}" - <<'PY'
import sys
import torch
print("Python:", sys.version.replace("\n", " "))
print("PyTorch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
if not torch.cuda.is_available():
    raise SystemExit("CUDA is unavailable; refusing an accidental CPU allocation")
print("GPU:", torch.cuda.get_device_name(0))
PY

if [[ "${RUN_TESTS}" == "1" ]]; then
    "${PYTHON_BIN}" -m unittest discover -s tests -v
fi

"${PYTHON_BIN}" train.py \
    --config configs/ratio_synthetic.json \
    --output-dir outputs/wisteria \
    --device cuda

echo "Finish: $(date --iso-8601=seconds)"
echo "Log: ${LOG_FILE}"
echo "Result: ${REPO_DIR}/outputs/wisteria/result.json"
