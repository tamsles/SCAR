#!/bin/bash
#PJM -N ratio_scar_s__SEED__
#PJM -L rscgrp=share
#PJM -L gpu=1
#PJM -L elapse=24:00:00
#PJM --fs /work
#PJM -j

set -Eeuo pipefail

SEED=__SEED__
RATIO_ARCHS="__RATIO_ARCHS__"
REPO_DIR="__REPO_DIR__"
SCARCE_REPO="__SCARCE_REPO__"
source "${REPO_DIR}/wisteria/config.sh"
cd "${REPO_DIR}"
mkdir -p logs outputs/scarce_comparison
JOB_TAG="${PJM_JOBID:-manual_$(date +%Y%m%d_%H%M%S)}"
LOG_FILE="${REPO_DIR}/logs/scarce_ratio_seed${SEED}_${JOB_TAG}.log"
exec > >(tee -a "${LOG_FILE}") 2>&1

echo "Start: $(date --iso-8601=seconds)"
echo "Host: $(hostname)"
echo "Seed: ${SEED}"
echo "Ratio repository: ${REPO_DIR}"
echo "SCARCE repository: ${SCARCE_REPO}"
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
import torch
print("PyTorch:", torch.__version__)
print("CUDA:", torch.cuda.is_available())
if not torch.cuda.is_available():
    raise SystemExit("CUDA unavailable")
print("GPU:", torch.cuda.get_device_name(0))
PY

for RATIO_ARCH in ${RATIO_ARCHS}; do
    echo "Starting ${RATIO_ARCH} ratio SCAR, seed ${SEED}"
    "${PYTHON_BIN}" -m experiments.scarce_ratio_comparison \
        --scarce-repo "${SCARCE_REPO}" \
        --output-dir outputs/scarce_comparison \
        --ratio-arch "${RATIO_ARCH}" \
        --seed "${SEED}" \
        --ratio-epochs 10 \
        --classifier-epochs 200 \
        --batch-size 256 \
        --workers 0 \
        --ratio-learning-rate 1e-3 \
        --learning-rate 1e-3 \
        --weight-decay 1e-5 \
        --ratio-clip-max 20 \
        --device cuda \
        --target-weak-size 1000 \
        --source-size 0 \
        --target-rotation 30
done

echo "Finish: $(date --iso-8601=seconds)"
echo "Log: ${LOG_FILE}"
