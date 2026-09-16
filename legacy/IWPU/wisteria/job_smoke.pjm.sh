#!/bin/bash
#PJM -N iwpu_smoke
#PJM -g gu15
#PJM -L rscgrp=share-debug
#PJM -L gpu=1
#PJM -L elapse=00:10:00
#PJM --fs /work
#PJM -j

set -Eeuo pipefail

# PJM executes a scheduler-generated copy of this file, so BASH_SOURCE no longer
# points at wisteria/.  PJM_O_WORKDIR is the directory where pjsub was invoked.
SUBMIT_DIR="${PJM_O_WORKDIR:-/work/04/gu15/j23000/iwpu_reproduction}"
# shellcheck source=config.sh
source "${SUBMIT_DIR}/wisteria/config.sh"
cd "${REMOTE_ROOT}"

mkdir -p logs results/smoke
JOB_TAG="${PJM_JOBID:-manual_$(date +%Y%m%d_%H%M%S)}"
LOG_FILE="${REMOTE_ROOT}/logs/smoke_${JOB_TAG}.log"
exec > >(tee -a "${LOG_FILE}") 2>&1

module purge >/dev/null 2>&1 || true
module load aquarius
module load "${CUDA_MODULE}"
module list

if [[ ! -x "${PYTHON_BIN}" ]]; then
    echo "Missing remote environment: ${PYTHON_BIN}" >&2
    echo "Run bash wisteria/bootstrap_env.sh on the login node first." >&2
    exit 2
fi

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-8}"
export TORCH_HOME="${TORCH_HOME:-${REMOTE_ROOT}/.cache/torch}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-${REMOTE_ROOT}/.cache}"
mkdir -p "${TORCH_HOME}" "${XDG_CACHE_HOME}"

"${PYTHON_BIN}" - <<'PY'
import torch

print("CUDA available:", torch.cuda.is_available())
if not torch.cuda.is_available():
    raise SystemExit("CUDA is unavailable; refusing to fall back to CPU.")
print("GPU:", torch.cuda.get_device_name(0))
PY

"${PYTHON_BIN}" -m pytest -q
"${PYTHON_BIN}" scripts/run_paper_grid.py \
    --config configs/paper.yaml \
    --output results/smoke \
    --data-root data \
    --device cuda \
    --smoke \
    --no-download \
    --force \
    --fail-fast

echo "Smoke job finished: $(date --iso-8601=seconds)"
echo "Log: ${LOG_FILE}"
