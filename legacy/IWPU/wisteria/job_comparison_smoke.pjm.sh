#!/bin/bash
#PJM -N iwpu_cmp_smoke
#PJM -g gu15
#PJM -L rscgrp=share-debug
#PJM -L gpu=1
#PJM -L elapse=00:10:00
#PJM --fs /work
#PJM -j

set -Eeuo pipefail

SUBMIT_DIR="${PJM_O_WORKDIR:-/work/04/gu15/j23000/iwpu_reproduction}"
# shellcheck source=config.sh
source "${SUBMIT_DIR}/wisteria/config.sh"
cd "${REMOTE_ROOT}"

module purge >/dev/null 2>&1 || true
module load aquarius
module load "${CUDA_MODULE}"

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-8}"
export TORCH_HOME="${TORCH_HOME:-${REMOTE_ROOT}/.cache/torch}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-${REMOTE_ROOT}/.cache}"

RESULT_ROOT="${REMOTE_ROOT}/results/diw_vs_two_step_smoke_v1"
mkdir -p "${REMOTE_ROOT}/logs" "${RESULT_ROOT}" "${TORCH_HOME}" "${XDG_CACHE_HOME}"

"${PYTHON_BIN}" - <<'PY'
import torch

if not torch.cuda.is_available():
    raise SystemExit("CUDA is unavailable")
name = torch.cuda.get_device_name(0)
print("GPU:", name)
if "A100" not in name.upper():
    raise SystemExit(f"Expected A100, got {name!r}")
PY

echo "CPU model: $(lscpu | awk -F: '/Model name/{sub(/^[[:space:]]+/, "", $2); print $2; exit}')"

run_cell() {
    local method="$1"
    local command=("${PYTHON_BIN}" scripts/run_experiment.py \
        --config configs/paper.yaml \
        --task mnist_io \
        --method "${method}" \
        --target-size 10,50 \
        --seed 0 \
        --device cuda \
        --data-root data \
        --output "${RESULT_ROOT}" \
        --no-download \
        --max-epochs 2 \
        --steps-per-epoch 2 \
        --patience 1)
    if [[ -x /usr/bin/time ]]; then
        /usr/bin/time -v "${command[@]}"
    else
        "${command[@]}"
    fi
}

run_cell iwpu
run_cell two_step

echo "Comparison smoke finished: $(date --iso-8601=seconds)"
