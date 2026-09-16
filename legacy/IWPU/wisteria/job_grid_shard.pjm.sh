#!/bin/bash
#PJM -N iwpu_grid
#PJM -g gu15
#PJM -L rscgrp=share
#PJM -L gpu=1
#PJM -L elapse=12:00:00
#PJM --fs /work
#PJM -j

set -Eeuo pipefail

# PJM executes a scheduler-owned copy of this file. PJM_O_WORKDIR, rather than
# BASH_SOURCE, identifies the directory from which pjsub was invoked.
SUBMIT_DIR="${PJM_O_WORKDIR:-/work/04/gu15/j23000/iwpu_reproduction}"
CONFIG_FILE="${SUBMIT_DIR}/wisteria/config.sh"
if [[ ! -f "${CONFIG_FILE}" ]]; then
    CONFIG_FILE="/work/04/gu15/j23000/iwpu_reproduction/wisteria/config.sh"
fi
if [[ ! -f "${CONFIG_FILE}" ]]; then
    echo "Cannot locate wisteria/config.sh from PJM_O_WORKDIR=${SUBMIT_DIR}" >&2
    exit 2
fi
# shellcheck source=config.sh
source "${CONFIG_FILE}"
cd "${REMOTE_ROOT}"

SHARD_ID="${IWPU_SHARD_ID:-${PJM_BULKNUM:-}}"
NUM_SHARDS="${IWPU_NUM_SHARDS:-8}"
GRID_MODE="${IWPU_GRID_MODE:-main}"
GRID_TASKS="${IWPU_TASKS:-}"
FORCE="${IWPU_FORCE:-0}"
FAIL_FAST="${IWPU_FAIL_FAST:-0}"
DATA_ROOT="${IWPU_DATA_ROOT:-${REMOTE_ROOT}/data}"
RESULT_ROOT="${IWPU_RESULT_ROOT:-${REMOTE_ROOT}/results/paper_grid}"
MAX_WALL_SECONDS="${IWPU_MAX_WALL_SECONDS:-41400}"
DIABETES_NPZ="${IWPU_DIABETES_NPZ:-${REMOTE_ROOT}/data/tableshift/brfss_diabetes.npz}"
FOODSTAMP_NPZ="${IWPU_FOODSTAMP_NPZ:-${REMOTE_ROOT}/data/tableshift/acsfoodstamps.npz}"

if [[ ! "${SHARD_ID}" =~ ^[0-9]+$ ]]; then
    echo "IWPU_SHARD_ID (or PJM_BULKNUM) must be a non-negative integer." >&2
    exit 2
fi
if [[ ! "${NUM_SHARDS}" =~ ^[1-9][0-9]*$ ]] || (( SHARD_ID >= NUM_SHARDS )); then
    echo "Require 0 <= shard ID (${SHARD_ID}) < shard count (${NUM_SHARDS})." >&2
    exit 2
fi
if [[ ! "${GRID_MODE}" =~ ^(main|ablations|comparison|all)$ ]]; then
    echo "IWPU_GRID_MODE must be main, ablations, comparison, or all." >&2
    exit 2
fi
if [[ ! "${FORCE}" =~ ^[01]$ || ! "${FAIL_FAST}" =~ ^[01]$ ]]; then
    echo "IWPU_FORCE and IWPU_FAIL_FAST must be 0 or 1." >&2
    exit 2
fi

mkdir -p "${REMOTE_ROOT}/logs" "${RESULT_ROOT}"
JOB_TAG="${PJM_SUBJOBID:-${PJM_JOBID:-manual_$(date +%Y%m%d_%H%M%S)}}"
LOG_FILE="${REMOTE_ROOT}/logs/grid_${GRID_MODE}_s${SHARD_ID}_of_${NUM_SHARDS}_${JOB_TAG}.log"
STATUS_FILE="${LOG_FILE%.log}.status"
exec > >(tee -a "${LOG_FILE}") 2>&1

on_exit() {
    local code=$?
    trap - EXIT
    set +e
    temporary="${STATUS_FILE}.tmp.$$"
    printf 'exit_code=%s\nfinished_at=%s\nlog=%s\n' \
        "${code}" "$(date --iso-8601=seconds)" "${LOG_FILE}" >"${temporary}"
    mv "${temporary}" "${STATUS_FILE}"
    echo "Shard exit code: ${code}"
    echo "Status: ${STATUS_FILE}"
    exit "${code}"
}
trap on_exit EXIT

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
gpu_name = torch.cuda.get_device_name(0)
print("GPU:", gpu_name)
if "A100" not in gpu_name.upper():
    raise SystemExit(f"Expected the paper's A100 environment, got {gpu_name!r}.")
PY
echo "CPU model: $(lscpu | awk -F: '/Model name/{sub(/^[[:space:]]+/, "", $2); print $2; exit}')"
echo "Kernel: $(uname -srmo)"

task_selected() {
    local wanted="$1"
    [[ -z "${GRID_TASKS}" || ":${GRID_TASKS}:" == *":${wanted}:"* ]]
}
grid_contains_task() {
    local wanted="$1"
    if [[ "${GRID_MODE}" =~ ^(ablations|comparison)$ && "${wanted}" == "foodstamp" ]]; then
        return 1
    fi
    task_selected "${wanted}"
}

if grid_contains_task mnist_io || grid_contains_task mnist_support || \
   grid_contains_task fmnist_io || grid_contains_task fmnist_support || \
   grid_contains_task cifar10_io || grid_contains_task cifar10_support; then
    if [[ ! -f "${DATA_ROOT}/.image_data_ready.json" ]]; then
        echo "Missing ${DATA_ROOT}/.image_data_ready.json." >&2
        echo "Run bash wisteria/prefetch_image_data.sh on the login node first." >&2
        exit 2
    fi
fi
if grid_contains_task diabetes && [[ ! -f "${DIABETES_NPZ}" ]]; then
    echo "The diabetes task is selected but IWPU_DIABETES_NPZ is not a file." >&2
    exit 2
fi
if grid_contains_task foodstamp && [[ ! -f "${FOODSTAMP_NPZ}" ]]; then
    echo "The foodstamp task is selected but IWPU_FOODSTAMP_NPZ is not a file." >&2
    exit 2
fi

command=(
    "${PYTHON_BIN}" wisteria/run_grid_shard.py
    --config configs/paper.yaml
    --output "${RESULT_ROOT}"
    --data-root "${DATA_ROOT}"
    --device cuda
    --mode "${GRID_MODE}"
    --shard-id "${SHARD_ID}"
    --num-shards "${NUM_SHARDS}"
    --max-wall-seconds "${MAX_WALL_SECONDS}"
)

if [[ -n "${GRID_TASKS}" ]]; then
    IFS=':' read -r -a task_array <<<"${GRID_TASKS}"
    for task in "${task_array[@]}"; do
        command+=(--task "${task}")
    done
fi
if [[ -n "${DIABETES_NPZ}" ]]; then
    command+=(--tabular-npz "diabetes=${DIABETES_NPZ}")
fi
if [[ -n "${FOODSTAMP_NPZ}" ]]; then
    command+=(--tabular-npz "foodstamp=${FOODSTAMP_NPZ}")
fi
if [[ "${FORCE}" == 1 ]]; then
    command+=(--force)
fi
if [[ "${FAIL_FAST}" == 1 ]]; then
    command+=(--fail-fast)
fi

printf 'Starting shard %s/%s at %s\n' "${SHARD_ID}" "${NUM_SHARDS}" "$(date --iso-8601=seconds)"
printf 'Command:'
printf ' %q' "${command[@]}"
printf '\n'
if [[ -x /usr/bin/time ]]; then
    # GNU time records per-job peak host RAM in the log, which is essential
    # after the earlier 24 GiB login-node memory incident.
    /usr/bin/time -v "${command[@]}"
else
    "${command[@]}"
fi

echo "Shard finished: $(date --iso-8601=seconds)"
echo "Log: ${LOG_FILE}"
