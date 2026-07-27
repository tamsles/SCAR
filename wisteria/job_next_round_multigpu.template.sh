#!/bin/bash
#PJM -N __JOB_NAME__
#PJM -L rscgrp=__RSCGRP__
#PJM -L gpu=__GPU_COUNT__
#PJM -L elapse=__ELAPSE__
#PJM --fs /work
#PJM -j

set -Eeuo pipefail

TASK_INDICES="__TASK_INDICES__"
GPU_COUNT=__GPU_COUNT__
CHUNK=__CHUNK__
REPO_DIR="__REPO_DIR__"
SCARCE_REPO="__SCARCE_REPO__"
CONFIG="__CONFIG__"
RUNNER="__RUNNER__"
PHASE="__PHASE__"
EXPERIMENT_MODE="__EXPERIMENT_MODE__"
RESULTS_ROOT="__RESULTS_ROOT__"
RUN_REPOSITORY_TESTS=__RUN_TESTS__

source "${REPO_DIR}/wisteria/config.sh"
cd "${REPO_DIR}"
mkdir -p "${RESULTS_ROOT}/task_logs" "${RESULTS_ROOT}/status" logs
JOB_TAG="${PJM_JOBID:-manual_$(date +%Y%m%d_%H%M%S)}"
LOG_FILE="${REPO_DIR}/logs/next_round_${PHASE}_chunk${CHUNK}_${JOB_TAG}.log"
exec > >(tee -a "${LOG_FILE}") 2>&1

echo "Start: $(date --iso-8601=seconds)"
echo "Host: $(hostname)"
echo "Job: ${JOB_TAG}; phase: ${PHASE}; mode: ${EXPERIMENT_MODE}"
echo "Requested GPUs: ${GPU_COUNT}; task indices: ${TASK_INDICES}"
echo "Results: ${RESULTS_ROOT}"

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

allocated_visible="${CUDA_VISIBLE_DEVICES:-}"
visible_devices=()
if [[ -n "${allocated_visible}" ]]; then
    IFS=',' read -r -a visible_devices <<< "${allocated_visible}"
fi
detected_gpu_count=$(
    "${PYTHON_BIN}" - <<'PY'
import torch
print(torch.cuda.device_count())
PY
)
echo "CUDA_VISIBLE_DEVICES=${allocated_visible:-<unset>}"
echo "PyTorch visible GPU count: ${detected_gpu_count}"
if [[ "${detected_gpu_count}" -lt "${GPU_COUNT}" ]]; then
    echo "Requested ${GPU_COUNT} GPUs but PyTorch sees ${detected_gpu_count}" >&2
    exit 3
fi

if [[ "${RUN_REPOSITORY_TESTS}" == "1" && "${CHUNK}" == "0" ]]; then
    echo "Running repository tests once on chunk 0"
    "${PYTHON_BIN}" -m unittest discover -s tests -v
fi

read -r -a task_array <<< "${TASK_INDICES}"
if [[ "${#task_array[@]}" -gt "${GPU_COUNT}" ]]; then
    echo "Task count exceeds requested GPU count" >&2
    exit 4
fi

pids=()
task_names=()
run_task() {
    local task_index="$1"
    local slot="$2"
    local device_token
    if [[ "${#visible_devices[@]}" -gt "${slot}" ]]; then
        device_token="${visible_devices[${slot}]}"
    else
        device_token="${slot}"
    fi
    local task_name="${PHASE}_task${task_index}"
    local task_log="${RESULTS_ROOT}/task_logs/${task_name}_${JOB_TAG}.log"
    local running="${RESULTS_ROOT}/status/${task_name}.running"
    local succeeded="${RESULTS_ROOT}/status/${task_name}.ok"
    local failed="${RESULTS_ROOT}/status/${task_name}.failed"
    rm -f "${succeeded}" "${failed}"
    printf 'job=%s gpu_slot=%s device=%s start=%s\n' \
        "${JOB_TAG}" "${slot}" "${device_token}" \
        "$(date --iso-8601=seconds)" > "${running}"
    echo "Launch ${task_name} on slot ${slot} (${device_token})"
    if CUDA_VISIBLE_DEVICES="${device_token}" \
        "${PYTHON_BIN}" "${RUNNER}" \
            --config "${CONFIG}" \
            --phase "${PHASE}" \
            --experiment-mode "${EXPERIMENT_MODE}" \
            --task-index "${task_index}" \
            --scarce-repo "${SCARCE_REPO}" \
            --repository-root "${REPO_DIR}" \
            --results-root "${RESULTS_ROOT}" \
            --device cuda \
            > "${task_log}" 2>&1; then
        rm -f "${running}"
        printf 'job=%s finish=%s log=%s\n' \
            "${JOB_TAG}" "$(date --iso-8601=seconds)" "${task_log}" \
            > "${succeeded}"
        echo "Completed ${task_name}"
    else
        exit_code=$?
        rm -f "${running}"
        printf 'job=%s exit=%s finish=%s log=%s\n' \
            "${JOB_TAG}" "${exit_code}" "$(date --iso-8601=seconds)" \
            "${task_log}" > "${failed}"
        echo "Failed ${task_name}; see ${task_log}" >&2
        return "${exit_code}"
    fi
}

slot=0
for task_index in "${task_array[@]}"; do
    run_task "${task_index}" "${slot}" &
    pids+=("$!")
    task_names+=("${PHASE}_task${task_index}")
    slot=$((slot + 1))
done

failed_count=0
for index in "${!pids[@]}"; do
    if ! wait "${pids[${index}]}"; then
        echo "Task failed: ${task_names[${index}]}" >&2
        failed_count=$((failed_count + 1))
    fi
done

echo "Finish: $(date --iso-8601=seconds)"
echo "Succeeded: $(find "${RESULTS_ROOT}/status" -maxdepth 1 -name '*.ok' | wc -l)"
echo "Failed in this chunk: ${failed_count}"
echo "Log: ${LOG_FILE}"
if [[ "${failed_count}" -ne 0 ]]; then
    exit 5
fi
