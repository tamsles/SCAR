#!/bin/bash
#PJM -N __JOB_NAME__
#PJM -L rscgrp=__RSCGRP__
#PJM -L gpu=__GPU_COUNT__
#PJM -L elapse=__ELAPSE__
#PJM --fs /work
#PJM -j

set -Eeuo pipefail

TASKS="__TASKS__"
GPU_COUNT=__GPU_COUNT__
CHUNK=__CHUNK__
REPO_DIR="__REPO_DIR__"
SCARCE_REPO="__SCARCE_REPO__"
OUTPUT_DIR="__OUTPUT_DIR__"
DATASET="__DATASET__"
SHIFT="__SHIFT__"
RATIO_EPOCHS=__RATIO_EPOCHS__
CLASSIFIER_EPOCHS=__CLASSIFIER_EPOCHS__
TARGET_WEAK_SIZE=__TARGET_WEAK_SIZE__
RUN_REPOSITORY_TESTS=__RUN_TESTS__

source "${REPO_DIR}/wisteria/config.sh"
cd "${REPO_DIR}"
mkdir -p "${OUTPUT_DIR}" "${OUTPUT_DIR}/task_logs" \
    "${OUTPUT_DIR}/status" logs
JOB_TAG="${PJM_JOBID:-manual_$(date +%Y%m%d_%H%M%S)}"
LOG_FILE="${REPO_DIR}/logs/scarce_multigpu_chunk${CHUNK}_${JOB_TAG}.log"
exec > >(tee -a "${LOG_FILE}") 2>&1

echo "Start: $(date --iso-8601=seconds)"
echo "Host: $(hostname)"
echo "Job: ${JOB_TAG}; chunk: ${CHUNK}; requested GPUs: ${GPU_COUNT}"
echo "Tasks: ${TASKS}"
echo "Ratio repository: ${REPO_DIR}"
echo "SCARCE repository: ${SCARCE_REPO}"
echo "Output: ${OUTPUT_DIR}"

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

read -r -a task_array <<< "${TASKS}"
if [[ "${#task_array[@]}" -gt "${GPU_COUNT}" ]]; then
    echo "Task count exceeds requested GPU count" >&2
    exit 4
fi

pids=()
task_names=()
run_task() {
    local architecture="$1"
    local seed="$2"
    local slot="$3"
    local device_token
    if [[ "${#visible_devices[@]}" -gt "${slot}" ]]; then
        device_token="${visible_devices[${slot}]}"
    else
        device_token="${slot}"
    fi
    local task_name="${architecture}_seed${seed}"
    local task_log="${OUTPUT_DIR}/task_logs/${task_name}_${JOB_TAG}.log"
    local running="${OUTPUT_DIR}/status/${task_name}.running"
    local succeeded="${OUTPUT_DIR}/status/${task_name}.ok"
    local failed="${OUTPUT_DIR}/status/${task_name}.failed"
    rm -f "${succeeded}" "${failed}"
    printf 'job=%s gpu_slot=%s device=%s start=%s\n' \
        "${JOB_TAG}" "${slot}" "${device_token}" \
        "$(date --iso-8601=seconds)" > "${running}"
    echo "Launch ${task_name} on slot ${slot} (${device_token})"
    if CUDA_VISIBLE_DEVICES="${device_token}" \
        "${PYTHON_BIN}" -m experiments.scarce_ratio_comparison \
            --scarce-repo "${SCARCE_REPO}" \
            --output-dir "${OUTPUT_DIR}" \
            --ratio-arch "${architecture}" \
            --dataset "${DATASET}" \
            --shift "${SHIFT}" \
            --seed "${seed}" \
            --ratio-epochs "${RATIO_EPOCHS}" \
            --classifier-epochs "${CLASSIFIER_EPOCHS}" \
            --batch-size 256 \
            --workers 0 \
            --ratio-learning-rate 1e-3 \
            --learning-rate 1e-3 \
            --weight-decay 1e-5 \
            --ratio-clip-max 20 \
            --device cuda \
            --target-weak-size "${TARGET_WEAK_SIZE}" \
            --source-size 0 \
            --target-rotation 30 \
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
for task in "${task_array[@]}"; do
    architecture="${task%%:*}"
    seed="${task##*:}"
    run_task "${architecture}" "${seed}" "${slot}" &
    pids+=("$!")
    task_names+=("${architecture}_seed${seed}")
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
echo "Succeeded: $(find "${OUTPUT_DIR}/status" -maxdepth 1 -name '*.ok' | wc -l)"
echo "Failed in this chunk: ${failed_count}"
echo "Log: ${LOG_FILE}"
if [[ "${failed_count}" -ne 0 ]]; then
    exit 5
fi
