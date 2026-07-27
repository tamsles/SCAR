#!/bin/bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
source "${SCRIPT_DIR}/config.sh"
cd "${REPO_DIR}"

SCARCE_REPO="${SCARCE_REPO:-${HOME}/adiw_scar_run_20260714_140713/SCARCE}"
CONFIG="${CONFIG:-configs/next_round.yaml}"
RUNNER="${RUNNER:-scripts/run_next_round.py}"
PHASE="${PHASE:-smoke}"
EXPERIMENT_MODE="${EXPERIMENT_MODE:-smoke}"
RESULTS_ROOT="${RESULTS_ROOT:-results/next_round_${PHASE}}"
GPUS_PER_JOB="${GPUS_PER_JOB:-4}"
OFFSET="${OFFSET:-0}"
MAX_JOBS="${MAX_JOBS:-16}"
RSCGRP="${RSCGRP:-share}"
ELAPSE="${ELAPSE:-24:00:00}"
DRY_RUN="${DRY_RUN:-0}"
JOB_RUN_TESTS="${JOB_RUN_TESTS:-1}"
TEMPLATE="${SCRIPT_DIR}/job_next_round_multigpu.template.sh"

if [[ "${GPUS_PER_JOB}" != "1" && "${GPUS_PER_JOB}" != "2" \
    && "${GPUS_PER_JOB}" != "4" ]]; then
    echo "Aquarius share queues accept gpu=1, gpu=2, or gpu=4" >&2
    exit 2
fi
if [[ "${EXPERIMENT_MODE}" != "smoke" \
    && "${EXPERIMENT_MODE}" != "diagnostic" \
    && "${EXPERIMENT_MODE}" != "full" ]]; then
    echo "EXPERIMENT_MODE must be smoke, diagnostic, or full" >&2
    exit 2
fi

total=$(
    "${PYTHON_BIN}" "${RUNNER}" \
        --config "${CONFIG}" --phase "${PHASE}" --count
)
if [[ ! "${total}" =~ ^[0-9]+$ ]]; then
    echo "Could not determine task count: ${total}" >&2
    exit 2
fi
if [[ "${OFFSET}" -ge "${total}" ]]; then
    echo "OFFSET ${OFFSET} is beyond ${total} tasks" >&2
    exit 2
fi
last=$((OFFSET + MAX_JOBS * GPUS_PER_JOB))
if [[ "${last}" -gt "${total}" ]]; then
    last="${total}"
fi

echo "Wisteria project limits:"
pjstat --limit || true
echo "Phase=${PHASE}; mode=${EXPERIMENT_MODE}; total=${total}"
echo "Submitting task indices [${OFFSET}, ${last}) using up to ${GPUS_PER_JOB} GPUs/job"

generated_dir="${SCRIPT_DIR}/generated_next_round"
mkdir -p "${generated_dir}" "${REPO_DIR}/logs"

escape_sed() {
    printf '%s' "$1" | sed 's/[\/&]/\\&/g'
}

escaped_repo="$(escape_sed "${REPO_DIR}")"
escaped_scarce="$(escape_sed "${SCARCE_REPO}")"
escaped_config="$(escape_sed "${CONFIG}")"
escaped_runner="$(escape_sed "${RUNNER}")"
escaped_results="$(escape_sed "${RESULTS_ROOT}")"
chunk=0
offset="${OFFSET}"
while [[ "${offset}" -lt "${last}" ]]; do
    remaining=$((last - offset))
    gpu_count="${GPUS_PER_JOB}"
    if [[ "${remaining}" -lt "${gpu_count}" ]]; then
        gpu_count="${remaining}"
    fi
    indices=()
    for ((slot=0; slot<gpu_count; slot++)); do
        indices+=("$((offset + slot))")
    done
    task_string="${indices[*]}"
    job_file="${generated_dir}/next_${PHASE}_${EXPERIMENT_MODE}_${offset}.sh"
    job_name="nxt_${PHASE}_${offset}"
    sed \
        -e "s/__JOB_NAME__/$(escape_sed "${job_name}")/g" \
        -e "s/__CHUNK__/${chunk}/g" \
        -e "s/__GPU_COUNT__/${gpu_count}/g" \
        -e "s/__RSCGRP__/$(escape_sed "${RSCGRP}")/g" \
        -e "s/__ELAPSE__/$(escape_sed "${ELAPSE}")/g" \
        -e "s/__TASK_INDICES__/$(escape_sed "${task_string}")/g" \
        -e "s/__REPO_DIR__/${escaped_repo}/g" \
        -e "s/__SCARCE_REPO__/${escaped_scarce}/g" \
        -e "s/__CONFIG__/${escaped_config}/g" \
        -e "s/__RUNNER__/${escaped_runner}/g" \
        -e "s/__PHASE__/$(escape_sed "${PHASE}")/g" \
        -e "s/__EXPERIMENT_MODE__/$(escape_sed "${EXPERIMENT_MODE}")/g" \
        -e "s/__RESULTS_ROOT__/${escaped_results}/g" \
        -e "s/__RUN_TESTS__/${JOB_RUN_TESTS}/g" \
        "${TEMPLATE}" > "${job_file}"
    chmod u+x "${job_file}"
    echo "Submitting gpu=${gpu_count}; tasks=${task_string}"
    if [[ "${DRY_RUN}" == "1" ]]; then
        echo "DRY RUN: pjsub -g ${PROJECT_GROUP} ${job_file}"
    else
        pjsub -g "${PROJECT_GROUP}" "${job_file}"
    fi
    offset=$((offset + gpu_count))
    chunk=$((chunk + 1))
done
echo "Submitted ${chunk} jobs. Next OFFSET=${last}."
