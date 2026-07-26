#!/bin/bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
source "${SCRIPT_DIR}/config.sh"

SCARCE_REPO="${SCARCE_REPO:-${HOME}/adiw_scar_run_20260714_140713/SCARCE}"
RATIO_ARCHS="${RATIO_ARCHS:-unified separate multihead fusion}"
SEEDS="${SEEDS:-0 1 2 3 4}"
RUN_MODE="${RUN_MODE:-debug}"
GPUS_PER_JOB="${GPUS_PER_JOB:-4}"
RSCGRP="${RSCGRP:-share}"
DATASET="${DATASET:-mnist}"
SHIFT="${SHIFT:-rotation}"
DRY_RUN="${DRY_RUN:-0}"
JOB_RUN_TESTS="${JOB_RUN_TESTS:-1}"
TEMPLATE="${SCRIPT_DIR}/job_scarce_ratio_multigpu.template.sh"

if [[ "${GPUS_PER_JOB}" -lt 1 || "${GPUS_PER_JOB}" -gt 4 ]]; then
    echo "GPUS_PER_JOB must be 1, 2, or 4 in the Aquarius share queues" >&2
    exit 2
fi
if [[ "${GPUS_PER_JOB}" != "1" && "${GPUS_PER_JOB}" != "2" \
    && "${GPUS_PER_JOB}" != "4" ]]; then
    echo "Aquarius share queues accept only gpu=1, gpu=2, or gpu=4" >&2
    exit 2
fi
for architecture in ${RATIO_ARCHS}; do
    case "${architecture}" in
        unified|separate|multihead|fusion) ;;
        *)
            echo "Invalid ratio architecture: ${architecture}" >&2
            exit 2
            ;;
    esac
done
for seed in ${SEEDS}; do
    if [[ ! "${seed}" =~ ^[0-9]+$ ]]; then
        echo "Invalid seed: ${seed}" >&2
        exit 2
    fi
done

case "${RUN_MODE}" in
    debug)
        RATIO_EPOCHS="${RATIO_EPOCHS:-1}"
        CLASSIFIER_EPOCHS="${CLASSIFIER_EPOCHS:-2}"
        TARGET_WEAK_SIZE="${TARGET_WEAK_SIZE:-256}"
        ELAPSE="${ELAPSE:-01:00:00}"
        OUTPUT_DIR="${OUTPUT_DIR:-outputs/scarce_multigpu_debug}"
        ;;
    full)
        RATIO_EPOCHS="${RATIO_EPOCHS:-10}"
        CLASSIFIER_EPOCHS="${CLASSIFIER_EPOCHS:-200}"
        TARGET_WEAK_SIZE="${TARGET_WEAK_SIZE:-1000}"
        ELAPSE="${ELAPSE:-24:00:00}"
        OUTPUT_DIR="${OUTPUT_DIR:-outputs/scarce_comparison}"
        ;;
    *)
        echo "RUN_MODE must be debug or full" >&2
        exit 2
        ;;
esac

tasks=()
for seed in ${SEEDS}; do
    for architecture in ${RATIO_ARCHS}; do
        tasks+=("${architecture}:${seed}")
    done
done
if [[ "${#tasks[@]}" -eq 0 ]]; then
    echo "No tasks selected" >&2
    exit 2
fi

echo "Wisteria project limits:"
if command -v pjstat >/dev/null 2>&1; then
    pjstat --limit || true
else
    echo "pjstat is unavailable in this shell"
fi
echo "Selected ${#tasks[@]} GPU tasks; up to ${GPUS_PER_JOB} GPUs per job"
echo "Mode=${RUN_MODE}; dataset=${DATASET}; shift=${SHIFT}"

generated_dir="${SCRIPT_DIR}/generated_multigpu"
mkdir -p "${generated_dir}" "${REPO_DIR}/logs"

escape_sed() {
    printf '%s' "$1" | sed 's/[\/&]/\\&/g'
}

escaped_repo="$(escape_sed "${REPO_DIR}")"
escaped_scarce="$(escape_sed "${SCARCE_REPO}")"
escaped_output="$(escape_sed "${OUTPUT_DIR}")"
total="${#tasks[@]}"
chunk=0
offset=0
while [[ "${offset}" -lt "${total}" ]]; do
    remaining=$((total - offset))
    gpu_count="${GPUS_PER_JOB}"
    if [[ "${remaining}" -lt "${gpu_count}" ]]; then
        gpu_count="${remaining}"
    fi
    chunk_tasks=("${tasks[@]:offset:gpu_count}")
    task_string="${chunk_tasks[*]}"
    job_file="${generated_dir}/scarce_${RUN_MODE}_chunk${chunk}.sh"
    job_name="scar_${RUN_MODE}_${chunk}"
    sed \
        -e "s/__JOB_NAME__/$(escape_sed "${job_name}")/g" \
        -e "s/__CHUNK__/${chunk}/g" \
        -e "s/__GPU_COUNT__/${gpu_count}/g" \
        -e "s/__RSCGRP__/$(escape_sed "${RSCGRP}")/g" \
        -e "s/__ELAPSE__/$(escape_sed "${ELAPSE}")/g" \
        -e "s/__TASKS__/$(escape_sed "${task_string}")/g" \
        -e "s/__REPO_DIR__/${escaped_repo}/g" \
        -e "s/__SCARCE_REPO__/${escaped_scarce}/g" \
        -e "s/__OUTPUT_DIR__/${escaped_output}/g" \
        -e "s/__DATASET__/$(escape_sed "${DATASET}")/g" \
        -e "s/__SHIFT__/$(escape_sed "${SHIFT}")/g" \
        -e "s/__RATIO_EPOCHS__/${RATIO_EPOCHS}/g" \
        -e "s/__CLASSIFIER_EPOCHS__/${CLASSIFIER_EPOCHS}/g" \
        -e "s/__TARGET_WEAK_SIZE__/${TARGET_WEAK_SIZE}/g" \
        -e "s/__RUN_TESTS__/${JOB_RUN_TESTS}/g" \
        "${TEMPLATE}" > "${job_file}"
    chmod u+x "${job_file}"
    echo "Submitting chunk ${chunk}: gpu=${gpu_count}; tasks=${task_string}"
    if [[ "${DRY_RUN}" == "1" ]]; then
        echo "DRY RUN: pjsub -g ${PROJECT_GROUP} ${job_file}"
    else
        pjsub -g "${PROJECT_GROUP}" "${job_file}"
    fi
    offset=$((offset + gpu_count))
    chunk=$((chunk + 1))
done

echo "Submitted ${chunk} jobs covering ${total} one-GPU tasks."
echo "Monitor: pjstat; find ${OUTPUT_DIR}/status -maxdepth 1 -type f -print"
