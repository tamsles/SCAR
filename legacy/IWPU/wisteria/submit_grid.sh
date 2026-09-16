#!/bin/bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=config.sh
source "${SCRIPT_DIR}/config.sh"

MODE="main"
NUM_SHARDS=8
SHARD_RANGE=""
TASKS=""
FORCE=0
DO_SUBMIT=0
RSCGRP="share"
ELAPSE="12:00:00"
MAX_WALL_SECONDS=41400
SUBMISSION_CAP="${IWPU_SUBMISSION_CAP:-8}"
DATA_ROOT="${IWPU_DATA_ROOT:-${REMOTE_ROOT}/data}"
RESULT_ROOT="${IWPU_RESULT_ROOT:-${REMOTE_ROOT}/results/paper_grid}"
DIABETES_NPZ="${IWPU_DIABETES_NPZ:-${REMOTE_ROOT}/data/tableshift/brfss_diabetes.npz}"
FOODSTAMP_NPZ="${IWPU_FOODSTAMP_NPZ:-${REMOTE_ROOT}/data/tableshift/acsfoodstamps.npz}"

usage() {
    cat <<'EOF'
Usage: bash wisteria/submit_grid.sh [options]

The default is a dry run. Add --submit only after smoke validation and a
pjstat --limit check.

Options:
  --mode main|ablations|comparison|all
                              Grid subset (comparison is 420 paired cells)
  --shards N                 Interleaved shard count (default: 8)
  --range A-B                Submit only shard IDs A through B
  --tasks A,B,...            Restrict to named task IDs
  --diabetes-npz PATH        Override the default preprocessed diabetes NPZ
  --foodstamp-npz PATH       Override the default preprocessed foodstamp NPZ
  --data-root PATH           Remote image-data root
  --result-root PATH         Remote result root
  --elapse HH:MM:SS          Scheduler wall-time per shard (default: 12:00:00)
  --max-wall-seconds N       Stop cleanly between cells after N seconds
  --force                    Rerun already-complete cells
  --submit                   Actually call pjsub (otherwise print commands)
  -h, --help                 Show this help

Wisteria's Aquarius resource policy prohibits PJM bulk jobs. This script submits one ordinary,
single-GPU PJM job per shard and passes its index with pjsub -x.
EOF
}

while (($#)); do
    case "$1" in
        --mode) MODE="$2"; shift 2 ;;
        --shards) NUM_SHARDS="$2"; shift 2 ;;
        --range) SHARD_RANGE="$2"; shift 2 ;;
        --tasks) TASKS="${2//,/:}"; shift 2 ;;
        --diabetes-npz) DIABETES_NPZ="$2"; shift 2 ;;
        --foodstamp-npz) FOODSTAMP_NPZ="$2"; shift 2 ;;
        --data-root) DATA_ROOT="$2"; shift 2 ;;
        --result-root) RESULT_ROOT="$2"; shift 2 ;;
        --elapse) ELAPSE="$2"; shift 2 ;;
        --max-wall-seconds) MAX_WALL_SECONDS="$2"; shift 2 ;;
        --force) FORCE=1; shift ;;
        --submit) DO_SUBMIT=1; shift ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
    esac
done

if [[ ! "${MODE}" =~ ^(main|ablations|comparison|all)$ ]]; then
    echo "--mode must be main, ablations, comparison, or all." >&2
    exit 2
fi
if [[ ! "${NUM_SHARDS}" =~ ^[1-9][0-9]*$ ]]; then
    echo "--shards must be a positive integer." >&2
    exit 2
fi
if [[ ! "${SUBMISSION_CAP}" =~ ^[1-9][0-9]*$ ]]; then
    echo "IWPU_SUBMISSION_CAP must be a positive integer." >&2
    exit 2
fi
if [[ ! "${ELAPSE}" =~ ^[0-9]+:[0-5][0-9]:[0-5][0-9]$ ]]; then
    echo "--elapse must have the form HH:MM:SS." >&2
    exit 2
fi
if [[ ! "${MAX_WALL_SECONDS}" =~ ^[1-9][0-9]*$ ]]; then
    echo "--max-wall-seconds must be a positive integer." >&2
    exit 2
fi

START_SHARD=0
END_SHARD=$((NUM_SHARDS - 1))
if [[ -n "${SHARD_RANGE}" ]]; then
    if [[ ! "${SHARD_RANGE}" =~ ^([0-9]+)-([0-9]+)$ ]]; then
        echo "--range must have the form A-B." >&2
        exit 2
    fi
    START_SHARD="${BASH_REMATCH[1]}"
    END_SHARD="${BASH_REMATCH[2]}"
    if (( START_SHARD > END_SHARD || END_SHARD >= NUM_SHARDS )); then
        echo "Require 0 <= range start <= range end < shard count." >&2
        exit 2
    fi
fi
DISPATCH_COUNT=$((END_SHARD - START_SHARD + 1))
if (( DISPATCH_COUNT > SUBMISSION_CAP )); then
    echo "This invocation would dispatch ${DISPATCH_COUNT} jobs; the cap is ${SUBMISSION_CAP}." >&2
    echo "Use --range to stage a smaller wave, or raise IWPU_SUBMISSION_CAP only after pjstat --limit." >&2
    exit 2
fi

valid_task() {
    case "$1" in
        mnist_io|mnist_support|fmnist_io|fmnist_support|cifar10_io|cifar10_support|diabetes|foodstamp) return 0 ;;
        *) return 1 ;;
    esac
}
if [[ -n "${TASKS}" ]]; then
    IFS=':' read -r -a task_array <<<"${TASKS}"
    for task in "${task_array[@]}"; do
        if ! valid_task "${task}"; then
            echo "Unknown task in --tasks: ${task}" >&2
            exit 2
        fi
    done
fi
if [[ "${MODE}" =~ ^(ablations|comparison)$ && -n "${TASKS}" ]]; then
    has_ablation_task=0
    for task in "${task_array[@]}"; do
        if [[ "${task}" != "foodstamp" ]]; then
            has_ablation_task=1
        fi
    done
    if (( ! has_ablation_task )); then
        echo "The selected paper grid excludes foodstamp; this filter selects no cells." >&2
        exit 2
    fi
fi

for value in "${DATA_ROOT}" "${RESULT_ROOT}" "${DIABETES_NPZ}" "${FOODSTAMP_NPZ}"; do
    if [[ "${value}" == *','* || "${value}" == *$'\n'* ]]; then
        echo "Paths passed through pjsub -x cannot contain commas or newlines: ${value}" >&2
        exit 2
    fi
done

task_selected() {
    local wanted="$1"
    [[ -z "${TASKS}" || ":${TASKS}:" == *":${wanted}:"* ]]
}
grid_contains_task() {
    local wanted="$1"
    if [[ "${MODE}" =~ ^(ablations|comparison)$ && "${wanted}" == "foodstamp" ]]; then
        return 1
    fi
    task_selected "${wanted}"
}
if (( DO_SUBMIT )); then
    if [[ ! -f "${REMOTE_ROOT}/wisteria/job_grid_shard.pjm.sh" ]]; then
        echo "Missing ${REMOTE_ROOT}/wisteria/job_grid_shard.pjm.sh" >&2
        exit 2
    fi
    if grid_contains_task mnist_io || grid_contains_task mnist_support || \
       grid_contains_task fmnist_io || grid_contains_task fmnist_support || \
       grid_contains_task cifar10_io || grid_contains_task cifar10_support; then
        if [[ ! -f "${DATA_ROOT}/.image_data_ready.json" ]]; then
            echo "Selected grid includes image tasks; run wisteria/prefetch_image_data.sh first." >&2
            exit 2
        fi
    fi
    if grid_contains_task diabetes && [[ ! -f "${DIABETES_NPZ}" ]]; then
        echo "Selected grid includes diabetes; pass --diabetes-npz PATH." >&2
        exit 2
    fi
    if grid_contains_task foodstamp && [[ ! -f "${FOODSTAMP_NPZ}" ]]; then
        echo "Selected grid includes foodstamp; pass --foodstamp-npz PATH." >&2
        exit 2
    fi
fi

mkdir -p "${REMOTE_ROOT}/logs"
cd "${REMOTE_ROOT}"  # Makes PJM_O_WORKDIR deterministic for every shard.

echo "Mode: ${MODE}"
echo "Shard layout: ${NUM_SHARDS} interleaved shards; dispatching ${START_SHARD}-${END_SHARD}"
echo "Per shard: 1 Aquarius GPU, rscgrp=${RSCGRP}, elapse=${ELAPSE}"
echo "Force complete cells: ${FORCE}"
if (( ! DO_SUBMIT )); then
    echo "DRY RUN: no jobs will be submitted. Add --submit to execute these commands."
else
    echo "Current project limits:"
    if ! pjstat --limit; then
        echo "pjstat --limit failed; refusing to submit an unverified GPU wave." >&2
        exit 2
    fi
fi

receipt="${REMOTE_ROOT}/logs/submission_${MODE}_$(date +%Y%m%d_%H%M%S).tsv"
for ((shard = START_SHARD; shard <= END_SHARD; shard++)); do
    env_spec="IWPU_SHARD_ID=${shard},IWPU_NUM_SHARDS=${NUM_SHARDS},IWPU_GRID_MODE=${MODE}"
    env_spec+=",IWPU_FORCE=${FORCE}"
    if [[ -n "${TASKS}" ]]; then
        env_spec+=",IWPU_TASKS=${TASKS}"
    fi
    env_spec+=",IWPU_DATA_ROOT=${DATA_ROOT},IWPU_RESULT_ROOT=${RESULT_ROOT}"
    env_spec+=",IWPU_MAX_WALL_SECONDS=${MAX_WALL_SECONDS}"
    if [[ -n "${DIABETES_NPZ}" ]]; then
        env_spec+=",IWPU_DIABETES_NPZ=${DIABETES_NPZ}"
    fi
    if [[ -n "${FOODSTAMP_NPZ}" ]]; then
        env_spec+=",IWPU_FOODSTAMP_NPZ=${FOODSTAMP_NPZ}"
    fi
    job_name="iwpu_${MODE:0:1}_s$(printf '%02d' "${shard}")"
    command=(
        pjsub -g "${PROJECT_GROUP}"
        -N "${job_name}"
        -L "rscgrp=${RSCGRP}"
        -L "gpu=1"
        -L "elapse=${ELAPSE}"
        -x "${env_spec}"
        wisteria/job_grid_shard.pjm.sh
    )
    if (( DO_SUBMIT )); then
        set +e
        submission_output="$("${command[@]}" 2>&1)"
        submission_code=$?
        set -e
        printf '%s\t%s\t%s\n' "${shard}" "${submission_code}" "${submission_output}" | tee -a "${receipt}"
        if (( submission_code != 0 )); then
            echo "Submission stopped at shard ${shard}; inspect ${receipt} and use --range for the remainder." >&2
            exit "${submission_code}"
        fi
    else
        printf '  '
        printf '%q ' "${command[@]}"
        printf '\n'
    fi
done

if (( DO_SUBMIT )); then
    echo "Submission receipt: ${receipt}"
    echo "Monitor with: pjstat"
fi
