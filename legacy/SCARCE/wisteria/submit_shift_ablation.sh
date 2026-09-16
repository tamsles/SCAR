#!/bin/bash

set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
CONFIG_FILE="${SCRIPT_DIR}/config.sh"
TEMPLATE_FILE="${SCRIPT_DIR}/job_shift_ablation.template.sh"
UNIFORM_RATE="0.1111111111"
SHIFTED_RATE="0.0555555556,0.1666666667,0.0555555556,0.1666666667,0.0555555556,0.1666666667,0.0555555556,0.1666666667,0.0555555556,0.1666666667"

# shellcheck source=config.sh
source "${CONFIG_FILE}"

if [[ "${PROJECT_GROUP}" == "CHANGE_ME" || -z "${PROJECT_GROUP}" ]]; then
    echo "Edit PROJECT_GROUP in ${CONFIG_FILE} before submission." >&2
    exit 2
fi
if ! command -v pjsub >/dev/null 2>&1; then
    echo "pjsub was not found. Run this script on a Wisteria login node." >&2
    exit 2
fi

if [[ "$#" -eq 0 ]]; then
    seeds=(0 1 2 3 4)
else
    seeds=("$@")
fi

conditions=(x_shift conditional_shift scar_shift joint_shift)
shifts=(rotation label_permutation none joint)
target_rates=(
    "${UNIFORM_RATE}"
    "${UNIFORM_RATE}"
    "${SHIFTED_RATE}"
    "${SHIFTED_RATE}"
)
requested_condition="${SHIFT_CONDITION:-all}"

mkdir -p "${SCRIPT_DIR}/generated" "${REPO_DIR}/logs"
escaped_repo_dir=$(printf '%s' "${REPO_DIR}" | sed 's/[\/&]/\\&/g')

cd "${REPO_DIR}"
for condition_index in 0 1 2 3; do
    condition="${conditions[condition_index]}"
    if [[ "${requested_condition}" != "all" && "${requested_condition}" != "${condition}" ]]; then
        continue
    fi
    shift_name="${shifts[condition_index]}"
    target_rate="${target_rates[condition_index]}"
    for seed in "${seeds[@]}"; do
        if [[ ! "${seed}" =~ ^[0-9]+$ ]]; then
            echo "Invalid seed: ${seed}" >&2
            exit 2
        fi
        run_tests=0
        if [[ "${seed}" == "0" ]]; then
            run_tests=1
        fi

        job_file="${SCRIPT_DIR}/generated/job_shift_${condition}_seed_${seed}.sh"
        sed \
            -e "s/__SEED__/${seed}/g" \
            -e "s/__COND__/${condition}/g" \
            -e "s/__SHIFT__/${shift_name}/g" \
            -e "s/__SOURCE_RATE__/${UNIFORM_RATE}/g" \
            -e "s/__TARGET_RATE__/${target_rate}/g" \
            -e "s/__RUN_TESTS__/${run_tests}/g" \
            -e "s/__REPO_DIR__/${escaped_repo_dir}/g" \
            "${TEMPLATE_FILE}" > "${job_file}"
        chmod u+x "${job_file}"

        echo "Submitting ${condition}, seed ${seed}: ${job_file}"
        pjsub -g "${PROJECT_GROUP}" "${job_file}"
    done
done

echo "Use 'pjstat' to inspect queued/running jobs."
