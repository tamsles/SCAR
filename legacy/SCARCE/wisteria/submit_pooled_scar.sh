#!/bin/bash

set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
CONFIG_FILE="${SCRIPT_DIR}/config.sh"
TEMPLATE_FILE="${SCRIPT_DIR}/job_scarce_domains_seed.template.sh"

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

mkdir -p "${SCRIPT_DIR}/generated" "${REPO_DIR}/logs"
escaped_repo_dir=$(printf '%s' "${REPO_DIR}" | sed 's/[\/&]/\\&/g')

cd "${REPO_DIR}"
for seed in "${seeds[@]}"; do
    if [[ ! "${seed}" =~ ^[0-9]+$ ]]; then
        echo "Invalid seed: ${seed}" >&2
        exit 2
    fi

    job_file="${SCRIPT_DIR}/generated/job_pooled_scar_seed_${seed}.sh"
    sed \
        -e "s/__SEED__/${seed}/g" \
        -e "s/__REPO_DIR__/${escaped_repo_dir}/g" \
        -e "s/__JOB_NAME__/pooled_scar/g" \
        -e "s/__ELAPSE__/48:00:00/g" \
        -e "s/__TARGET_WEAK_SIZE__/1000/g" \
        -e "s/__SOURCE_SIZE__/0/g" \
        -e "s/__EPOCHS__/200/g" \
        -e "s/__WORKERS__/0/g" \
        -e "s/__DOMAINS__/pooled/g" \
        "${TEMPLATE_FILE}" > "${job_file}"
    chmod u+x "${job_file}"

    echo "Submitting unweighted pooled SCAR seed ${seed}: ${job_file}"
    pjsub -g "${PROJECT_GROUP}" "${job_file}"
done

echo "Use 'pjstat' to inspect queued/running jobs."
