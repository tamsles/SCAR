#!/bin/bash

set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
CONFIG_FILE="${SCRIPT_DIR}/config.sh"
TEMPLATE_FILE="${SCRIPT_DIR}/job_scarce_domains_seed.template.sh"
SEED="${1:-0}"

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
if [[ ! "${SEED}" =~ ^[0-9]+$ ]]; then
    echo "Invalid seed: ${SEED}" >&2
    exit 2
fi

mkdir -p "${SCRIPT_DIR}/generated" "${REPO_DIR}/logs"
escaped_repo_dir=$(printf '%s' "${REPO_DIR}" | sed 's/[\/&]/\\&/g')
job_file="${SCRIPT_DIR}/generated/job_scarce_domains_smoke_seed_${SEED}.sh"

sed \
    -e "s/__SEED__/${SEED}/g" \
    -e "s/__REPO_DIR__/${escaped_repo_dir}/g" \
    -e "s/__JOB_NAME__/scarce_smoke/g" \
    -e "s/__ELAPSE__/00:15:00/g" \
    -e "s/__TARGET_WEAK_SIZE__/64/g" \
    -e "s/__SOURCE_SIZE__/64/g" \
    -e "s/__EPOCHS__/1/g" \
    -e "s/__WORKERS__/0/g" \
    -e "s/__DOMAINS__/train test/g" \
    "${TEMPLATE_FILE}" > "${job_file}"
chmod u+x "${job_file}"

cd "${REPO_DIR}"
echo "Submitting train/test-domain SCARCE smoke seed ${SEED}: ${job_file}"
pjsub -g "${PROJECT_GROUP}" "${job_file}"
echo "Use 'pjstat' to inspect the smoke job."
