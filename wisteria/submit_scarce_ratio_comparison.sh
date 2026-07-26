#!/bin/bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
SCARCE_REPO="${SCARCE_REPO:-${HOME}/adiw_scar_run_20260714_140713/SCARCE}"
RATIO_ARCHS="${RATIO_ARCHS:-${ESTIMATORS:-unified separate multihead fusion}}"
source "${SCRIPT_DIR}/config.sh"
TEMPLATE="${SCRIPT_DIR}/job_scarce_ratio_seed.template.sh"

for architecture in ${RATIO_ARCHS}; do
    if [[ "${architecture}" != "unified" && "${architecture}" != "separate" \
        && "${architecture}" != "multihead" && "${architecture}" != "fusion" ]]; then
        echo "Invalid ratio architecture: ${architecture}" >&2
        exit 2
    fi
done

if [[ "$#" -eq 0 ]]; then
    seeds=(0 1 2 3 4)
else
    seeds=("$@")
fi
mkdir -p "${SCRIPT_DIR}/generated" "${REPO_DIR}/logs"
escaped_repo=$(printf '%s' "${REPO_DIR}" | sed 's/[\/&]/\\&/g')
escaped_scarce=$(printf '%s' "${SCARCE_REPO}" | sed 's/[\/&]/\\&/g')
escaped_architectures=$(printf '%s' "${RATIO_ARCHS}" | sed 's/[\/&]/\\&/g')
for seed in "${seeds[@]}"; do
    if [[ ! "${seed}" =~ ^[0-9]+$ ]]; then
        echo "Invalid seed: ${seed}" >&2
        exit 2
    fi
    job_file="${SCRIPT_DIR}/generated/job_scarce_ratio_seed_${seed}.sh"
    sed \
        -e "s/__SEED__/${seed}/g" \
        -e "s/__RATIO_ARCHS__/${escaped_architectures}/g" \
        -e "s/__REPO_DIR__/${escaped_repo}/g" \
        -e "s/__SCARCE_REPO__/${escaped_scarce}/g" \
        "${TEMPLATE}" > "${job_file}"
    chmod u+x "${job_file}"
    echo "Submitting neural ratio SCAR seed ${seed}: ${RATIO_ARCHS}"
    pjsub -g "${PROJECT_GROUP}" "${job_file}"
done
