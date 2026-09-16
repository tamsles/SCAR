#!/bin/bash
#PJM -N iwpu_tabular_export
#PJM -g gu15
#PJM -L rscgrp=share-short
#PJM -L gpu=1
#PJM -L elapse=02:00:00
#PJM --fs /work
#PJM -j

set -Eeuo pipefail

SUBMIT_DIR="${PJM_O_WORKDIR:-/work/04/gu15/j23000/iwpu_reproduction}"
# shellcheck source=config.sh
source "${SUBMIT_DIR}/wisteria/config.sh"
cd "${REMOTE_ROOT}"

TABLESHIFT_COMMIT="${TABLESHIFT_COMMIT:-fca9429814703a07e3902d005d46563a207b7f0a}"
TABLESHIFT_SOURCE="${TABLESHIFT_SOURCE:-${REMOTE_ROOT}/.sources/tableshift}"
TABLESHIFT_CONTAINER_REFERENCE="${TABLESHIFT_CONTAINER_REFERENCE:-docker://ghcr.io/jpgard/tableshift:latest}"
TABLESHIFT_SIF="${TABLESHIFT_SIF:-${REMOTE_ROOT}/.containers/tableshift.sif}"
TABLESHIFT_SANDBOX="${TABLESHIFT_SANDBOX:-${REMOTE_ROOT}/.containers/tableshift-sandbox}"
TABLESHIFT_CACHE="${TABLESHIFT_CACHE:-${REMOTE_ROOT}/data/tableshift_cache}"
TABLESHIFT_OUTPUT="${TABLESHIFT_OUTPUT:-${REMOTE_ROOT}/data/tableshift}"

mkdir -p logs "${TABLESHIFT_CACHE}" "${TABLESHIFT_OUTPUT}" \
    "${REMOTE_ROOT}/.cache/singularity" "${REMOTE_ROOT}/.tmp"
JOB_TAG="${PJM_JOBID:-manual_$(date +%Y%m%d_%H%M%S)}"
LOG_FILE="${REMOTE_ROOT}/logs/tableshift_export_${JOB_TAG}.log"
exec > >(tee -a "${LOG_FILE}") 2>&1

if [[ ! -s "${TABLESHIFT_SIF}" || ! -d "${TABLESHIFT_SANDBOX}" || ! -d "${TABLESHIFT_SOURCE}/.git" ]]; then
    echo "Run bootstrap_tabular_env.sh and build_tableshift_sandbox.sh first." >&2
    exit 2
fi
actual_commit="$(git -C "${TABLESHIFT_SOURCE}" rev-parse HEAD)"
if [[ "${actual_commit}" != "${TABLESHIFT_COMMIT}" ]]; then
    echo "TableShift checkout mismatch: ${actual_commit}" >&2
    exit 3
fi

module purge >/dev/null 2>&1 || true
module load singularity/3.9.5
export SINGULARITY_CACHEDIR="${REMOTE_ROOT}/.cache/singularity"
export SINGULARITY_TMPDIR="${REMOTE_ROOT}/.tmp"
export IWPU_TABLESHIFT_CONTAINER_REFERENCE="${TABLESHIFT_CONTAINER_REFERENCE}"
export IWPU_TABLESHIFT_CONTAINER_SHA256
IWPU_TABLESHIFT_CONTAINER_SHA256="$(sha256sum "${TABLESHIFT_SIF}" | awk '{print $1}')"

for task in brfss_diabetes acsfoodstamps; do
    destination="${TABLESHIFT_OUTPUT}/${task}.npz"
    if [[ -s "${destination}" && -s "${destination}.sha256" ]]; then
        echo "Reusing completed export: ${destination}"
        continue
    fi
    singularity exec --userns \
        --bind "${REMOTE_ROOT}:${REMOTE_ROOT}" \
        --pwd "${REMOTE_ROOT}" \
        "${TABLESHIFT_SANDBOX}" \
        env \
            PYTHONPATH="${TABLESHIFT_SOURCE}" \
            IWPU_TABLESHIFT_SOURCE_COMMIT="${actual_commit}" \
            IWPU_TABLESHIFT_CONTAINER_REFERENCE="${IWPU_TABLESHIFT_CONTAINER_REFERENCE}" \
            IWPU_TABLESHIFT_CONTAINER_SHA256="${IWPU_TABLESHIFT_CONTAINER_SHA256}" \
        python scripts/export_tableshift.py \
            --task "${task}" \
            --output-dir "${TABLESHIFT_OUTPUT}" \
            --cache-dir "${TABLESHIFT_CACHE}" \
            --tableshift-source "${TABLESHIFT_SOURCE}" \
            --strict-feature-dim
    sha256sum "${destination}" > "${destination}.sha256"
done

echo "TableShift exports complete: $(date --iso-8601=seconds)"
find "${TABLESHIFT_OUTPUT}" -maxdepth 1 -type f -printf '%f %s bytes\n' | sort
echo "Log: ${LOG_FILE}"
