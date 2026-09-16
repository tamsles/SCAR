#!/bin/bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=config.sh
source "${SCRIPT_DIR}/config.sh"

if [[ -n "${PJM_JOBID:-}" || "${PJM_ENVIRONMENT:-}" == "BATCH" ]]; then
    echo "Build the TableShift sandbox on a Wisteria login node." >&2
    exit 2
fi

TABLESHIFT_SIF="${TABLESHIFT_SIF:-${REMOTE_ROOT}/.containers/tableshift.sif}"
TABLESHIFT_SANDBOX="${TABLESHIFT_SANDBOX:-${REMOTE_ROOT}/.containers/tableshift-sandbox}"

if [[ ! -s "${TABLESHIFT_SIF}" ]]; then
    echo "Missing ${TABLESHIFT_SIF}; run bootstrap_tabular_env.sh first." >&2
    exit 2
fi
if [[ -d "${TABLESHIFT_SANDBOX}" ]]; then
    echo "Reusing TableShift sandbox: ${TABLESHIFT_SANDBOX}"
    exit 0
fi

module purge >/dev/null 2>&1 || true
module load singularity/3.9.5
export SINGULARITY_CACHEDIR="${REMOTE_ROOT}/.cache/singularity"
export SINGULARITY_TMPDIR="${REMOTE_ROOT}/.tmp"
mkdir -p "$(dirname "${TABLESHIFT_SANDBOX}")" "${SINGULARITY_CACHEDIR}" "${SINGULARITY_TMPDIR}"

temporary="${TABLESHIFT_SANDBOX}.partial.${BASHPID}"
echo "Expanding ${TABLESHIFT_SIF} to ${temporary}"
singularity build --sandbox "${temporary}" "${TABLESHIFT_SIF}"
mv "${temporary}" "${TABLESHIFT_SANDBOX}"

singularity exec "${TABLESHIFT_SANDBOX}" python -c \
    'import pandas, sklearn; print("sandbox ready", pandas.__version__, sklearn.__version__)'
echo "TableShift sandbox ready: ${TABLESHIFT_SANDBOX}"
