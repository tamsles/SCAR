#!/bin/bash
#PJM -N iwpu_ts_runtime
#PJM -g gu15
#PJM -L rscgrp=share-debug
#PJM -L gpu=1
#PJM -L elapse=00:10:00
#PJM --fs /work
#PJM -j

set -Eeuo pipefail

SUBMIT_DIR="${PJM_O_WORKDIR:-/work/04/gu15/j23000/iwpu_reproduction}"
# shellcheck source=config.sh
source "${SUBMIT_DIR}/wisteria/config.sh"
cd "${REMOTE_ROOT}"

module purge >/dev/null 2>&1 || true
module load singularity/3.9.5
export SINGULARITY_CACHEDIR="${REMOTE_ROOT}/.cache/singularity"
export SINGULARITY_TMPDIR="${REMOTE_ROOT}/.tmp"
TABLESHIFT_SANDBOX="${TABLESHIFT_SANDBOX:-${REMOTE_ROOT}/.containers/tableshift-sandbox}"

singularity exec --userns \
    --bind "${REMOTE_ROOT}:${REMOTE_ROOT}" \
    --pwd "${REMOTE_ROOT}" \
    "${TABLESHIFT_SANDBOX}" \
    python -c 'import pandas, sklearn; print("compute sandbox OK", pandas.__version__, sklearn.__version__)'
