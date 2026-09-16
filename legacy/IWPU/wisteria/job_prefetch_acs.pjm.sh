#!/bin/bash
#PJM -N iwpu_acs_fetch
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

mkdir -p logs
JOB_TAG="${PJM_JOBID:-manual_$(date +%Y%m%d_%H%M%S)}"
LOG_FILE="${REMOTE_ROOT}/logs/acs_household_prefetch_${JOB_TAG}.log"
exec > >(tee -a "${LOG_FILE}") 2>&1

IWPU_ALLOW_BATCH_PREFETCH=1 \
ACS_PREFETCH_PAUSE_SECONDS=3 \
    bash wisteria/prefetch_acs_household.sh

echo "ACS prefetch job complete: $(date --iso-8601=seconds)"
echo "Log: ${LOG_FILE}"
