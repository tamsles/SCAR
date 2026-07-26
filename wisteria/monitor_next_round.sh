#!/bin/bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
PHASE="${PHASE:-smoke}"
RESULTS_ROOT="${RESULTS_ROOT:-${REPO_DIR}/results/next_round_${PHASE}}"
STATUS_DIR="${RESULTS_ROOT}/status"

pjstat || true
echo
echo "Phase ${PHASE} task status under ${STATUS_DIR}:"
if [[ ! -d "${STATUS_DIR}" ]]; then
    echo "No status directory yet."
    exit 0
fi
for state in running ok failed; do
    count=$(find "${STATUS_DIR}" -maxdepth 1 -name "*.${state}" | wc -l)
    echo "${state}: ${count}"
    find "${STATUS_DIR}" -maxdepth 1 -name "*.${state}" -printf '  %f\n' \
        | sort
done
