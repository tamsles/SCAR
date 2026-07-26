#!/bin/bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_DIR}/outputs/scarce_multigpu_debug}"
STATUS_DIR="${OUTPUT_DIR}/status"

pjstat || true
echo
echo "Task status under ${STATUS_DIR}:"
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

