#!/bin/bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "${SCRIPT_DIR}/config.sh"
cd "${SCRIPT_DIR}/.."
pjsub -g "${PROJECT_GROUP}" "${SCRIPT_DIR}/job_ratio_estimator.sh"

