#!/bin/bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=config.sh
source "${SCRIPT_DIR}/config.sh"

TABLESHIFT_COMMIT="${TABLESHIFT_COMMIT:-fca9429814703a07e3902d005d46563a207b7f0a}"
TABLESHIFT_SOURCE="${TABLESHIFT_SOURCE:-${REMOTE_ROOT}/.sources/tableshift}"
TABLESHIFT_CONTAINER_REFERENCE="${TABLESHIFT_CONTAINER_REFERENCE:-docker://ghcr.io/jpgard/tableshift:latest}"
TABLESHIFT_SIF="${TABLESHIFT_SIF:-${REMOTE_ROOT}/.containers/tableshift.sif}"

mkdir -p \
    "${REMOTE_ROOT}/.sources" \
    "${REMOTE_ROOT}/.containers" \
    "${REMOTE_ROOT}/.cache/singularity" \
    "${REMOTE_ROOT}/.tmp"

if [[ ! -d "${TABLESHIFT_SOURCE}/.git" ]]; then
    git clone https://github.com/mlfoundations/tableshift.git \
        "${TABLESHIFT_SOURCE}"
fi

if ! git -C "${TABLESHIFT_SOURCE}" diff --quiet || \
   ! git -C "${TABLESHIFT_SOURCE}" diff --cached --quiet; then
    echo "Refusing to overwrite a modified TableShift source checkout." >&2
    exit 2
fi
git -C "${TABLESHIFT_SOURCE}" fetch --depth 1 origin "${TABLESHIFT_COMMIT}"
git -C "${TABLESHIFT_SOURCE}" checkout --detach "${TABLESHIFT_COMMIT}"

module purge >/dev/null 2>&1 || true
module load singularity/3.9.5
export SINGULARITY_CACHEDIR="${REMOTE_ROOT}/.cache/singularity"
export SINGULARITY_TMPDIR="${REMOTE_ROOT}/.tmp"

if [[ ! -s "${TABLESHIFT_SIF}" ]]; then
    temporary="${TABLESHIFT_SIF}.partial.${BASHPID}"
    trap 'rm -f "${temporary}"' EXIT
    singularity pull "${temporary}" "${TABLESHIFT_CONTAINER_REFERENCE}"
    mv "${temporary}" "${TABLESHIFT_SIF}"
    trap - EXIT
fi

actual_commit="$(git -C "${TABLESHIFT_SOURCE}" rev-parse HEAD)"
image_sha256="$(sha256sum "${TABLESHIFT_SIF}" | awk '{print $1}')"
if [[ "${actual_commit}" != "${TABLESHIFT_COMMIT}" ]]; then
    echo "TableShift checkout mismatch: ${actual_commit}" >&2
    exit 3
fi

singularity exec \
    --bind "${REMOTE_ROOT}:${REMOTE_ROOT}" \
    --pwd "${REMOTE_ROOT}" \
    "${TABLESHIFT_SIF}" \
    env PYTHONPATH="${TABLESHIFT_SOURCE}" python - <<'PY'
import tableshift
from tableshift import get_dataset

print("TableShift module:", tableshift.__file__)
print("get_dataset callable:", callable(get_dataset))
PY

printf 'TableShift commit: %s\n' "${actual_commit}"
printf 'TableShift SIF SHA-256: %s\n' "${image_sha256}"
printf 'TableShift container ready: %s\n' "${TABLESHIFT_SIF}"
