#!/bin/bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=config.sh
source "${SCRIPT_DIR}/config.sh"

mkdir -p "${REMOTE_ROOT}/.tools" "${MAMBA_ROOT_PREFIX}" "${REMOTE_ROOT}/data"

if [[ ! -x "${MAMBA_BIN}" ]]; then
    tmp_dir="$(mktemp -d "${REMOTE_ROOT}/.tools/micromamba.XXXXXX")"
    trap 'rm -rf "${tmp_dir}"' EXIT
    curl -fL --retry 3 \
        https://micro.mamba.pm/api/micromamba/linux-64/latest \
        -o "${tmp_dir}/micromamba.tar.bz2"
    tar -xjf "${tmp_dir}/micromamba.tar.bz2" -C "${tmp_dir}" bin/micromamba
    install -m 0755 "${tmp_dir}/bin/micromamba" "${MAMBA_BIN}"
fi

export MAMBA_ROOT_PREFIX
if [[ ! -x "${PYTHON_BIN}" ]]; then
    "${MAMBA_BIN}" create -y -p "${ENV_PREFIX}" python=3.10 pip
fi

"${PYTHON_BIN}" -m pip install --upgrade pip setuptools wheel
"${PYTHON_BIN}" -m pip install \
    --index-url https://download.pytorch.org/whl/cu118 \
    --extra-index-url https://pypi.org/simple \
    torch==2.2.2 torchvision==0.17.2
"${PYTHON_BIN}" -m pip install -e "${REMOTE_ROOT}[dev]"

if [[ -d "${LEGACY_MNIST}" && ! -e "${REMOTE_ROOT}/data/MNIST" ]]; then
    ln -s "${LEGACY_MNIST}" "${REMOTE_ROOT}/data/MNIST"
fi

"${PYTHON_BIN}" - <<'PY'
import platform
import torch
import torchvision

print("Python:", platform.python_version())
print("PyTorch:", torch.__version__)
print("torchvision:", torchvision.__version__)
print("CUDA build:", torch.version.cuda)
print("Environment bootstrap complete; GPU availability is checked in the PJM job.")
PY

