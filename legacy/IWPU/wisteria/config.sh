#!/bin/bash

# Wisteria/BDEC-01 deployment defaults. Override these in the environment when
# using another account/project rather than editing experiment code.
PROJECT_GROUP="${PROJECT_GROUP:-gu15}"
REMOTE_ROOT="${REMOTE_ROOT:-/work/04/gu15/j23000/iwpu_reproduction}"
MAMBA_ROOT_PREFIX="${MAMBA_ROOT_PREFIX:-${REMOTE_ROOT}/.micromamba}"
MAMBA_BIN="${MAMBA_BIN:-${REMOTE_ROOT}/.tools/micromamba}"
ENV_PREFIX="${ENV_PREFIX:-${REMOTE_ROOT}/.venv}"
PYTHON_BIN="${PYTHON_BIN:-${ENV_PREFIX}/bin/python}"
CUDA_MODULE="${CUDA_MODULE:-cuda/11.8}"

# Reuse the already-present remote MNIST copy if available. This avoids
# relying on outbound internet access from an Aquarius compute node.
LEGACY_MNIST="${LEGACY_MNIST:-/work/04/gu15/j23000/adiw_scar_run_20260714_140713/SCARCE/dataset/mnist/MNIST}"

