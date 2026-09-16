#!/bin/bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=config.sh
source "${SCRIPT_DIR}/config.sh"

DATA_ROOT="${IWPU_DATA_ROOT:-${REMOTE_ROOT}/data}"

if [[ -n "${PJM_JOBID:-}" || "${PJM_ENVIRONMENT:-}" == "BATCH" ]]; then
    echo "Run this prefetch script on a Wisteria login node, not inside a GPU job." >&2
    exit 2
fi
if [[ ! -x "${PYTHON_BIN}" ]]; then
    echo "Missing remote Python environment: ${PYTHON_BIN}" >&2
    echo "Run bash wisteria/bootstrap_env.sh first." >&2
    exit 2
fi
if ! command -v flock >/dev/null 2>&1; then
    echo "The login node does not provide flock; refusing an unlocked dataset download." >&2
    exit 2
fi

mkdir -p "${DATA_ROOT}"
if [[ -L "${DATA_ROOT}/MNIST" && ! -e "${DATA_ROOT}/MNIST" ]]; then
    echo "Broken MNIST symlink: ${DATA_ROOT}/MNIST" >&2
    exit 2
fi
if [[ -d "${LEGACY_MNIST}" && ! -e "${DATA_ROOT}/MNIST" && ! -L "${DATA_ROOT}/MNIST" ]]; then
    ln -s "${LEGACY_MNIST}" "${DATA_ROOT}/MNIST"
fi

exec 9>"${DATA_ROOT}/.image-prefetch.lock"
if ! flock -n 9; then
    echo "Another image-data prefetch is already using ${DATA_ROOT}." >&2
    exit 3
fi

export IWPU_PREFETCH_DATA_ROOT="${DATA_ROOT}"

# torchvision's downloader restarts an interrupted CIFAR-10 transfer.  Resume
# the official archive explicitly so a slow mirror cannot waste prior bytes.
CIFAR_ARCHIVE="${DATA_ROOT}/cifar-10-python.tar.gz"
CIFAR_URL="https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
CIFAR_MD5="c58f30108f718f92721af3b95e74349a"
if [[ -f "${CIFAR_ARCHIVE}" ]]; then
    current_size="$(stat -c '%s' "${CIFAR_ARCHIVE}")"
    echo "Existing CIFAR-10 archive: ${current_size} bytes"
fi
if ! echo "${CIFAR_MD5}  ${CIFAR_ARCHIVE}" | md5sum --check --status 2>/dev/null; then
    echo "Resuming CIFAR-10 archive from the official source"
    curl --fail --location --silent --show-error \
        --retry 20 --retry-delay 5 --retry-all-errors \
        --continue-at - --output "${CIFAR_ARCHIVE}" "${CIFAR_URL}"
fi
echo "${CIFAR_MD5}  ${CIFAR_ARCHIVE}" | md5sum --check

"${PYTHON_BIN}" - <<'PY'
from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path

import torch
import torchvision
from torchvision.datasets import CIFAR10, FashionMNIST, MNIST


root = Path(os.environ["IWPU_PREFETCH_DATA_ROOT"]).resolve()
expected = (
    ("mnist", MNIST, 60_000, 10_000),
    ("fashion_mnist", FashionMNIST, 60_000, 10_000),
    ("cifar10", CIFAR10, 50_000, 10_000),
)
datasets: dict[str, dict[str, int]] = {}

for name, constructor, expected_train, expected_test in expected:
    print(f"Prefetching {name} under {root}", flush=True)
    train = constructor(root=str(root), train=True, download=True)
    test = constructor(root=str(root), train=False, download=True)
    train_count, test_count = len(train), len(test)
    if (train_count, test_count) != (expected_train, expected_test):
        raise RuntimeError(
            f"{name} length mismatch: got {(train_count, test_count)}, "
            f"expected {(expected_train, expected_test)}"
        )

    # Re-open with downloads disabled. This is the same availability condition
    # required by formal compute jobs, which always use download=False.
    verify_train = constructor(root=str(root), train=True, download=False)
    verify_test = constructor(root=str(root), train=False, download=False)
    if (len(verify_train), len(verify_test)) != (expected_train, expected_test):
        raise RuntimeError(f"No-download verification failed for {name}")
    datasets[name] = {"train": train_count, "test": test_count}

manifest = {
    "schema_version": 1,
    "created_at": datetime.now(timezone.utc).isoformat(),
    "data_root": str(root),
    "torch": torch.__version__,
    "torchvision": torchvision.__version__,
    "datasets": datasets,
    "verified_with_download_false": True,
}
destination = root / ".image_data_ready.json"
temporary = destination.with_name(f"{destination.name}.tmp.{os.getpid()}")
temporary.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
temporary.replace(destination)
print(json.dumps(manifest, indent=2, sort_keys=True), flush=True)
print(f"Image data ready marker: {destination}", flush=True)
PY
