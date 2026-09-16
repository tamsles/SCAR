from __future__ import annotations

import os

import pytest
import torch

from iwpu.utils import seed_everything


def test_seed_everything_uses_strict_pytorch_determinism(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[bool, bool]] = []

    def fake_use_deterministic_algorithms(
        enabled: bool,
        *,
        warn_only: bool = False,
    ) -> None:
        calls.append((enabled, warn_only))

    monkeypatch.setattr(
        torch, "use_deterministic_algorithms", fake_use_deterministic_algorithms
    )
    monkeypatch.delenv("CUBLAS_WORKSPACE_CONFIG", raising=False)
    monkeypatch.setattr(torch.backends.cudnn, "benchmark", True)
    monkeypatch.setattr(torch.backends.cudnn, "deterministic", False)

    seed_everything(17, deterministic=True)

    assert calls == [(True, False)]
    assert os.environ["CUBLAS_WORKSPACE_CONFIG"] == ":4096:8"
    assert torch.backends.cudnn.benchmark is False
    assert torch.backends.cudnn.deterministic is True
