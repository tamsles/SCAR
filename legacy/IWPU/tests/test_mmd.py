import torch

from iwpu.mmd import multi_rbf_mmd


def test_mmd_is_zero_for_identical_batches() -> None:
    x = torch.randn(16, 5)
    value = multi_rbf_mmd(x, x)
    assert torch.allclose(value, torch.tensor(0.0), atol=1e-6)


def test_mmd_is_nonnegative_up_to_roundoff() -> None:
    x = torch.randn(12, 3)
    y = torch.randn(11, 3) + 2.0
    assert multi_rbf_mmd(x, y).item() > -1e-6
