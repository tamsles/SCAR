from __future__ import annotations

import torch


def _squared_distances(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return torch.cdist(x, y, p=2).square()


def multi_rbf_mmd(
    source: torch.Tensor,
    target: torch.Tensor,
    multipliers: tuple[float, ...] = (0.25, 0.5, 1.0, 2.0, 4.0),
) -> torch.Tensor:
    """Biased MMD with five RBF kernels and a detached median bandwidth.

    The paper reports five RBF kernels but omits bandwidths and estimator form.
    This explicit median-heuristic choice is therefore a reproduction assumption.
    """
    if source.ndim != 2 or target.ndim != 2:
        raise ValueError("MMD inputs must have shape [batch, features]")
    combined = torch.cat((source, target), dim=0)
    distances = _squared_distances(combined, combined)
    positive = distances.detach()[distances.detach() > 0]
    bandwidth = positive.median() if positive.numel() else distances.new_tensor(1.0)
    bandwidth = bandwidth.clamp_min(torch.finfo(distances.dtype).eps)

    k_xx = distances[: source.shape[0], : source.shape[0]]
    k_yy = distances[source.shape[0] :, source.shape[0] :]
    k_xy = distances[: source.shape[0], source.shape[0] :]

    value = distances.new_zeros(())
    for multiplier in multipliers:
        scale = 2.0 * bandwidth * multiplier
        value = value + torch.exp(-k_xx / scale).mean()
        value = value + torch.exp(-k_yy / scale).mean()
        value = value - 2.0 * torch.exp(-k_xy / scale).mean()
    return value / len(multipliers)

