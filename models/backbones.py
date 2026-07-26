"""Small default backbones.

The ratio estimator accepts any ``nn.Module`` that maps an input batch to a
``[B, feature_dim]`` tensor.  This module only supplies a vector-data default;
image projects can inject their existing CNN without copying its code.
"""

from typing import Sequence

import torch
from torch import nn


def _make_mlp(input_dim: int, hidden_dims: Sequence[int], output_dim: int) -> nn.Sequential:
    dims = [input_dim] + list(hidden_dims) + [output_dim]
    layers = []
    for index in range(len(dims) - 1):
        layers.append(nn.Linear(dims[index], dims[index + 1]))
        if index < len(dims) - 2:
            layers.append(nn.ReLU(inplace=True))
    return nn.Sequential(*layers)


class VectorMLPBackbone(nn.Module):
    """MLP feature extractor for vector inputs."""

    def __init__(
        self,
        input_dim: int,
        feature_dim: int,
        hidden_dims: Sequence[int] = (64, 64),
    ) -> None:
        super().__init__()
        if input_dim <= 0 or feature_dim <= 0:
            raise ValueError("input_dim and feature_dim must be positive")
        self.network = _make_mlp(input_dim, hidden_dims, feature_dim)
        self.output_dim = feature_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return features with shape ``[B, feature_dim]``."""
        if x.ndim != 2:
            raise ValueError("VectorMLPBackbone expects x with shape [B, input_dim]")
        return self.network(x)

