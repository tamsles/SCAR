"""Synthetic source/target data for a runnable density-ratio demonstration."""

from typing import Dict, Optional, Sequence, Tuple, Union

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


def synthetic_label_parameters(
    input_dim: int,
    num_classes: int,
    dtype: torch.dtype = torch.float32,
    device: Optional[torch.device] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return the fixed label projection and bias used by all domains."""
    generator = torch.Generator().manual_seed(104729)
    projection = torch.randn(
        input_dim, num_classes, generator=generator, dtype=dtype
    ) / max(1.0, float(input_dim) ** 0.5)
    bias = torch.linspace(-0.6, 0.6, num_classes, dtype=dtype)
    if device is not None:
        projection = projection.to(device)
        bias = bias.to(device)
    return projection, bias


def empirical_branch_priors(bar_y: torch.Tensor) -> torch.Tensor:
    """Return empirical ``P(bar_y_k=z)`` with shape ``[q,2]``."""
    probability_one = bar_y.float().mean(dim=0)
    return torch.stack((1.0 - probability_one, probability_one), dim=1)


def _label_shift_tensor(
    label_shift: Union[float, Sequence[float]],
    num_classes: int,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    shift = torch.as_tensor(label_shift, dtype=dtype, device=device)
    if shift.ndim == 0:
        return shift.expand(num_classes)
    if shift.ndim != 1 or shift.numel() != num_classes:
        raise ValueError("label_shift must be a scalar or contain num_classes values")
    return shift


def synthetic_oracle_log_ratios(
    x: torch.Tensor,
    bar_y: torch.Tensor,
    config: Dict[str, object],
    source_branch_prior: torch.Tensor,
    target_branch_prior: torch.Tensor,
) -> torch.Tensor:
    """Compute known ``log p_target(x|z)/p_source(x|z)`` for selected branches.

    The Gaussian density term is analytic. Empirical branch priors are supplied
    because logistic-normal Bernoulli marginals do not have a simple closed
    form.
    """
    input_dim = int(config["input_dim"])
    num_classes = int(config["num_classes"])
    if x.ndim != 2 or x.shape[1] != input_dim:
        raise ValueError("x must have shape [B, input_dim]")
    if tuple(bar_y.shape) != (x.shape[0], num_classes):
        raise ValueError("bar_y must have shape [B, num_classes]")
    projection, bias = synthetic_label_parameters(
        input_dim, num_classes, x.dtype, x.device
    )
    mean_shift = torch.tensor(
        config.get("target_mean_shift", [0.8] + [0.0] * (input_dim - 1)),
        dtype=x.dtype,
        device=x.device,
    )
    label_shift = _label_shift_tensor(
        config.get("target_label_shift", 0.2),
        num_classes,
        x.dtype,
        x.device,
    )

    # log N(x; mean_shift, I) - log N(x; 0, I), shape [B,1].
    feature_log_ratio = (
        x.matmul(mean_shift) - 0.5 * mean_shift.square().sum()
    )[:, None]
    source_label_logit = x.matmul(projection) + bias[None, :]
    target_label_logit = source_label_logit + label_shift[None, :]
    states = bar_y.long()
    source_log_probability = torch.where(
        states == 1,
        F.logsigmoid(source_label_logit),
        F.logsigmoid(-source_label_logit),
    )
    target_log_probability = torch.where(
        states == 1,
        F.logsigmoid(target_label_logit),
        F.logsigmoid(-target_label_logit),
    )
    source_prior = source_branch_prior.to(device=x.device, dtype=x.dtype)
    target_prior = target_branch_prior.to(device=x.device, dtype=x.dtype)
    if tuple(source_prior.shape) != (num_classes, 2):
        raise ValueError("source_branch_prior must have shape [q,2]")
    if tuple(target_prior.shape) != (num_classes, 2):
        raise ValueError("target_branch_prior must have shape [q,2]")
    source_selected = source_prior[None, :, :].expand(x.shape[0], -1, -1)
    source_selected = source_selected.gather(2, states.unsqueeze(-1)).squeeze(-1)
    target_selected = target_prior[None, :, :].expand(x.shape[0], -1, -1)
    target_selected = target_selected.gather(2, states.unsqueeze(-1)).squeeze(-1)
    prior_log_ratio = torch.log(source_selected.clamp_min(1.0e-8)) - torch.log(
        target_selected.clamp_min(1.0e-8)
    )
    return (
        feature_log_ratio
        + target_log_probability
        - source_log_probability
        + prior_log_ratio
    )


class SyntheticComplementaryDataset(Dataset):
    """Gaussian vectors with feature-dependent binary complementary labels."""

    def __init__(
        self,
        num_samples: int,
        input_dim: int,
        num_classes: int,
        seed: int,
        mean_shift: Optional[Sequence[float]] = None,
        label_shift: Union[float, Sequence[float]] = 0.0,
    ) -> None:
        if num_samples <= 0 or input_dim <= 0 or num_classes <= 0:
            raise ValueError("dataset dimensions must be positive")
        generator = torch.Generator().manual_seed(seed)
        x = torch.randn(num_samples, input_dim, generator=generator)
        if mean_shift is not None:
            shift = torch.tensor(mean_shift, dtype=x.dtype)
            if shift.numel() != input_dim:
                raise ValueError("mean_shift must contain input_dim values")
            x = x + shift[None, :]

        # Reuse one labeling mechanism across source/target datasets. Domain
        # differences then come only from the configured feature/label shifts.
        projection, bias = synthetic_label_parameters(input_dim, num_classes)
        shift = _label_shift_tensor(
            label_shift, num_classes, x.dtype, x.device
        )
        logits = x.matmul(projection) + bias[None, :] + shift[None, :]
        probabilities = torch.sigmoid(logits).clamp(0.05, 0.95)
        draws = torch.rand(num_samples, num_classes, generator=generator)
        bar_y = (draws < probabilities).long()
        self.x = x
        self.bar_y = bar_y

    def __len__(self) -> int:
        return self.x.shape[0]

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.x[index], self.bar_y[index]


def make_synthetic_loaders(
    config: Dict[str, object]
) -> Tuple[DataLoader, DataLoader, DataLoader, DataLoader]:
    """Build source/target train and validation loaders from configuration."""
    input_dim = int(config["input_dim"])
    num_classes = int(config["num_classes"])
    train_size = int(config.get("train_size", 4096))
    validation_size = int(config.get("validation_size", 1024))
    batch_size = int(config.get("batch_size", 256))
    seed = int(config.get("seed", 7))
    target_shift = config.get("target_mean_shift", [0.8] + [0.0] * (input_dim - 1))
    target_label_shift = config.get("target_label_shift", 0.2)

    source_train = SyntheticComplementaryDataset(
        train_size, input_dim, num_classes, seed, [0.0] * input_dim, 0.0
    )
    target_train = SyntheticComplementaryDataset(
        train_size,
        input_dim,
        num_classes,
        seed + 1,
        target_shift,
        target_label_shift,
    )
    source_validation = SyntheticComplementaryDataset(
        validation_size, input_dim, num_classes, seed + 2, [0.0] * input_dim, 0.0
    )
    target_validation = SyntheticComplementaryDataset(
        validation_size,
        input_dim,
        num_classes,
        seed + 3,
        target_shift,
        target_label_shift,
    )
    loader_kwargs = {
        "batch_size": batch_size,
        "num_workers": int(config.get("num_workers", 0)),
        "pin_memory": bool(config.get("pin_memory", False)),
    }
    return (
        DataLoader(source_train, shuffle=True, **loader_kwargs),
        DataLoader(target_train, shuffle=True, **loader_kwargs),
        DataLoader(source_validation, shuffle=False, **loader_kwargs),
        DataLoader(target_validation, shuffle=False, **loader_kwargs),
    )
