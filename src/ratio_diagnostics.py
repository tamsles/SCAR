"""Density-ratio direction, saturation, and distribution-matching audits."""

import math
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


DOMAIN_LABEL_CONVENTION = {
    "source": 0,
    "target": 1,
    "probability": "D(x) = P(domain=target | x)",
    "ratio": (
        "p_target(x)/p_source(x) = D(x)/(1-D(x)) "
        "* pi_source/pi_target"
    ),
}


def ratio_components_from_logits(
    logits: torch.Tensor,
    source_prior: float = 0.5,
    target_prior: float = 0.5,
    lower_clip: float = 0.01,
    upper_clip: float = 20.0,
    eps: float = 1.0e-8,
) -> Dict[str, torch.Tensor]:
    """Expose every conversion stage for target-positive domain logits."""
    if source_prior <= 0 or target_prior <= 0:
        raise ValueError("domain priors must be positive")
    if lower_clip < 0 or upper_clip <= 0 or lower_clip > upper_clip:
        raise ValueError("invalid ratio clipping bounds")
    logits64 = logits.double()
    probability = torch.sigmoid(logits64)
    probability = probability.clamp(min=eps, max=1.0 - eps)
    raw_odds = probability / (1.0 - probability)
    prior_correction = torch.full_like(
        raw_odds, float(source_prior) / float(target_prior)
    )
    unclipped = raw_odds * prior_correction
    clipped = unclipped.clamp(min=lower_clip, max=upper_clip)
    return {
        "discriminator_probability": probability.float(),
        "raw_odds": raw_odds.float(),
        "prior_correction": prior_correction.float(),
        "unclipped_ratio": unclipped.float(),
        "clipped_ratio": clipped.float(),
    }


def normalize_ratios(
    ratios: torch.Tensor,
    scope: str,
    states: Optional[torch.Tensor] = None,
    eps: float = 1.0e-8,
) -> torch.Tensor:
    """Normalize a ``[N,q]`` ratio tensor with Phase B scopes."""
    if ratios.ndim != 2:
        raise ValueError("ratios must have shape [N,q]")
    if scope == "none":
        return ratios.clone()
    if scope == "global":
        return ratios / ratios.mean().clamp_min(eps)
    if scope != "conditional_branch":
        raise ValueError("scope must be none, global, or conditional_branch")
    if states is None or states.shape != ratios.shape:
        raise ValueError("conditional normalization requires matching states")
    output = ratios.clone()
    for class_index in range(ratios.shape[1]):
        for state in (0, 1):
            mask = states[:, class_index].long() == state
            if bool(mask.any().item()):
                output[mask, class_index] = (
                    ratios[mask, class_index]
                    / ratios[mask, class_index].mean().clamp_min(eps)
                )
    return output


def weight_summary(
    values: torch.Tensor,
    raw_reference: Optional[torch.Tensor] = None,
    lower_clip: Optional[float] = None,
    upper_clip: Optional[float] = None,
    eps: float = 1.0e-8,
) -> Dict[str, Any]:
    """Compute the Phase B scalar diagnostics for one ratio stage."""
    flat = values.detach().float().reshape(-1)
    finite = torch.isfinite(flat)
    clean = flat[finite]
    if clean.numel() == 0:
        raise ValueError("weight summary has no finite values")
    quantiles = torch.quantile(
        clean,
        torch.tensor(
            [0.01, 0.05, 0.50, 0.95, 0.99],
            dtype=clean.dtype,
            device=clean.device,
        ),
    )
    mean = clean.mean()
    std = clean.std(unbiased=False)
    ess = clean.sum().square() / clean.square().sum().clamp_min(eps)
    reference = flat if raw_reference is None else raw_reference.reshape(-1)
    raw_finite = torch.isfinite(reference)
    lower_rate = None
    upper_rate = None
    if lower_clip is not None:
        lower_rate = float(
            ((~raw_finite) | (reference < lower_clip)).float().mean().item()
        )
    if upper_clip is not None:
        upper_rate = float(
            ((~raw_finite) | (reference > upper_clip)).float().mean().item()
        )
    return {
        "count": int(flat.numel()),
        "finite_rate": float(finite.float().mean().item()),
        "mean": float(mean.item()),
        "std": float(std.item()),
        "cv": float(std.div(mean.abs().clamp_min(eps)).item()),
        "min": float(clean.min().item()),
        "max": float(clean.max().item()),
        "p01": float(quantiles[0].item()),
        "p05": float(quantiles[1].item()),
        "p50": float(quantiles[2].item()),
        "p95": float(quantiles[3].item()),
        "p99": float(quantiles[4].item()),
        "lower_clipping_rate": lower_rate,
        "upper_clipping_rate": upper_rate,
        "ess": float(ess.item()),
        "normalized_ess": float(ess.div(clean.numel()).item()),
    }


def _normalized_weights(
    weights: torch.Tensor, eps: float = 1.0e-8
) -> torch.Tensor:
    flat = weights.detach().double().reshape(-1)
    flat = torch.nan_to_num(flat, nan=0.0, posinf=0.0, neginf=0.0)
    flat = flat.clamp_min(0.0)
    return flat / flat.sum().clamp_min(eps)


def weighted_moment_errors(
    source_features: torch.Tensor,
    target_features: torch.Tensor,
    source_weights: torch.Tensor,
) -> Dict[str, float]:
    """Compare weighted source and target first/second moments."""
    source = source_features.detach().double().reshape(
        source_features.shape[0], -1
    )
    target = target_features.detach().double().reshape(
        target_features.shape[0], -1
    )
    if source_weights.ndim == 2:
        source_weights = source_weights.mean(dim=1)
    weights = _normalized_weights(source_weights).to(source.device)
    source_mean = (weights[:, None] * source).sum(dim=0)
    target_mean = target.mean(dim=0)
    source_centered = source - source_mean
    target_centered = target - target_mean
    source_covariance = (
        source_centered.transpose(0, 1)
        @ (source_centered * weights[:, None])
    )
    target_covariance = (
        target_centered.transpose(0, 1) @ target_centered
    ) / float(max(target.shape[0], 1))
    return {
        "weighted_feature_mean_error": float(
            torch.mean((source_mean - target_mean) ** 2).sqrt().item()
        ),
        "weighted_covariance_error": float(
            torch.mean((source_covariance - target_covariance) ** 2)
            .sqrt()
            .item()
        ),
    }


def weighted_mmd(
    source_features: torch.Tensor,
    target_features: torch.Tensor,
    source_weights: torch.Tensor,
    maximum_samples: int = 1024,
    seed: int = 0,
    eps: float = 1.0e-8,
) -> float:
    """Biased weighted RBF MMD with a deterministic median bandwidth."""
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    source_count = min(source_features.shape[0], maximum_samples)
    target_count = min(target_features.shape[0], maximum_samples)
    source_index = torch.randperm(
        source_features.shape[0], generator=generator
    )[:source_count]
    target_index = torch.randperm(
        target_features.shape[0], generator=generator
    )[:target_count]
    source = source_features[source_index].detach().float().reshape(
        source_count, -1
    )
    target = target_features[target_index].detach().float().reshape(
        target_count, -1
    )
    weights = source_weights[source_index]
    if weights.ndim == 2:
        weights = weights.mean(dim=1)
    weights = _normalized_weights(weights, eps=eps).float()
    combined = torch.cat((source, target), dim=0)
    probe = combined[: min(combined.shape[0], 512)]
    distances = torch.pdist(probe).square()
    positive = distances[distances > 0]
    bandwidth = (
        positive.median()
        if positive.numel() > 0
        else torch.tensor(1.0, device=combined.device)
    ).clamp_min(eps)

    def kernel(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        return torch.exp(-torch.cdist(left, right).square() / bandwidth)

    source_kernel = kernel(source, source)
    target_kernel = kernel(target, target)
    cross_kernel = kernel(source, target)
    target_weights = torch.full(
        (target_count,), 1.0 / float(target_count), device=target.device
    )
    value = (
        torch.dot(weights, torch.mv(source_kernel, weights))
        + torch.dot(target_weights, torch.mv(target_kernel, target_weights))
        - 2.0 * torch.dot(weights, torch.mv(cross_kernel, target_weights))
    )
    return float(value.clamp_min(0.0).sqrt().item())


def _weighted_auc(
    labels: np.ndarray, scores: np.ndarray, sample_weights: np.ndarray
) -> float:
    order = np.argsort(scores, kind="mergesort")
    labels = labels[order]
    weights = sample_weights[order]
    negative_total = weights[labels == 0].sum()
    positive_total = weights[labels == 1].sum()
    if negative_total <= 0 or positive_total <= 0:
        return float("nan")
    cumulative_negative = 0.0
    concordant = 0.0
    index = 0
    while index < labels.size:
        end = index + 1
        while end < labels.size and scores[order[end]] == scores[order[index]]:
            end += 1
        group_labels = labels[index:end]
        group_weights = weights[index:end]
        group_negative = group_weights[group_labels == 0].sum()
        group_positive = group_weights[group_labels == 1].sum()
        concordant += group_positive * (
            cumulative_negative + 0.5 * group_negative
        )
        cumulative_negative += group_negative
        index = end
    return float(concordant / (negative_total * positive_total))


def posthoc_domain_diagnostics(
    source_features: torch.Tensor,
    target_features: torch.Tensor,
    source_weights: torch.Tensor,
    seed: int,
    epochs: int = 100,
    learning_rate: float = 0.05,
) -> Dict[str, float]:
    """Fit an independent linear domain classifier on weighted source/target."""
    torch.manual_seed(int(seed))
    source = source_features.detach().float().reshape(
        source_features.shape[0], -1
    ).cpu()
    target = target_features.detach().float().reshape(
        target_features.shape[0], -1
    ).cpu()
    if source_weights.ndim == 2:
        source_weights = source_weights.mean(dim=1)
    source_weights = source_weights.detach().float().cpu().clamp_min(0.0)
    source_weights = source_weights / source_weights.mean().clamp_min(1.0e-8)
    features = torch.cat((source, target), dim=0)
    labels = torch.cat(
        (torch.zeros(source.shape[0]), torch.ones(target.shape[0]))
    )
    sample_weights = torch.cat(
        (source_weights, torch.ones(target.shape[0]))
    )
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed) + 193)
    permutation = torch.randperm(features.shape[0], generator=generator)
    validation_count = max(2, int(round(features.shape[0] * 0.25)))
    validation_index = permutation[:validation_count]
    train_index = permutation[validation_count:]
    train_features = features[train_index]
    validation_features = features[validation_index]
    feature_mean = train_features.mean(dim=0)
    feature_std = train_features.std(dim=0, unbiased=False).clamp_min(1.0e-4)
    train_features = (train_features - feature_mean) / feature_std
    validation_features = (validation_features - feature_mean) / feature_std
    model = nn.Linear(features.shape[1], 1)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=learning_rate, weight_decay=1.0e-4
    )
    for _ in range(epochs):
        optimizer.zero_grad()
        logits = model(train_features).squeeze(1)
        losses = F.binary_cross_entropy_with_logits(
            logits, labels[train_index], reduction="none"
        )
        loss = (
            losses * sample_weights[train_index]
        ).sum() / sample_weights[train_index].sum().clamp_min(1.0e-8)
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        probabilities = torch.sigmoid(
            model(validation_features).squeeze(1)
        )
    validation_labels = labels[validation_index]
    validation_weights = sample_weights[validation_index]
    accuracy = (
        ((probabilities >= 0.5) == validation_labels.bool()).float()
        * validation_weights
    ).sum() / validation_weights.sum().clamp_min(1.0e-8)
    return {
        "posthoc_domain_accuracy": float(accuracy.item()),
        "posthoc_domain_auc": _weighted_auc(
            validation_labels.numpy().astype(np.int64),
            probabilities.numpy(),
            validation_weights.numpy(),
        ),
    }


def distribution_matching_diagnostics(
    source_features: torch.Tensor,
    target_features: torch.Tensor,
    source_weights: torch.Tensor,
    seed: int,
) -> Dict[str, float]:
    output = weighted_moment_errors(
        source_features, target_features, source_weights
    )
    output["weighted_mmd"] = weighted_mmd(
        source_features, target_features, source_weights, seed=seed
    )
    output.update(
        posthoc_domain_diagnostics(
            source_features,
            target_features,
            source_weights,
            seed=seed + 100003,
        )
    )
    return output

