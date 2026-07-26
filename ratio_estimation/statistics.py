"""Branch-wise weight processing and machine-readable diagnostics.

The next-round experiments keep the four weight stages separate:

``raw -> clipped -> normalized -> classifier_used``.

Keeping these tensors distinct prevents diagnostics from accidentally
describing a normalized value as an estimated density ratio.
"""

from typing import Any, Dict, List, Optional, Tuple

import torch

from .conditional_masks import validate_complementary_labels


QUANTILE_LEVELS = (0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99)
NORMALIZATION_SCOPES = (
    "conditional_branch",
    "class_level",
    "global",
    "none",
)


def _rankdata(values: torch.Tensor) -> torch.Tensor:
    """Return deterministic average-free ranks for correlation diagnostics."""
    order = torch.argsort(values)
    ranks = torch.empty_like(values, dtype=torch.float32)
    ranks[order] = torch.arange(
        values.numel(), device=values.device, dtype=torch.float32
    )
    return ranks


def _spearman(left: torch.Tensor, right: torch.Tensor) -> Optional[float]:
    finite = torch.isfinite(left) & torch.isfinite(right)
    left = left[finite].float()
    right = right[finite].float()
    if left.numel() < 2:
        return None
    left_rank = _rankdata(left)
    right_rank = _rankdata(right)
    left_rank = left_rank - left_rank.mean()
    right_rank = right_rank - right_rank.mean()
    denominator = (
        left_rank.square().sum().sqrt()
        * right_rank.square().sum().sqrt()
    )
    if denominator.item() <= torch.finfo(torch.float32).eps:
        return None
    return float((left_rank * right_rank).sum().div(denominator).item())


def summarize_weight_values(
    values: torch.Tensor,
    eps: float = 1.0e-8,
    raw_reference: Optional[torch.Tensor] = None,
    min_weight: Optional[float] = None,
    max_weight: Optional[float] = None,
    normalization_scale: Optional[torch.Tensor] = None,
) -> Dict[str, Any]:
    """Summarize one ratio stage without mutating or sanitizing the input."""
    if values.numel() == 0:
        empty = {
            "count": 0,
            "mean": None,
            "std": None,
            "min": None,
            "max": None,
            "q01": None,
            "q05": None,
            "q25": None,
            "q50": None,
            "q75": None,
            "q95": None,
            "q99": None,
            "lower_clipping_rate": None,
            "upper_clipping_rate": None,
            "finite_ratio_rate": None,
            "log_ratio_mean": None,
            "log_ratio_std": None,
            "ess": 0.0,
            "mean_absolute_deviation_from_one": None,
            "raw_spearman": None,
            "normalization_scale_factor": None,
            "histogram": None,
            "histogram_min": None,
            "histogram_max": None,
        }
        empty["clipping_rate"] = None
        return empty

    values_float = values.float().reshape(-1)
    finite_mask = torch.isfinite(values_float)
    finite_values = values_float[finite_mask]
    finite_rate = float(finite_mask.float().mean().item())
    if finite_values.numel() == 0:
        output = summarize_weight_values(values_float[:0], eps=eps)
        output["count"] = int(values_float.numel())
        output["finite_ratio_rate"] = finite_rate
        return output

    quantile_tensor = torch.tensor(
        QUANTILE_LEVELS, device=finite_values.device, dtype=finite_values.dtype
    )
    quantiles = torch.quantile(finite_values, quantile_tensor)
    weight_sum = finite_values.sum()
    ess = weight_sum.square() / finite_values.square().sum().clamp_min(eps)
    positive = finite_values[finite_values > 0]
    if positive.numel() == 0:
        log_mean = None
        log_std = None
    else:
        log_values = positive.log()
        log_mean = float(log_values.mean().item())
        log_std = float(log_values.std(unbiased=False).item())

    raw = values_float if raw_reference is None else raw_reference.float().reshape(-1)
    raw_finite = torch.isfinite(raw)
    if min_weight is None:
        lower_rate = None
    else:
        lower_rate = float(((~raw_finite) | (raw < min_weight)).float().mean().item())
    if max_weight is None:
        upper_rate = None
    else:
        upper_rate = float(((~raw_finite) | (raw > max_weight)).float().mean().item())
    clipping_rate = None
    if lower_rate is not None and upper_rate is not None:
        clipping_rate = float(
            (
                (~raw_finite)
                | (raw < float(min_weight))
                | (raw > float(max_weight))
            ).float().mean().item()
        )
    histogram = torch.histc(
        finite_values.cpu(), bins=20,
        min=float(finite_values.min().item()),
        max=float(finite_values.max().item())
        if finite_values.max().item() > finite_values.min().item()
        else float(finite_values.min().item()) + 1.0,
    )
    scale = None
    if normalization_scale is not None and normalization_scale.numel() > 0:
        finite_scale = normalization_scale.float().reshape(-1)
        finite_scale = finite_scale[torch.isfinite(finite_scale)]
        if finite_scale.numel() > 0:
            scale = float(finite_scale.mean().item())
    raw_spearman = None
    if raw_reference is not None and raw.numel() == values_float.numel():
        raw_spearman = _spearman(raw, values_float)
    output = {
        "count": int(values_float.numel()),
        "mean": float(finite_values.mean().item()),
        "std": float(finite_values.std(unbiased=False).item()),
        "min": float(finite_values.min().item()),
        "max": float(finite_values.max().item()),
        "q01": float(quantiles[0].item()),
        "q05": float(quantiles[1].item()),
        "q25": float(quantiles[2].item()),
        "q50": float(quantiles[3].item()),
        "q75": float(quantiles[4].item()),
        "q95": float(quantiles[5].item()),
        "q99": float(quantiles[6].item()),
        "lower_clipping_rate": lower_rate,
        "upper_clipping_rate": upper_rate,
        "clipping_rate": clipping_rate,
        "finite_ratio_rate": finite_rate,
        "log_ratio_mean": log_mean,
        "log_ratio_std": log_std,
        "ess": float(ess.item()),
        "mean_absolute_deviation_from_one": float(
            (finite_values - 1.0).abs().mean().item()
        ),
        "raw_spearman": raw_spearman,
        "normalization_scale_factor": scale,
        "histogram": ";".join(str(int(value)) for value in histogram.tolist()),
        "histogram_min": float(finite_values.min().item()),
        "histogram_max": float(finite_values.max().item())
        if finite_values.max().item() > finite_values.min().item()
        else float(finite_values.min().item()) + 1.0,
    }
    return output


def process_weight_stages(
    weights: torch.Tensor,
    y_bar: torch.Tensor,
    min_weight: float = 0.01,
    max_weight: float = 20.0,
    normalization_scope: str = "conditional_branch",
    eps: float = 1.0e-8,
) -> Dict[str, torch.Tensor]:
    """Return separate raw, clipped, normalized, and used weight tensors."""
    if weights.ndim != 2:
        raise ValueError("weights must have shape [B,q]")
    if min_weight <= 0 or max_weight < min_weight:
        raise ValueError("invalid positive weight bounds")
    if eps <= 0:
        raise ValueError("eps must be positive")
    if normalization_scope not in NORMALIZATION_SCOPES:
        raise ValueError(
            "normalization_scope must be one of {}".format(
                ", ".join(NORMALIZATION_SCOPES)
            )
        )
    states = validate_complementary_labels(y_bar, weights.shape[1])
    raw = weights.float()
    safe = torch.nan_to_num(
        raw, nan=1.0, posinf=max_weight, neginf=min_weight
    )
    clipped = safe.clamp(min=min_weight, max=max_weight)
    normalized = clipped.clone()
    scales = torch.ones_like(clipped)
    if normalization_scope == "global":
        scale = clipped.mean().clamp_min(eps)
        normalized = clipped / scale
        scales.fill_(float(scale.detach().item()))
    elif normalization_scope == "class_level":
        class_scales = clipped.mean(dim=0).clamp_min(eps)
        normalized = clipped / class_scales.unsqueeze(0)
        scales = class_scales.unsqueeze(0).expand_as(clipped)
    elif normalization_scope == "conditional_branch":
        for k in range(weights.shape[1]):
            for b in range(2):
                mask = states[:, k] == b
                if bool(mask.any().item()):
                    branch_mean = clipped[mask, k].mean().clamp_min(eps)
                    normalized[mask, k] = clipped[mask, k] / branch_mean
                    scales[mask, k] = branch_mean
    return {
        "raw": raw,
        "clipped": clipped,
        "normalized": normalized,
        "classifier_used": normalized,
        "normalization_scale": scales,
    }


def weight_stage_rows(
    stages: Dict[str, torch.Tensor],
    y_bar: torch.Tensor,
    epoch: int,
    min_weight: float,
    max_weight: float,
    domain: str = "source",
    eps: float = 1.0e-8,
) -> Dict[str, List[Dict[str, Any]]]:
    """Build per-branch CSV rows for every weight stage."""
    states = validate_complementary_labels(
        y_bar, stages["raw"].shape[1]
    )
    rows = {
        "raw": [],
        "clipped": [],
        "normalized": [],
        "classifier_used": [],
    }
    for stage_name in rows:
        tensor = stages[stage_name]
        for k in range(tensor.shape[1]):
            for b in range(2):
                mask = states[:, k] == b
                raw_branch = stages["raw"][mask, k]
                scale_branch = stages["normalization_scale"][mask, k]
                summary = summarize_weight_values(
                    tensor[mask, k],
                    eps=eps,
                    raw_reference=raw_branch,
                    min_weight=min_weight,
                    max_weight=max_weight,
                    normalization_scale=scale_branch,
                )
                summary.update(
                    {
                        "epoch": int(epoch),
                        "domain": domain,
                        "class_index": int(k),
                        "state": int(b),
                        "stage": stage_name,
                    }
                )
                rows[stage_name].append(summary)
    return rows


def process_conditional_weights(
    weights: torch.Tensor,
    y_bar: torch.Tensor,
    min_weight: float = 0.01,
    max_weight: float = 20.0,
    self_normalize: bool = True,
    normalization_scope: str = "conditional_branch",
    eps: float = 1.0e-8,
) -> Tuple[torch.Tensor, Dict[str, Dict[str, Any]]]:
    """Sanitize, clip, then optionally normalize observed source branches."""
    effective_scope = normalization_scope if self_normalize else "none"
    stages = process_weight_stages(
        weights,
        y_bar,
        min_weight=min_weight,
        max_weight=max_weight,
        normalization_scope=effective_scope,
        eps=eps,
    )
    states = validate_complementary_labels(y_bar, weights.shape[1])
    processed = stages["classifier_used"]
    statistics = {}
    for k in range(weights.shape[1]):
        for b in range(2):
            mask = states[:, k] == b
            statistics["{}_{}".format(k, b)] = summarize_weight_values(
                processed[mask, k],
                eps=eps,
                raw_reference=stages["raw"][mask, k],
                min_weight=min_weight,
                max_weight=max_weight,
                normalization_scale=stages["normalization_scale"][mask, k],
            )
    return processed.to(dtype=weights.dtype), statistics
