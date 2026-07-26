"""Downstream class-wise complementary-risk integration."""

from typing import Any, Dict, Optional, Tuple, Union

import torch

from models.ratio_estimator import LabelConditionedRatioEstimator
from .statistics import process_conditional_weights


def weighted_classwise_risk(
    estimator: LabelConditionedRatioEstimator,
    x: torch.Tensor,
    bar_y: torch.Tensor,
    classwise_loss: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    joint_training: bool = False,
    allow_ratio_gradients: bool = False,
    ratio_config: Optional[Dict[str, Any]] = None,
    return_statistics: bool = False,
) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, Dict[str, Any]]]]:
    """Apply ``[B,q]`` ratios before reducing a class-wise loss.

    Even with ``joint_training=True``, weights remain detached unless
    ``allow_ratio_gradients`` is explicitly enabled. This guard prevents the
    classifier objective from obtaining a degenerate solution by shrinking
    the estimated ratios.
    """
    if classwise_loss.shape != bar_y.shape:
        raise ValueError("classwise_loss and bar_y must both have shape [B, q]")
    detach = not (joint_training and allow_ratio_gradients)
    config = ratio_config or {}
    default_minimum = (
        estimator.ratio_clip_min
        if estimator.ratio_clip_min is not None
        else 0.01
    )
    default_maximum = (
        estimator.ratio_clip_max
        if estimator.ratio_clip_max is not None
        else 20.0
    )
    weights = estimator.get_conditional_weights(
        x, bar_y, detach=False, self_normalize=False
    )
    weights, statistics = process_conditional_weights(
        weights,
        bar_y,
        min_weight=float(config.get("min_weight", default_minimum)),
        max_weight=float(config.get("max_weight", default_maximum)),
        self_normalize=bool(
            config.get(
                "self_normalize", estimator.default_self_normalize
            )
        ),
        normalization_scope=str(
            config.get("normalization_scope", "conditional_branch")
        ),
        eps=float(config.get("eps", 1.0e-8)),
    )
    if detach:
        weights = weights.detach()
    weighted = weights * classwise_loss
    if mask is None:
        risk = weighted.mean()
        return (risk, statistics) if return_statistics else risk
    if mask.shape != weighted.shape:
        raise ValueError("mask must have shape [B, q]")
    mask_float = mask.to(dtype=weighted.dtype)
    risk = (weighted * mask_float).sum() / mask_float.sum().clamp_min(1.0)
    return (risk, statistics) if return_statistics else risk
