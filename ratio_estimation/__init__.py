"""Training and integration utilities for density-ratio estimation."""

from .risk import weighted_classwise_risk
from .conditional_masks import build_conditional_masks, valid_condition_mask
from .statistics import (
    NORMALIZATION_SCOPES,
    process_conditional_weights,
    process_weight_stages,
    summarize_weight_values,
    weight_stage_rows,
)
from .trainer import RatioEstimatorTrainer

__all__ = [
    "RatioEstimatorTrainer",
    "weighted_classwise_risk",
    "build_conditional_masks",
    "valid_condition_mask",
    "NORMALIZATION_SCOPES",
    "process_conditional_weights",
    "process_weight_stages",
    "summarize_weight_values",
    "weight_stage_rows",
]
