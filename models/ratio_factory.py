"""Factory isolating ratio-architecture configuration from training code."""

from typing import Any, Dict, Optional

import torch

from .backbones import VectorMLPBackbone
from .conditional_ratio import (
    FusionConditionalRatioEstimator,
    MultiHeadConditionalRatioEstimator,
    SeparateConditionalRatioEstimator,
)
from .ratio_estimator import LabelConditionedRatioEstimator


def _section(config: Dict[str, Any], name: str) -> Dict[str, Any]:
    value = config.get(name, {})
    return value if isinstance(value, dict) else {}


def build_ratio_estimator(
    config: Dict[str, Any],
    backbone: Optional[torch.nn.Module] = None,
) -> LabelConditionedRatioEstimator:
    """Build a legacy or direct-conditional estimator from one configuration."""
    if backbone is None:
        backbone = VectorMLPBackbone(
            input_dim=int(config["input_dim"]),
            feature_dim=int(config["feature_dim"]),
            hidden_dims=config.get("backbone_hidden_dims", [64, 64]),
        )
    ratio_config = _section(config, "ratio")
    fusion_config = _section(config, "fusion")
    # Missing ratio_arch intentionally preserves the previous implementation.
    architecture = str(config.get("ratio_arch", "unified")).lower()
    common = {
        "backbone": backbone,
        "num_classes": int(config["num_classes"]),
        "feature_dim": int(config["feature_dim"]),
        "hidden_dims": config.get("hidden_dims", [64, 32]),
        "balanced_domain_sampling": bool(
            config.get("balanced_domain_sampling", False)
        ),
        "log_ratio_clip_min": float(config.get("log_ratio_clip_min", -10.0)),
        "log_ratio_clip_max": float(config.get("log_ratio_clip_max", 10.0)),
        "ratio_clip_min": ratio_config.get(
            "min_weight", config.get("ratio_clip_min", 1.0e-6)
        ),
        "ratio_clip_max": ratio_config.get(
            "max_weight", config.get("ratio_clip_max", 1.0e6)
        ),
        "self_normalize": bool(
            ratio_config.get(
                "self_normalize", config.get("self_normalize", False)
            )
        ),
        "joint_training": not bool(
            ratio_config.get("detach_classifier_update", True)
        ),
    }
    if architecture == "unified":
        return LabelConditionedRatioEstimator(
            class_embedding_dim=int(config.get("class_embedding_dim", 16)),
            state_embedding_dim=int(config.get("state_embedding_dim", 4)),
            estimator_type=str(config.get("estimator_type", "conditioned")),
            **common,
        )
    if architecture == "separate":
        return SeparateConditionalRatioEstimator(**common)
    if architecture == "multihead":
        return MultiHeadConditionalRatioEstimator(**common)
    if architecture == "fusion":
        return FusionConditionalRatioEstimator(
            gate_regularization=float(
                fusion_config.get("gate_regularization", 0.0)
            ),
            gate_init=float(fusion_config.get("gate_init", 0.7)),
            detach_global_for_gate=bool(
                fusion_config.get("detach_global_for_gate", False)
            ),
            **common,
        )
    raise ValueError(
        "ratio_arch must be one of unified, separate, multihead, fusion"
    )

