"""Complementary-label-only conditional subset construction."""

from typing import Dict, Tuple

import torch


def validate_complementary_labels(
    y_bar: torch.Tensor, num_classes: int
) -> torch.Tensor:
    """Validate and return binary states without accepting ordinary labels."""
    if y_bar.ndim != 2 or y_bar.shape[1] != num_classes:
        raise ValueError("y_bar must have shape [B,q]")
    states = y_bar.long()
    if not bool(((states == 0) | (states == 1)).all().item()):
        raise ValueError("y_bar must contain only 0/1")
    return states


def build_conditional_masks(
    source_y_bar: torch.Tensor,
    target_y_bar: torch.Tensor,
    num_classes: int,
) -> Dict[Tuple[int, int], Tuple[torch.Tensor, torch.Tensor]]:
    """Return strict source/target masks for every ``(k,b)`` branch."""
    source = validate_complementary_labels(source_y_bar, num_classes)
    target = validate_complementary_labels(target_y_bar, num_classes)
    return {
        (k, b): (source[:, k] == b, target[:, k] == b)
        for k in range(num_classes)
        for b in range(2)
    }


def valid_condition_mask(
    source_y_bar: torch.Tensor,
    target_y_bar: torch.Tensor,
    num_classes: int,
    min_condition_samples: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return validity plus source/target counts, all shaped ``[q,2]``."""
    if min_condition_samples < 1:
        raise ValueError("min_condition_samples must be at least one")
    masks = build_conditional_masks(
        source_y_bar, target_y_bar, num_classes
    )
    source_count = torch.zeros(
        num_classes, 2, dtype=torch.long, device=source_y_bar.device
    )
    target_count = torch.zeros(
        num_classes, 2, dtype=torch.long, device=target_y_bar.device
    )
    for (k, b), (source_mask, target_mask) in masks.items():
        source_count[k, b] = source_mask.sum()
        target_count[k, b] = target_mask.sum()
    valid = (
        (source_count >= min_condition_samples)
        & (target_count >= min_condition_samples)
    )
    return valid, source_count, target_count

