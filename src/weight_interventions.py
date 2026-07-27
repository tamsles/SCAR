"""Classifier-side interventions for the ADIW causal decomposition."""

from typing import Optional

import torch


INTERVENTIONS = ("original", "ones", "mean", "shuffle")


def apply_weight_intervention(
    weights: torch.Tensor,
    intervention: str,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """Return classifier-used weights while leaving ratio updates untouched.

    Shuffle uses one row permutation for the complete ``[B, q]`` tensor.  It
    therefore preserves every class-wise weight distribution, its scale, and
    ESS while breaking the sample-to-weight correspondence.
    """
    if weights.ndim != 2:
        raise ValueError("weights must have shape [batch, classes]")
    if intervention not in INTERVENTIONS:
        raise ValueError(
            "intervention must be one of {}".format(", ".join(INTERVENTIONS))
        )
    if intervention == "original":
        return weights
    if intervention == "ones":
        return torch.ones_like(weights)
    if intervention == "mean":
        return torch.ones_like(weights) * weights.mean()
    if generator is None:
        raise ValueError("shuffle requires an independent generator")
    permutation = torch.randperm(
        weights.shape[0], generator=generator, device="cpu"
    ).to(weights.device)
    return weights.index_select(0, permutation)


def classifier_method_contract(method: str) -> str:
    """Map a Phase B method name to its classifier-side intervention."""
    mapping = {
        "adiw_original": "original",
        "adiw_ones": "ones",
        "adiw_mean": "mean",
        "adiw_shuffle": "shuffle",
    }
    if method not in mapping:
        raise ValueError("method has no ADIW intervention: {}".format(method))
    return mapping[method]

