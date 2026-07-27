"""Held-out domain-label calibration without target class labels."""

from typing import Dict

import torch
import torch.nn.functional as F


def _flatten(
    source_logits: torch.Tensor, target_logits: torch.Tensor
) -> tuple:
    source = source_logits.detach().double().reshape(-1)
    target = target_logits.detach().double().reshape(-1)
    logits = torch.cat((source, target))
    labels = torch.cat((torch.zeros_like(source), torch.ones_like(target)))
    return logits, labels


def fit_temperature_scaling(
    source_logits: torch.Tensor,
    target_logits: torch.Tensor,
    iterations: int = 100,
) -> Dict[str, float]:
    """Fit a positive temperature using held-out domain labels only."""
    logits, labels = _flatten(source_logits, target_logits)
    log_temperature = torch.zeros(
        (), dtype=torch.double, requires_grad=True
    )
    optimizer = torch.optim.LBFGS(
        [log_temperature], lr=0.25, max_iter=iterations
    )

    def closure() -> torch.Tensor:
        optimizer.zero_grad()
        temperature = log_temperature.exp().clamp(0.05, 20.0)
        loss = F.binary_cross_entropy_with_logits(
            logits / temperature, labels
        )
        loss.backward()
        return loss

    optimizer.step(closure)
    return {
        "method": "temperature",
        "temperature": float(
            log_temperature.detach().exp().clamp(0.05, 20.0).item()
        ),
        "slope": None,
        "intercept": None,
    }


def fit_platt_scaling(
    source_logits: torch.Tensor,
    target_logits: torch.Tensor,
    iterations: int = 100,
) -> Dict[str, float]:
    """Fit Platt slope/intercept using held-out domain labels only."""
    logits, labels = _flatten(source_logits, target_logits)
    slope = torch.ones((), dtype=torch.double, requires_grad=True)
    intercept = torch.zeros((), dtype=torch.double, requires_grad=True)
    optimizer = torch.optim.LBFGS(
        [slope, intercept], lr=0.25, max_iter=iterations
    )

    def closure() -> torch.Tensor:
        optimizer.zero_grad()
        calibrated = slope * logits + intercept
        loss = F.binary_cross_entropy_with_logits(calibrated, labels)
        loss.backward()
        return loss

    optimizer.step(closure)
    return {
        "method": "platt",
        "temperature": None,
        "slope": float(slope.detach().item()),
        "intercept": float(intercept.detach().item()),
    }


def apply_calibration(
    logits: torch.Tensor, parameters: Dict[str, float]
) -> torch.Tensor:
    method = parameters["method"]
    if method == "none":
        return logits
    if method == "temperature":
        return logits / float(parameters["temperature"])
    if method == "platt":
        return (
            float(parameters["slope"]) * logits
            + float(parameters["intercept"])
        )
    raise ValueError("unknown calibration method: {}".format(method))

