"""Direct conditional density-ratio architectures.

All architectures estimate the same object directly:

    p_target(x | bar_y[k] = b) / p_source(x | bar_y[k] = b)

The models deliberately accept only features and complementary-label vectors.
Ordinary labels are neither part of the interface nor stored on the model.
"""

import copy
import math
from typing import Any, Dict, Iterable, Optional, Sequence, Tuple

import torch
from torch import nn

from .ratio_estimator import LabelConditionedRatioEstimator, _mlp


def _ratio_encoder(
    input_dim: int, hidden_dims: Sequence[int]
) -> Tuple[nn.Module, int]:
    hidden_dims = tuple(int(value) for value in hidden_dims)
    if not hidden_dims:
        return nn.Identity(), int(input_dim)
    return (
        _mlp(input_dim, hidden_dims[:-1], hidden_dims[-1]),
        hidden_dims[-1],
    )


class _IndependentBranchEstimator(nn.Module):
    """One private backbone, encoder, and scalar output for a single branch."""

    def __init__(
        self,
        backbone: nn.Module,
        feature_dim: int,
        hidden_dims: Sequence[int],
    ) -> None:
        super().__init__()
        self.backbone = copy.deepcopy(backbone)
        self.feature_dim = int(feature_dim)
        self.encoder, representation_dim = _ratio_encoder(
            self.feature_dim, hidden_dims
        )
        self.output = nn.Linear(representation_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        if features.ndim > 2:
            features = features.flatten(start_dim=1)
        if features.ndim != 2 or features.shape[1] != self.feature_dim:
            raise ValueError("branch backbone must return [B, feature_dim]")
        return self.output(self.encoder(features)).squeeze(-1)


class SeparateConditionalRatioEstimator(LabelConditionedRatioEstimator):
    """Use ``2q`` completely independent ratio networks."""

    def __init__(
        self,
        backbone: nn.Module,
        num_classes: int,
        feature_dim: int,
        hidden_dims: Sequence[int] = (64, 32),
        **kwargs: Any,
    ) -> None:
        # Initialize the stable odds conversion and compatibility API, then
        # replace the legacy output module with private branch networks.
        super().__init__(
            backbone=nn.Identity(),
            num_classes=num_classes,
            feature_dim=feature_dim,
            hidden_dims=(),
            estimator_type="multihead",
            **kwargs,
        )
        self.multihead = None
        self.architecture = "separate"
        self.branch_estimators = nn.ModuleList(
            [
                nn.ModuleList(
                    [
                        _IndependentBranchEstimator(
                            backbone, feature_dim, hidden_dims
                        )
                        for _ in range(2)
                    ]
                )
                for _ in range(num_classes)
            ]
        )
        self._branch_initialized.zero_()

    def forward_all(self, x: torch.Tensor) -> torch.Tensor:
        rows = [
            torch.stack(
                [self.branch_estimators[k][b](x) for b in range(2)], dim=1
            )
            for k in range(self.num_classes)
        ]
        return torch.stack(rows, dim=1)

    def ratio_training_outputs(
        self, x: torch.Tensor, y_bar: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        if y_bar is None:
            return {"conditional_logits": self.forward_all(x)}
        states = self._validate_bar_y(y_bar, x.shape[0])
        # During estimation, a private branch sees only its strict conditional
        # subset. The dense tensor is assembled solely for the common trainer.
        logits = x.new_zeros((x.shape[0], self.num_classes, 2))
        for k in range(self.num_classes):
            for b in range(2):
                mask = states[:, k] == b
                if bool(mask.any().item()):
                    branch_logits = self.branch_estimators[k][b](x[mask])
                    # CUDA AMP produces FP16 branch outputs while ``x`` and
                    # the dense compatibility tensor remain FP32. Explicitly
                    # cast before index_put; this operation remains
                    # differentiable and is required by PyTorch 1.8.
                    logits[mask, k, b] = branch_logits.to(dtype=logits.dtype)
        return {"conditional_logits": logits}

    def branch_parameters(self, k: int, b: int) -> Iterable[nn.Parameter]:
        return self.branch_estimators[k][b].parameters()


class MultiHeadConditionalRatioEstimator(LabelConditionedRatioEstimator):
    """Share a ratio encoder and use one scalar output head per ``(k,b)``."""

    def __init__(
        self,
        backbone: nn.Module,
        num_classes: int,
        feature_dim: int,
        hidden_dims: Sequence[int] = (64, 32),
        **kwargs: Any,
    ) -> None:
        super().__init__(
            backbone=backbone,
            num_classes=num_classes,
            feature_dim=feature_dim,
            hidden_dims=(),
            estimator_type="multihead",
            **kwargs,
        )
        self.multihead = None
        self.architecture = "multihead"
        self.ratio_encoder, representation_dim = _ratio_encoder(
            feature_dim, hidden_dims
        )
        self.ratio_heads = nn.ModuleList(
            [
                nn.ModuleList(
                    [nn.Linear(representation_dim, 1) for _ in range(2)]
                )
                for _ in range(num_classes)
            ]
        )
        self._branch_initialized.zero_()

    def forward_all(self, x: torch.Tensor) -> torch.Tensor:
        z = self.ratio_encoder(self._features(x))
        rows = [
            torch.cat([self.ratio_heads[k][b](z) for b in range(2)], dim=1)
            for k in range(self.num_classes)
        ]
        return torch.stack(rows, dim=1)

    def branch_parameters(self, k: int, b: int) -> Iterable[nn.Parameter]:
        return self.ratio_heads[k][b].parameters()


class FusionConditionalRatioEstimator(LabelConditionedRatioEstimator):
    """Learn global/conditional log-ratios and sample-dependent gates."""

    def __init__(
        self,
        backbone: nn.Module,
        num_classes: int,
        feature_dim: int,
        hidden_dims: Sequence[int] = (64, 32),
        gate_regularization: float = 0.0,
        gate_init: float = 0.7,
        detach_global_for_gate: bool = False,
        **kwargs: Any,
    ) -> None:
        if not 0.0 < gate_init < 1.0:
            raise ValueError("fusion gate_init must be strictly between 0 and 1")
        if gate_regularization < 0:
            raise ValueError("fusion gate_regularization cannot be negative")
        super().__init__(
            backbone=backbone,
            num_classes=num_classes,
            feature_dim=feature_dim,
            hidden_dims=(),
            estimator_type="multihead",
            **kwargs,
        )
        self.multihead = None
        self.architecture = "fusion"
        self.ratio_encoder, representation_dim = _ratio_encoder(
            feature_dim, hidden_dims
        )
        self.global_head = nn.Linear(representation_dim, 1)
        self.conditional_heads = nn.ModuleList(
            [
                nn.ModuleList(
                    [nn.Linear(representation_dim, 1) for _ in range(2)]
                )
                for _ in range(num_classes)
            ]
        )
        self.gate_heads = nn.ModuleList(
            [
                nn.ModuleList(
                    [nn.Linear(representation_dim, 1) for _ in range(2)]
                )
                for _ in range(num_classes)
            ]
        )
        gate_bias = math.log(gate_init / (1.0 - gate_init))
        for row in self.gate_heads:
            for head in row:
                nn.init.zeros_(head.weight)
                nn.init.constant_(head.bias, gate_bias)
        self.gate_regularization = float(gate_regularization)
        self.gate_init = float(gate_init)
        self.detach_global_for_gate = bool(detach_global_for_gate)
        self.register_buffer("_global_log_prior_correction", torch.zeros(()))
        self._branch_initialized.zero_()

    def _components(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        z = self.ratio_encoder(self._features(x))
        global_logit = self.global_head(z).squeeze(-1)
        conditional = torch.stack(
            [
                torch.cat(
                    [self.conditional_heads[k][b](z) for b in range(2)],
                    dim=1,
                )
                for k in range(self.num_classes)
            ],
            dim=1,
        )
        gates = torch.stack(
            [
                torch.cat(
                    [torch.sigmoid(self.gate_heads[k][b](z)) for b in range(2)],
                    dim=1,
                )
                for k in range(self.num_classes)
            ],
            dim=1,
        )
        global_for_fusion = (
            global_logit.detach() if self.detach_global_for_gate else global_logit
        )
        fused = (
            gates * conditional
            + (1.0 - gates) * global_for_fusion[:, None, None]
        )
        return global_logit, conditional, gates, fused

    def forward_all(self, x: torch.Tensor) -> torch.Tensor:
        return self._components(x)[3]

    def forward_global(self, x: torch.Tensor) -> torch.Tensor:
        return self._components(x)[0]

    def ratio_training_outputs(
        self, x: torch.Tensor, y_bar: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        global_logit, conditional, gates, fused = self._components(x)
        return {
            "conditional_logits": fused,
            "global_logits": global_logit,
            "raw_conditional_logits": conditional,
            "gates": gates,
        }

    def fusion_regularization(self, gates: torch.Tensor) -> torch.Tensor:
        if self.gate_regularization == 0.0:
            return gates.sum() * 0.0
        return self.gate_regularization * (gates - self.gate_init).square().mean()

    def branch_parameters(self, k: int, b: int) -> Iterable[nn.Parameter]:
        return list(self.conditional_heads[k][b].parameters()) + list(
            self.gate_heads[k][b].parameters()
        )

    @torch.no_grad()
    def set_global_domain_counts(
        self, source_count: int, target_count: int, balanced_sampling: bool
    ) -> None:
        if source_count <= 0 or target_count <= 0:
            self._global_log_prior_correction.zero_()
        elif balanced_sampling:
            self._global_log_prior_correction.zero_()
        else:
            value = math.log(float(source_count) / float(target_count))
            self._global_log_prior_correction.fill_(value)

    @torch.no_grad()
    def extra_ratio_statistics(self, x: torch.Tensor) -> Dict[str, Any]:
        global_logit, conditional_logit, gates, fused_logit = self._components(x)
        global_ratio = torch.exp(
            torch.clamp(
                global_logit.float() + self._global_log_prior_correction,
                self.log_ratio_clip_min,
                self.log_ratio_clip_max,
            )
        )
        conditional_ratio = self.ratios_from_logits(conditional_logit)
        fused_ratio = self.ratios_from_logits(fused_logit)

        def distribution(values: torch.Tensor) -> Dict[str, float]:
            flat = values.float().reshape(-1)
            return {
                "mean": float(flat.mean().item()),
                "std": float(flat.std(unbiased=False).item()),
                "min": float(flat.min().item()),
                "max": float(flat.max().item()),
            }

        return {
            "gate_mean": gates.mean(dim=0).cpu().tolist(),
            "gate_std": gates.std(dim=0, unbiased=False).cpu().tolist(),
            "gate_variance": gates.var(dim=0, unbiased=False).cpu().tolist(),
            "gate_histogram": torch.histc(
                gates.float().cpu(), bins=10, min=0.0, max=1.0
            ).long().tolist(),
            "global_ratio": distribution(global_ratio),
            "conditional_ratio": distribution(conditional_ratio),
            "fused_ratio": distribution(fused_ratio),
            "global_conditional_log_difference": float(
                (
                    conditional_logit
                    - global_logit[:, None, None]
                ).abs().mean().item()
            ),
        }
