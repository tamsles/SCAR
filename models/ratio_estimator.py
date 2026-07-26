"""Shared-backbone, label-conditioned density-ratio estimator."""

import math
import warnings
from typing import Any, Dict, Optional, Sequence, Tuple

import torch
from torch import nn


def _mlp(input_dim: int, hidden_dims: Sequence[int], output_dim: int) -> nn.Sequential:
    dims = [input_dim] + list(hidden_dims) + [output_dim]
    layers = []
    for index in range(len(dims) - 1):
        layers.append(nn.Linear(dims[index], dims[index + 1]))
        if index < len(dims) - 2:
            layers.append(nn.ReLU(inplace=True))
    return nn.Sequential(*layers)


class LabelConditionedRatioEstimator(nn.Module):
    """Estimate all class/state-specific target-to-source density ratios.

    ``backbone`` is executed exactly once by :meth:`forward_all`.  In
    ``conditioned`` mode, class/state embeddings and a shared MLP produce all
    ``2q`` logits using tensor broadcasting.  In ``multihead`` mode, one shared
    backbone and one output head produce the same tensor directly.
    """

    def __init__(
        self,
        backbone: nn.Module,
        num_classes: int,
        feature_dim: int,
        class_embedding_dim: int = 16,
        state_embedding_dim: int = 4,
        hidden_dims: Sequence[int] = (64, 32),
        estimator_type: str = "conditioned",
        balanced_domain_sampling: bool = False,
        log_ratio_clip_min: float = -10.0,
        log_ratio_clip_max: float = 10.0,
        ratio_clip_min: Optional[float] = 1.0e-6,
        ratio_clip_max: Optional[float] = 1.0e6,
        self_normalize: bool = False,
        joint_training: bool = False,
    ) -> None:
        super().__init__()
        if num_classes <= 0 or feature_dim <= 0:
            raise ValueError("num_classes and feature_dim must be positive")
        if estimator_type not in ("conditioned", "multihead"):
            raise ValueError("estimator_type must be 'conditioned' or 'multihead'")
        if log_ratio_clip_min >= log_ratio_clip_max:
            raise ValueError("log_ratio_clip_min must be smaller than log_ratio_clip_max")
        if ratio_clip_min is not None and ratio_clip_min <= 0:
            raise ValueError("ratio_clip_min must be strictly positive")
        if (
            ratio_clip_min is not None
            and ratio_clip_max is not None
            and ratio_clip_min > ratio_clip_max
        ):
            raise ValueError("ratio_clip_min cannot exceed ratio_clip_max")

        self.backbone = backbone
        self.num_classes = int(num_classes)
        self.feature_dim = int(feature_dim)
        self.estimator_type = estimator_type
        self.balanced_domain_sampling = bool(balanced_domain_sampling)
        self.log_ratio_clip_min = float(log_ratio_clip_min)
        self.log_ratio_clip_max = float(log_ratio_clip_max)
        self.ratio_clip_min = ratio_clip_min
        self.ratio_clip_max = ratio_clip_max
        self.default_self_normalize = bool(self_normalize)
        self.joint_training = bool(joint_training)
        self.architecture = "unified"

        if estimator_type == "conditioned":
            if class_embedding_dim < 0 or state_embedding_dim < 0:
                raise ValueError("embedding dimensions cannot be negative")
            self.class_embedding = (
                nn.Embedding(num_classes, class_embedding_dim)
                if class_embedding_dim > 0
                else None
            )
            self.state_embedding = (
                nn.Embedding(2, state_embedding_dim)
                if state_embedding_dim > 0
                else None
            )
            head_input_dim = feature_dim + class_embedding_dim + state_embedding_dim
            self.ratio_head = _mlp(head_input_dim, hidden_dims, 1)
            self.multihead = None
        else:
            self.class_embedding = None
            self.state_embedding = None
            self.ratio_head = None
            self.multihead = _mlp(feature_dim, hidden_dims, num_classes * 2)

        # log(pi_source / pi_target), one value for every (k, z) branch.
        self.register_buffer("_log_prior_correction", torch.zeros(num_classes, 2))
        self.register_buffer(
            "_prior_supported", torch.ones(num_classes, 2, dtype=torch.bool)
        )
        # The legacy unified estimator is immediately usable. New conditional
        # architectures override this buffer to implement the unit-weight
        # fallback until a branch has received a valid source/target update.
        self.register_buffer(
            "_branch_initialized", torch.ones(num_classes, 2, dtype=torch.bool)
        )
        self.register_buffer(
            "_skipped_update_count", torch.zeros(num_classes, 2, dtype=torch.long)
        )

    def _features(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        if not torch.is_tensor(features):
            raise TypeError("backbone must return a Tensor")
        if features.ndim > 2:
            features = features.flatten(start_dim=1)
        if features.ndim != 2 or features.shape[1] != self.feature_dim:
            raise ValueError(
                "backbone output must have shape [B, feature_dim={}]".format(
                    self.feature_dim
                )
            )
        return features

    def forward_all(self, x: torch.Tensor) -> torch.Tensor:
        """Return domain-classification logits with shape ``[B, q, 2]``."""
        features = self._features(x)  # [B, F], one shared-backbone call.
        batch_size = features.shape[0]

        if self.estimator_type == "multihead":
            logits = self.multihead(features)
            return logits.reshape(batch_size, self.num_classes, 2)

        device = features.device
        class_ids = torch.arange(self.num_classes, device=device)
        state_ids = torch.arange(2, device=device)
        # Broadcast to [B, q, 2, *]. No Python loop over class/state branches.
        h = features[:, None, None, :].expand(-1, self.num_classes, 2, -1)
        inputs = [h]
        if self.class_embedding is not None:
            class_features = self.class_embedding(class_ids)  # [q, C]
            inputs.append(
                class_features[None, :, None, :].expand(batch_size, -1, 2, -1)
            )
        if self.state_embedding is not None:
            state_features = self.state_embedding(state_ids)  # [2, S]
            inputs.append(
                state_features[None, None, :, :].expand(
                    batch_size, self.num_classes, -1, -1
                )
            )
        conditioned = torch.cat(inputs, dim=-1)
        return self.ratio_head(conditioned).squeeze(-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Alias for :meth:`forward_all`."""
        return self.forward_all(x)

    def ratio_training_outputs(
        self, x: torch.Tensor, y_bar: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """Return architecture-neutral tensors consumed by the ratio trainer."""
        return {"conditional_logits": self.forward_all(x)}

    def _validate_bar_y(self, bar_y: torch.Tensor, batch_size: int) -> torch.Tensor:
        if bar_y.ndim != 2 or tuple(bar_y.shape) != (batch_size, self.num_classes):
            raise ValueError(
                "bar_y must have shape [B, q]=[{}, {}]".format(
                    batch_size, self.num_classes
                )
            )
        states = bar_y.to(dtype=torch.long)
        if not bool(((states == 0) | (states == 1)).all().item()):
            raise ValueError("bar_y values must be binary (0 or 1)")
        return states

    def forward_selected(self, x: torch.Tensor, bar_y: torch.Tensor) -> torch.Tensor:
        """Return logits for ``z=bar_y[i,k]`` with shape ``[B, q]``."""
        all_logits = self.forward_all(x)
        states = self._validate_bar_y(bar_y, all_logits.shape[0])
        return all_logits.gather(dim=2, index=states.unsqueeze(-1)).squeeze(-1)

    def set_domain_priors(
        self,
        source_prior: torch.Tensor,
        target_prior: torch.Tensor,
        balanced_sampling: Optional[bool] = None,
    ) -> None:
        """Set branch-wise domain priors used by the odds correction.

        Priors have shape ``[q, 2]`` and represent
        ``P(d=source | k,z)`` and ``P(d=target | k,z)`` in discriminator
        training. Unsupported zero-prior branches receive zero correction and
        are marked in ``prior_supported``.
        """
        source = torch.as_tensor(
            source_prior, dtype=self._log_prior_correction.dtype,
            device=self._log_prior_correction.device,
        )
        target = torch.as_tensor(
            target_prior, dtype=self._log_prior_correction.dtype,
            device=self._log_prior_correction.device,
        )
        expected = (self.num_classes, 2)
        if tuple(source.shape) != expected or tuple(target.shape) != expected:
            raise ValueError("domain priors must both have shape [q, 2]")
        if not bool((torch.isfinite(source) & torch.isfinite(target)).all().item()):
            raise ValueError("domain priors must be finite")
        if bool(((source < 0) | (target < 0)).any().item()):
            raise ValueError("domain priors cannot be negative")

        balanced = (
            self.balanced_domain_sampling
            if balanced_sampling is None
            else bool(balanced_sampling)
        )
        supported = (source > 0) & (target > 0)
        correction = torch.zeros_like(source)
        if not balanced:
            correction[supported] = (
                torch.log(source[supported]) - torch.log(target[supported])
            )
        self._log_prior_correction.copy_(correction)
        self._prior_supported.copy_(supported)

    def set_domain_counts(
        self,
        source_count: torch.Tensor,
        target_count: torch.Tensor,
        balanced_sampling: Optional[bool] = None,
    ) -> None:
        """Derive branch-wise domain priors from sample counts."""
        source = torch.as_tensor(source_count, dtype=torch.float32)
        target = torch.as_tensor(target_count, dtype=torch.float32)
        expected = (self.num_classes, 2)
        if tuple(source.shape) != expected or tuple(target.shape) != expected:
            raise ValueError("domain counts must both have shape [q, 2]")
        if bool(((source < 0) | (target < 0)).any().item()):
            raise ValueError("domain counts cannot be negative")
        total = source + target
        source_prior = torch.zeros_like(source)
        target_prior = torch.zeros_like(target)
        nonempty = total > 0
        source_prior[nonempty] = source[nonempty] / total[nonempty]
        target_prior[nonempty] = target[nonempty] / total[nonempty]
        self.set_domain_priors(source_prior, target_prior, balanced_sampling)

    @property
    def log_prior_correction(self) -> torch.Tensor:
        """Branch-wise ``log(pi_source/pi_target)`` with shape ``[q,2]``."""
        return self._log_prior_correction

    @property
    def prior_supported(self) -> torch.Tensor:
        """Whether both domains have support for each ``(k,z)`` branch."""
        return self._prior_supported

    def ratios_from_logits(
        self, logits: torch.Tensor, bar_y: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Convert discriminator logits to stable target/source ratios."""
        logits_float = logits.float()
        if logits.ndim == 3:
            expected = (self.num_classes, 2)
            if tuple(logits.shape[1:]) != expected:
                raise ValueError("all-branch logits must have shape [B, q, 2]")
            correction = self._log_prior_correction[None, :, :]
        elif logits.ndim == 2:
            if logits.shape[1] != self.num_classes or bar_y is None:
                raise ValueError("selected logits require bar_y and shape [B, q]")
            states = self._validate_bar_y(bar_y, logits.shape[0])
            correction = self._log_prior_correction[None, :, :].expand(
                logits.shape[0], -1, -1
            )
            correction = correction.gather(2, states.unsqueeze(-1)).squeeze(-1)
        else:
            raise ValueError("logits must have shape [B,q,2] or [B,q]")

        log_ratio = torch.clamp(
            logits_float + correction,
            min=self.log_ratio_clip_min,
            max=self.log_ratio_clip_max,
        )
        ratio = torch.exp(log_ratio)
        if self.ratio_clip_min is not None or self.ratio_clip_max is not None:
            min_value = (
                self.ratio_clip_min
                if self.ratio_clip_min is not None
                else math.exp(self.log_ratio_clip_min)
            )
            max_value = (
                self.ratio_clip_max
                if self.ratio_clip_max is not None
                else math.exp(self.log_ratio_clip_max)
            )
            ratio = torch.clamp(ratio, min=min_value, max=max_value)

        safe_min = (
            self.ratio_clip_min
            if self.ratio_clip_min is not None
            else math.exp(self.log_ratio_clip_min)
        )
        safe_max = (
            self.ratio_clip_max
            if self.ratio_clip_max is not None
            else math.exp(self.log_ratio_clip_max)
        )
        ratio = torch.where(torch.isnan(ratio), torch.full_like(ratio, safe_min), ratio)
        ratio = torch.where(
            torch.isposinf(ratio), torch.full_like(ratio, safe_max), ratio
        )
        ratio = torch.where(
            torch.isneginf(ratio), torch.full_like(ratio, safe_min), ratio
        )
        return ratio

    def estimate_all_ratios(self, x: torch.Tensor) -> torch.Tensor:
        """Return all ratios with shape ``[B, q, 2]``."""
        return self.ratios_from_logits(self.forward_all(x))

    def predict_all_ratios(self, features: torch.Tensor) -> torch.Tensor:
        """Unified API alias returning all ``[B,q,2]`` positive ratios."""
        return self.estimate_all_ratios(features)

    def estimate_selected_ratios(
        self, x: torch.Tensor, bar_y: torch.Tensor
    ) -> torch.Tensor:
        """Return ``r[k,bar_y_k](x)`` with shape ``[B, q]``."""
        logits = self.forward_selected(x, bar_y)
        return self.ratios_from_logits(logits, bar_y)

    def select_observed_ratios(
        self, features: torch.Tensor, bar_y: torch.Tensor
    ) -> torch.Tensor:
        """Unified API alias returning ``r[k,bar_y_k](x)`` as ``[B,q]``."""
        return self.estimate_selected_ratios(features, bar_y)

    @property
    def branch_initialized(self) -> torch.Tensor:
        return self._branch_initialized

    @property
    def skipped_update_count(self) -> torch.Tensor:
        return self._skipped_update_count

    @torch.no_grad()
    def record_branch_updates(self, active: torch.Tensor) -> None:
        """Record valid and skipped conditional updates without label access."""
        active = torch.as_tensor(
            active, dtype=torch.bool, device=self._branch_initialized.device
        )
        if tuple(active.shape) != (self.num_classes, 2):
            raise ValueError("active must have shape [q,2]")
        self._branch_initialized.logical_or_(active)
        self._skipped_update_count.add_((~active).long())

    def compute_importance_weights(
        self,
        x: torch.Tensor,
        bar_y: torch.Tensor,
        detach: bool = True,
        clip_min: Optional[float] = None,
        clip_max: Optional[float] = None,
        self_normalize: Optional[bool] = None,
    ) -> torch.Tensor:
        """Return branch-specific importance weights with shape ``[B, q]``.

        Self-normalization, when enabled, is performed independently for every
        observed ``(k,z)`` branch using vectorized ``scatter_add`` operations.
        """
        weights = self.estimate_selected_ratios(x, bar_y)
        if clip_min is not None or clip_max is not None:
            minimum = clip_min if clip_min is not None else 0.0
            maximum = clip_max if clip_max is not None else float("inf")
            if minimum < 0 or minimum > maximum:
                raise ValueError("invalid importance-weight clipping bounds")
            weights = torch.clamp(weights, min=minimum, max=maximum)

        normalize = (
            self.default_self_normalize
            if self_normalize is None
            else bool(self_normalize)
        )
        if normalize and weights.numel() > 0:
            states = self._validate_bar_y(bar_y, weights.shape[0])
            class_offset = 2 * torch.arange(
                self.num_classes, device=weights.device, dtype=torch.long
            )
            branch_ids = states + class_offset[None, :]  # [B, q]
            flat_ids = branch_ids.reshape(-1)
            flat_weights = weights.reshape(-1)
            branch_sums = torch.zeros(
                self.num_classes * 2, device=weights.device, dtype=weights.dtype
            )
            branch_counts = torch.zeros_like(branch_sums)
            branch_sums.scatter_add_(0, flat_ids, flat_weights)
            branch_counts.scatter_add_(0, flat_ids, torch.ones_like(flat_weights))
            means = branch_sums / branch_counts.clamp_min(1.0)
            normalizers = means.gather(0, flat_ids).clamp_min(
                torch.finfo(weights.dtype).tiny
            )
            weights = (flat_weights / normalizers).reshape_as(weights)

        return weights.detach() if detach else weights

    def get_conditional_weights(
        self,
        x: torch.Tensor,
        y_bar: torch.Tensor,
        detach: bool = True,
        clip_min: Optional[float] = None,
        clip_max: Optional[float] = None,
        self_normalize: Optional[bool] = None,
        fallback_uninitialized: bool = True,
    ) -> torch.Tensor:
        """Return the ratio matching every observed class/state branch.

        ``weights[i,k]`` is exactly ``r[k,int(y_bar[i,k])](x[i])``. For a new
        architecture whose branch has never had a valid update, the conservative
        fallback is one; no ordinary label is accepted by this interface.
        """
        weights = self.compute_importance_weights(
            x,
            y_bar,
            detach=False,
            clip_min=clip_min,
            clip_max=clip_max,
            self_normalize=self_normalize,
        )
        if fallback_uninitialized:
            states = self._validate_bar_y(y_bar, weights.shape[0])
            initialized = self._branch_initialized[None, :, :].expand(
                weights.shape[0], -1, -1
            )
            selected_initialized = initialized.gather(
                2, states.unsqueeze(-1)
            ).squeeze(-1)
            weights = torch.where(
                selected_initialized, weights, torch.ones_like(weights)
            )
        return weights.detach() if detach else weights

    def extra_ratio_statistics(self, x: torch.Tensor) -> Dict[str, Any]:
        """Architecture-specific diagnostics; empty for the legacy baseline."""
        return {}
