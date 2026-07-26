"""Vectorized trainer for label-conditioned density-ratio estimation."""

import json
import os
import warnings
from typing import Any, Dict, Iterable, List, Optional, Tuple

import torch
import torch.nn.functional as F

from models.ratio_estimator import LabelConditionedRatioEstimator
from .conditional_masks import valid_condition_mask
from .data import unpack_batch


class RatioEstimatorTrainer:
    """Train one shared discriminator over all ``(class, state)`` branches."""

    def __init__(
        self,
        model: LabelConditionedRatioEstimator,
        optimizer: torch.optim.Optimizer,
        device: Optional[torch.device] = None,
        branch_balancing: str = "balanced_bce",
        min_branch_count: int = 1,
        max_branch_weight: float = 20.0,
        balanced_domain_sampling: bool = False,
        use_amp: bool = False,
        x_key: str = "x",
        bar_y_key: str = "bar_y",
        grad_clip_norm: Optional[float] = None,
    ) -> None:
        if branch_balancing not in ("none", "inverse_frequency", "balanced_bce"):
            raise ValueError(
                "branch_balancing must be none, inverse_frequency, or balanced_bce"
            )
        if min_branch_count < 1:
            raise ValueError("min_branch_count must be at least 1")
        if max_branch_weight <= 0:
            raise ValueError("max_branch_weight must be positive")
        self.model = model
        self.optimizer = optimizer
        self.device = (
            torch.device(device)
            if device is not None
            else next(model.parameters()).device
        )
        self.model.to(self.device)
        self.branch_balancing = branch_balancing
        self.min_branch_count = int(min_branch_count)
        self.max_branch_weight = float(max_branch_weight)
        self.balanced_domain_sampling = bool(balanced_domain_sampling)
        self.use_amp = bool(use_amp and self.device.type == "cuda")
        self.x_key = x_key
        self.bar_y_key = bar_y_key
        self.grad_clip_norm = grad_clip_norm
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)
        self.source_count = None  # type: Optional[torch.Tensor]
        self.target_count = None  # type: Optional[torch.Tensor]
        self.branch_optimizers = None  # type: Optional[List[torch.optim.Optimizer]]
        if getattr(self.model, "architecture", "unified") == "separate":
            optimizer_type = type(self.optimizer)
            defaults = dict(self.optimizer.defaults)
            self.branch_optimizers = []
            for k in range(self.model.num_classes):
                for b in range(2):
                    self.branch_optimizers.append(
                        optimizer_type(
                            list(self.model.branch_parameters(k, b)), **defaults
                        )
                    )

    def _prepare_batch(self, batch: Any) -> Tuple[torch.Tensor, torch.Tensor]:
        x, bar_y = unpack_batch(batch, self.x_key, self.bar_y_key)
        x = x.to(self.device, non_blocking=True)
        bar_y = bar_y.to(self.device, non_blocking=True)
        return x, bar_y

    def _branch_ids(self, bar_y: torch.Tensor) -> torch.Tensor:
        states = bar_y.to(dtype=torch.long)
        if states.ndim != 2 or states.shape[1] != self.model.num_classes:
            raise ValueError("bar_y must have shape [B, q]")
        if not bool(((states == 0) | (states == 1)).all().item()):
            raise ValueError("bar_y must contain only 0/1")
        offsets = 2 * torch.arange(
            self.model.num_classes, device=states.device, dtype=torch.long
        )
        return states + offsets[None, :]

    def _scatter_sum(
        self, values: torch.Tensor, branch_ids: torch.Tensor
    ) -> torch.Tensor:
        output = torch.zeros(
            self.model.num_classes * 2,
            device=values.device,
            dtype=values.dtype,
        )
        output.scatter_add_(0, branch_ids.reshape(-1), values.reshape(-1))
        return output.reshape(self.model.num_classes, 2)

    def _batch_counts(self, branch_ids: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
        return self._scatter_sum(torch.ones_like(branch_ids, dtype=dtype), branch_ids)

    def _dataset_support(self, device: torch.device) -> Optional[torch.Tensor]:
        if self.source_count is None or self.target_count is None:
            return None
        return (
            (self.source_count.to(device) >= self.min_branch_count)
            & (self.target_count.to(device) >= self.min_branch_count)
        )

    def _loss(
        self,
        source_logits: torch.Tensor,
        target_logits: torch.Tensor,
        source_bar_y: torch.Tensor,
        target_bar_y: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        source_element = F.binary_cross_entropy_with_logits(
            source_logits, torch.zeros_like(source_logits), reduction="none"
        )
        target_element = F.binary_cross_entropy_with_logits(
            target_logits, torch.ones_like(target_logits), reduction="none"
        )
        source_ids = self._branch_ids(source_bar_y)
        target_ids = self._branch_ids(target_bar_y)
        source_sum = self._scatter_sum(source_element, source_ids)
        target_sum = self._scatter_sum(target_element, target_ids)
        source_batch_count = self._batch_counts(source_ids, source_element.dtype)
        target_batch_count = self._batch_counts(target_ids, target_element.dtype)

        # A conditional update is valid only when this mini-batch contains the
        # configured minimum on both sides. Dataset-level support alone is not
        # enough and must not trigger an empty/small-tensor objective.
        active, _, _ = valid_condition_mask(
            source_bar_y,
            target_bar_y,
            self.model.num_classes,
            self.min_branch_count,
        )
        dataset_support = self._dataset_support(source_logits.device)
        if dataset_support is not None:
            active = active & dataset_support
        active_float = active.to(source_element.dtype)

        if not bool(active.any().item()):
            # Keep a valid graph so backward remains safe for a missing batch.
            total_loss = (source_logits.sum() + target_logits.sum()) * 0.0
        elif self.branch_balancing == "balanced_bce":
            source_mean = source_sum / source_batch_count.clamp_min(1.0)
            target_mean = target_sum / target_batch_count.clamp_min(1.0)
            per_branch = 0.5 * (source_mean + target_mean)
            total_loss = (per_branch * active_float).sum() / active_float.sum()
        elif self.branch_balancing == "inverse_frequency":
            combined_count = source_batch_count + target_batch_count
            active_total = (combined_count * active_float).sum()
            nominal = active_total / active_float.sum().clamp_min(1.0)
            branch_weight = nominal / combined_count.clamp_min(1.0)
            branch_weight = torch.clamp(
                branch_weight, max=self.max_branch_weight
            ) * active_float
            source_weights = branch_weight.gather(1, source_bar_y.long().t()).t()
            target_weights = branch_weight.gather(1, target_bar_y.long().t()).t()
            weighted_sum = (
                (source_element * source_weights).sum()
                + (target_element * target_weights).sum()
            )
            weight_sum = source_weights.sum() + target_weights.sum()
            total_loss = weighted_sum / weight_sum.clamp_min(1.0)
        else:
            source_active = active.gather(1, source_bar_y.long().t()).t()
            target_active = active.gather(1, target_bar_y.long().t()).t()
            loss_sum = (
                (source_element * source_active).sum()
                + (target_element * target_active).sum()
            )
            count = source_active.sum() + target_active.sum()
            total_loss = loss_sum / count.clamp_min(1)

        source_denominator = (source_batch_count * active_float).sum().clamp_min(1.0)
        target_denominator = (target_batch_count * active_float).sum().clamp_min(1.0)
        source_loss = (source_sum * active_float).sum() / source_denominator
        target_loss = (target_sum * active_float).sum() / target_denominator
        details = {
            "source_loss": source_loss,
            "target_loss": target_loss,
            "source_count": source_batch_count,
            "target_count": target_batch_count,
            "active": active,
            "source_ids": source_ids,
            "target_ids": target_ids,
            "per_branch_loss": 0.5
            * (
                source_sum / source_batch_count.clamp_min(1.0)
                + target_sum / target_batch_count.clamp_min(1.0)
            ),
        }
        return total_loss, details

    def _selected_logits(
        self, all_logits: torch.Tensor, bar_y: torch.Tensor
    ) -> torch.Tensor:
        states = bar_y.long()
        return all_logits.gather(2, states.unsqueeze(-1)).squeeze(-1)

    def _add_architecture_objective(
        self,
        total_loss: torch.Tensor,
        details: Dict[str, torch.Tensor],
        source_outputs: Dict[str, torch.Tensor],
        target_outputs: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Add optional global/fusion objectives through the unified output API."""
        if "global_logits" not in source_outputs:
            details["global_loss"] = total_loss.detach() * 0.0
            details["gate_regularization"] = total_loss.detach() * 0.0
            return total_loss
        source_global = source_outputs["global_logits"]
        target_global = target_outputs["global_logits"]
        global_loss = 0.5 * (
            F.binary_cross_entropy_with_logits(
                source_global, torch.zeros_like(source_global)
            )
            + F.binary_cross_entropy_with_logits(
                target_global, torch.ones_like(target_global)
            )
        )
        gates = torch.cat(
            (source_outputs["gates"], target_outputs["gates"]), dim=0
        )
        gate_regularization = self.model.fusion_regularization(gates)
        details["global_loss"] = global_loss
        details["gate_regularization"] = gate_regularization
        return total_loss + global_loss + gate_regularization

    def _clear_inactive_branch_gradients(self, active: torch.Tensor) -> None:
        branch_parameters = getattr(self.model, "branch_parameters", None)
        if branch_parameters is None:
            return
        for k in range(self.model.num_classes):
            for b in range(2):
                if bool(active[k, b].item()):
                    continue
                for parameter in branch_parameters(k, b):
                    parameter.grad = None

    def _optimizer_step(self, active: torch.Tensor) -> None:
        if self.grad_clip_norm is not None:
            if self.branch_optimizers is None:
                self.scaler.unscale_(self.optimizer)
            else:
                for branch_optimizer in self.branch_optimizers:
                    self.scaler.unscale_(branch_optimizer)
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.grad_clip_norm
            )
        if self.branch_optimizers is None:
            self.scaler.step(self.optimizer)
        else:
            for k in range(self.model.num_classes):
                for b in range(2):
                    if bool(active[k, b].item()):
                        self.scaler.step(
                            self.branch_optimizers[2 * k + b]
                        )
        self.scaler.update()

    def _step_metrics(
        self,
        total_loss: torch.Tensor,
        details: Dict[str, torch.Tensor],
        source_logits: torch.Tensor,
        target_logits: torch.Tensor,
        source_bar_y: torch.Tensor,
        target_bar_y: torch.Tensor,
    ) -> Dict[str, Any]:
        active = details["active"]
        source_active = active.gather(1, source_bar_y.long().t()).t()
        target_active = active.gather(1, target_bar_y.long().t()).t()
        correct_source = ((source_logits < 0) & source_active).sum()
        correct_target = ((target_logits >= 0) & target_active).sum()
        prediction_count = source_active.sum() + target_active.sum()
        accuracy = (correct_source + correct_target).float() / prediction_count.clamp_min(1)

        source_ratios = self.model.ratios_from_logits(source_logits, source_bar_y)
        target_ratios = self.model.ratios_from_logits(target_logits, target_bar_y)
        ratios = torch.cat((source_ratios.reshape(-1), target_ratios.reshape(-1)))
        finite = torch.isfinite(ratios)
        finite_ratios = ratios[finite]
        if finite_ratios.numel() == 0:
            ratio_mean = ratio_std = ratio_min = ratio_max = float("nan")
        else:
            ratio_mean = float(finite_ratios.mean().item())
            ratio_std = float(finite_ratios.std(unbiased=False).item())
            ratio_min = float(finite_ratios.min().item())
            ratio_max = float(finite_ratios.max().item())
        nonfinite = (
            not bool(torch.isfinite(total_loss).item())
            or not bool(torch.isfinite(source_logits).all().item())
            or not bool(torch.isfinite(target_logits).all().item())
            or not bool(finite.all().item())
        )
        return {
            "total_loss": float(total_loss.detach().item()),
            "source_loss": float(details["source_loss"].detach().item()),
            "target_loss": float(details["target_loss"].detach().item()),
            "discriminator_accuracy": float(accuracy.detach().item()),
            "source_branch_count": details["source_count"].detach().cpu().long().tolist(),
            "target_branch_count": details["target_count"].detach().cpu().long().tolist(),
            "active_branches": details["active"].detach().cpu().tolist(),
            "branch_objective": details["per_branch_loss"].detach().cpu().tolist(),
            "global_objective": float(details["global_loss"].detach().item()),
            "gate_regularization": float(
                details["gate_regularization"].detach().item()
            ),
            "skipped_update_count": self.model.skipped_update_count.detach()
            .cpu()
            .tolist(),
            "ratio_mean": ratio_mean,
            "ratio_std": ratio_std,
            "ratio_min": ratio_min,
            "ratio_max": ratio_max,
            "has_nan_or_inf": nonfinite,
        }

    def train_step(self, source_batch: Any, target_batch: Any) -> Dict[str, Any]:
        """Run one vectorized discriminator update and return diagnostics."""
        self.model.train()
        source_x, source_bar_y = self._prepare_batch(source_batch)
        target_x, target_bar_y = self._prepare_batch(target_batch)
        self.optimizer.zero_grad()
        if self.branch_optimizers is not None:
            for branch_optimizer in self.branch_optimizers:
                branch_optimizer.zero_grad()
        with torch.cuda.amp.autocast(enabled=self.use_amp):
            source_outputs = self.model.ratio_training_outputs(
                source_x, source_bar_y
            )
            target_outputs = self.model.ratio_training_outputs(
                target_x, target_bar_y
            )
            source_logits = self._selected_logits(
                source_outputs["conditional_logits"], source_bar_y
            )
            target_logits = self._selected_logits(
                target_outputs["conditional_logits"], target_bar_y
            )
            total_loss, details = self._loss(
                source_logits, target_logits, source_bar_y, target_bar_y
            )
            total_loss = self._add_architecture_objective(
                total_loss, details, source_outputs, target_outputs
            )

        if bool(torch.isfinite(total_loss).item()):
            self.scaler.scale(total_loss).backward()
            self._clear_inactive_branch_gradients(details["active"])
            self._optimizer_step(details["active"])
            self.model.record_branch_updates(details["active"])
        else:
            warnings.warn("non-finite ratio-estimator loss; optimizer step skipped")
            self.model.record_branch_updates(
                torch.zeros_like(details["active"])
            )
        return self._step_metrics(
            total_loss,
            details,
            source_logits.detach(),
            target_logits.detach(),
            source_bar_y,
            target_bar_y,
        )

    @torch.no_grad()
    def validate(
        self, source_loader: Iterable[Any], target_loader: Iterable[Any]
    ) -> Dict[str, Any]:
        """Evaluate paired source/target loaders without optimizer updates."""
        was_training = self.model.training
        self.model.eval()
        aggregate = []  # type: List[Dict[str, Any]]
        for source_batch, target_batch in zip(source_loader, target_loader):
            source_x, source_bar_y = self._prepare_batch(source_batch)
            target_x, target_bar_y = self._prepare_batch(target_batch)
            with torch.cuda.amp.autocast(enabled=self.use_amp):
                source_outputs = self.model.ratio_training_outputs(
                    source_x, source_bar_y
                )
                target_outputs = self.model.ratio_training_outputs(
                    target_x, target_bar_y
                )
                source_logits = self._selected_logits(
                    source_outputs["conditional_logits"], source_bar_y
                )
                target_logits = self._selected_logits(
                    target_outputs["conditional_logits"], target_bar_y
                )
                total_loss, details = self._loss(
                    source_logits, target_logits, source_bar_y, target_bar_y
                )
                total_loss = self._add_architecture_objective(
                    total_loss, details, source_outputs, target_outputs
                )
            aggregate.append(
                self._step_metrics(
                    total_loss,
                    details,
                    source_logits,
                    target_logits,
                    source_bar_y,
                    target_bar_y,
                )
            )
        self.model.train(was_training)
        return self._aggregate_metrics(aggregate)

    def _aggregate_metrics(self, metrics: List[Dict[str, Any]]) -> Dict[str, Any]:
        if not metrics:
            return {"steps": 0, "has_nan_or_inf": False}
        scalar_keys = (
            "total_loss",
            "source_loss",
            "target_loss",
            "discriminator_accuracy",
            "ratio_mean",
            "ratio_std",
            "ratio_min",
            "ratio_max",
        )
        output = {"steps": len(metrics)}  # type: Dict[str, Any]
        for key in scalar_keys:
            values = [item[key] for item in metrics]
            output[key] = sum(values) / float(len(values))
        matrix_keys = (
            "branch_objective",
            "source_branch_count",
            "target_branch_count",
        )
        for key in matrix_keys:
            tensors = [
                torch.as_tensor(item[key], dtype=torch.float64)
                for item in metrics
            ]
            output[key] = torch.stack(tensors).mean(dim=0).tolist()
        output["skipped_update_count"] = metrics[-1][
            "skipped_update_count"
        ]
        output["has_nan_or_inf"] = any(item["has_nan_or_inf"] for item in metrics)
        return output

    @torch.no_grad()
    def compute_branch_statistics(
        self,
        source_loader: Iterable[Any],
        target_loader: Iterable[Any],
        update_prior: bool = True,
    ) -> Dict[str, Any]:
        """Count source/target observations for every branch.

        Returns ``source_count`` and ``target_count`` with shape ``[q,2]``.
        No branch-specific datasets are materialized.
        """

        def count(loader: Iterable[Any]) -> torch.Tensor:
            result = torch.zeros(self.model.num_classes, 2, dtype=torch.long)
            for batch in loader:
                _, bar_y = unpack_batch(batch, self.x_key, self.bar_y_key)
                states = bar_y.detach().cpu().long()
                if states.ndim != 2 or states.shape[1] != self.model.num_classes:
                    raise ValueError("bar_y must have shape [B, q]")
                result[:, 1] += states.sum(dim=0)
                result[:, 0] += states.shape[0] - states.sum(dim=0)
            return result

        self.source_count = count(source_loader)
        self.target_count = count(target_loader)
        supported = (
            (self.source_count >= self.min_branch_count)
            & (self.target_count >= self.min_branch_count)
        )
        if not bool(supported.all().item()):
            missing = (~supported).nonzero(as_tuple=False).tolist()
            warnings.warn(
                "insufficient source/target support for branches {}; "
                "they will be excluded from loss".format(missing)
            )
        if update_prior:
            # balanced_bce averages source and target losses separately inside
            # each branch, so its *effective* discriminator domain prior is
            # exactly 1:1 even when raw branch counts differ.
            effective_balanced_sampling = (
                self.balanced_domain_sampling
                or self.branch_balancing == "balanced_bce"
            )
            self.model.set_domain_counts(
                self.source_count,
                self.target_count,
                balanced_sampling=effective_balanced_sampling,
            )
            set_global_counts = getattr(
                self.model, "set_global_domain_counts", None
            )
            if set_global_counts is not None:
                set_global_counts(
                    int(self.source_count.sum().item() // self.model.num_classes),
                    int(self.target_count.sum().item() // self.model.num_classes),
                    effective_balanced_sampling,
                )
        return {
            "source_count": self.source_count.tolist(),
            "target_count": self.target_count.tolist(),
            "supported": supported.tolist(),
        }

    def fit(
        self,
        source_loader: Iterable[Any],
        target_loader: Iterable[Any],
        epochs: int,
        validation_loaders: Optional[
            Tuple[Iterable[Any], Iterable[Any]]
        ] = None,
        checkpoint_path: Optional[str] = None,
        max_steps_per_epoch: Optional[int] = None,
        verbose: bool = True,
    ) -> List[Dict[str, Any]]:
        """Fit the estimator, optionally validating and saving checkpoints."""
        if epochs <= 0:
            raise ValueError("epochs must be positive")
        statistics = self.compute_branch_statistics(source_loader, target_loader)
        if verbose:
            print("Branch statistics: {}".format(json.dumps(statistics)))
        history = []  # type: List[Dict[str, Any]]
        for epoch in range(epochs):
            epoch_metrics = []
            for step, (source_batch, target_batch) in enumerate(
                zip(source_loader, target_loader)
            ):
                if max_steps_per_epoch is not None and step >= max_steps_per_epoch:
                    break
                epoch_metrics.append(self.train_step(source_batch, target_batch))
            summary = self._aggregate_metrics(epoch_metrics)
            summary["epoch"] = epoch + 1
            if validation_loaders is not None:
                summary["validation"] = self.validate(
                    validation_loaders[0], validation_loaders[1]
                )
            history.append(summary)
            if verbose:
                print("Epoch: {}".format(json.dumps(summary, sort_keys=True)))
            if checkpoint_path is not None:
                directory = os.path.dirname(os.path.abspath(checkpoint_path))
                if not os.path.isdir(directory):
                    os.makedirs(directory)
                torch.save(
                    {
                        "epoch": epoch + 1,
                        "model_state_dict": self.model.state_dict(),
                        "optimizer_state_dict": self.optimizer.state_dict(),
                        "branch_optimizer_state_dicts": (
                            [
                                branch_optimizer.state_dict()
                                for branch_optimizer in self.branch_optimizers
                            ]
                            if self.branch_optimizers is not None
                            else None
                        ),
                        "source_count": self.source_count,
                        "target_count": self.target_count,
                        "history": history,
                    },
                    checkpoint_path,
                )
        return history
