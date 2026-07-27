"""Matched next-round SCAR experiments with auditable diagnostics.

This module runs one method/setting/seed task.  Matrix construction and
Wisteria scheduling live in :mod:`scripts.run_next_round`; keeping one task per
process makes failures and GPU ownership unambiguous.

Ordinary target labels are consumed only by the evaluation helpers.  They are
never passed to ratio estimation, classifier training, early stopping, or
hyperparameter selection.
"""

import argparse
import csv
import datetime as dt
import hashlib
import json
import math
import os
import subprocess
import time
import traceback
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import ConcatDataset, DataLoader, Dataset, Subset

from experiments.scarce_ratio_comparison import (
    ScarceMLPFeatureBackbone,
    WeakPairView,
    _next_batch,
    load_scarce_modules,
    set_random_seed,
)
from models import build_ratio_estimator as create_ratio_estimator
from ratio_estimation import (
    RatioEstimatorTrainer,
    process_weight_stages,
    summarize_weight_values,
    weight_stage_rows,
)
from src.weight_interventions import (
    apply_weight_intervention,
    classifier_method_contract,
)
from src.ratio_diagnostics import distribution_matching_diagnostics


METHODS = (
    "no_iw",
    "current_no_iw",
    "matched_no_iw",
    "pooled_scar",
    "adiw",
    "adiw_original",
    "adiw_ones",
    "adiw_mean",
    "adiw_shuffle",
    "unified",
    "multihead",
    "fusion",
    "separate_full",
    "separate_matched_capacity",
)
RATIO_METHODS = (
    "unified",
    "multihead",
    "fusion",
    "separate_full",
    "separate_matched_capacity",
)
CSV_FILES = {
    "epoch_metrics.csv": [
        "epoch",
        "classifier_training_loss",
        "source_ovr_risk",
        "target_ovr_risk",
        "source_accuracy",
        "target_accuracy",
        "target_prediction_confidence",
        "target_ece",
        "target_nll",
        "true_target_risk",
        "estimated_weak_risk",
        "estimated_iw_risk",
        "source_risk_contribution",
        "target_risk_contribution",
        "total_loss_scale",
        "classifier_gradient_norm",
        "ratio_estimator_loss",
        "weight_mean",
        "weight_std",
        "weight_cv",
        "weight_min",
        "weight_max",
        "weight_q01",
        "weight_q05",
        "weight_q50",
        "weight_q95",
        "weight_q99",
        "weight_ess",
        "normalized_ess",
    ],
    "branch_metrics.csv": [
        "epoch",
        "stage",
        "class_index",
        "state",
        "source_count",
        "target_count",
        "branch_initialized",
        "skipped_update_count",
    ],
    "ratio_raw_stats.csv": [],
    "ratio_clipped_stats.csv": [],
    "ratio_normalized_stats.csv": [],
    "classifier_used_ratio_stats.csv": [],
    "discriminator_metrics.csv": [
        "epoch",
        "split",
        "class_index",
        "state",
        "source_count",
        "target_count",
        "domain_loss",
        "domain_accuracy",
        "source_probability_mean",
        "target_probability_mean",
        "brier_score",
        "ece",
        "source_probability_histogram",
        "target_probability_histogram",
    ],
    "per_class_metrics.csv": [
        "epoch",
        "domain",
        "class_index",
        "accuracy",
        "ovr_risk",
        "support",
    ],
    "gate_metrics.csv": [
        "epoch",
        "split",
        "domain",
        "class_index",
        "state",
        "count",
        "gate_mean",
        "gate_std",
        "gate_min",
        "gate_max",
        "gate_q01",
        "gate_q05",
        "gate_q25",
        "gate_q50",
        "gate_q75",
        "gate_q95",
        "gate_q99",
        "gate_histogram",
        "fraction_gate_lt_0_1",
        "fraction_gate_gt_0_9",
        "global_log_ratio_mean",
        "global_log_ratio_std",
        "conditional_log_ratio_mean",
        "conditional_log_ratio_std",
        "fused_log_ratio_mean",
        "fused_log_ratio_std",
        "abs_global_minus_conditional",
        "abs_fused_minus_global",
        "abs_fused_minus_conditional",
    ],
}


def _ensure_directory(path: str) -> None:
    if not os.path.isdir(path):
        os.makedirs(path)


def _json_dump(path: str, value: Any) -> None:
    with open(path, "w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)


def _write_csv(
    path: str,
    rows: Sequence[Dict[str, Any]],
    default_fields: Optional[Sequence[str]] = None,
) -> None:
    fields = list(default_fields or [])
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with open(path, "w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def initialize_result_files(run_dir: str) -> None:
    """Create every required CSV even when a stage is not applicable."""
    _ensure_directory(run_dir)
    for filename, fields in CSV_FILES.items():
        _write_csv(os.path.join(run_dir, filename), [], fields)


def _git_commit_hash(repository: str) -> Optional[str]:
    try:
        output = subprocess.check_output(
            ["git", "-C", repository, "rev-parse", "HEAD"],
            stderr=subprocess.STDOUT,
        )
        return output.decode("utf-8").strip()
    except Exception:
        return None


def _config_hash(config: Dict[str, Any]) -> str:
    payload = json.dumps(
        config, sort_keys=True, separators=(",", ":"), default=str
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _hash_tensor(digest: Any, tensor: torch.Tensor) -> None:
    array = torch.as_tensor(tensor).detach().cpu().contiguous().numpy()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(str(tuple(array.shape)).encode("ascii"))
    digest.update(array.tobytes())


def paired_split_hash(data: Dict[str, Any], shift_spec: Dict[str, Any]) -> str:
    """Hash only split, weak labels, and fixed transformation metadata."""
    digest = hashlib.sha256()
    for name in ("source_weak", "target_weak"):
        dataset = data[name]
        if not hasattr(dataset, "base_indices"):
            raise ValueError("{} must expose base_indices".format(name))
        if not hasattr(dataset, "complementary_vectors"):
            raise ValueError("{} must expose complementary_vectors".format(name))
        digest.update(name.encode("ascii"))
        _hash_tensor(digest, dataset.base_indices)
        _hash_tensor(digest, dataset.complementary_vectors)
    digest.update(
        json.dumps(shift_spec, sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        )
    )
    return digest.hexdigest()


class OrdinarySourceEvaluationView(Dataset):
    """Expose source ordinary labels only inside evaluation code."""

    def __init__(self, weak_dataset: Dataset) -> None:
        self.weak_dataset = weak_dataset

    def __len__(self) -> int:
        return len(self.weak_dataset)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        dataset = self.weak_dataset
        base_index = int(dataset.base_indices[index].item())
        image, label = dataset.base_dataset[base_index]
        return image, int(label)


def _heterogeneous_transform(
    image: torch.Tensor,
    label: int,
    transform_functional: Any,
    class_rotations: Optional[Sequence[float]] = None,
) -> torch.Tensor:
    if class_rotations is not None:
        if len(class_rotations) != 10:
            raise ValueError("class_rotations must contain ten angles")
        return transform_functional.rotate(
            image, float(class_rotations[int(label)])
        )
    if label <= 2:
        return transform_functional.rotate(image, 10.0)
    if label <= 5:
        return transform_functional.rotate(image, 20.0)
    if label <= 7:
        return transform_functional.rotate(image, -15.0)
    if label == 8:
        return transform_functional.affine(
            image, angle=0.0, translate=[2, 0], scale=1.0, shear=0.0
        )
    return transform_functional.adjust_contrast(image, 1.5)


class HeterogeneousWeakDataset(Dataset):
    """Apply a fixed class-specific transform without returning labels."""

    def __init__(
        self,
        weak_dataset: Dataset,
        transform_functional: Any,
        class_rotations: Optional[Sequence[float]] = None,
    ) -> None:
        self.base_dataset = weak_dataset.base_dataset
        self.base_indices = weak_dataset.base_indices
        self.complementary_vectors = weak_dataset.complementary_vectors
        self.rotation_degrees = 0.0
        self.transform_functional = transform_functional
        self.class_rotations = (
            None
            if class_rotations is None
            else tuple(float(value) for value in class_rotations)
        )

    def __len__(self) -> int:
        return len(self.base_indices)

    def __getitem__(
        self, local_index: int
    ) -> Tuple[int, torch.Tensor, torch.Tensor]:
        base_index = int(self.base_indices[local_index].item())
        image, label = self.base_dataset[base_index]
        image = _heterogeneous_transform(
            image,
            int(label),
            self.transform_functional,
            self.class_rotations,
        )
        return local_index, image, self.complementary_vectors[local_index]


class HeterogeneousEvaluationDataset(Dataset):
    def __init__(
        self,
        base_dataset: Dataset,
        transform_functional: Any,
        class_rotations: Optional[Sequence[float]] = None,
    ) -> None:
        self.base_dataset = base_dataset
        self.transform_functional = transform_functional
        self.class_rotations = (
            None
            if class_rotations is None
            else tuple(float(value) for value in class_rotations)
        )

    def __len__(self) -> int:
        return len(self.base_dataset)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        image, label = self.base_dataset[index]
        return (
            _heterogeneous_transform(
                image,
                int(label),
                self.transform_functional,
                self.class_rotations,
            ),
            int(label),
        )


def prepare_matched_data(
    modules: Dict[str, Any],
    scarce_repository: str,
    dataset: str,
    target_weak_size: int,
    source_size: int,
    seed: int,
    shift: str,
    target_rotation: float,
    class_rotations: Optional[Sequence[float]] = None,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Build one paired split; ordinary labels only define controlled shifts."""
    domain_dataset = "fashion" if dataset == "fashionmnist" else dataset
    controlled_class_shift = (
        shift == "heterogeneous" or class_rotations is not None
    )
    base_shift = "none" if controlled_class_shift else shift
    original_directory = os.getcwd()
    os.chdir(scarce_repository)
    try:
        data = modules["domain_data"].prepare_dual_domain_datasets(
            dataname=domain_dataset,
            target_weak_size=target_weak_size,
            source_size=source_size,
            seed=seed,
            shift=base_shift,
            target_rotation=target_rotation,
            label_shift=1,
            source_generation="single",
            target_generation="single",
            source_selection_probability=None,
            target_selection_probability=None,
        )
    finally:
        os.chdir(original_directory)
    if controlled_class_shift:
        from torchvision.transforms import functional as transform_functional

        target = data["target_weak"]
        evaluation = data["target_evaluation"]
        data["target_weak"] = HeterogeneousWeakDataset(
            target, transform_functional, class_rotations=class_rotations
        )
        data["target_evaluation"] = HeterogeneousEvaluationDataset(
            evaluation.base_dataset,
            transform_functional,
            class_rotations=class_rotations,
        )
    shift_spec = {
        "type": shift,
        "rotation_degrees": float(target_rotation)
        if shift == "rotation"
        else None,
        "heterogeneous_map": {
            "0-2": "rotation:+10",
            "3-5": "rotation:+20",
            "6-7": "rotation:-15",
            "8": "translation:+2px-x",
            "9": "contrast:1.5",
        }
        if shift == "heterogeneous"
        else None,
        "class_rotations": (
            None
            if class_rotations is None
            else [float(value) for value in class_rotations]
        ),
    }
    data["source_evaluation"] = OrdinarySourceEvaluationView(
        data["source_weak"]
    )
    return data, shift_spec


def _split_indices(
    length: int, seed: int, validation_fraction: float
) -> Tuple[List[int], List[int]]:
    if length < 2:
        raise ValueError("ratio split needs at least two samples")
    random_state = np.random.RandomState(seed)
    indices = random_state.permutation(length)
    validation_size = max(1, int(round(length * validation_fraction)))
    validation_size = min(validation_size, length - 1)
    return (
        indices[validation_size:].tolist(),
        indices[:validation_size].tolist(),
    )


def ratio_split_hash(
    source_train: Sequence[int],
    source_validation: Sequence[int],
    target_train: Sequence[int],
    target_validation: Sequence[int],
) -> str:
    digest = hashlib.sha256()
    for values in (
        source_train,
        source_validation,
        target_train,
        target_validation,
    ):
        _hash_tensor(digest, torch.as_tensor(values, dtype=torch.long))
    return digest.hexdigest()


def build_next_round_ratio_estimator(
    scarce_models: Any,
    input_dim: int,
    num_classes: int,
    method: str,
    device: torch.device,
    ratio_clip_min: float,
    ratio_clip_max: float,
    normalization: str,
    gate_init: float,
) -> Tuple[nn.Module, Dict[str, Any]]:
    """Build a full or matched-capacity estimator with recorded dimensions."""
    architecture = method
    backbone_hidden = 500
    ratio_hidden = [128, 64]
    if method == "separate_full":
        architecture = "separate"
    elif method == "separate_matched_capacity":
        architecture = "separate"
        backbone_hidden = 30
        ratio_hidden = []
    original_mlp = scarce_models.mlp_model(
        input_dim=input_dim,
        hidden_dim=backbone_hidden,
        output_dim=num_classes,
    )
    backbone = ScarceMLPFeatureBackbone(original_mlp)
    config = {
        "input_dim": input_dim,
        "num_classes": num_classes,
        "feature_dim": backbone_hidden,
        "class_embedding_dim": 16,
        "state_embedding_dim": 4,
        "hidden_dims": ratio_hidden,
        "ratio_arch": architecture,
        "estimator_type": "conditioned",
        "balanced_domain_sampling": True,
        # Keep log-space safety wider than every ratio sweep bound.
        "log_ratio_clip_min": -20.0,
        "log_ratio_clip_max": 20.0,
        "ratio": {
            # Model conversion remains wide; experiment-stage clipping and
            # normalization are applied explicitly to raw prior-corrected odds.
            "self_normalize": False,
            "normalization_scope": normalization,
            "min_weight": 1.0e-8,
            "max_weight": 1.0e8,
            "min_condition_samples": 8,
            "eps": 1.0e-8,
            "detach_classifier_update": True,
        },
        "fusion": {
            "gate_regularization": 0.0,
            "gate_init": gate_init,
            "detach_global_for_gate": False,
        },
    }
    model = create_ratio_estimator(config, backbone=backbone).to(device)
    metadata = {
        "architecture": architecture,
        "backbone_hidden_dim": backbone_hidden,
        "ratio_hidden_dims": ratio_hidden,
        "parameter_count": sum(
            parameter.numel() for parameter in model.parameters()
        ),
        "classifier_ratio_clip_min": ratio_clip_min,
        "classifier_ratio_clip_max": ratio_clip_max,
        "normalization": normalization,
        "fusion_gate_init": gate_init,
    }
    return model, metadata


def _selected_logits(
    model: nn.Module, images: torch.Tensor, complements: torch.Tensor
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    outputs = model.ratio_training_outputs(images, complements)
    all_logits = outputs["conditional_logits"]
    states = complements.long()
    selected = all_logits.gather(
        2, states.unsqueeze(-1)
    ).squeeze(-1)
    return selected, outputs


def raw_ratios_from_model(
    model: nn.Module, images: torch.Tensor, complements: torch.Tensor
) -> torch.Tensor:
    """Return prior-corrected odds before all clipping and normalization."""
    selected, _ = _selected_logits(model, images, complements)
    states = complements.long()
    correction = model.log_prior_correction.unsqueeze(0).expand(
        selected.shape[0], -1, -1
    )
    correction = correction.gather(
        2, states.unsqueeze(-1)
    ).squeeze(-1)
    log_ratio = selected.double() + correction.double()
    return torch.exp(log_ratio).float()


def _histogram(values: torch.Tensor, bins: int = 10) -> str:
    if values.numel() == 0:
        return ""
    counts = torch.histc(values.float().cpu(), bins=bins, min=0.0, max=1.0)
    return ";".join(str(int(value)) for value in counts.tolist())


def _ece(probabilities: torch.Tensor, labels: torch.Tensor, bins: int = 10) -> float:
    if probabilities.numel() == 0:
        return float("nan")
    score = probabilities.new_zeros(())
    boundaries = torch.linspace(
        0.0, 1.0, bins + 1, device=probabilities.device
    )
    for index in range(bins):
        if index == bins - 1:
            mask = (
                (probabilities >= boundaries[index])
                & (probabilities <= boundaries[index + 1])
            )
        else:
            mask = (
                (probabilities >= boundaries[index])
                & (probabilities < boundaries[index + 1])
            )
        if bool(mask.any().item()):
            score = score + mask.float().mean() * (
                probabilities[mask].mean() - labels[mask].float().mean()
            ).abs()
    return float(score.item())


@torch.no_grad()
def _collect_domain_outputs(
    model: nn.Module,
    source_dataset: Dataset,
    target_dataset: Dataset,
    device: torch.device,
    batch_size: int,
    workers: int,
    maximum: int,
) -> Dict[str, torch.Tensor]:
    collected = {}
    for domain, dataset in (
        ("source", source_dataset),
        ("target", target_dataset),
    ):
        bounded = Subset(dataset, list(range(min(len(dataset), maximum))))
        loader = DataLoader(
            WeakPairView(bounded),
            batch_size=batch_size,
            shuffle=False,
            num_workers=workers,
        )
        logits_parts = []
        complements_parts = []
        raw_parts = []
        gate_parts = []
        global_parts = []
        conditional_parts = []
        fused_parts = []
        for images, complements in loader:
            images = images.to(device)
            complements = complements.to(device)
            selected, outputs = _selected_logits(model, images, complements)
            logits_parts.append(selected.cpu())
            complements_parts.append(complements.cpu())
            raw_parts.append(
                raw_ratios_from_model(model, images, complements).cpu()
            )
            if "gates" in outputs:
                states = complements.long()
                gate_parts.append(
                    outputs["gates"].gather(
                        2, states.unsqueeze(-1)
                    ).squeeze(-1).cpu()
                )
                conditional_parts.append(
                    outputs["raw_conditional_logits"].gather(
                        2, states.unsqueeze(-1)
                    ).squeeze(-1).cpu()
                )
                fused_parts.append(
                    outputs["conditional_logits"].gather(
                        2, states.unsqueeze(-1)
                    ).squeeze(-1).cpu()
                )
                global_logit = outputs["global_logits"]
                if global_logit.ndim > 1:
                    global_logit = global_logit.reshape(global_logit.shape[0], -1)[
                        :, 0
                    ]
                global_parts.append(
                    global_logit[:, None].expand_as(selected).cpu()
                )
        collected[domain + "_logits"] = torch.cat(logits_parts, dim=0)
        collected[domain + "_complements"] = torch.cat(
            complements_parts, dim=0
        )
        collected[domain + "_raw"] = torch.cat(raw_parts, dim=0)
        if gate_parts:
            collected[domain + "_gates"] = torch.cat(gate_parts, dim=0)
            collected[domain + "_global"] = torch.cat(global_parts, dim=0)
            collected[domain + "_conditional"] = torch.cat(
                conditional_parts, dim=0
            )
            collected[domain + "_fused"] = torch.cat(fused_parts, dim=0)
    return collected


def discriminator_metric_rows(
    collected: Dict[str, torch.Tensor], epoch: int, split: str
) -> List[Dict[str, Any]]:
    rows = []
    source_logits = collected["source_logits"]
    target_logits = collected["target_logits"]
    source_states = collected["source_complements"].long()
    target_states = collected["target_complements"].long()
    for k in range(source_logits.shape[1]):
        for b in range(2):
            source_values = source_logits[source_states[:, k] == b, k]
            target_values = target_logits[target_states[:, k] == b, k]
            logits = torch.cat((source_values, target_values))
            labels = torch.cat(
                (
                    torch.zeros_like(source_values),
                    torch.ones_like(target_values),
                )
            )
            probabilities = torch.sigmoid(logits)
            source_probabilities = torch.sigmoid(source_values)
            target_probabilities = torch.sigmoid(target_values)
            rows.append(
                {
                    "epoch": epoch,
                    "split": split,
                    "class_index": k,
                    "state": b,
                    "source_count": int(source_values.numel()),
                    "target_count": int(target_values.numel()),
                    "domain_loss": float(
                        F.binary_cross_entropy_with_logits(
                            logits, labels
                        ).item()
                    ),
                    "domain_accuracy": float(
                        (
                            (probabilities >= 0.5)
                            == (labels >= 0.5)
                        ).float().mean().item()
                    ),
                    "source_probability_mean": float(
                        source_probabilities.mean().item()
                    ),
                    "target_probability_mean": float(
                        target_probabilities.mean().item()
                    ),
                    "brier_score": float(
                        (probabilities - labels).square().mean().item()
                    ),
                    "ece": _ece(probabilities, labels),
                    "source_probability_histogram": _histogram(
                        source_probabilities
                    ),
                    "target_probability_histogram": _histogram(
                        target_probabilities
                    ),
                }
            )
    return rows


def _gate_distribution(values: torch.Tensor) -> Dict[str, Any]:
    quantiles = torch.quantile(
        values.float(),
        torch.tensor(
            [0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99],
            dtype=torch.float32,
            device=values.device,
        ),
    )
    return {
        "gate_mean": float(values.mean().item()),
        "gate_std": float(values.std(unbiased=False).item()),
        "gate_min": float(values.min().item()),
        "gate_max": float(values.max().item()),
        "gate_q01": float(quantiles[0].item()),
        "gate_q05": float(quantiles[1].item()),
        "gate_q25": float(quantiles[2].item()),
        "gate_q50": float(quantiles[3].item()),
        "gate_q75": float(quantiles[4].item()),
        "gate_q95": float(quantiles[5].item()),
        "gate_q99": float(quantiles[6].item()),
        "gate_histogram": _histogram(values),
        "fraction_gate_lt_0_1": float((values < 0.1).float().mean().item()),
        "fraction_gate_gt_0_9": float((values > 0.9).float().mean().item()),
    }


def gate_metric_rows(
    collected: Dict[str, torch.Tensor], epoch: int, split: str
) -> List[Dict[str, Any]]:
    rows = []
    for domain in ("source", "target"):
        gate_key = domain + "_gates"
        if gate_key not in collected:
            continue
        gates = collected[gate_key]
        states = collected[domain + "_complements"].long()
        global_log = collected[domain + "_global"]
        conditional_log = collected[domain + "_conditional"]
        fused_log = collected[domain + "_fused"]
        for k in range(gates.shape[1]):
            for b in range(2):
                mask = states[:, k] == b
                gate = gates[mask, k]
                global_values = global_log[mask, k]
                conditional_values = conditional_log[mask, k]
                fused_values = fused_log[mask, k]
                row = {
                    "epoch": epoch,
                    "split": split,
                    "domain": domain,
                    "class_index": k,
                    "state": b,
                    "count": int(gate.numel()),
                    "global_log_ratio_mean": float(global_values.mean().item()),
                    "global_log_ratio_std": float(
                        global_values.std(unbiased=False).item()
                    ),
                    "conditional_log_ratio_mean": float(
                        conditional_values.mean().item()
                    ),
                    "conditional_log_ratio_std": float(
                        conditional_values.std(unbiased=False).item()
                    ),
                    "fused_log_ratio_mean": float(fused_values.mean().item()),
                    "fused_log_ratio_std": float(
                        fused_values.std(unbiased=False).item()
                    ),
                    "abs_global_minus_conditional": float(
                        (global_values - conditional_values).abs().mean().item()
                    ),
                    "abs_fused_minus_global": float(
                        (fused_values - global_values).abs().mean().item()
                    ),
                    "abs_fused_minus_conditional": float(
                        (fused_values - conditional_values).abs().mean().item()
                    ),
                }
                row.update(_gate_distribution(gate))
                rows.append(row)
    return rows


def _branch_metric_rows(
    model: nn.Module,
    statistics: Dict[str, Any],
    epoch: int,
) -> List[Dict[str, Any]]:
    source_count = torch.as_tensor(statistics["source_count"])
    target_count = torch.as_tensor(statistics["target_count"])
    rows = []
    for k in range(source_count.shape[0]):
        for b in range(2):
            rows.append(
                {
                    "epoch": epoch,
                    "stage": "ratio_training",
                    "class_index": k,
                    "state": b,
                    "source_count": int(source_count[k, b].item()),
                    "target_count": int(target_count[k, b].item()),
                    "branch_initialized": bool(
                        model.branch_initialized[k, b].item()
                    ),
                    "skipped_update_count": int(
                        model.skipped_update_count[k, b].item()
                    ),
                }
            )
    return rows


def pretrain_ratio_with_diagnostics(
    model: nn.Module,
    source_dataset: Dataset,
    target_dataset: Dataset,
    device: torch.device,
    batch_size: int,
    epochs: int,
    learning_rate: float,
    weight_decay: float,
    workers: int,
    validation_fraction: float,
    seed: int,
    ratio_clip_min: float,
    ratio_clip_max: float,
    normalization: str,
    diagnostic_max_samples: int,
) -> Dict[str, Any]:
    source_train_indices, source_validation_indices = _split_indices(
        len(source_dataset), seed + 701, validation_fraction
    )
    target_train_indices, target_validation_indices = _split_indices(
        len(target_dataset), seed + 907, validation_fraction
    )
    source_train = Subset(source_dataset, source_train_indices)
    target_train = Subset(target_dataset, target_train_indices)
    source_validation = Subset(source_dataset, source_validation_indices)
    target_validation = Subset(target_dataset, target_validation_indices)
    source_loader = DataLoader(
        WeakPairView(source_train),
        batch_size=batch_size,
        shuffle=True,
        num_workers=workers,
    )
    target_loader = DataLoader(
        WeakPairView(target_train),
        batch_size=batch_size,
        shuffle=True,
        num_workers=workers,
    )
    optimizer = torch.optim.Adam(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    trainer = RatioEstimatorTrainer(
        model,
        optimizer,
        device=device,
        branch_balancing="balanced_bce",
        min_branch_count=8,
        max_branch_weight=20.0,
        balanced_domain_sampling=True,
        use_amp=device.type == "cuda",
        grad_clip_norm=5.0,
    )
    statistics = trainer.compute_branch_statistics(
        source_loader, target_loader
    )
    output = {
        "ratio_raw_stats.csv": [],
        "ratio_clipped_stats.csv": [],
        "ratio_normalized_stats.csv": [],
        "discriminator_metrics.csv": [],
        "gate_metrics.csv": [],
        "branch_metrics.csv": [],
        "epoch_summaries": [],
        "split_hash": ratio_split_hash(
            source_train_indices,
            source_validation_indices,
            target_train_indices,
            target_validation_indices,
        ),
    }
    start = time.perf_counter()
    for epoch in range(1, epochs + 1):
        target_iterator = iter(target_loader)
        step_metrics = []
        for source_batch in source_loader:
            target_batch, target_iterator = _next_batch(
                target_loader, target_iterator
            )
            step_metrics.append(
                trainer.train_step(source_batch, target_batch)
            )
        summary = trainer._aggregate_metrics(step_metrics)
        summary["epoch"] = epoch
        output["epoch_summaries"].append(summary)

        train_collected = _collect_domain_outputs(
            model,
            source_train,
            target_train,
            device,
            batch_size,
            workers,
            diagnostic_max_samples,
        )
        validation_collected = _collect_domain_outputs(
            model,
            source_validation,
            target_validation,
            device,
            batch_size,
            workers,
            diagnostic_max_samples,
        )
        output["discriminator_metrics.csv"].extend(
            discriminator_metric_rows(train_collected, epoch, "train")
        )
        output["discriminator_metrics.csv"].extend(
            discriminator_metric_rows(
                validation_collected, epoch, "validation"
            )
        )
        stages = process_weight_stages(
            validation_collected["source_raw"],
            validation_collected["source_complements"],
            min_weight=ratio_clip_min,
            max_weight=ratio_clip_max,
            normalization_scope=normalization,
        )
        stage_rows = weight_stage_rows(
            stages,
            validation_collected["source_complements"],
            epoch,
            ratio_clip_min,
            ratio_clip_max,
            domain="source_validation",
        )
        output["ratio_raw_stats.csv"].extend(stage_rows["raw"])
        output["ratio_clipped_stats.csv"].extend(stage_rows["clipped"])
        output["ratio_normalized_stats.csv"].extend(
            stage_rows["normalized"]
        )
        output["branch_metrics.csv"].extend(
            _branch_metric_rows(model, statistics, epoch)
        )
        output["gate_metrics.csv"].extend(
            gate_metric_rows(validation_collected, epoch, "validation")
        )
        print(
            "Ratio epoch {}: loss {:.6f}, domain accuracy {:.4f}, "
            "raw ratio {:.4f}+/-{:.4f}".format(
                epoch,
                summary["total_loss"],
                summary["discriminator_accuracy"],
                summary["ratio_mean"],
                summary["ratio_std"],
            )
        )
    output["seconds"] = time.perf_counter() - start
    output["statistics"] = statistics
    output["validation"] = trainer.validate(
        DataLoader(
            WeakPairView(source_validation),
            batch_size=batch_size,
            shuffle=False,
            num_workers=workers,
        ),
        DataLoader(
            WeakPairView(target_validation),
            batch_size=batch_size,
            shuffle=False,
            num_workers=workers,
        ),
    )
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return output


@torch.no_grad()
def cache_source_raw_weights(
    model: nn.Module,
    source_dataset: Dataset,
    num_classes: int,
    device: torch.device,
    batch_size: int,
    workers: int,
) -> Tuple[torch.Tensor, float]:
    loader = DataLoader(
        source_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
    )
    cache = torch.empty(
        len(source_dataset), num_classes, dtype=torch.float32
    )
    start = time.perf_counter()
    for indices, images, complements in loader:
        cache[indices.long()] = raw_ratios_from_model(
            model, images.to(device), complements.to(device)
        ).cpu()
    return cache, time.perf_counter() - start


@torch.no_grad()
def evaluate_classifier(
    loader: DataLoader,
    model: nn.Module,
    device: torch.device,
    num_classes: int,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    model.eval()
    total = 0
    correct = 0
    confidence_sum = 0.0
    nll_sum = 0.0
    probability_parts = []
    correctness_parts = []
    per_class_correct = torch.zeros(num_classes, dtype=torch.long)
    per_class_total = torch.zeros(num_classes, dtype=torch.long)
    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device).long()
        logits = model(images)
        probabilities = torch.softmax(logits, dim=1)
        confidence, prediction = probabilities.max(dim=1)
        correctness = prediction.eq(labels)
        batch_size = labels.numel()
        total += batch_size
        correct += int(correctness.sum().item())
        confidence_sum += float(confidence.sum().item())
        nll_sum += float(
            F.cross_entropy(logits, labels, reduction="sum").item()
        )
        probability_parts.append(confidence.cpu())
        correctness_parts.append(correctness.float().cpu())
        for k in range(num_classes):
            mask = labels == k
            per_class_total[k] += int(mask.sum().item())
            per_class_correct[k] += int(
                (correctness & mask).sum().item()
            )
    probabilities = torch.cat(probability_parts)
    correctness = torch.cat(correctness_parts)
    summary = {
        "accuracy": 100.0 * correct / float(max(total, 1)),
        "prediction_confidence": confidence_sum / float(max(total, 1)),
        "ece": _ece(probabilities, correctness),
        "nll": nll_sum / float(max(total, 1)),
    }
    per_class = []
    for k in range(num_classes):
        support = int(per_class_total[k].item())
        per_class.append(
            {
                "class_index": k,
                "accuracy": 100.0
                * int(per_class_correct[k].item())
                / float(max(support, 1)),
                "support": support,
            }
        )
    return summary, per_class


@torch.no_grad()
def collect_classifier_features(
    dataset: Dataset,
    classifier: nn.Module,
    device: torch.device,
    batch_size: int,
    workers: int,
    maximum: int,
) -> torch.Tensor:
    """Collect the reused SCARCE MLP hidden representation."""
    loader = DataLoader(
        Subset(dataset, list(range(min(len(dataset), maximum)))),
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
    )
    classifier.eval()
    parts = []
    for batch in loader:
        images = batch[1].to(device)
        flattened = images.reshape(images.shape[0], -1)
        parts.append(
            classifier.relu1(classifier.fc1(flattened)).detach().cpu()
        )
    return torch.cat(parts, dim=0)


@torch.no_grad()
def evaluate_weak_risk(
    modules: Dict[str, Any],
    dataset: Dataset,
    model: nn.Module,
    device: torch.device,
    batch_size: int,
    workers: int,
    class_prior: torch.Tensor,
    complement_prior: torch.Tensor,
    raw_weight_cache: Optional[torch.Tensor],
    ratio_clip_min: float,
    ratio_clip_max: float,
    normalization: str,
) -> Tuple[float, List[float]]:
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
    )
    total_risk = 0.0
    total_samples = 0
    per_class_sum = None
    for indices, images, complements in loader:
        images = images.to(device)
        complements = complements.to(device)
        outputs = model(images)
        weights = None
        if raw_weight_cache is not None:
            raw = raw_weight_cache[indices.long()].to(device)
            weights = process_weight_stages(
                raw,
                complements,
                min_weight=ratio_clip_min,
                max_weight=ratio_clip_max,
                normalization_scope=normalization,
            )["classifier_used"]
        risk, diagnostics = modules["adiw_scar"].scar_branch_risk(
            outputs,
            complements,
            class_prior,
            complement_prior,
            branch_weights=weights,
            correction="abs",
        )
        class_risk = (
            diagnostics["negative"]
            + diagnostics["positive_corrected"]
        )
        current = images.shape[0]
        total_risk += float(risk.item()) * current
        total_samples += current
        if per_class_sum is None:
            per_class_sum = class_risk.detach().cpu() * current
        else:
            per_class_sum += class_risk.detach().cpu() * current
    return (
        total_risk / float(max(total_samples, 1)),
        (per_class_sum / float(max(total_samples, 1))).tolist(),
    )


def _pearson(left: Sequence[float], right: Sequence[float]) -> Optional[float]:
    if len(left) < 2:
        return None
    left_array = np.asarray(left, dtype=np.float64)
    right_array = np.asarray(right, dtype=np.float64)
    if left_array.std() == 0.0 or right_array.std() == 0.0:
        return None
    return float(np.corrcoef(left_array, right_array)[0, 1])


def _rank(values: Sequence[float]) -> np.ndarray:
    order = np.argsort(np.asarray(values), kind="mergesort")
    ranks = np.empty(len(order), dtype=np.float64)
    ranks[order] = np.arange(len(order), dtype=np.float64)
    return ranks


def _spearman_list(
    left: Sequence[float], right: Sequence[float]
) -> Optional[float]:
    if len(left) < 2:
        return None
    return _pearson(_rank(left), _rank(right))


def train_classifier(
    modules: Dict[str, Any],
    method: str,
    data: Dict[str, Any],
    device: torch.device,
    model_seed: int,
    batch_seed: int,
    weight_shuffle_seed: int,
    batch_size: int,
    epochs: int,
    learning_rate: float,
    weight_decay: float,
    workers: int,
    raw_weight_cache: Optional[torch.Tensor],
    ratio_clip_min: float,
    ratio_clip_max: float,
    normalization: str,
) -> Dict[str, Any]:
    """Train without target-test-driven selection and record every epoch."""
    set_random_seed(model_seed)
    source_dataset = data["source_weak"]
    target_dataset = data["target_weak"]
    source_generator = torch.Generator(device="cpu")
    source_generator.manual_seed(int(batch_seed))
    target_generator = torch.Generator(device="cpu")
    target_generator.manual_seed(int(batch_seed) + 1)
    source_loader = DataLoader(
        source_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=workers,
        generator=source_generator,
    )
    target_loader = DataLoader(
        target_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=workers,
        generator=target_generator,
    )
    target_evaluation_loader = DataLoader(
        data["target_evaluation"],
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
    )
    source_evaluation_loader = DataLoader(
        data["source_evaluation"],
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
    )
    classifier = modules["models"].mlp_model(
        input_dim=data["input_dim"],
        hidden_dim=500,
        output_dim=data["num_classes"],
    ).to(device)
    optimizer = torch.optim.Adam(
        classifier.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    num_classes = data["num_classes"]
    uniform_prior = torch.full(
        (num_classes,), 1.0 / num_classes, device=device
    )
    source_complement_prior = data["source_complement_prior"].to(device)
    target_complement_prior = data["target_complement_prior"].to(device)
    target_fraction = float(len(target_dataset)) / float(
        len(source_dataset) + len(target_dataset)
    )

    pooled_loader = None
    pooled_complement_prior = None
    if method == "pooled_scar":
        pooled_generator = torch.Generator(device="cpu")
        pooled_generator.manual_seed(int(batch_seed))
        pooled_loader = DataLoader(
            ConcatDataset((source_dataset, target_dataset)),
            batch_size=batch_size,
            shuffle=True,
            num_workers=workers,
            generator=pooled_generator,
        )
        source_fraction = 1.0 - target_fraction
        pooled_complement_prior = (
            source_fraction * source_complement_prior
            + target_fraction * target_complement_prior
        )
    weight_memory = None
    adiw_methods = (
        "adiw",
        "adiw_original",
        "adiw_ones",
        "adiw_mean",
        "adiw_shuffle",
    )
    if method in adiw_methods:
        weight_memory = modules["adiw_scar"].BranchWeightMemory(
            len(source_dataset), num_classes, device
        )
    shuffle_generator = torch.Generator(device="cpu")
    shuffle_generator.manual_seed(int(weight_shuffle_seed))
    classifier_used_cache = torch.ones(
        len(source_dataset), num_classes, dtype=torch.float32
    )

    epoch_rows = []
    per_class_rows = []
    classifier_weight_rows = []
    final_raw_weight_summary = summarize_weight_values(
        classifier_used_cache,
        raw_reference=classifier_used_cache,
        min_weight=ratio_clip_min,
        max_weight=ratio_clip_max,
    )
    target_iterator = iter(target_loader)
    start = time.perf_counter()
    for epoch in range(1, epochs + 1):
        classifier.train()
        total_loss = 0.0
        total_samples = 0
        source_contribution_sum = 0.0
        target_contribution_sum = 0.0
        gradient_norm_sum = 0.0
        ratio_loss_values = []
        epoch_raw_weights = []
        epoch_used_weights = []
        epoch_complements = []
        training_loader = (
            pooled_loader if method == "pooled_scar" else source_loader
        )
        for batch in training_loader:
            indices, images, complements = batch
            images = images.to(device)
            complements = complements.to(device)
            optimizer.zero_grad()
            if method in ("no_iw", "current_no_iw", "pooled_scar"):
                outputs = classifier(images)
                complement_prior = (
                    pooled_complement_prior
                    if method == "pooled_scar"
                    else source_complement_prior
                )
                loss, risk_diagnostics = modules[
                    "adiw_scar"
                ].scar_branch_risk(
                    outputs,
                    complements,
                    uniform_prior,
                    complement_prior,
                    branch_weights=None,
                    correction="abs",
                )
                source_contribution = loss
                target_contribution = loss.new_zeros(())
            else:
                target_batch, target_iterator = _next_batch(
                    target_loader, target_iterator
                )
                _, target_images, target_complements = target_batch
                target_images = target_images.to(device)
                target_complements = target_complements.to(device)
                source_outputs = classifier(images)
                target_outputs = classifier(target_images)
                if method in adiw_methods:
                    source_representation = modules[
                        "adiw_scar"
                    ].loss_value_representation(source_outputs.detach())
                    target_representation = modules[
                        "adiw_scar"
                    ].loss_value_representation(target_outputs.detach())
                    raw_weights = weight_memory.estimate(
                        indices.to(device),
                        source_representation,
                        complements,
                        target_representation,
                        target_complements,
                        source_complement_prior,
                        target_complement_prior,
                        step_size=1.0,
                        num_steps=1,
                        max_weight=50.0,
                        kernel_quantile=0.5,
                        sum_tolerance=None,
                    )
                    if not ratio_loss_values:
                        branch_objectives = []
                        for class_index in range(num_classes):
                            for observed in (False, True):
                                source_mask = (
                                    complements[:, class_index] > 0.5
                                ) == observed
                                target_mask = (
                                    target_complements[:, class_index] > 0.5
                                ) == observed
                                if not bool(source_mask.any().item()) or not bool(
                                    target_mask.any().item()
                                ):
                                    continue
                                source_values = source_representation[
                                    source_mask, class_index
                                ].reshape(-1, 1)
                                target_values = target_representation[
                                    target_mask, class_index
                                ].reshape(-1, 1)
                                branch_weights = raw_weights[
                                    source_mask, class_index
                                ]
                                gamma = modules[
                                    "adiw_scar"
                                ].rbf_gamma_from_quantile(
                                    source_values,
                                    target_values,
                                    quantile=0.5,
                                )
                                branch_objectives.append(
                                    modules[
                                        "adiw_scar"
                                    ].mmd_weight_objective(
                                        source_values,
                                        target_values,
                                        branch_weights,
                                        gamma,
                                    )
                                )
                        if branch_objectives:
                            ratio_loss_values.append(
                                float(
                                    torch.stack(branch_objectives)
                                    .mean()
                                    .detach()
                                    .item()
                                )
                            )
                elif method == "matched_no_iw":
                    raw_weights = torch.ones(
                        images.shape[0],
                        num_classes,
                        device=device,
                    )
                else:
                    raw_weights = raw_weight_cache[indices.long()].to(device)
                stages = process_weight_stages(
                    raw_weights,
                    complements,
                    min_weight=ratio_clip_min,
                    max_weight=ratio_clip_max,
                    normalization_scope=normalization,
                )
                source_weights = stages["classifier_used"]
                if method in (
                    "adiw_original",
                    "adiw_ones",
                    "adiw_mean",
                    "adiw_shuffle",
                ):
                    source_weights = apply_weight_intervention(
                        source_weights,
                        classifier_method_contract(method),
                        generator=shuffle_generator,
                    )
                stages["classifier_used"] = source_weights
                classifier_used_cache[indices.long().cpu()] = (
                    source_weights.detach().cpu()
                )
                epoch_raw_weights.append(raw_weights.detach().cpu())
                epoch_used_weights.append(source_weights.detach().cpu())
                epoch_complements.append(complements.detach().cpu())
                loss, risk_diagnostics = modules[
                    "adiw_scar"
                ].combined_adiw_scar_loss(
                    source_outputs,
                    complements,
                    source_weights,
                    target_outputs,
                    target_complements,
                    uniform_prior,
                    target_complement_prior,
                    target_fraction,
                    correction="abs",
                )
                source_contribution = (
                    (1.0 - target_fraction)
                    * risk_diagnostics["source_risk"]
                )
                target_contribution = (
                    target_fraction * risk_diagnostics["target_risk"]
                )
            if not bool(torch.isfinite(loss).item()):
                raise FloatingPointError(
                    "non-finite classifier loss at epoch {}".format(epoch)
                )
            loss.backward()
            squared_gradient_norm = loss.new_zeros(())
            for parameter in classifier.parameters():
                if parameter.grad is not None:
                    squared_gradient_norm = (
                        squared_gradient_norm
                        + parameter.grad.detach().square().sum()
                    )
            gradient_norm = squared_gradient_norm.sqrt()
            optimizer.step()
            current = images.shape[0]
            total_loss += float(loss.item()) * current
            source_contribution_sum += (
                float(source_contribution.detach().item()) * current
            )
            target_contribution_sum += (
                float(target_contribution.detach().item()) * current
            )
            gradient_norm_sum += float(gradient_norm.item()) * current
            total_samples += current

        if epoch_raw_weights:
            raw_epoch = torch.cat(epoch_raw_weights, dim=0)
            complements_epoch = torch.cat(epoch_complements, dim=0)
            stages = process_weight_stages(
                raw_epoch,
                complements_epoch,
                min_weight=ratio_clip_min,
                max_weight=ratio_clip_max,
                normalization_scope=normalization,
            )
            final_raw_weight_summary = summarize_weight_values(
                stages["raw"],
                raw_reference=stages["raw"],
                min_weight=ratio_clip_min,
                max_weight=ratio_clip_max,
            )
            if epoch_used_weights:
                stages["classifier_used"] = torch.cat(
                    epoch_used_weights, dim=0
                )
            stage_rows = weight_stage_rows(
                stages,
                complements_epoch,
                epoch,
                ratio_clip_min,
                ratio_clip_max,
                domain="source_training",
            )
            classifier_weight_rows.extend(
                stage_rows["classifier_used"]
            )
            weight_summary = summarize_weight_values(stages["classifier_used"])
        else:
            weight_summary = summarize_weight_values(classifier_used_cache)

        source_cache_for_eval = raw_weight_cache
        if method in adiw_methods:
            source_cache_for_eval = weight_memory.weights.detach().cpu()
        if method in (
            "no_iw",
            "current_no_iw",
            "matched_no_iw",
            "pooled_scar",
        ):
            source_cache_for_eval = None
        source_risk, source_class_risk = evaluate_weak_risk(
            modules,
            source_dataset,
            classifier,
            device,
            batch_size,
            workers,
            uniform_prior,
            source_complement_prior,
            source_cache_for_eval,
            ratio_clip_min,
            ratio_clip_max,
            normalization,
        )
        target_risk, target_class_risk = evaluate_weak_risk(
            modules,
            target_dataset,
            classifier,
            device,
            batch_size,
            workers,
            uniform_prior,
            target_complement_prior,
            None,
            ratio_clip_min,
            ratio_clip_max,
            normalization,
        )
        source_eval, source_per_class = evaluate_classifier(
            source_evaluation_loader, classifier, device, num_classes
        )
        target_eval, target_per_class = evaluate_classifier(
            target_evaluation_loader, classifier, device, num_classes
        )
        epoch_rows.append(
            {
                "epoch": epoch,
                "classifier_training_loss": total_loss
                / float(max(total_samples, 1)),
                "source_ovr_risk": source_risk,
                "target_ovr_risk": target_risk,
                "source_accuracy": source_eval["accuracy"],
                "target_accuracy": target_eval["accuracy"],
                "target_prediction_confidence": target_eval[
                    "prediction_confidence"
                ],
                "target_ece": target_eval["ece"],
                "target_nll": target_eval["nll"],
                "true_target_risk": target_eval["nll"],
                "estimated_weak_risk": target_risk,
                "estimated_iw_risk": source_risk,
                "source_risk_contribution": source_contribution_sum
                / float(max(total_samples, 1)),
                "target_risk_contribution": target_contribution_sum
                / float(max(total_samples, 1)),
                "total_loss_scale": total_loss
                / float(max(total_samples, 1)),
                "classifier_gradient_norm": gradient_norm_sum
                / float(max(total_samples, 1)),
                "ratio_estimator_loss": (
                    float(np.mean(ratio_loss_values))
                    if ratio_loss_values
                    else None
                ),
                "weight_mean": weight_summary["mean"],
                "weight_std": weight_summary["std"],
                "weight_cv": weight_summary["cv"],
                "weight_min": weight_summary["min"],
                "weight_max": weight_summary["max"],
                "weight_q01": weight_summary["q01"],
                "weight_q05": weight_summary["q05"],
                "weight_q50": weight_summary["q50"],
                "weight_q95": weight_summary["q95"],
                "weight_q99": weight_summary["q99"],
                "weight_ess": weight_summary["ess"],
                "normalized_ess": weight_summary["normalized_ess"],
            }
        )
        for domain, class_metrics, class_risks in (
            ("source", source_per_class, source_class_risk),
            ("target", target_per_class, target_class_risk),
        ):
            for item in class_metrics:
                per_class_rows.append(
                    {
                        "epoch": epoch,
                        "domain": domain,
                        "class_index": item["class_index"],
                        "accuracy": item["accuracy"],
                        "ovr_risk": class_risks[item["class_index"]],
                        "support": item["support"],
                    }
                )
        print(
            "Classifier epoch {}: loss {:.4f}, source acc {:.3f}, "
            "target acc {:.3f}, target risk {:.4f}".format(
                epoch,
                epoch_rows[-1]["classifier_training_loss"],
                source_eval["accuracy"],
                target_eval["accuracy"],
                target_risk,
            )
        )

    target_accuracies = [row["target_accuracy"] for row in epoch_rows]
    target_risks = [row["target_ovr_risk"] for row in epoch_rows]
    last_count = min(10, len(epoch_rows))
    best_index = int(np.argmax(target_accuracies))
    minimum_risk_index = int(np.argmin(target_risks))
    return {
        "model": classifier,
        "source_weight_cache": (
            None
            if weight_memory is None
            else weight_memory.weights.detach().cpu()
        ),
        "classifier_used_cache": classifier_used_cache,
        "final_raw_weight_summary": final_raw_weight_summary,
        "epoch_rows": epoch_rows,
        "per_class_rows": per_class_rows,
        "classifier_weight_rows": classifier_weight_rows,
        "seconds": time.perf_counter() - start,
        "classifier_parameter_count": sum(
            parameter.numel() for parameter in classifier.parameters()
        ),
        "final_target_accuracy": target_accuracies[-1],
        "last10_target_accuracy": float(
            np.mean(target_accuracies[-last_count:])
        ),
        "last10_target_risk": float(np.mean(target_risks[-last_count:])),
        "best_target_accuracy": target_accuracies[best_index],
        "best_target_accuracy_epoch": best_index + 1,
        "minimum_target_risk": target_risks[minimum_risk_index],
        "minimum_target_risk_epoch": minimum_risk_index + 1,
        "final_source_accuracy": epoch_rows[-1]["source_accuracy"],
        "final_source_ovr_risk": epoch_rows[-1]["source_ovr_risk"],
        "final_target_ovr_risk": epoch_rows[-1]["target_ovr_risk"],
        "risk_accuracy_pearson": _pearson(
            target_risks, target_accuracies
        ),
        "risk_accuracy_spearman": _spearman_list(
            target_risks, target_accuracies
        ),
        "target_fraction": target_fraction,
    }


def _stage_summary(
    rows: Sequence[Dict[str, Any]], epoch: int
) -> Dict[str, Optional[float]]:
    selected = [row for row in rows if int(row["epoch"]) == int(epoch)]
    if not selected:
        return {
            "mean": None,
            "std": None,
            "ess": None,
            "lower_clipping_rate": None,
            "upper_clipping_rate": None,
        }
    total = float(sum(max(int(row["count"]), 0) for row in selected))
    if total <= 0:
        total = float(len(selected))

    def weighted(field: str) -> Optional[float]:
        applicable = [
            row for row in selected if row.get(field) not in (None, "")
        ]
        if not applicable:
            return None
        denominator = float(
            sum(max(int(row["count"]), 0) for row in applicable)
        )
        if denominator <= 0:
            denominator = float(len(applicable))
        return sum(
            float(row[field]) * max(int(row["count"]), 1)
            for row in applicable
        ) / denominator

    return {
        "mean": weighted("mean"),
        "std": weighted("std"),
        "ess": weighted("ess"),
        "lower_clipping_rate": weighted("lower_clipping_rate"),
        "upper_clipping_rate": weighted("upper_clipping_rate"),
    }


def run_experiment(args: argparse.Namespace) -> Dict[str, Any]:
    if args.method not in METHODS:
        raise ValueError("unknown method: {}".format(args.method))
    if args.normalization not in (
        "conditional_branch",
        "class_level",
        "global",
        "none",
    ):
        raise ValueError("invalid normalization")
    if args.device == "auto":
        device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
    else:
        device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")

    if args.class_rotations is not None:
        shift_name = args.shift_label or "class_rotation"
    elif args.shift == "rotation":
        shift_name = "rotation_{:g}".format(args.target_rotation)
    else:
        shift_name = "heterogeneous_class_specific"
    setting_name = "{}_tw{}_clip{:g}-{:g}_norm-{}".format(
        shift_name,
        args.target_weak_size,
        args.ratio_clip_min,
        args.ratio_clip_max,
        args.normalization,
    )
    if (
        args.method == "fusion"
        and abs(args.fusion_gate_init - 0.5) > 1.0e-12
    ):
        setting_name += "_gate{:g}".format(args.fusion_gate_init)
    run_dir = os.path.abspath(
        os.path.join(
            args.results_root,
            args.dataset,
            setting_name,
            args.method,
            "seed_{}".format(args.seed),
        )
    )
    initialize_result_files(run_dir)
    config = vars(args).copy()
    config.update(
        {
            "run_dir": run_dir,
            "shift_name": shift_name,
            "setting_name": setting_name,
            "primary_model_selection_metric": (
                "last_10_classifier_epochs_mean_target_accuracy"
            ),
            "target_test_used_for_selection": False,
        }
    )
    config["config_hash"] = _config_hash(config)
    _json_dump(os.path.join(run_dir, "config.yaml"), config)

    started_at = dt.datetime.now(dt.timezone.utc).isoformat()
    summary = {
        "experiment": args.experiment,
        "dataset": args.dataset,
        "shift_type": shift_name,
        "shift": shift_name,
        "setting": setting_name,
        "seed": args.seed,
        "method": args.method,
        "status": "running",
        "failure_reason": None,
        "last_stable_classifier_epoch": 0,
        "started_at": started_at,
        "git_commit_hash": _git_commit_hash(args.repository_root),
        "git_commit": _git_commit_hash(args.repository_root),
        "config_hash": config["config_hash"],
        "data_seed": args.data_seed,
        "model_seed": args.model_seed,
        "batch_seed": args.batch_seed,
        "ratio_seed": args.ratio_seed,
        "weight_shuffle_seed": args.weight_shuffle_seed,
        "class_rotations": args.class_rotations,
        "target_test_used_for_selection": False,
    }
    _json_dump(os.path.join(run_dir, "summary.json"), summary)
    try:
        set_random_seed(args.data_seed)
        modules = load_scarce_modules(args.scarce_repo)
        data, shift_spec = prepare_matched_data(
            modules,
            args.scarce_repo,
            args.dataset,
            args.target_weak_size,
            args.source_size,
            args.data_seed,
            args.shift,
            args.target_rotation,
            class_rotations=args.class_rotations,
        )
        split_hash = paired_split_hash(data, shift_spec)
        ratio_model = None
        ratio_metadata = {
            "architecture": None,
            "parameter_count": 0,
        }
        ratio_output = {
            "ratio_raw_stats.csv": [],
            "ratio_clipped_stats.csv": [],
            "ratio_normalized_stats.csv": [],
            "discriminator_metrics.csv": [],
            "gate_metrics.csv": [],
            "branch_metrics.csv": [],
            "epoch_summaries": [],
            "seconds": 0.0,
            "split_hash": None,
            "validation": {},
        }
        raw_weight_cache = None
        cache_seconds = 0.0
        if args.method in RATIO_METHODS:
            ratio_model, ratio_metadata = build_next_round_ratio_estimator(
                modules["models"],
                data["input_dim"],
                data["num_classes"],
                args.method,
                device,
                args.ratio_clip_min,
                args.ratio_clip_max,
                args.normalization,
                args.fusion_gate_init,
            )
            ratio_output = pretrain_ratio_with_diagnostics(
                ratio_model,
                data["source_weak"],
                data["target_weak"],
                device,
                args.batch_size,
                args.ratio_epochs,
                args.ratio_learning_rate,
                args.weight_decay,
                args.workers,
                args.ratio_validation_fraction,
                args.ratio_seed,
                args.ratio_clip_min,
                args.ratio_clip_max,
                args.normalization,
                args.diagnostic_max_samples,
            )
            raw_weight_cache, cache_seconds = cache_source_raw_weights(
                ratio_model,
                data["source_weak"],
                data["num_classes"],
                device,
                args.batch_size,
                args.workers,
            )
        for filename in (
            "ratio_raw_stats.csv",
            "ratio_clipped_stats.csv",
            "ratio_normalized_stats.csv",
            "discriminator_metrics.csv",
            "gate_metrics.csv",
            "branch_metrics.csv",
        ):
            _write_csv(
                os.path.join(run_dir, filename),
                ratio_output[filename],
                CSV_FILES[filename],
            )

        classifier_output = train_classifier(
            modules,
            args.method,
            data,
            device,
            args.model_seed,
            args.batch_seed,
            args.weight_shuffle_seed,
            args.batch_size,
            args.classifier_epochs,
            args.learning_rate,
            args.weight_decay,
            args.workers,
            raw_weight_cache,
            args.ratio_clip_min,
            args.ratio_clip_max,
            args.normalization,
        )
        diagnostic_sample_count = min(
            int(args.diagnostic_max_samples), 2048
        )
        source_features = collect_classifier_features(
            data["source_weak"],
            classifier_output["model"],
            device,
            args.batch_size,
            args.workers,
            diagnostic_sample_count,
        )
        target_features = collect_classifier_features(
            data["target_weak"],
            classifier_output["model"],
            device,
            args.batch_size,
            args.workers,
            diagnostic_sample_count,
        )
        distribution_matching = distribution_matching_diagnostics(
            source_features,
            target_features,
            classifier_output["classifier_used_cache"][
                : source_features.shape[0]
            ],
            seed=args.seed + 50000,
        )
        _json_dump(
            os.path.join(run_dir, "distribution_matching.json"),
            distribution_matching,
        )
        _write_csv(
            os.path.join(run_dir, "epoch_metrics.csv"),
            classifier_output["epoch_rows"],
            CSV_FILES["epoch_metrics.csv"],
        )
        _write_csv(
            os.path.join(run_dir, "per_class_metrics.csv"),
            classifier_output["per_class_rows"],
            CSV_FILES["per_class_metrics.csv"],
        )
        _write_csv(
            os.path.join(run_dir, "classifier_used_ratio_stats.csv"),
            classifier_output["classifier_weight_rows"],
            CSV_FILES["classifier_used_ratio_stats.csv"],
        )
        checkpoint = {
            "classifier": classifier_output["model"].state_dict(),
            "ratio": None
            if ratio_model is None
            else ratio_model.state_dict(),
            "source_weight_cache": classifier_output[
                "source_weight_cache"
            ],
            "classifier_used_cache": classifier_output[
                "classifier_used_cache"
            ],
            "config": config,
            "split_hash": split_hash,
        }
        torch.save(checkpoint, os.path.join(run_dir, "checkpoint.pt"))

        raw_summary = _stage_summary(
            ratio_output["ratio_raw_stats.csv"], args.ratio_epochs
        )
        if raw_summary["mean"] is None:
            classifier_raw = classifier_output[
                "final_raw_weight_summary"
            ]
            raw_summary = {
                "mean": classifier_raw["mean"],
                "std": classifier_raw["std"],
                "ess": classifier_raw["ess"],
                "lower_clipping_rate": classifier_raw[
                    "lower_clipping_rate"
                ],
                "upper_clipping_rate": classifier_raw[
                    "upper_clipping_rate"
                ],
            }
        clipped_summary = _stage_summary(
            ratio_output["ratio_clipped_stats.csv"], args.ratio_epochs
        )
        normalized_summary = _stage_summary(
            ratio_output["ratio_normalized_stats.csv"], args.ratio_epochs
        )
        gate_rows = [
            row
            for row in ratio_output["gate_metrics.csv"]
            if int(row["epoch"]) == args.ratio_epochs
            and row["split"] == "validation"
        ]
        gate_summary = None
        if gate_rows:
            gate_summary = {
                "mean": float(
                    np.mean([float(row["gate_mean"]) for row in gate_rows])
                ),
                "std": float(
                    np.mean([float(row["gate_std"]) for row in gate_rows])
                ),
                "fraction_below_0_1": float(
                    np.mean(
                        [
                            float(row["fraction_gate_lt_0_1"])
                            for row in gate_rows
                        ]
                    )
                ),
                "fraction_above_0_9": float(
                    np.mean(
                        [
                            float(row["fraction_gate_gt_0_9"])
                            for row in gate_rows
                        ]
                    )
                ),
            }
        epoch_rows = classifier_output["epoch_rows"]
        summary.update(
            {
                "status": "ok",
                "finished_at": dt.datetime.now(dt.timezone.utc).isoformat(),
                "split_hash": split_hash,
                "ratio_split_hash": (
                    ratio_output["split_hash"]
                    if ratio_output["split_hash"] is not None
                    else split_hash
                ),
                "source_size": len(data["source_weak"]),
                "target_weak_size": len(data["target_weak"]),
                "ratio_parameter_count": ratio_metadata["parameter_count"],
                "classifier_parameter_count": classifier_output[
                    "classifier_parameter_count"
                ],
                "ratio_train_seconds": ratio_output["seconds"],
                "ratio_cache_seconds": cache_seconds,
                "classifier_train_seconds": classifier_output["seconds"],
                "source_accuracy": classifier_output[
                    "final_source_accuracy"
                ],
                "target_accuracy": classifier_output[
                    "last10_target_accuracy"
                ],
                "source_ovr_risk": classifier_output[
                    "final_source_ovr_risk"
                ],
                "target_ovr_risk": classifier_output[
                    "final_target_ovr_risk"
                ],
                "final_target_accuracy": classifier_output[
                    "final_target_accuracy"
                ],
                "last10_target_accuracy": classifier_output[
                    "last10_target_accuracy"
                ],
                "last10_target_ovr_risk": classifier_output[
                    "last10_target_risk"
                ],
                "best_target_accuracy": classifier_output[
                    "best_target_accuracy"
                ],
                "best_target_accuracy_epoch": classifier_output[
                    "best_target_accuracy_epoch"
                ],
                "minimum_target_risk": classifier_output[
                    "minimum_target_risk"
                ],
                "minimum_target_risk_epoch": classifier_output[
                    "minimum_target_risk_epoch"
                ],
                "risk_accuracy_pearson": classifier_output[
                    "risk_accuracy_pearson"
                ],
                "risk_accuracy_spearman": classifier_output[
                    "risk_accuracy_spearman"
                ],
                "raw_ratio": raw_summary,
                "clipped_ratio": clipped_summary,
                "normalized_ratio": normalized_summary,
                "classifier_used_weight_mean": epoch_rows[-1][
                    "weight_mean"
                ],
                "classifier_used_weight_std": epoch_rows[-1]["weight_std"],
                "classifier_used_weight_ess": epoch_rows[-1]["weight_ess"],
                "classifier_used_weight_cv": epoch_rows[-1]["weight_cv"],
                "normalized_ESS": epoch_rows[-1]["normalized_ess"],
                "ESS": epoch_rows[-1]["weight_ess"],
                "true_target_risk": epoch_rows[-1]["true_target_risk"],
                "estimated_weak_risk": epoch_rows[-1][
                    "estimated_weak_risk"
                ],
                "estimated_iw_risk": epoch_rows[-1]["estimated_iw_risk"],
                "runtime": ratio_output["seconds"]
                + cache_seconds
                + classifier_output["seconds"],
                "weighted_MMD": distribution_matching["weighted_mmd"],
                "weighted_mmd": distribution_matching["weighted_mmd"],
                "weighted_feature_mean_error": distribution_matching[
                    "weighted_feature_mean_error"
                ],
                "weighted_covariance_error": distribution_matching[
                    "weighted_covariance_error"
                ],
                "posthoc_domain_accuracy": distribution_matching[
                    "posthoc_domain_accuracy"
                ],
                "posthoc_domain_AUC": distribution_matching[
                    "posthoc_domain_auc"
                ],
                "posthoc_domain_auc": distribution_matching[
                    "posthoc_domain_auc"
                ],
                "raw_ratio_mean": raw_summary["mean"],
                "raw_ratio_std": raw_summary["std"],
                "raw_lower_clip_rate": raw_summary[
                    "lower_clipping_rate"
                ],
                "raw_upper_clip_rate": raw_summary[
                    "upper_clipping_rate"
                ],
                "used_weight_mean": epoch_rows[-1]["weight_mean"],
                "used_weight_std": epoch_rows[-1]["weight_std"],
                "used_weight_CV": epoch_rows[-1]["weight_cv"],
                "lower_clipping_rate": raw_summary[
                    "lower_clipping_rate"
                ],
                "upper_clipping_rate": raw_summary[
                    "upper_clipping_rate"
                ],
                "skipped_branch_count": int(
                    sum(
                        1
                        for row in ratio_output["branch_metrics.csv"]
                        if int(row["epoch"]) == args.ratio_epochs
                        and not bool(row["branch_initialized"])
                    )
                ),
                "ratio_validation": ratio_output["validation"],
                "gate_summary": gate_summary,
                "ratio_architecture": ratio_metadata,
                "last_stable_classifier_epoch": args.classifier_epochs,
                "target_test_used_for_selection": False,
                "primary_result": (
                    "last10_target_accuracy; best target-test epoch is "
                    "diagnostic only"
                ),
            }
        )
        _json_dump(os.path.join(run_dir, "summary.json"), summary)
        print(
            "NEXT_ROUND_RESULT {}".format(
                json.dumps(summary, sort_keys=True)
            )
        )
        return summary
    except Exception as error:
        summary.update(
            {
                "status": "failed",
                "finished_at": dt.datetime.now(dt.timezone.utc).isoformat(),
                "failure_reason": "{}: {}".format(
                    type(error).__name__, error
                ),
            }
        )
        _json_dump(os.path.join(run_dir, "summary.json"), summary)
        with open(
            os.path.join(run_dir, "failure.txt"), "w", encoding="utf-8"
        ) as stream:
            stream.write(traceback.format_exc())
        raise


def parse_args(
    argv: Optional[Sequence[str]] = None
) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scarce-repo", required=True)
    parser.add_argument("--experiment", default="next_round")
    parser.add_argument("--repository-root", default=os.getcwd())
    parser.add_argument("--results-root", default="results/next_round")
    parser.add_argument("--dataset", default="mnist")
    parser.add_argument(
        "--shift",
        choices=("rotation", "heterogeneous", "class_rotation"),
        default="rotation",
    )
    parser.add_argument("--shift-label")
    parser.add_argument(
        "--class-rotations",
        help="comma-separated per-class target rotations",
    )
    parser.add_argument("--target-rotation", type=float, default=30.0)
    parser.add_argument("--target-weak-size", type=int, default=1000)
    parser.add_argument("--source-size", type=int, default=0)
    parser.add_argument("--method", choices=METHODS, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--data-seed", type=int)
    parser.add_argument("--model-seed", type=int)
    parser.add_argument("--batch-seed", type=int)
    parser.add_argument("--ratio-seed", type=int)
    parser.add_argument("--weight-shuffle-seed", type=int)
    parser.add_argument("--ratio-epochs", type=int, default=10)
    parser.add_argument("--classifier-epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--ratio-learning-rate", type=float, default=1.0e-3)
    parser.add_argument("--learning-rate", type=float, default=1.0e-3)
    parser.add_argument("--weight-decay", type=float, default=1.0e-5)
    parser.add_argument("--ratio-clip-min", type=float, default=0.01)
    parser.add_argument("--ratio-clip-max", type=float, default=20.0)
    parser.add_argument(
        "--normalization",
        choices=("conditional_branch", "class_level", "global", "none"),
        default="conditional_branch",
    )
    parser.add_argument("--fusion-gate-init", type=float, default=0.5)
    parser.add_argument(
        "--ratio-validation-fraction", type=float, default=0.2
    )
    parser.add_argument("--diagnostic-max-samples", type=int, default=4096)
    parser.add_argument(
        "--device", choices=("auto", "cpu", "cuda"), default="auto"
    )
    args = parser.parse_args(argv)
    if args.ratio_epochs <= 0 or args.classifier_epochs <= 0:
        parser.error("epoch counts must be positive")
    if not 0.0 < args.ratio_validation_fraction < 1.0:
        parser.error("ratio validation fraction must be between zero and one")
    if args.ratio_clip_min < 0 or args.ratio_clip_max < args.ratio_clip_min:
        parser.error("invalid ratio clip bounds")
    if args.class_rotations:
        try:
            args.class_rotations = [
                float(value) for value in args.class_rotations.split(",")
            ]
        except ValueError:
            parser.error("--class-rotations must contain numeric angles")
        if len(args.class_rotations) != 10:
            parser.error("--class-rotations must contain ten angles")
    else:
        args.class_rotations = None
    args.data_seed = (
        args.seed if args.data_seed is None else args.data_seed
    )
    args.model_seed = (
        args.seed + 10000 if args.model_seed is None else args.model_seed
    )
    args.batch_seed = (
        args.seed + 20000 if args.batch_seed is None else args.batch_seed
    )
    args.ratio_seed = (
        args.seed + 30000 if args.ratio_seed is None else args.ratio_seed
    )
    args.weight_shuffle_seed = (
        args.seed + 40000
        if args.weight_shuffle_seed is None
        else args.weight_shuffle_seed
    )
    return args


def main(argv: Optional[Sequence[str]] = None) -> Dict[str, Any]:
    return run_experiment(parse_args(argv))


if __name__ == "__main__":
    main()
