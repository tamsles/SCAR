"""Compare neural branch-ratio estimators with the original SCAR baselines.

This bridge intentionally imports the existing SCARCE data, model, risk, and
evaluation modules from a caller-provided repository. The only new components
are the ratio estimator and its frozen class/state-specific weights.
"""

import argparse
import csv
import importlib.util
import json
import os
import random
import sys
import time
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from models import LabelConditionedRatioEstimator
from models import build_ratio_estimator as create_ratio_estimator
from ratio_estimation import RatioEstimatorTrainer, process_conditional_weights


def _load_module(name: str, path: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError("cannot load {} from {}".format(name, path))
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def load_scarce_modules(repository: str) -> Dict[str, Any]:
    """Load original SCARCE modules without shadowing this project's models."""
    repository = os.path.abspath(repository)
    required = ("models.py", "utils_algo.py", "adiw_scar.py", "domain_data.py")
    missing = [
        filename for filename in required
        if not os.path.isfile(os.path.join(repository, filename))
    ]
    if missing:
        raise ValueError("SCARCE repository is missing files: {}".format(missing))
    utils_algo = _load_module(
        "utils_algo", os.path.join(repository, "utils_algo.py")
    )
    return {
        "models": _load_module(
            "scarce_models", os.path.join(repository, "models.py")
        ),
        "utils_algo": utils_algo,
        # adiw_scar imports the registered original utils_algo name.
        "adiw_scar": _load_module(
            "scarce_adiw_scar", os.path.join(repository, "adiw_scar.py")
        ),
        "domain_data": _load_module(
            "scarce_domain_data", os.path.join(repository, "domain_data.py")
        ),
    }


def set_random_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class WeakPairView(Dataset):
    """Zero-copy view changing ``(index,x,bar_y)`` to ``(x,bar_y)``."""

    def __init__(self, weak_dataset: Dataset) -> None:
        self.weak_dataset = weak_dataset

    def __len__(self) -> int:
        return len(self.weak_dataset)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        _, image, complementary_vector = self.weak_dataset[index]
        return image, complementary_vector


class ScarceMLPFeatureBackbone(nn.Module):
    """Reuse the feature layer created by the original SCARCE MLP class."""

    def __init__(self, scarce_mlp: nn.Module) -> None:
        super().__init__()
        self.fc1 = scarce_mlp.fc1
        self.relu1 = scarce_mlp.relu1
        self.output_dim = int(scarce_mlp.fc1.out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu1(self.fc1(x.reshape(x.shape[0], -1)))


def build_ratio_estimator(
    scarce_models: Any,
    input_dim: int,
    num_classes: int,
    ratio_arch: str,
    device: torch.device,
    ratio_clip_max: float,
) -> LabelConditionedRatioEstimator:
    """Build a ratio model using the original MLP's 500-unit feature layer."""
    original_mlp = scarce_models.mlp_model(
        input_dim=input_dim, hidden_dim=500, output_dim=num_classes
    )
    backbone = ScarceMLPFeatureBackbone(original_mlp)
    config = {
        "input_dim": input_dim,
        "num_classes": num_classes,
        "feature_dim": 500,
        "class_embedding_dim": 16,
        "state_embedding_dim": 4,
        "hidden_dims": [128, 64],
        "ratio_arch": ratio_arch,
        "estimator_type": "conditioned",
        "balanced_domain_sampling": True,
        "log_ratio_clip_min": -6.0,
        "log_ratio_clip_max": 6.0,
        "ratio": {
            "self_normalize": True,
            "normalization_scope": "conditional_branch",
            "min_weight": 0.01,
            "max_weight": ratio_clip_max,
            "min_condition_samples": 8,
            "eps": 1.0e-8,
            "detach_classifier_update": True,
        },
        "fusion": {
            "gate_regularization": 0.0,
            "gate_init": 0.7,
            "detach_global_for_gate": False,
        },
    }
    return create_ratio_estimator(config, backbone=backbone).to(device)


def _next_batch(loader: DataLoader, iterator: Any) -> Tuple[Any, Any]:
    try:
        batch = next(iterator)
    except StopIteration:
        iterator = iter(loader)
        batch = next(iterator)
    return batch, iterator


def pretrain_ratio_estimator(
    model: LabelConditionedRatioEstimator,
    source_dataset: Dataset,
    target_dataset: Dataset,
    device: torch.device,
    batch_size: int,
    epochs: int,
    learning_rate: float,
    weight_decay: float,
    workers: int,
    use_amp: bool,
) -> Tuple[RatioEstimatorTrainer, Dict[str, Any]]:
    """Pretrain on all source batches while cycling the smaller target set."""
    source_loader = DataLoader(
        WeakPairView(source_dataset),
        batch_size=batch_size,
        shuffle=True,
        num_workers=workers,
        drop_last=False,
    )
    target_loader = DataLoader(
        WeakPairView(target_dataset),
        batch_size=batch_size,
        shuffle=True,
        num_workers=workers,
        drop_last=False,
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
        use_amp=use_amp,
        grad_clip_norm=5.0,
    )
    statistics = trainer.compute_branch_statistics(source_loader, target_loader)
    start = time.perf_counter()
    last_metrics = None
    for epoch in range(epochs):
        target_iterator = iter(target_loader)
        epoch_loss = 0.0
        steps = 0
        for source_batch in source_loader:
            target_batch, target_iterator = _next_batch(
                target_loader, target_iterator
            )
            last_metrics = trainer.train_step(source_batch, target_batch)
            epoch_loss += last_metrics["total_loss"]
            steps += 1
        print(
            "Ratio epoch {}: loss {:.6f}, accuracy {:.4f}, ratio "
            "{:.4f}+/-{:.4f}".format(
                epoch + 1,
                epoch_loss / max(steps, 1),
                last_metrics["discriminator_accuracy"],
                last_metrics["ratio_mean"],
                last_metrics["ratio_std"],
            )
        )
    elapsed = time.perf_counter() - start
    validation = trainer.validate(source_loader, target_loader)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return trainer, {
        "seconds": elapsed,
        "statistics": statistics,
        "validation": validation,
    }


@torch.no_grad()
def cache_source_weights(
    model: LabelConditionedRatioEstimator,
    source_dataset: Dataset,
    num_classes: int,
    device: torch.device,
    batch_size: int,
    workers: int,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Cache frozen source weights by the original dataset's local index."""
    loader = DataLoader(
        source_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        drop_last=False,
    )
    cache = torch.empty(len(source_dataset), num_classes, dtype=torch.float32)
    start = time.perf_counter()
    model.eval()
    for indices, images, complementary_vectors in loader:
        weights = model.compute_importance_weights(
            images.to(device),
            complementary_vectors.to(device),
            detach=True,
            self_normalize=False,
        )
        cache[indices.long()] = weights.cpu()
    elapsed = time.perf_counter() - start
    if not bool(torch.isfinite(cache).all().item()):
        raise RuntimeError("cached source weights contain NaN or Inf")
    return cache, {
        "seconds": elapsed,
        "mean": float(cache.mean().item()),
        "std": float(cache.std(unbiased=False).item()),
        "min": float(cache.min().item()),
        "max": float(cache.max().item()),
    }


def train_scar_classifier(
    modules: Dict[str, Any],
    source_weight_cache: torch.Tensor,
    data: Dict[str, Any],
    device: torch.device,
    seed: int,
    batch_size: int,
    epochs: int,
    learning_rate: float,
    weight_decay: float,
    workers: int,
    risk_correction: str,
    detail_path: str,
) -> Dict[str, Any]:
    """Train the original SCAR classifier with frozen neural ratio weights."""
    # Reset here so ratio pretraining cannot change classifier initialization.
    set_random_seed(seed)
    source_dataset = data["source_weak"]
    target_dataset = data["target_weak"]
    source_loader = DataLoader(
        source_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=workers,
        drop_last=False,
    )
    target_loader = DataLoader(
        target_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=workers,
        drop_last=False,
    )
    evaluation_loader = DataLoader(
        data["target_evaluation"],
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        drop_last=False,
    )
    source_evaluation_loader = None
    if data.get("source_evaluation") is not None:
        source_evaluation_loader = DataLoader(
            data["source_evaluation"],
            batch_size=batch_size,
            shuffle=False,
            num_workers=workers,
            drop_last=False,
        )
    classifier = modules["models"].mlp_model(
        input_dim=data["input_dim"],
        hidden_dim=500,
        output_dim=data["num_classes"],
    ).to(device)
    optimizer = torch.optim.Adam(
        classifier.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    target_class_prior = torch.full(
        (data["num_classes"],), 1.0 / data["num_classes"], device=device
    )
    target_complement_prior = data["target_complement_prior"].to(device)
    target_fraction = float(len(target_dataset)) / float(
        len(source_dataset) + len(target_dataset)
    )
    target_iterator = iter(target_loader)
    last_accuracies = []
    final_target_risk = float("nan")
    final_source_risk = float("nan")
    start = time.perf_counter()
    with open(detail_path, "w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            [
                "epoch",
                "loss",
                "source_risk",
                "target_risk",
                "target_accuracy",
                "weight_mean",
                "weight_std",
            ]
        )
        for epoch in range(epochs):
            classifier.train()
            total_loss = 0.0
            total_source_risk = 0.0
            total_target_risk = 0.0
            total_samples = 0
            weight_sum = 0.0
            weight_square_sum = 0.0
            weight_count = 0
            for source_batch in source_loader:
                target_batch, target_iterator = _next_batch(
                    target_loader, target_iterator
                )
                source_indices, source_images, source_complements = source_batch
                _, target_images, target_complements = target_batch
                source_images = source_images.to(device)
                source_complements = source_complements.to(device)
                target_images = target_images.to(device)
                target_complements = target_complements.to(device)

                source_weights = source_weight_cache[
                    source_indices.long()
                ].to(device, non_blocking=True)
                source_weights, _ = process_conditional_weights(
                    source_weights,
                    source_complements,
                    min_weight=0.01,
                    max_weight=20.0,
                    self_normalize=True,
                    normalization_scope="conditional_branch",
                    eps=1.0e-8,
                )
                optimizer.zero_grad()
                source_outputs = classifier(source_images)
                target_outputs = classifier(target_images)
                loss, diagnostics = modules[
                    "adiw_scar"
                ].combined_adiw_scar_loss(
                    source_outputs,
                    source_complements,
                    source_weights,
                    target_outputs,
                    target_complements,
                    target_class_prior,
                    target_complement_prior,
                    target_fraction,
                    correction=risk_correction,
                )
                loss.backward()
                optimizer.step()

                current_batch_size = source_images.shape[0]
                total_samples += current_batch_size
                total_loss += loss.item() * current_batch_size
                total_source_risk += (
                    diagnostics["source_risk"].item() * current_batch_size
                )
                total_target_risk += (
                    diagnostics["target_risk"].item() * current_batch_size
                )
                weights = source_weights.detach()
                weight_sum += weights.sum().item()
                weight_square_sum += weights.square().sum().item()
                weight_count += weights.numel()

            target_accuracy = modules["utils_algo"].accuracy_check(
                evaluation_loader, classifier, device
            )
            if epoch >= epochs - 10:
                last_accuracies.append(target_accuracy)
            mean_weight = weight_sum / weight_count
            weight_variance = max(
                weight_square_sum / weight_count - mean_weight * mean_weight,
                0.0,
            )
            row = [
                epoch + 1,
                total_loss / total_samples,
                total_source_risk / total_samples,
                total_target_risk / total_samples,
                target_accuracy,
                mean_weight,
                weight_variance ** 0.5,
            ]
            final_source_risk = float(row[2])
            final_target_risk = float(row[3])
            writer.writerow(row)
            print(
                "Classifier epoch {}: loss {:.4f}, target acc {:.3f}, "
                "weight {:.4f}+/-{:.4f}".format(
                    epoch + 1, row[1], target_accuracy, row[5], row[6]
                )
            )
    source_accuracy = None
    if source_evaluation_loader is not None:
        source_accuracy = float(
            modules["utils_algo"].accuracy_check(
                source_evaluation_loader, classifier, device
            )
        )
    return {
        "target_accuracy": float(np.mean(last_accuracies)),
        "source_accuracy": source_accuracy,
        "target_ovr_risk": final_target_risk,
        "source_ovr_risk": final_source_risk,
        "classifier_seconds": time.perf_counter() - start,
        "classifier_parameter_count": sum(
            parameter.numel() for parameter in classifier.parameters()
        ),
        "target_fraction": target_fraction,
    }


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scarce-repo", required=True)
    parser.add_argument("--output-dir", default="outputs/scarce_comparison")
    parser.add_argument(
        "--estimator-type",
        choices=("conditioned", "multihead"),
        default=None,
        help="deprecated legacy alias; use --ratio-arch",
    )
    parser.add_argument(
        "--ratio-arch",
        choices=("unified", "separate", "multihead", "fusion"),
        default=None,
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--ratio-epochs", type=int, default=10)
    parser.add_argument("--classifier-epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--ratio-learning-rate", type=float, default=1.0e-3)
    parser.add_argument("--learning-rate", type=float, default=1.0e-3)
    parser.add_argument("--weight-decay", type=float, default=1.0e-5)
    parser.add_argument("--ratio-clip-max", type=float, default=50.0)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--target-weak-size", type=int, default=1000)
    parser.add_argument("--source-size", type=int, default=0)
    parser.add_argument("--target-rotation", type=float, default=30.0)
    parser.add_argument(
        "--dataset",
        choices=("mnist", "fashionmnist", "cifar10"),
        default="mnist",
    )
    parser.add_argument(
        "--shift",
        choices=("rotation", "input_output_relation", "support"),
        default="rotation",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> Dict[str, Any]:
    args = parse_args(argv)
    if args.ratio_epochs <= 0 or args.classifier_epochs <= 0:
        raise ValueError("epoch counts must be positive")
    output_dir = os.path.abspath(args.output_dir)
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
    scarce_repository = os.path.abspath(args.scarce_repo)
    modules = load_scarce_modules(scarce_repository)
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")

    set_random_seed(args.seed)
    original_directory = os.getcwd()
    os.chdir(scarce_repository)
    try:
        data = modules["domain_data"].prepare_dual_domain_datasets(
            dataname=args.dataset,
            target_weak_size=args.target_weak_size,
            source_size=args.source_size,
            seed=args.seed,
            shift=args.shift,
            target_rotation=args.target_rotation,
            label_shift=1,
            source_generation="single",
            target_generation="single",
            source_selection_probability=None,
            target_selection_probability=None,
        )
    finally:
        os.chdir(original_directory)

    ratio_arch = args.ratio_arch
    if ratio_arch is None:
        # Preserve the old command form while routing it through the new
        # architecture factory.
        ratio_arch = (
            "multihead" if args.estimator_type == "multihead" else "unified"
        )
    ratio_model = build_ratio_estimator(
        modules["models"],
        data["input_dim"],
        data["num_classes"],
        ratio_arch,
        device,
        args.ratio_clip_max,
    )
    _, ratio_result = pretrain_ratio_estimator(
        ratio_model,
        data["source_weak"],
        data["target_weak"],
        device,
        args.batch_size,
        args.ratio_epochs,
        args.ratio_learning_rate,
        args.weight_decay,
        args.workers,
        use_amp=device.type == "cuda",
    )
    method = "{}_ratio_scar".format(ratio_arch)
    checkpoint_path = os.path.join(
        output_dir, "{}_seed{}_ratio.pt".format(method, args.seed)
    )
    torch.save(ratio_model.state_dict(), checkpoint_path)
    source_weight_cache, cache_result = cache_source_weights(
        ratio_model,
        data["source_weak"],
        data["num_classes"],
        device,
        args.batch_size,
        args.workers,
    )
    detail_path = os.path.join(
        output_dir, "{}_seed{}_detail.csv".format(method, args.seed)
    )
    classifier_result = train_scar_classifier(
        modules,
        source_weight_cache,
        data,
        device,
        args.seed,
        args.batch_size,
        args.classifier_epochs,
        args.learning_rate,
        args.weight_decay,
        args.workers,
        "abs",
        detail_path,
    )
    result = {
        "method": method,
        "seed": args.seed,
        "dataset": args.dataset,
        "shift": args.shift,
        "ratio_arch": ratio_arch,
        "target_rotation": args.target_rotation,
        "source_size": len(data["source_weak"]),
        "target_weak_size": len(data["target_weak"]),
        "ratio_epochs": args.ratio_epochs,
        "classifier_epochs": args.classifier_epochs,
        "ratio_parameter_count": sum(
            parameter.numel() for parameter in ratio_model.parameters()
        ),
        "ratio_training": ratio_result,
        "source_weight_cache": cache_result,
        "checkpoint": checkpoint_path,
        "detail": detail_path,
    }
    result.update(classifier_result)
    result_path = os.path.join(
        output_dir, "{}_seed{}.json".format(method, args.seed)
    )
    with open(result_path, "w") as stream:
        json.dump(result, stream, indent=2, sort_keys=True)
    print("SCAR comparison result: {}".format(json.dumps(result, sort_keys=True)))
    return result


if __name__ == "__main__":
    main()
