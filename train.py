"""Main training entry point for the label-conditioned ratio estimator."""

import argparse
import csv
import json
import os
import random
import subprocess
from typing import Any, Dict, Optional, Sequence

import torch

from datasets import make_synthetic_loaders
from models import LabelConditionedRatioEstimator
from models import build_ratio_estimator as create_ratio_estimator
from ratio_estimation import RatioEstimatorTrainer, process_conditional_weights


def load_config(path: str) -> Dict[str, Any]:
    """Load JSON, or YAML when PyYAML is already available."""
    with open(path, "r") as stream:
        text = stream.read()
    try:
        config = json.loads(text)
    except json.JSONDecodeError:
        try:
            import yaml
        except ImportError as error:
            raise ValueError(
                "non-JSON YAML requires optional PyYAML; JSON syntax is "
                "accepted in .yaml files without it"
            ) from error
        config = yaml.safe_load(text)
    required = ("input_dim", "num_classes", "feature_dim")
    missing = [key for key in required if key not in config]
    if missing:
        raise ValueError("missing configuration keys: {}".format(missing))
    return config


def build_estimator(
    config: Dict[str, Any], backbone: Optional[torch.nn.Module] = None
) -> LabelConditionedRatioEstimator:
    """Build an estimator, optionally reusing a caller-provided backbone."""
    return create_ratio_estimator(config, backbone=backbone)


def _apply_override(config: Dict[str, Any], override: str) -> None:
    if "=" not in override:
        raise ValueError("config override must use key=value: {}".format(override))
    key, raw_value = override.split("=", 1)
    try:
        value = json.loads(raw_value)
    except json.JSONDecodeError:
        value = raw_value
    target = config
    parts = key.split(".")
    for part in parts[:-1]:
        child = target.get(part)
        if not isinstance(child, dict):
            child = {}
            target[part] = child
        target = child
    target[parts[-1]] = value


def _git_commit() -> str:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True,
        )
        return completed.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unavailable"


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", default="configs/ratio_synthetic.json", help="JSON config path"
    )
    parser.add_argument("--output-dir", default="outputs/ratio_synthetic")
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "overrides",
        nargs="*",
        help="configuration overrides such as ratio_arch=separate",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> Dict[str, Any]:
    args = parse_args(argv)
    config = load_config(args.config)
    for override in args.overrides:
        _apply_override(config, override)
    if args.epochs is not None:
        config["epochs"] = args.epochs
    if args.seed is not None:
        config["seed"] = args.seed
    seed = int(config.get("seed", 7))
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")

    loaders = make_synthetic_loaders(config)
    model = build_estimator(config).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(config.get("learning_rate", 1.0e-3)),
        weight_decay=float(config.get("weight_decay", 0.0)),
    )
    ratio_config = config.get("ratio", {})
    if not isinstance(ratio_config, dict):
        ratio_config = {}
    trainer = RatioEstimatorTrainer(
        model=model,
        optimizer=optimizer,
        device=device,
        branch_balancing=str(config.get("branch_balancing", "balanced_bce")),
        min_branch_count=int(
            ratio_config.get(
                "min_condition_samples", config.get("min_branch_count", 1)
            )
        ),
        max_branch_weight=float(config.get("max_branch_weight", 20.0)),
        balanced_domain_sampling=bool(
            config.get("balanced_domain_sampling", False)
        ),
        use_amp=bool(config.get("use_amp", False)),
        grad_clip_norm=config.get("grad_clip_norm"),
    )
    if not os.path.isdir(args.output_dir):
        os.makedirs(args.output_dir)
    checkpoint_path = os.path.join(args.output_dir, "ratio_estimator.pt")
    history = trainer.fit(
        loaders[0],
        loaders[1],
        epochs=int(config.get("epochs", 25)),
        validation_loaders=(loaders[2], loaders[3]),
        checkpoint_path=checkpoint_path,
    )

    model.eval()
    source_x, source_bar_y = next(iter(loaders[2]))
    target_x, target_bar_y = next(iter(loaders[3]))
    with torch.no_grad():
        sample_weights = model.get_conditional_weights(
            target_x.to(device),
            target_bar_y.to(device),
            detach=True,
            self_normalize=False,
        )
        source_raw_weights = model.get_conditional_weights(
            source_x.to(device),
            source_bar_y.to(device),
            detach=True,
            self_normalize=False,
        )
        _, branch_statistics = process_conditional_weights(
            source_raw_weights,
            source_bar_y.to(device),
            min_weight=float(ratio_config.get("min_weight", 0.01)),
            max_weight=float(ratio_config.get("max_weight", 20.0)),
            self_normalize=bool(ratio_config.get("self_normalize", True)),
            normalization_scope=str(
                ratio_config.get(
                    "normalization_scope", "conditional_branch"
                )
            ),
            eps=float(ratio_config.get("eps", 1.0e-8)),
        )
        architecture_statistics = model.extra_ratio_statistics(
            source_x.to(device)
        )
    git_commit = _git_commit()
    result = {
        "device": str(device),
        "ratio_arch": getattr(model, "architecture", "unified"),
        "git_commit": git_commit,
        "checkpoint": os.path.abspath(checkpoint_path),
        "final_epoch": history[-1],
        "sample_weight_mean": float(sample_weights.mean().item()),
        "sample_weight_std": float(sample_weights.std(unbiased=False).item()),
        "sample_weight_min": float(sample_weights.min().item()),
        "sample_weight_max": float(sample_weights.max().item()),
        "skipped_conditional_updates": model.skipped_update_count.cpu().tolist(),
        "architecture_statistics": architecture_statistics,
    }
    result_path = os.path.join(args.output_dir, "result.json")
    with open(result_path, "w") as stream:
        json.dump(result, stream, indent=2, sort_keys=True)
    with open(os.path.join(args.output_dir, "summary.json"), "w") as stream:
        json.dump(result, stream, indent=2, sort_keys=True)
    saved_config = dict(config)
    saved_config["git_commit"] = git_commit
    with open(os.path.join(args.output_dir, "config.yaml"), "w") as stream:
        # JSON is valid YAML and avoids imposing a new runtime dependency.
        json.dump(saved_config, stream, indent=2, sort_keys=True)
    with open(
        os.path.join(args.output_dir, "metrics.csv"), "w", newline=""
    ) as stream:
        fields = [
            "epoch",
            "steps",
            "total_loss",
            "source_loss",
            "target_loss",
            "discriminator_accuracy",
            "ratio_mean",
            "ratio_std",
            "ratio_min",
            "ratio_max",
            "has_nan_or_inf",
        ]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in history:
            writer.writerow({key: row.get(key) for key in fields})
    with open(
        os.path.join(args.output_dir, "ratio_statistics.csv"),
        "w",
        newline="",
    ) as stream:
        fields = [
            "class",
            "state",
            "count",
            "mean",
            "std",
            "min",
            "max",
            "q05",
            "q50",
            "q95",
            "clipping_rate",
            "ess",
            "source_sample_count",
            "target_sample_count",
            "ratio_objective",
            "skipped_updates",
        ]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for k in range(model.num_classes):
            for b in range(2):
                row = dict(branch_statistics["{}_{}".format(k, b)])
                row.update(
                    {
                        "class": k,
                        "state": b,
                        "source_sample_count": int(
                            trainer.source_count[k, b].item()
                        ),
                        "target_sample_count": int(
                            trainer.target_count[k, b].item()
                        ),
                        "ratio_objective": float(
                            history[-1]["branch_objective"][k][b]
                        ),
                        "skipped_updates": int(
                            model.skipped_update_count[k, b].item()
                        ),
                    }
                )
                writer.writerow(row)
    print("Training result: {}".format(json.dumps(result, sort_keys=True)))
    return result


if __name__ == "__main__":
    main()
