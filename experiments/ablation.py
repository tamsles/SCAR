"""Multi-seed ablation and baseline comparison on the synthetic shift."""

import argparse
import copy
import csv
import json
import math
import os
import random
import time
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import torch

from datasets import (
    empirical_branch_priors,
    make_synthetic_loaders,
    synthetic_oracle_log_ratios,
)
from ratio_estimation import RatioEstimatorTrainer
from train import _git_commit, build_estimator, load_config


CONDITIONS = [
    {
        "name": "unified_ratio",
        "category": "architecture",
        "description": "legacy unified conditioned ratio baseline",
        "overrides": {"ratio_arch": "unified"},
    },
    {
        "name": "separate_conditional",
        "category": "architecture",
        "description": "2q independent direct conditional estimators",
        "overrides": {"ratio_arch": "separate"},
    },
    {
        "name": "multihead_conditional",
        "category": "architecture",
        "description": "shared ratio encoder with 2q scalar heads",
        "overrides": {"ratio_arch": "multihead"},
    },
    {
        "name": "fusion_conditional",
        "category": "architecture",
        "description": "learned global-conditional log-space fusion",
        "overrides": {"ratio_arch": "fusion"},
    },
    {
        "name": "proposed",
        "category": "proposed",
        "description": "conditioned + balanced BCE + effective-prior correction",
        "overrides": {},
    },
    {
        "name": "multihead",
        "category": "ablation",
        "description": "shared backbone with direct q-by-2 output head",
        "overrides": {"estimator_type": "multihead"},
    },
    {
        "name": "no_class_condition",
        "category": "ablation",
        "description": "remove class embedding",
        "overrides": {"class_embedding_dim": 0},
    },
    {
        "name": "no_state_condition",
        "category": "ablation",
        "description": "remove complementary-state embedding",
        "overrides": {"state_embedding_dim": 0},
    },
    {
        "name": "no_prior_correction",
        "category": "ablation",
        "description": "unbalanced branch loss but incorrectly force prior to one",
        "overrides": {
            "branch_balancing": "none",
            "balanced_domain_sampling": True,
        },
    },
    {
        "name": "branch_none",
        "category": "ablation",
        "description": "no branch balancing",
        "overrides": {"branch_balancing": "none"},
    },
    {
        "name": "branch_inverse_frequency",
        "category": "ablation",
        "description": "inverse-frequency branch weights",
        "overrides": {"branch_balancing": "inverse_frequency"},
    },
    {
        "name": "marginal_ratio",
        "category": "baseline",
        "description": "one pooled domain ratio r(x), broadcast to all classes",
        "overrides": {
            "class_embedding_dim": 0,
            "state_embedding_dim": 0,
            "branch_balancing": "none",
            "balanced_domain_sampling": True,
        },
    },
    {
        "name": "unit_ratio",
        "category": "baseline",
        "description": "no distribution correction (all weights are one)",
        "kind": "unit",
        "overrides": {},
    },
    {
        "name": "oracle_ratio",
        "category": "reference",
        "description": "known synthetic conditional density ratio",
        "kind": "oracle",
        "overrides": {},
    },
]

METRIC_KEYS = (
    "ratio_mse",
    "log_ratio_mse",
    "spearman_correlation",
    "log_ratio_rmse",
    "log_ratio_mae",
    "log_ratio_correlation",
    "moment_rmse",
    "source_ratio_mean_abs_error",
    "ess_fraction",
    "validation_loss",
    "validation_accuracy",
    "parameter_count",
    "train_seconds",
)


def _collect(loader: Iterable[Any]) -> Tuple[torch.Tensor, torch.Tensor]:
    all_x = []
    all_bar_y = []
    for x, bar_y in loader:
        all_x.append(x)
        all_bar_y.append(bar_y)
    return torch.cat(all_x, dim=0), torch.cat(all_bar_y, dim=0)


def _safe_correlation(left: torch.Tensor, right: torch.Tensor) -> float:
    left = left.reshape(-1).double()
    right = right.reshape(-1).double()
    left = left - left.mean()
    right = right - right.mean()
    denominator = torch.sqrt(left.square().sum() * right.square().sum())
    if float(denominator) <= 1.0e-12:
        return 0.0
    return float((left * right).sum() / denominator)


def _safe_spearman(left: torch.Tensor, right: torch.Tensor) -> float:
    """Dependency-free Spearman correlation for continuous ratio diagnostics."""
    left = left.reshape(-1)
    right = right.reshape(-1)
    if left.numel() < 2:
        return 0.0
    if bool((left == left[0]).all()) or bool((right == right[0]).all()):
        return 0.0
    left_rank = torch.argsort(torch.argsort(left)).double()
    right_rank = torch.argsort(torch.argsort(right)).double()
    return _safe_correlation(left_rank, right_rank)


def _branch_metrics(
    source_x: torch.Tensor,
    source_bar_y: torch.Tensor,
    target_x: torch.Tensor,
    target_bar_y: torch.Tensor,
    source_weights: torch.Tensor,
) -> Dict[str, float]:
    """Evaluate conditional moment matching, ratio means, and branch ESS."""
    moment_errors = []
    ratio_mean_errors = []
    ess_fractions = []
    num_classes = source_bar_y.shape[1]
    for k in range(num_classes):
        for z in range(2):
            source_mask = source_bar_y[:, k] == z
            target_mask = target_bar_y[:, k] == z
            if not bool(source_mask.any()) or not bool(target_mask.any()):
                continue
            weights = source_weights[source_mask, k].double()
            source_values = source_x[source_mask].double()
            target_values = target_x[target_mask].double()
            weight_sum = weights.sum().clamp_min(1.0e-12)
            weighted_source_mean = (source_values * weights[:, None]).sum(0) / weight_sum
            target_mean = target_values.mean(0)
            moment_errors.append((weighted_source_mean - target_mean).square().mean())
            ratio_mean_errors.append(torch.abs(weights.mean() - 1.0))
            ess = weight_sum.square() / weights.square().sum().clamp_min(1.0e-12)
            ess_fractions.append(ess / float(weights.numel()))
    return {
        "moment_rmse": float(torch.stack(moment_errors).mean().sqrt()),
        "source_ratio_mean_abs_error": float(torch.stack(ratio_mean_errors).mean()),
        "ess_fraction": float(torch.stack(ess_fractions).mean()),
    }


@torch.no_grad()
def evaluate_condition(
    model: Optional[torch.nn.Module],
    kind: str,
    config: Dict[str, Any],
    loaders: Tuple[Any, Any, Any, Any],
    device: torch.device,
) -> Dict[str, float]:
    """Compare estimated selected ratios with the known synthetic oracle."""
    source_train_x, source_train_y = _collect(loaders[0])
    target_train_x, target_train_y = _collect(loaders[1])
    source_x, source_y = _collect(loaders[2])
    target_x, target_y = _collect(loaders[3])
    source_prior = empirical_branch_priors(source_train_y)
    target_prior = empirical_branch_priors(target_train_y)
    combined_x = torch.cat((source_x, target_x), dim=0).to(device)
    combined_y = torch.cat((source_y, target_y), dim=0).to(device)
    oracle_log = synthetic_oracle_log_ratios(
        combined_x, combined_y, config, source_prior, target_prior
    )
    oracle_log = oracle_log.clamp(
        float(config.get("log_ratio_clip_min", -8.0)),
        float(config.get("log_ratio_clip_max", 8.0)),
    )

    if kind == "unit":
        predicted_log = torch.zeros_like(oracle_log)
        source_weights = torch.ones_like(source_y, dtype=torch.float32)
    elif kind == "oracle":
        predicted_log = oracle_log
        source_log = synthetic_oracle_log_ratios(
            source_x.to(device),
            source_y.to(device),
            config,
            source_prior,
            target_prior,
        ).clamp(
            float(config.get("log_ratio_clip_min", -8.0)),
            float(config.get("log_ratio_clip_max", 8.0)),
        )
        source_weights = torch.exp(source_log).cpu()
    else:
        model.eval()
        predicted = model.compute_importance_weights(
            combined_x, combined_y, detach=True, self_normalize=False
        )
        predicted_log = torch.log(predicted.clamp_min(1.0e-12))
        source_weights = model.compute_importance_weights(
            source_x.to(device),
            source_y.to(device),
            detach=True,
            self_normalize=False,
        ).cpu()

    predicted_ratio = torch.exp(predicted_log)
    oracle_ratio = torch.exp(oracle_log)
    difference = predicted_log - oracle_log
    branch_ratio_metrics = {}
    for k in range(combined_y.shape[1]):
        for b in range(2):
            mask = combined_y[:, k] == b
            branch_predicted = predicted_ratio[mask, k]
            branch_oracle = oracle_ratio[mask, k]
            branch_log_predicted = predicted_log[mask, k]
            branch_log_oracle = oracle_log[mask, k]
            key = "{}_{}".format(k, b)
            branch_ratio_metrics[key] = {
                "count": int(mask.sum().item()),
                "ratio_mse": float(
                    (branch_predicted - branch_oracle).square().mean().item()
                ),
                "log_ratio_mse": float(
                    (
                        branch_log_predicted - branch_log_oracle
                    ).square().mean().item()
                ),
                "spearman_correlation": _safe_spearman(
                    branch_predicted, branch_oracle
                ),
            }
    metrics = {
        "ratio_mse": float(
            (predicted_ratio - oracle_ratio).square().mean().item()
        ),
        "log_ratio_mse": float(difference.square().mean().item()),
        "spearman_correlation": _safe_spearman(
            predicted_ratio, oracle_ratio
        ),
        "log_ratio_rmse": float(difference.square().mean().sqrt()),
        "log_ratio_mae": float(difference.abs().mean()),
        "log_ratio_correlation": _safe_correlation(predicted_log, oracle_log),
        "branch_ratio_metrics": branch_ratio_metrics,
    }
    metrics.update(
        _branch_metrics(
            source_x, source_y, target_x, target_y, source_weights
        )
    )
    return metrics


def run_one(
    base_config: Dict[str, Any],
    condition: Dict[str, Any],
    seed: int,
    device: torch.device,
    epochs: int,
) -> Dict[str, Any]:
    """Train/evaluate one condition and one seed."""
    config = copy.deepcopy(base_config)
    config.update(condition.get("overrides", {}))
    config["seed"] = seed
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    loaders = make_synthetic_loaders(config)
    kind = str(condition.get("kind", "model"))
    model = None
    parameter_count = 0
    train_seconds = 0.0
    validation_loss = math.log(2.0)
    validation_accuracy = 0.5

    if kind == "model":
        model = build_estimator(config).to(device)
        parameter_count = sum(
            parameter.numel() for parameter in model.parameters()
            if parameter.requires_grad
        )
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=float(config.get("learning_rate", 1.0e-3)),
            weight_decay=float(config.get("weight_decay", 0.0)),
        )
        ratio_config = config.get("ratio", {})
        if not isinstance(ratio_config, dict):
            ratio_config = {}
        trainer = RatioEstimatorTrainer(
            model,
            optimizer,
            device=device,
            branch_balancing=str(config.get("branch_balancing", "balanced_bce")),
            min_branch_count=int(
                ratio_config.get(
                    "min_condition_samples",
                    config.get("min_branch_count", 1),
                )
            ),
            max_branch_weight=float(config.get("max_branch_weight", 20.0)),
            balanced_domain_sampling=bool(
                config.get("balanced_domain_sampling", False)
            ),
            use_amp=bool(config.get("use_amp", False)),
            grad_clip_norm=config.get("grad_clip_norm"),
        )
        start = time.perf_counter()
        trainer.fit(
            loaders[0],
            loaders[1],
            epochs=epochs,
            validation_loaders=None,
            checkpoint_path=None,
            verbose=False,
        )
        train_seconds = time.perf_counter() - start
        validation = trainer.validate(loaders[2], loaders[3])
        validation_loss = float(validation["total_loss"])
        validation_accuracy = float(validation["discriminator_accuracy"])

    metrics = evaluate_condition(model, kind, config, loaders, device)
    record = {
        "name": condition["name"],
        "category": condition["category"],
        "description": condition["description"],
        "seed": seed,
        "epochs": epochs if kind == "model" else 0,
        "parameter_count": parameter_count,
        "train_seconds": train_seconds,
        "validation_loss": validation_loss,
        "validation_accuracy": validation_accuracy,
    }
    record.update(metrics)
    return record


def aggregate(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Aggregate numeric metrics as mean/std over seeds."""
    summaries = []
    names = []
    for record in records:
        if record["name"] not in names:
            names.append(record["name"])
    for name in names:
        group = [record for record in records if record["name"] == name]
        summary = {
            "name": name,
            "category": group[0]["category"],
            "description": group[0]["description"],
            "seeds": len(group),
        }
        for key in METRIC_KEYS:
            values = []
            for record in group:
                if key in record:
                    value = record[key]
                elif key in ("ratio_mse", "log_ratio_mse"):
                    value = float(record["log_ratio_rmse"]) ** 2
                elif key == "spearman_correlation":
                    value = record["log_ratio_correlation"]
                else:
                    raise KeyError(key)
                values.append(float(value))
            mean = sum(values) / float(len(values))
            variance = sum((value - mean) ** 2 for value in values) / float(len(values))
            summary[key + "_mean"] = mean
            summary[key + "_std"] = variance ** 0.5
        summaries.append(summary)
    return summaries


def _markdown(summaries: List[Dict[str, Any]]) -> str:
    ranked = sorted(summaries, key=lambda row: row["log_ratio_rmse_mean"])
    lines = [
        "# Density-ratio ablation and baseline report",
        "",
        "Lower is better for RMSE/MAE; higher is better for correlation and ESS.",
        "",
        "| Rank | Method | Type | Log-ratio RMSE | Moment RMSE | Corr. | ESS | Params | Train s |",
        "|---:|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for rank, row in enumerate(ranked, 1):
        lines.append(
            "| {rank} | {name} | {category} | {rmse:.4f} +/- {rmse_std:.4f} | "
            "{moment:.4f} +/- {moment_std:.4f} | {corr:.4f} | {ess:.4f} | "
            "{params:.0f} | {seconds:.2f} |".format(
                rank=rank,
                name=row["name"],
                category=row["category"],
                rmse=row["log_ratio_rmse_mean"],
                rmse_std=row["log_ratio_rmse_std"],
                moment=row["moment_rmse_mean"],
                moment_std=row["moment_rmse_std"],
                corr=row["log_ratio_correlation_mean"],
                ess=row["ess_fraction_mean"],
                params=row["parameter_count_mean"],
                seconds=row["train_seconds_mean"],
            )
        )
    by_name = dict((row["name"], row) for row in summaries)
    lines.extend(["", "## Main comparison", ""])
    if "proposed" in by_name and "unit_ratio" in by_name:
        proposed = by_name["proposed"]
        unit = by_name["unit_ratio"]
        lines.append(
            "- Proposed vs unit-ratio log-RMSE reduction: {:.1f}%.".format(
                100.0
                * (unit["log_ratio_rmse_mean"] - proposed["log_ratio_rmse_mean"])
                / max(unit["log_ratio_rmse_mean"], 1.0e-12)
            )
        )
    if "proposed" in by_name and "marginal_ratio" in by_name:
        proposed = by_name["proposed"]
        marginal = by_name["marginal_ratio"]
        lines.append(
            "- Proposed vs marginal-ratio log-RMSE reduction: {:.1f}%.".format(
                100.0
                * (
                    marginal["log_ratio_rmse_mean"]
                    - proposed["log_ratio_rmse_mean"]
                )
                / max(marginal["log_ratio_rmse_mean"], 1.0e-12)
            )
        )
    if "oracle_ratio" in by_name:
        lines.extend(
            [
                "",
                "The oracle row is a finite-sample reference, not a trainable baseline.",
            ]
        )
    lines.append("")
    return "\n".join(lines)


def write_reports(
    output_dir: str,
    config: Dict[str, Any],
    records: List[Dict[str, Any]],
    summaries: List[Dict[str, Any]],
) -> None:
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
    with open(os.path.join(output_dir, "raw_results.json"), "w") as stream:
        json.dump(
            {"config": config, "conditions": CONDITIONS, "records": records},
            stream,
            indent=2,
            sort_keys=True,
        )
    saved_config = copy.deepcopy(config)
    saved_config["git_commit"] = _git_commit()
    with open(os.path.join(output_dir, "config.yaml"), "w") as stream:
        json.dump(saved_config, stream, indent=2, sort_keys=True)
    with open(os.path.join(output_dir, "summary.json"), "w") as stream:
        json.dump(summaries, stream, indent=2, sort_keys=True)
    metric_fields = sorted(
        set(key for record in records for key in record.keys())
    )
    with open(os.path.join(output_dir, "metrics.csv"), "w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=metric_fields)
        writer.writeheader()
        writer.writerows(records)
    ratio_fields = [
        "name",
        "seed",
        "class",
        "state",
        "count",
        "ratio_mse",
        "log_ratio_mse",
        "spearman_correlation",
        "log_ratio_rmse",
        "log_ratio_mae",
        "log_ratio_correlation",
        "source_ratio_mean_abs_error",
        "ess_fraction",
    ]
    with open(
        os.path.join(output_dir, "ratio_statistics.csv"), "w", newline=""
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=ratio_fields)
        writer.writeheader()
        for record in records:
            branches = record.get("branch_ratio_metrics", {})
            if not branches:
                writer.writerow({key: record.get(key) for key in ratio_fields})
                continue
            for branch, values in branches.items():
                k, b = branch.split("_")
                row = {key: record.get(key) for key in ratio_fields}
                row.update(values)
                row["class"] = k
                row["state"] = b
                writer.writerow(row)
    fieldnames = ["name", "category", "description", "seeds"]
    for key in METRIC_KEYS:
        fieldnames.extend((key + "_mean", key + "_std"))
    with open(os.path.join(output_dir, "summary.csv"), "w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summaries)
    report = _markdown(summaries)
    with open(os.path.join(output_dir, "REPORT.md"), "w") as stream:
        stream.write(report)
    print(report)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/ablation_synthetic.json")
    parser.add_argument("--output-dir", default="outputs/ablation")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--seeds", default="7,17,29")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument(
        "--conditions",
        default=None,
        help="optional comma-separated condition names",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> List[Dict[str, Any]]:
    args = parse_args(argv)
    config = load_config(args.config)
    seeds = [int(item) for item in args.seeds.split(",") if item.strip()]
    epochs = int(args.epochs if args.epochs is not None else config.get("epochs", 15))
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    selected_names = (
        set(args.conditions.split(",")) if args.conditions is not None else None
    )
    conditions = [
        condition
        for condition in CONDITIONS
        if selected_names is None or condition["name"] in selected_names
    ]
    if not conditions:
        raise ValueError("no experiment conditions selected")
    records = []
    for condition in conditions:
        for seed in seeds:
            record = run_one(config, condition, seed, device, epochs)
            records.append(record)
            print("Experiment: {}".format(json.dumps(record, sort_keys=True)))
    summaries = aggregate(records)
    write_reports(args.output_dir, config, records, summaries)
    return summaries


if __name__ == "__main__":
    main()
