"""Run one B3 seed across all preregistered Gaussian settings/methods."""

import argparse
import csv
import hashlib
import json
import os
import subprocess
import sys
import time
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import torch

REPOSITORY_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPOSITORY_ROOT not in sys.path:
    sys.path.insert(0, REPOSITORY_ROOT)

from ratio_estimation import process_weight_stages
from src.ratio_diagnostics import (
    distribution_matching_diagnostics,
    weight_summary,
)
from src.risk_recovery_metrics import risk_recovery_summary
from src.synthetic_gaussian import (
    estimated_ratios,
    fit_domain_estimators,
    generate_gaussian_shift,
    oracle_branch_ratios,
    oracle_global_ratio,
    train_weak_classifier,
)


def _json_dump(path: str, value: Any) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)


def _write_csv(path: str, rows: List[Dict[str, Any]]) -> None:
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with open(path, "w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _hash_data(data: Dict[str, Any]) -> str:
    digest = hashlib.sha256()
    for domain in ("source", "target_weak", "target_test"):
        for key in ("x", "y", "bar_y"):
            array = data[domain][key].detach().cpu().numpy()
            digest.update(domain.encode("ascii"))
            digest.update(key.encode("ascii"))
            digest.update(array.tobytes())
    return digest.hexdigest()


def _git_commit(repository: str) -> Optional[str]:
    try:
        return (
            subprocess.check_output(
                ["git", "-C", repository, "rev-parse", "HEAD"]
            )
            .decode("utf-8")
            .strip()
        )
    except Exception:
        return None


def _ratio_metrics(
    raw: torch.Tensor, oracle: torch.Tensor, states: torch.Tensor
) -> Dict[str, float]:
    raw_flat = raw.double().reshape(-1)
    oracle_flat = oracle.double().reshape(-1).clamp_min(1.0e-8)
    error = raw_flat - oracle_flat
    log_error = raw_flat.clamp_min(1.0e-8).log() - oracle_flat.log()
    correlation = (
        float(np.corrcoef(raw_flat.numpy(), oracle_flat.numpy())[0, 1])
        if raw_flat.std() > 0 and oracle_flat.std() > 0
        else 0.0
    )
    normalization_errors = []
    for class_index in range(raw.shape[1]):
        for state in (0, 1):
            mask = states[:, class_index].long() == state
            if bool(mask.any().item()):
                normalization_errors.append(
                    abs(float(raw[mask, class_index].mean().item()) - 1.0)
                )
    return {
        "ratio_mse": float((error ** 2).mean().item()),
        "log_ratio_mse": float((log_error ** 2).mean().item()),
        "normalization_error": float(np.mean(normalization_errors)),
        "ratio_oracle_correlation": correlation,
    }


def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as stream:
        return json.load(stream)


def parse_args(
    argv: Optional[Sequence[str]] = None,
) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", default="configs/phase_b/b3_gaussian/config.json"
    )
    parser.add_argument("--phase", default="B3")
    parser.add_argument(
        "--experiment-mode",
        choices=("smoke", "diagnostic", "full"),
        default="smoke",
    )
    parser.add_argument("--task-index", type=int)
    parser.add_argument("--count", action="store_true")
    parser.add_argument("--list-tasks", action="store_true")
    parser.add_argument("--scarce-repo")
    parser.add_argument("--repository-root", default=os.getcwd())
    parser.add_argument("--results-root", default="outputs/phase_b/runs/B3")
    parser.add_argument(
        "--device", choices=("auto", "cpu", "cuda"), default="auto"
    )
    return parser.parse_args(argv)


def run_seed(
    config: Dict[str, Any], seed: int, args: argparse.Namespace
) -> List[Dict[str, Any]]:
    device = torch.device(
        "cuda"
        if args.device == "auto" and torch.cuda.is_available()
        else args.device
    )
    if args.device == "auto" and not torch.cuda.is_available():
        device = torch.device("cpu")
    smoke = args.experiment_mode == "smoke"
    ratio_epochs = (
        config["smoke_ratio_epochs"] if smoke else config["ratio_epochs"]
    )
    classifier_epochs = (
        config["smoke_classifier_epochs"]
        if smoke
        else config["classifier_epochs"]
    )
    summaries = []
    for setting_index, setting in enumerate(config["settings"]):
        data_seed = seed
        data = generate_gaussian_shift(
            setting,
            seed=data_seed,
            source_size=config["source_size"],
            target_weak_size=config["target_weak_size"],
            target_test_size=config["target_test_size"],
            q=config["q"],
        )
        split_hash = _hash_data(data)
        ratio_start = time.perf_counter()
        global_model, conditional_model, ratio_history = (
            fit_domain_estimators(
                data["source"],
                data["target_weak"],
                q=config["q"],
                seed=seed + 30000 + setting_index,
                device=device,
                epochs=ratio_epochs,
                learning_rate=config["learning_rate"],
            )
        )
        ratio_seconds = time.perf_counter() - ratio_start
        estimated_global, estimated_conditional = estimated_ratios(
            global_model,
            conditional_model,
            data["source"]["x"],
            data["source"]["bar_y"],
            device,
        )
        oracle_global = oracle_global_ratio(
            data["source"]["x"], data["parameters"]
        )
        oracle_conditional = oracle_branch_ratios(
            data["source"]["x"],
            data["source"]["bar_y"],
            data["parameters"],
        )
        raw_by_method = {
            "no_iw": torch.ones_like(oracle_conditional),
            "global_estimated_ratio": estimated_global[:, None].expand_as(
                oracle_conditional
            ),
            "conditional_estimated_ratio": estimated_conditional,
            "oracle_global_ratio": oracle_global[:, None].expand_as(
                oracle_conditional
            ),
            "oracle_conditional_ratio": oracle_conditional,
        }
        setting_summaries = []
        for method_index, method in enumerate(config["methods"]):
            start = time.perf_counter()
            raw = raw_by_method[method]
            normalization = (
                "conditional_branch"
                if "conditional" in method
                else "global"
            )
            stages = process_weight_stages(
                raw,
                data["source"]["bar_y"],
                min_weight=config["lower_clip"],
                max_weight=config["upper_clip"],
                normalization_scope=normalization,
            )
            used = stages["classifier_used"]
            classifier = train_weak_classifier(
                data["source"],
                data["target_weak"],
                data["target_test"],
                source_weights=used,
                seed=seed + 10000,
                device=device,
                epochs=classifier_epochs,
                learning_rate=config["learning_rate"],
            )
            epoch_rows = classifier["epoch_rows"]
            reference = (
                oracle_conditional
                if "conditional" in method
                else oracle_global[:, None].expand_as(oracle_conditional)
            )
            ratio_metrics = _ratio_metrics(
                raw, reference, data["source"]["bar_y"]
            )
            matching = distribution_matching_diagnostics(
                data["source"]["x"],
                data["target_weak"]["x"],
                used,
                seed=seed + 50000 + method_index,
            )
            used_stats = weight_summary(used)
            raw_stats = weight_summary(
                raw,
                raw_reference=raw,
                lower_clip=config["lower_clip"],
                upper_clip=config["upper_clip"],
            )
            run_dir = os.path.join(
                args.results_root,
                "gaussian",
                setting,
                method,
                "seed_{}".format(seed),
            )
            os.makedirs(run_dir, exist_ok=True)
            run_config = {
                "config": config,
                "setting": setting,
                "method": method,
                "seed": seed,
                "data_seed": data_seed,
                "model_seed": seed + 10000,
                "ratio_seed": seed + 30000 + setting_index,
                "target_test_used_for_selection": False,
            }
            config_hash = hashlib.sha256(
                json.dumps(run_config, sort_keys=True).encode("utf-8")
            ).hexdigest()
            _json_dump(os.path.join(run_dir, "config.yaml"), run_config)
            _write_csv(os.path.join(run_dir, "epoch_metrics.csv"), epoch_rows)
            recovery_weak = risk_recovery_summary(
                [row["true_target_risk"] for row in epoch_rows],
                [row["estimated_weak_risk"] for row in epoch_rows],
            )
            recovery_iw = risk_recovery_summary(
                [row["true_target_risk"] for row in epoch_rows],
                [row["estimated_iw_risk"] for row in epoch_rows],
            )
            last_count = min(10, len(epoch_rows))
            summary = {
                "experiment": "B3",
                "method": method,
                "dataset": "gaussian_mixture_q{}".format(config["q"]),
                "shift_type": setting,
                "class_rotations": None,
                "target_weak_size": config["target_weak_size"],
                "seed": seed,
                "data_seed": data_seed,
                "model_seed": seed + 10000,
                "batch_seed": seed + 20000,
                "weight_shuffle_seed": seed + 40000,
                "split_hash": split_hash,
                "ratio_split_hash": split_hash,
                "git_commit": _git_commit(args.repository_root),
                "config_hash": config_hash,
                "target_test_used_for_selection": False,
                "last10_target_accuracy": float(
                    np.mean(
                        [
                            row["target_accuracy"]
                            for row in epoch_rows[-last_count:]
                        ]
                    )
                ),
                "true_target_risk": epoch_rows[-1]["true_target_risk"],
                "estimated_weak_risk": epoch_rows[-1][
                    "estimated_weak_risk"
                ],
                "estimated_iw_risk": epoch_rows[-1][
                    "estimated_iw_risk"
                ],
                "raw_ratio_mean": raw_stats["mean"],
                "raw_ratio_std": raw_stats["std"],
                "raw_lower_clip_rate": raw_stats[
                    "lower_clipping_rate"
                ],
                "raw_upper_clip_rate": raw_stats[
                    "upper_clipping_rate"
                ],
                "used_weight_mean": used_stats["mean"],
                "used_weight_std": used_stats["std"],
                "used_weight_CV": used_stats["cv"],
                "ESS": used_stats["ess"],
                "normalized_ESS": used_stats["normalized_ess"],
                "weighted_MMD": matching["weighted_mmd"],
                "weighted_feature_mean_error": matching[
                    "weighted_feature_mean_error"
                ],
                "weighted_covariance_error": matching[
                    "weighted_covariance_error"
                ],
                "posthoc_domain_accuracy": matching[
                    "posthoc_domain_accuracy"
                ],
                "posthoc_domain_AUC": matching["posthoc_domain_auc"],
                "runtime": time.perf_counter() - start + ratio_seconds,
                "ratio_estimator_loss": ratio_history[-1],
                "ratio_seconds_shared": ratio_seconds,
                "risk_recovery_weak": recovery_weak,
                "risk_recovery_iw": recovery_iw,
                "status": "ok",
            }
            summary.update(ratio_metrics)
            _json_dump(os.path.join(run_dir, "summary.json"), summary)
            setting_summaries.append(summary)
            summaries.append(summary)
        oracle_risk = min(
            item["true_target_risk"]
            for item in setting_summaries
            if item["method"].startswith("oracle_")
        )
        for item in setting_summaries:
            item["excess_risk_relative_to_oracle"] = (
                item["true_target_risk"] - oracle_risk
            )
            run_dir = os.path.join(
                args.results_root,
                "gaussian",
                setting,
                item["method"],
                "seed_{}".format(seed),
            )
            _json_dump(os.path.join(run_dir, "summary.json"), item)
    return summaries


def main(argv: Optional[Sequence[str]] = None) -> Any:
    args = parse_args(argv)
    config = load_config(args.config)
    seeds = [0] if args.experiment_mode == "smoke" else config["seeds"]
    if args.count:
        print(len(seeds))
        return len(seeds)
    if args.list_tasks:
        for index, seed in enumerate(seeds):
            print("{}\t{}".format(index, json.dumps({"seed": seed})))
        return seeds
    if args.task_index is None:
        raise ValueError("--task-index is required")
    return run_seed(config, int(seeds[args.task_index]), args)


if __name__ == "__main__":
    main()
