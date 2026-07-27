"""Frozen-score clipping, normalization, and calibration experiment."""

import argparse
import csv
import hashlib
import json
import os
import subprocess
import sys
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

REPOSITORY_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPOSITORY_ROOT not in sys.path:
    sys.path.insert(0, REPOSITORY_ROOT)

from experiments.next_round import (
    _collect_domain_outputs,
    _selected_logits,
    _split_indices,
    build_next_round_ratio_estimator,
    collect_classifier_features,
    paired_split_hash,
    prepare_matched_data,
    pretrain_ratio_with_diagnostics,
    train_classifier,
)
from experiments.scarce_ratio_comparison import load_scarce_modules, set_random_seed
from ratio_estimation import summarize_weight_values
from src.ratio_calibration import (
    apply_calibration,
    fit_platt_scaling,
    fit_temperature_scaling,
)
from src.ratio_diagnostics import (
    distribution_matching_diagnostics,
    ratio_components_from_logits,
)
from src.risk_recovery_metrics import risk_recovery_summary


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


def _cache_logits(
    model: torch.nn.Module,
    dataset: Any,
    device: torch.device,
    batch_size: int,
    workers: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    logits = torch.empty(
        len(dataset), model.num_classes, dtype=torch.float32
    )
    states = torch.empty(
        len(dataset), model.num_classes, dtype=torch.float32
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
    )
    model.eval()
    with torch.no_grad():
        for indices, images, complements in loader:
            selected, _ = _selected_logits(
                model, images.to(device), complements.to(device)
            )
            logits[indices.long()] = selected.cpu()
            states[indices.long()] = complements
    return logits, states


def _calibration_parameters(
    method: str,
    source_validation_logits: torch.Tensor,
    target_validation_logits: torch.Tensor,
) -> Dict[str, Any]:
    if method == "none":
        return {
            "method": "none",
            "temperature": None,
            "slope": None,
            "intercept": None,
        }
    if method == "temperature":
        return fit_temperature_scaling(
            source_validation_logits, target_validation_logits
        )
    if method == "platt":
        return fit_platt_scaling(
            source_validation_logits, target_validation_logits
        )
    raise ValueError("unknown calibration method")


def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as stream:
        return json.load(stream)


def parse_args(
    argv: Optional[Sequence[str]] = None,
) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="configs/phase_b/b2_postprocessing/config.json",
    )
    parser.add_argument("--phase", default="B2_grid")
    parser.add_argument(
        "--experiment-mode",
        choices=("smoke", "diagnostic", "full"),
        default="smoke",
    )
    parser.add_argument(
        "--stage",
        choices=("freeze", "grid", "calibration"),
        default=None,
    )
    parser.add_argument("--task-index", type=int)
    parser.add_argument("--count", action="store_true")
    parser.add_argument("--list-tasks", action="store_true")
    parser.add_argument("--scarce-repo")
    parser.add_argument("--repository-root", default=os.getcwd())
    parser.add_argument("--results-root", default="outputs/phase_b/runs/B2")
    parser.add_argument(
        "--device", choices=("auto", "cpu", "cuda"), default="auto"
    )
    return parser.parse_args(argv)


def _run_classifier_configuration(
    config: Dict[str, Any],
    args: argparse.Namespace,
    data: Dict[str, Any],
    modules: Dict[str, Any],
    raw_weights: torch.Tensor,
    ratio_output: Dict[str, Any],
    split_hash: str,
    seed: int,
    lower_clip: float,
    normalization: str,
    calibration: str,
    calibration_parameters: Dict[str, Any],
    ratio_seconds: float,
    classifier_epochs: int,
    device: torch.device,
) -> Dict[str, Any]:
    start = time.perf_counter()
    classifier = train_classifier(
        modules,
        "unified",
        data,
        device,
        seed + 10000,
        seed + 20000,
        seed + 40000,
        config["batch_size"],
        classifier_epochs,
        config["classifier_learning_rate"],
        config["weight_decay"],
        config["workers"],
        raw_weights,
        lower_clip,
        config["upper_clip"],
        normalization,
    )
    source_features = collect_classifier_features(
        data["source_weak"],
        classifier["model"],
        device,
        config["batch_size"],
        config["workers"],
        min(config["diagnostic_max_samples"], 2048),
    )
    target_features = collect_classifier_features(
        data["target_weak"],
        classifier["model"],
        device,
        config["batch_size"],
        config["workers"],
        min(config["diagnostic_max_samples"], 2048),
    )
    matching = distribution_matching_diagnostics(
        source_features,
        target_features,
        classifier["classifier_used_cache"][: source_features.shape[0]],
        seed=seed + 50000,
    )
    raw_stats = summarize_weight_values(
        raw_weights,
        raw_reference=raw_weights,
        min_weight=lower_clip,
        max_weight=config["upper_clip"],
    )
    used_stats = summarize_weight_values(
        classifier["classifier_used_cache"]
    )
    setting = "lower{:g}_norm-{}_cal-{}".format(
        lower_clip, normalization, calibration
    )
    run_dir = os.path.join(
        args.results_root,
        "mnist",
        setting,
        "unified",
        "seed_{}".format(seed),
    )
    os.makedirs(run_dir, exist_ok=True)
    _write_csv(
        os.path.join(run_dir, "epoch_metrics.csv"),
        classifier["epoch_rows"],
    )
    _json_dump(
        os.path.join(run_dir, "calibration.json"),
        calibration_parameters,
    )
    epoch_rows = classifier["epoch_rows"]
    last_count = min(10, len(epoch_rows))
    config_record = {
        "config": config,
        "experiment": "B2",
        "seed": seed,
        "lower_clip": lower_clip,
        "upper_clip": config["upper_clip"],
        "normalization": normalization,
        "calibration": calibration,
        "target_test_used_for_selection": False,
    }
    config_hash = hashlib.sha256(
        json.dumps(config_record, sort_keys=True).encode("utf-8")
    ).hexdigest()
    summary = {
        "experiment": "B2",
        "method": "unified",
        "dataset": "mnist",
        "shift_type": "rotation_30",
        "class_rotations": None,
        "target_weak_size": config["target_weak_size"],
        "seed": seed,
        "data_seed": seed,
        "model_seed": seed + 10000,
        "batch_seed": seed + 20000,
        "weight_shuffle_seed": seed + 40000,
        "split_hash": split_hash,
        "ratio_split_hash": ratio_output["split_hash"],
        "git_commit": _git_commit(args.repository_root),
        "config_hash": config_hash,
        "target_test_used_for_selection": False,
        "lower_clip": lower_clip,
        "upper_clip": config["upper_clip"],
        "normalization": normalization,
        "calibration": calibration,
        "last10_target_accuracy": float(
            np.mean(
                [
                    row["target_accuracy"]
                    for row in epoch_rows[-last_count:]
                ]
            )
        ),
        "true_target_risk": epoch_rows[-1]["true_target_risk"],
        "estimated_weak_risk": epoch_rows[-1]["estimated_weak_risk"],
        "estimated_iw_risk": epoch_rows[-1]["estimated_iw_risk"],
        "raw_ratio_mean": raw_stats["mean"],
        "raw_ratio_std": raw_stats["std"],
        "raw_lower_clip_rate": raw_stats["lower_clipping_rate"],
        "raw_upper_clip_rate": raw_stats["upper_clipping_rate"],
        "used_weight_mean": used_stats["mean"],
        "used_weight_std": used_stats["std"],
        "used_weight_CV": used_stats["cv"],
        "ESS": used_stats["ess"],
        "normalized_ESS": used_stats["normalized_ess"],
        "weighted_MMD": matching["weighted_mmd"],
        "posthoc_domain_accuracy": matching["posthoc_domain_accuracy"],
        "posthoc_domain_AUC": matching["posthoc_domain_auc"],
        "weighted_feature_mean_error": matching[
            "weighted_feature_mean_error"
        ],
        "weighted_covariance_error": matching[
            "weighted_covariance_error"
        ],
        "ratio_estimator_loss": ratio_output["epoch_summaries"][-1][
            "total_loss"
        ],
        "ratio_seconds_shared": ratio_seconds,
        "runtime": time.perf_counter() - start + ratio_seconds,
        "risk_recovery_weak": risk_recovery_summary(
            [row["true_target_risk"] for row in epoch_rows],
            [row["estimated_weak_risk"] for row in epoch_rows],
        ),
        "risk_recovery_iw": risk_recovery_summary(
            [row["true_target_risk"] for row in epoch_rows],
            [row["estimated_iw_risk"] for row in epoch_rows],
        ),
        "status": "ok",
    }
    _json_dump(os.path.join(run_dir, "config.yaml"), config_record)
    _json_dump(os.path.join(run_dir, "summary.json"), summary)
    return summary


def run_seed(
    config: Dict[str, Any],
    seed: int,
    args: argparse.Namespace,
    specification: Optional[Tuple[float, str, str]] = None,
) -> Any:
    device = torch.device(
        "cuda"
        if args.device == "auto" and torch.cuda.is_available()
        else args.device
    )
    if args.device == "auto" and not torch.cuda.is_available():
        device = torch.device("cpu")
    stage = args.stage
    if stage is None:
        phase = args.phase.lower()
        if "freeze" in phase:
            stage = "freeze"
        elif "calibration" in phase:
            stage = "calibration"
        else:
            stage = "grid"
    smoke = args.experiment_mode == "smoke"
    ratio_epochs = (
        config["smoke_ratio_epochs"] if smoke else config["ratio_epochs"]
    )
    classifier_epochs = (
        config["smoke_classifier_epochs"]
        if smoke
        else config["classifier_epochs"]
    )
    set_random_seed(seed)
    modules = load_scarce_modules(args.scarce_repo)
    data, shift_spec = prepare_matched_data(
        modules,
        args.scarce_repo,
        config["dataset"],
        config["target_weak_size"],
        config["source_size"],
        seed,
        config["shift"],
        config["target_rotation"],
    )
    split_hash = paired_split_hash(data, shift_spec)
    frozen_dir = os.path.join(
        args.results_root, "frozen_scores", "seed_{}".format(seed)
    )
    os.makedirs(frozen_dir, exist_ok=True)
    frozen_path = os.path.join(frozen_dir, "frozen_scores.pt")
    metadata_path = os.path.join(frozen_dir, "metadata.json")
    if stage in ("grid", "calibration") and os.path.isfile(frozen_path):
        frozen = torch.load(frozen_path, map_location="cpu")
        with open(metadata_path, "r", encoding="utf-8") as stream:
            frozen_metadata = json.load(stream)
        if frozen["split_hash"] != split_hash:
            raise ValueError("frozen-score split hash does not match data")
        ratio_output = {
            "split_hash": frozen["ratio_split_hash"],
            "seconds": float(frozen_metadata["ratio_seconds"]),
            "epoch_summaries": [
                {
                    "total_loss": float(
                        frozen_metadata["ratio_estimator_loss"]
                    )
                }
            ],
        }
        source_logits = frozen["raw_domain_logits"]
        source_states = frozen["branch_id"]
        target_logits = frozen["target_domain_logits"]
        target_states = frozen["target_branch_id"]
        source_validation = frozen["source_validation_indices"]
        target_validation = frozen["target_validation_indices"]
    elif stage == "freeze":
        ratio_model, _ = build_next_round_ratio_estimator(
            modules["models"],
            data["input_dim"],
            data["num_classes"],
            config["ratio_method"],
            device,
            0.0,
            config["upper_clip"],
            "none",
            0.5,
        )
        ratio_output = pretrain_ratio_with_diagnostics(
            ratio_model,
            data["source_weak"],
            data["target_weak"],
            device,
            config["batch_size"],
            ratio_epochs,
            config["ratio_learning_rate"],
            config["weight_decay"],
            config["workers"],
            config["ratio_validation_fraction"],
            seed + 30000,
            0.0,
            config["upper_clip"],
            "none",
            config["diagnostic_max_samples"],
        )
        source_logits, source_states = _cache_logits(
            ratio_model,
            data["source_weak"],
            device,
            config["batch_size"],
            config["workers"],
        )
        target_logits, target_states = _cache_logits(
            ratio_model,
            data["target_weak"],
            device,
            config["batch_size"],
            config["workers"],
        )
        source_validation = _split_indices(
            len(data["source_weak"]),
            seed + 30000 + 701,
            config["ratio_validation_fraction"],
        )[1]
        target_validation = _split_indices(
            len(data["target_weak"]),
            seed + 30000 + 907,
            config["ratio_validation_fraction"],
        )[1]
        frozen = {
            "sample_id": torch.arange(source_logits.shape[0]),
            "branch_id": source_states,
            "raw_domain_logits": source_logits,
            "probabilities": torch.sigmoid(source_logits),
            "raw_unclipped_ratios": torch.exp(
                source_logits.double().clamp(-20.0, 20.0)
            ).float(),
            "target_domain_logits": target_logits,
            "target_branch_id": target_states,
            "source_validation_indices": source_validation,
            "target_validation_indices": target_validation,
            "split_hash": split_hash,
            "ratio_split_hash": ratio_output["split_hash"],
        }
        torch.save(frozen, frozen_path)
        _json_dump(
            metadata_path,
            {
                "seed": seed,
                "split_hash": split_hash,
                "ratio_split_hash": ratio_output["split_hash"],
                "ratio_seconds": ratio_output["seconds"],
                "ratio_estimator_loss": ratio_output[
                    "epoch_summaries"
                ][-1]["total_loss"],
                "domain_label_convention": {
                    "source": 0,
                    "target": 1,
                    "probability": "P(target|x,branch)",
                    "ratio_formula": "exp(logit) * pi_source/pi_target",
                    "prior_correction": 1.0,
                },
                "target_test_used_for_selection": False,
            },
        )
    else:
        raise FileNotFoundError(
            "run B2_freeze first; missing {}".format(frozen_path)
        )
    if stage == "freeze":
        return {
            "experiment": "B2_freeze",
            "seed": seed,
            "split_hash": split_hash,
            "ratio_split_hash": ratio_output["split_hash"],
            "ratio_estimator_loss": ratio_output["epoch_summaries"][-1][
                "total_loss"
            ],
            "ratio_seconds": ratio_output["seconds"],
            "target_test_used_for_selection": False,
            "status": "ok",
        }
    outputs = []
    if specification is None:
        raise ValueError("classifier stages require one specification")
    specifications = [specification]
    for lower_clip, normalization, calibration in specifications:
        parameters = _calibration_parameters(
            calibration,
            source_logits[source_validation],
            target_logits[target_validation],
        )
        calibrated_logits = apply_calibration(source_logits, parameters)
        raw = ratio_components_from_logits(
            calibrated_logits,
            lower_clip=0.0,
            upper_clip=config["upper_clip"],
        )["unclipped_ratio"]
        outputs.append(
            _run_classifier_configuration(
                config,
                args,
                data,
                modules,
                raw,
                ratio_output,
                split_hash,
                seed,
                lower_clip,
                normalization,
                calibration,
                parameters,
                ratio_output["seconds"],
                classifier_epochs,
                device,
            )
        )
    return outputs


def main(argv: Optional[Sequence[str]] = None) -> Any:
    args = parse_args(argv)
    config = load_config(args.config)
    seeds = [0] if args.experiment_mode == "smoke" else config["seeds"]
    stage = args.stage
    if stage is None:
        phase = args.phase.lower()
        if "freeze" in phase:
            stage = "freeze"
        elif "calibration" in phase:
            stage = "calibration"
        else:
            stage = "grid"
    if stage == "freeze":
        tasks = [
            {"seed": int(seed), "specification": None} for seed in seeds
        ]
    elif stage == "grid":
        lower_clips = (
            config["smoke_lower_clips"]
            if args.experiment_mode == "smoke"
            else config["lower_clips"]
        )
        tasks = [
            {
                "seed": int(seed),
                "specification": (float(lower), str(normalization), "none"),
            }
            for seed in seeds
            for lower in lower_clips
            for normalization in config["normalizations"]
        ]
    else:
        selected_path = os.path.join(
            args.results_root, "selected_configs.json"
        )
        with open(selected_path, "r", encoding="utf-8") as stream:
            selected = json.load(stream)["selected_configs"]
        tasks = [
            {
                "seed": int(seed),
                "specification": (
                    float(item["lower_clip"]),
                    str(item["normalization"]),
                    str(calibration),
                ),
            }
            for seed in seeds
            for item in selected
            for calibration in config["calibrations"]
        ]
    if args.count:
        print(len(tasks))
        return len(tasks)
    if args.list_tasks:
        for index, task in enumerate(tasks):
            print("{}\t{}".format(index, json.dumps(task)))
        return tasks
    if args.task_index is None:
        raise ValueError("--task-index is required")
    if not args.scarce_repo:
        raise ValueError("--scarce-repo is required")
    task = tasks[args.task_index]
    return run_seed(
        config,
        int(task["seed"]),
        args,
        specification=task["specification"],
    )


if __name__ == "__main__":
    main()
