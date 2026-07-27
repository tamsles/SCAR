"""Ratio correctness and distribution-matching audit."""

import argparse
import csv
import hashlib
import json
import os
import subprocess
import sys
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
from torch.utils.data import DataLoader, Subset

REPOSITORY_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPOSITORY_ROOT not in sys.path:
    sys.path.insert(0, REPOSITORY_ROOT)

from experiments.next_round import (
    _collect_domain_outputs,
    _selected_logits,
    build_next_round_ratio_estimator,
    paired_split_hash,
    prepare_matched_data,
    pretrain_ratio_with_diagnostics,
)
from experiments.scarce_ratio_comparison import load_scarce_modules, set_random_seed
from ratio_estimation import process_weight_stages, weight_stage_rows
from src.ratio_diagnostics import (
    DOMAIN_LABEL_CONVENTION,
    distribution_matching_diagnostics,
    ratio_components_from_logits,
    weight_summary,
    weighted_mmd,
    weighted_moment_errors,
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


def _weak_labels(dataset: Any, maximum: int) -> torch.Tensor:
    labels = []
    for local_index in range(min(len(dataset), maximum)):
        base_index = int(dataset.base_indices[local_index].item())
        _, label = dataset.base_dataset[base_index]
        labels.append(int(label))
    return torch.tensor(labels, dtype=torch.long)


@torch.no_grad()
def _ratio_features(
    model: torch.nn.Module,
    dataset: Any,
    device: torch.device,
    batch_size: int,
    workers: int,
    maximum: int,
) -> torch.Tensor:
    loader = DataLoader(
        Subset(dataset, list(range(min(len(dataset), maximum)))),
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
    )
    parts = []
    model.eval()
    for _, images, _ in loader:
        parts.append(model.backbone(images.to(device)).detach().cpu())
    return torch.cat(parts)


def _classwise_rows(
    source_features: torch.Tensor,
    target_features: torch.Tensor,
    source_weights: torch.Tensor,
    source_labels: torch.Tensor,
    target_labels: torch.Tensor,
    seed: int,
) -> List[Dict[str, Any]]:
    rows = []
    for class_index in range(10):
        source_mask = source_labels == class_index
        target_mask = target_labels == class_index
        if source_mask.sum() < 5 or target_mask.sum() < 5:
            continue
        weights = source_weights[source_mask]
        moments = weighted_moment_errors(
            source_features[source_mask],
            target_features[target_mask],
            weights,
        )
        rows.append(
            {
                "class_index": class_index,
                "source_count": int(source_mask.sum().item()),
                "target_count": int(target_mask.sum().item()),
                "weighted_mmd": weighted_mmd(
                    source_features[source_mask],
                    target_features[target_mask],
                    weights,
                    maximum_samples=512,
                    seed=seed + class_index,
                ),
                **moments,
            }
        )
    return rows


def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as stream:
        return json.load(stream)


def parse_args(
    argv: Optional[Sequence[str]] = None,
) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", default="configs/phase_b/b1_ratio_audit/config.json"
    )
    parser.add_argument("--phase", default="B1")
    parser.add_argument(
        "--experiment-mode",
        choices=("smoke", "diagnostic", "full"),
        default="smoke",
    )
    parser.add_argument("--task-index", type=int)
    parser.add_argument("--count", action="store_true")
    parser.add_argument("--list-tasks", action="store_true")
    parser.add_argument("--scarce-repo")
    parser.add_argument("--b0-root")
    parser.add_argument("--repository-root", default=os.getcwd())
    parser.add_argument("--results-root", default="outputs/phase_b/runs/B1")
    parser.add_argument(
        "--device", choices=("auto", "cpu", "cuda"), default="auto"
    )
    return parser.parse_args(argv)


def run_task(
    config: Dict[str, Any],
    method: str,
    seed: int,
    args: argparse.Namespace,
) -> Dict[str, Any]:
    started = time.perf_counter()
    device = torch.device(
        "cuda"
        if args.device == "auto" and torch.cuda.is_available()
        else args.device
    )
    if args.device == "auto" and not torch.cuda.is_available():
        device = torch.device("cpu")
    set_random_seed(seed)
    modules = load_scarce_modules(args.scarce_repo)
    data, shift_spec = prepare_matched_data(
        modules,
        args.scarce_repo,
        config["dataset"],
        config["target_weak_size"],
        config["source_size"],
        seed,
        "rotation",
        config["target_rotation"],
    )
    split_hash = paired_split_hash(data, shift_spec)
    maximum = min(config["diagnostic_max_samples"], 2048)
    ratio_stage_rows = []
    if method == "adiw_original":
        b0_root = args.b0_root or os.path.join(
            os.path.dirname(args.results_root), "B0"
        )
        setting = (
            "rotation_30_tw1000_clip0.01-20_"
            "norm-conditional_branch"
        )
        checkpoint_path = os.path.join(
            b0_root,
            "mnist",
            setting,
            "adiw_original",
            "seed_{}".format(seed),
            "checkpoint.pt",
        )
        checkpoint = torch.load(checkpoint_path, map_location=device)
        classifier = modules["models"].mlp_model(
            input_dim=data["input_dim"],
            hidden_dim=500,
            output_dim=data["num_classes"],
        ).to(device)
        classifier.load_state_dict(checkpoint["classifier"])
        raw = checkpoint["source_weight_cache"].float()
        used = checkpoint["classifier_used_cache"].float()
        source_states = torch.stack(
            [
                data["source_weak"][index][2]
                for index in range(min(len(data["source_weak"]), maximum))
            ]
        )
        adiw_stages = process_weight_stages(
            raw[:maximum],
            source_states,
            min_weight=config["lower_clip"],
            max_weight=config["upper_clip"],
            normalization_scope=config["normalization"],
        )
        adiw_stages["classifier_used"] = used[:maximum]
        adiw_rows = weight_stage_rows(
            adiw_stages,
            source_states,
            epoch=1,
            min_weight=config["lower_clip"],
            max_weight=config["upper_clip"],
            domain="source_diagnostic",
        )
        ratio_stage_rows = (
            adiw_rows["raw"]
            + adiw_rows["clipped"]
            + adiw_rows["normalized"]
            + adiw_rows["classifier_used"]
        )
        source_features = []
        target_features = []
        with torch.no_grad():
            for dataset, output in (
                (data["source_weak"], source_features),
                (data["target_weak"], target_features),
            ):
                loader = DataLoader(
                    Subset(dataset, list(range(min(len(dataset), maximum)))),
                    batch_size=config["batch_size"],
                    shuffle=False,
                )
                for _, images, _ in loader:
                    images = images.to(device)
                    output.append(
                        classifier.relu1(
                            classifier.fc1(images.reshape(images.shape[0], -1))
                        )
                        .cpu()
                    )
        source_features = torch.cat(source_features)
        target_features = torch.cat(target_features)
        raw_logits = None
        probabilities = None
        raw_odds = None
        prior_correction = None
        ratio_split = split_hash
        ratio_loss = None
        direction_note = (
            "ADIW uses online KMM weights; discriminator probability and "
            "odds are not applicable."
        )
    else:
        ratio_model, _ = build_next_round_ratio_estimator(
            modules["models"],
            data["input_dim"],
            data["num_classes"],
            method,
            device,
            config["lower_clip"],
            config["upper_clip"],
            config["normalization"],
            0.5,
        )
        ratio_epochs = (
            config["smoke_ratio_epochs"]
            if args.experiment_mode == "smoke"
            else config["ratio_epochs"]
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
            config["lower_clip"],
            config["upper_clip"],
            config["normalization"],
            config["diagnostic_max_samples"],
        )
        collected = _collect_domain_outputs(
            ratio_model,
            data["source_weak"],
            data["target_weak"],
            device,
            config["batch_size"],
            config["workers"],
            maximum,
        )
        raw = collected["source_raw"]
        states = collected["source_complements"]
        stages = process_weight_stages(
            raw,
            states,
            min_weight=config["lower_clip"],
            max_weight=config["upper_clip"],
            normalization_scope=config["normalization"],
        )
        used = stages["classifier_used"]
        raw_logits = collected["source_logits"]
        components = ratio_components_from_logits(
            raw_logits,
            lower_clip=config["lower_clip"],
            upper_clip=config["upper_clip"],
        )
        probabilities = components["discriminator_probability"]
        raw_odds = components["raw_odds"]
        prior_correction = components["prior_correction"]
        source_features = _ratio_features(
            ratio_model,
            data["source_weak"],
            device,
            config["batch_size"],
            config["workers"],
            maximum,
        )
        target_features = _ratio_features(
            ratio_model,
            data["target_weak"],
            device,
            config["batch_size"],
            config["workers"],
            maximum,
        )
        ratio_split = ratio_output["split_hash"]
        ratio_loss = ratio_output["epoch_summaries"][-1]["total_loss"]
        direction_note = (
            "source=0,target=1; logit=log P(target|x)/P(source|x); "
            "balanced branch BCE gives prior correction 1."
        )
        ratio_stage_rows = (
            ratio_output["ratio_raw_stats.csv"]
            + ratio_output["ratio_clipped_stats.csv"]
            + ratio_output["ratio_normalized_stats.csv"]
        )
    source_features = source_features[:maximum]
    target_features = target_features[:maximum]
    used = used[:maximum]
    raw = raw[:maximum]
    matching = distribution_matching_diagnostics(
        source_features,
        target_features,
        used,
        seed=seed + 50000,
    )
    classwise = _classwise_rows(
        source_features,
        target_features,
        used,
        _weak_labels(data["source_weak"], maximum),
        _weak_labels(data["target_weak"], maximum),
        seed,
    )
    raw_stats = weight_summary(
        raw,
        raw_reference=raw,
        lower_clip=config["lower_clip"],
        upper_clip=config["upper_clip"],
    )
    used_stats = weight_summary(used)
    run_dir = os.path.join(
        args.results_root, method, "seed_{}".format(seed)
    )
    os.makedirs(run_dir, exist_ok=True)
    _write_csv(os.path.join(run_dir, "ratio_stage_stats.csv"), ratio_stage_rows)
    _write_csv(os.path.join(run_dir, "classwise_diagnostics.csv"), classwise)
    torch.save(
        {
            "raw_domain_logits": raw_logits,
            "discriminator_probability": probabilities,
            "raw_odds": raw_odds,
            "prior_correction": prior_correction,
            "unclipped_ratio": raw,
            "classifier_used_ratio": used,
        },
        os.path.join(run_dir, "ratio_samples.pt"),
    )
    summary = {
        "experiment": "B1",
        "method": method,
        "dataset": "mnist",
        "shift_type": "rotation_30",
        "seed": seed,
        "data_seed": seed,
        "model_seed": seed + 10000,
        "batch_seed": seed + 20000,
        "weight_shuffle_seed": seed + 40000,
        "split_hash": split_hash,
        "ratio_split_hash": ratio_split,
        "git_commit": _git_commit(args.repository_root),
        "config_hash": hashlib.sha256(
            json.dumps(
                {"config": config, "method": method, "seed": seed},
                sort_keys=True,
            ).encode("utf-8")
        ).hexdigest(),
        "target_test_used_for_selection": False,
        "target_weak_size": config["target_weak_size"],
        "class_rotations": None,
        "last10_target_accuracy": None,
        "true_target_risk": None,
        "estimated_weak_risk": None,
        "estimated_iw_risk": None,
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
        "posthoc_domain_accuracy": matching[
            "posthoc_domain_accuracy"
        ],
        "posthoc_domain_AUC": matching["posthoc_domain_auc"],
        "runtime": time.perf_counter() - started,
        "domain_label_convention": DOMAIN_LABEL_CONVENTION,
        "direction_note": direction_note,
        "raw_ratio": raw_stats,
        "used_ratio": used_stats,
        "ratio_estimator_loss": ratio_loss,
        **matching,
        "status": "ok",
    }
    _json_dump(
        os.path.join(run_dir, "config.yaml"),
        {"config": config, "method": method, "seed": seed},
    )
    _json_dump(os.path.join(run_dir, "summary.json"), summary)
    return summary


def main(argv: Optional[Sequence[str]] = None) -> Any:
    args = parse_args(argv)
    config = load_config(args.config)
    seeds = [0] if args.experiment_mode == "smoke" else config["seeds"]
    tasks = [
        (method, int(seed)) for method in config["methods"] for seed in seeds
    ]
    if args.count:
        print(len(tasks))
        return len(tasks)
    if args.list_tasks:
        for index, (method, seed) in enumerate(tasks):
            print(
                "{}\t{}".format(
                    index, json.dumps({"method": method, "seed": seed})
                )
            )
        return tasks
    if args.task_index is None:
        raise ValueError("--task-index is required")
    if not args.scarce_repo:
        raise ValueError("--scarce-repo is required")
    method, seed = tasks[args.task_index]
    return run_task(config, method, seed, args)


if __name__ == "__main__":
    main()
