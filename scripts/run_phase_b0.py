"""Run one matched B0 ADIW causal-decomposition task."""

import argparse
import itertools
import json
import os
import sys
from typing import Any, Dict, List, Optional, Sequence

REPOSITORY_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPOSITORY_ROOT not in sys.path:
    sys.path.insert(0, REPOSITORY_ROOT)


def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as stream:
        return json.load(stream)


def expand_tasks(
    config: Dict[str, Any], experiment_mode: str
) -> List[Dict[str, Any]]:
    seeds = [0] if experiment_mode == "smoke" else config["seeds"]
    return [
        {"method": method, "seed": int(seed)}
        for method, seed in itertools.product(config["methods"], seeds)
    ]


def task_arguments(
    config: Dict[str, Any],
    task: Dict[str, Any],
    experiment_mode: str,
    args: argparse.Namespace,
) -> List[str]:
    seed = int(task["seed"])
    classifier_epochs = (
        config["smoke_classifier_epochs"]
        if experiment_mode == "smoke"
        else config["classifier_epochs"]
    )
    return [
        "--experiment",
        "B0",
        "--scarce-repo",
        args.scarce_repo,
        "--repository-root",
        args.repository_root,
        "--results-root",
        args.results_root,
        "--dataset",
        config["dataset"],
        "--shift",
        config["shift"],
        "--target-rotation",
        str(config["target_rotation"]),
        "--target-weak-size",
        str(config["target_weak_size"]),
        "--source-size",
        str(config["source_size"]),
        "--method",
        task["method"],
        "--seed",
        str(seed),
        "--data-seed",
        str(seed),
        "--model-seed",
        str(seed + 10000),
        "--batch-seed",
        str(seed + 20000),
        "--ratio-seed",
        str(seed + 30000),
        "--weight-shuffle-seed",
        str(seed + 40000),
        "--ratio-epochs",
        str(config["ratio_epochs"]),
        "--classifier-epochs",
        str(classifier_epochs),
        "--batch-size",
        str(config["batch_size"]),
        "--workers",
        str(config["workers"]),
        "--ratio-learning-rate",
        str(config["ratio_learning_rate"]),
        "--learning-rate",
        str(config["classifier_learning_rate"]),
        "--weight-decay",
        str(config["weight_decay"]),
        "--ratio-clip-min",
        str(config["lower_clip"]),
        "--ratio-clip-max",
        str(config["upper_clip"]),
        "--normalization",
        config["normalization"],
        "--ratio-validation-fraction",
        str(config["ratio_validation_fraction"]),
        "--diagnostic-max-samples",
        str(config["diagnostic_max_samples"]),
        "--device",
        args.device,
    ]


def parse_args(
    argv: Optional[Sequence[str]] = None,
) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="configs/phase_b/b0_adiw_causal/config.json",
    )
    parser.add_argument("--phase", default="B0")
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
    parser.add_argument("--results-root", default="outputs/phase_b/runs/B0")
    parser.add_argument(
        "--device", choices=("auto", "cpu", "cuda"), default="auto"
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> Any:
    args = parse_args(argv)
    config = load_config(args.config)
    tasks = expand_tasks(config, args.experiment_mode)
    if args.count:
        print(len(tasks))
        return len(tasks)
    if args.list_tasks:
        for index, task in enumerate(tasks):
            print("{}\t{}".format(index, json.dumps(task, sort_keys=True)))
        return tasks
    if args.task_index is None:
        raise ValueError("--task-index is required")
    if not args.scarce_repo:
        raise ValueError("--scarce-repo is required")
    task = tasks[args.task_index]
    from experiments.next_round import main as run_one

    return run_one(
        task_arguments(config, task, args.experiment_mode, args)
    )


if __name__ == "__main__":
    main()

