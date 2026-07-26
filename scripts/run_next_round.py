"""Expand a bounded next-round matrix and run one reproducible task."""

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
    # The .yaml file deliberately uses JSON-compatible YAML so Wisteria does
    # not need an additional PyYAML installation.
    with open(path, "r", encoding="utf-8") as stream:
        return json.load(stream)


def expand_phase(config: Dict[str, Any], phase_name: str) -> List[Dict[str, Any]]:
    if phase_name not in config["phases"]:
        raise ValueError("unknown phase: {}".format(phase_name))
    phase = config["phases"][phase_name]
    keys = (
        "datasets",
        "shifts",
        "rotations",
        "target_weak_sizes",
        "methods",
        "clips",
        "normalizations",
        "fusion_gate_inits",
        "seeds",
    )
    tasks = []
    for values in itertools.product(*(phase[key] for key in keys)):
        task = dict(zip(keys, values))
        task["phase"] = phase_name
        tasks.append(task)
    return tasks


def task_arguments(
    config: Dict[str, Any],
    task: Dict[str, Any],
    mode: str,
    scarce_repo: str,
    repository_root: str,
    results_root: str,
    device: str,
) -> List[str]:
    common = config["common"]
    mode_config = config["experiment_modes"][mode]
    clip_min, clip_max = task["clips"]
    return [
        "--scarce-repo",
        scarce_repo,
        "--repository-root",
        repository_root,
        "--results-root",
        results_root,
        "--dataset",
        str(task["datasets"]),
        "--shift",
        str(task["shifts"]),
        "--target-rotation",
        str(task["rotations"]),
        "--target-weak-size",
        str(task["target_weak_sizes"]),
        "--source-size",
        str(common["source_size"]),
        "--method",
        str(task["methods"]),
        "--seed",
        str(task["seeds"]),
        "--ratio-epochs",
        str(mode_config["ratio_epochs"]),
        "--classifier-epochs",
        str(mode_config["classifier_epochs"]),
        "--batch-size",
        str(common["batch_size"]),
        "--workers",
        str(common["workers"]),
        "--ratio-learning-rate",
        str(common["ratio_learning_rate"]),
        "--learning-rate",
        str(common["classifier_learning_rate"]),
        "--weight-decay",
        str(common["weight_decay"]),
        "--ratio-clip-min",
        str(clip_min),
        "--ratio-clip-max",
        str(clip_max),
        "--normalization",
        str(task["normalizations"]),
        "--fusion-gate-init",
        str(task["fusion_gate_inits"]),
        "--ratio-validation-fraction",
        str(common["ratio_validation_fraction"]),
        "--diagnostic-max-samples",
        str(common["diagnostic_max_samples"]),
        "--device",
        device,
    ]


def parse_args(
    argv: Optional[Sequence[str]] = None
) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/next_round.yaml")
    parser.add_argument("--phase", default="smoke")
    parser.add_argument(
        "--experiment-mode",
        choices=("smoke", "diagnostic", "full"),
        default="smoke",
    )
    parser.add_argument("--task-index", type=int)
    parser.add_argument("--list-tasks", action="store_true")
    parser.add_argument("--count", action="store_true")
    parser.add_argument("--scarce-repo")
    parser.add_argument("--repository-root", default=os.getcwd())
    parser.add_argument("--results-root", default="results/next_round")
    parser.add_argument(
        "--device", choices=("auto", "cpu", "cuda"), default="auto"
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> Any:
    args = parse_args(argv)
    config = load_config(args.config)
    tasks = expand_phase(config, args.phase)
    if args.count:
        print(len(tasks))
        return len(tasks)
    if args.list_tasks:
        for index, task in enumerate(tasks):
            print("{}\t{}".format(index, json.dumps(task, sort_keys=True)))
        return tasks
    if args.task_index is None:
        raise ValueError("--task-index is required unless listing/counting")
    if args.task_index < 0 or args.task_index >= len(tasks):
        raise IndexError(
            "task index {} outside [0,{})".format(
                args.task_index, len(tasks)
            )
        )
    if not args.scarce_repo:
        raise ValueError("--scarce-repo is required to execute a task")
    task = tasks[args.task_index]
    print(
        "NEXT_ROUND_TASK index={} {}".format(
            args.task_index, json.dumps(task, sort_keys=True)
        )
    )
    from experiments.next_round import main as run_one

    return run_one(
        task_arguments(
            config,
            task,
            args.experiment_mode,
            args.scarce_repo,
            args.repository_root,
            args.results_root,
            args.device,
        )
    )


if __name__ == "__main__":
    main()
