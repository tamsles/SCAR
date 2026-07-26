"""Run the fair multi-seed conditional-ratio architecture comparison."""

import argparse
import os
import sys
from typing import Optional, Sequence

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from experiments.ablation import main as ablation_main
from train import load_config


ARCHITECTURE_CONDITIONS = ",".join(
    (
        "unified_ratio",
        "separate_conditional",
        "multihead_conditional",
        "fusion_conditional",
        "unit_ratio",
        "oracle_ratio",
    )
)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", default="configs/conditional_synthetic.json"
    )
    parser.add_argument(
        "--output-dir", default="results/conditional_ratio_ablation"
    )
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--seeds", default=None)
    parser.add_argument("--epochs", type=int, default=None)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None):
    args = parse_args(argv)
    forwarded = [
        "--config",
        args.config,
        "--output-dir",
        args.output_dir,
        "--device",
        args.device,
        "--conditions",
        ARCHITECTURE_CONDITIONS,
    ]
    if args.seeds is not None:
        forwarded.extend(("--seeds", args.seeds))
    else:
        config = load_config(args.config)
        experiment = config.get("experiment", {})
        configured_seeds = (
            experiment.get("seeds", [0, 1, 2, 3, 4])
            if isinstance(experiment, dict)
            else [0, 1, 2, 3, 4]
        )
        forwarded.extend(
            ("--seeds", ",".join(str(seed) for seed in configured_seeds))
        )
    if args.epochs is not None:
        forwarded.extend(("--epochs", str(args.epochs)))
    return ablation_main(forwarded)


if __name__ == "__main__":
    main()
