from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))

from iwpu.reporting import (
    aggregate_ablation_table,
    aggregate_foodstamp_table,
    aggregate_main_table,
    completeness_table,
    read_results,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate paper reproduction runs.")
    parser.add_argument("--results", type=Path, default=PROJECT / "results")
    parser.add_argument("--output", type=Path, default=PROJECT / "reports")
    args = parser.parse_args()

    frame = read_results(args.results)
    args.output.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output / "all_runs.csv", index=False)
    aggregate_main_table(frame).to_csv(args.output / "main_table.csv", index=False)
    aggregate_ablation_table(frame).to_csv(
        args.output / "ablation_table.csv", index=False
    )
    aggregate_foodstamp_table(frame).to_csv(
        args.output / "appendix_foodstamp.csv", index=False
    )
    completeness_table(frame).to_csv(args.output / "completeness.csv", index=False)
    print(f"Wrote aggregate reports to {args.output}")


if __name__ == "__main__":
    main()
