from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


PROJECT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare aggregate reproduction results to Table 5.")
    parser.add_argument(
        "--reproduced",
        type=Path,
        default=PROJECT / "reports" / "main_table.csv",
    )
    parser.add_argument(
        "--reference",
        type=Path,
        default=PROJECT / "reports" / "paper_table5_reference.csv",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT / "reports" / "comparison_to_table5.csv",
    )
    args = parser.parse_args()

    reproduced = pd.read_csv(args.reproduced).rename(
        columns={"mean": "reproduced_mean", "std": "reproduced_std"}
    )
    reference = pd.read_csv(args.reference).rename(
        columns={"mean_accuracy": "paper_mean", "std_accuracy": "paper_std"}
    )
    merged = reference.merge(
        reproduced,
        on=["dataset", "shift", "method"],
        how="outer",
        validate="one_to_one",
    )
    merged["mean_delta"] = merged["reproduced_mean"] - merged["paper_mean"]
    merged["absolute_delta"] = merged["mean_delta"].abs()
    merged["delta_in_paper_std"] = merged["mean_delta"] / merged["paper_std"]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(args.output, index=False)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
