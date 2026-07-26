"""Combine original SCAR/ADIW results with neural-ratio SCAR runs."""

import argparse
import csv
import glob
import json
import os
from typing import Any, Dict, List, Optional, Sequence

import numpy as np


BASELINE_ALIASES = {
    "ADIW-SCAR (train + test)": "ADIW-SCAR",
    "SCAR (pooled train + test, no ADIW)": "ftSCAR / pooled SCAR",
    "SCAR (train only)": "trainSCAR",
    "SCAR (test only)": "testSCAR",
}


def read_baselines(path: str) -> List[Dict[str, Any]]:
    rows = []
    with open(path, "r", newline="") as stream:
        for row in csv.DictReader(stream):
            values = [
                float(value) for value in row["seed_accuracies"].split(";")
            ]
            rows.append(
                {
                    "method": BASELINE_ALIASES.get(row["method"], row["method"]),
                    "seeds": list(range(len(values))),
                    "accuracies": values,
                }
            )
    return rows


def read_neural_results(directory: str) -> List[Dict[str, Any]]:
    grouped = {}  # type: Dict[str, Dict[int, float]]
    pattern = os.path.join(directory, "*_ratio_scar_seed*.json")
    for path in glob.glob(pattern):
        with open(path, "r") as stream:
            record = json.load(stream)
        grouped.setdefault(record["method"], {})[int(record["seed"])] = float(
            record["target_accuracy"]
        )
    rows = []
    display = {
        "conditioned_ratio_scar": "Conditioned-Ratio-SCAR",
        "multihead_ratio_scar": "Multihead-Ratio-SCAR",
    }
    for method in sorted(grouped):
        seed_map = grouped[method]
        seeds = sorted(seed_map)
        rows.append(
            {
                "method": display.get(method, method),
                "seeds": seeds,
                "accuracies": [seed_map[seed] for seed in seeds],
            }
        )
    return rows


def summarize(
    baseline_path: str, result_directory: str
) -> List[Dict[str, Any]]:
    rows = read_baselines(baseline_path) + read_neural_results(result_directory)
    pooled = next(row for row in rows if row["method"] == "ftSCAR / pooled SCAR")
    adiw = next(row for row in rows if row["method"] == "ADIW-SCAR")
    pooled_values = np.asarray(pooled["accuracies"], dtype=np.float64)
    adiw_values = np.asarray(adiw["accuracies"], dtype=np.float64)
    summaries = []
    for row in rows:
        values = np.asarray(row["accuracies"], dtype=np.float64)
        if row["seeds"] != pooled["seeds"]:
            raise ValueError(
                "{} has seeds {}; expected {}".format(
                    row["method"], row["seeds"], pooled["seeds"]
                )
            )
        pooled_deltas = values - pooled_values
        summaries.append(
            {
                "method": row["method"],
                "num_seeds": len(values),
                "mean_accuracy": float(values.mean()),
                "std_accuracy": float(values.std()),
                "delta_vs_pooled": float(pooled_deltas.mean()),
                "paired_delta_std_vs_pooled": float(
                    pooled_deltas.std()
                ),
                "wins_vs_pooled": int((pooled_deltas > 0.0).sum()),
                "delta_vs_adiw": float((values - adiw_values).mean()),
                "seed_accuracies": ";".join(
                    "{:.6f}".format(value) for value in values
                ),
            }
        )
    return summaries


def write_report(summaries: List[Dict[str, Any]], output_dir: str) -> None:
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
    csv_path = os.path.join(output_dir, "scar_method_comparison.csv")
    fields = [
        "method",
        "num_seeds",
        "mean_accuracy",
        "std_accuracy",
        "delta_vs_pooled",
        "paired_delta_std_vs_pooled",
        "wins_vs_pooled",
        "delta_vs_adiw",
        "seed_accuracies",
    ]
    with open(csv_path, "w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(summaries)
    ranked = sorted(summaries, key=lambda row: row["mean_accuracy"], reverse=True)
    lines = [
        "# Neural ratio estimator versus SCAR baselines",
        "",
        "Matched MNIST rotation-30 setting, seeds 0--4. Accuracy is the mean "
        "over each run's final 10 classifier epochs.",
        "",
        "| Rank | Method | Accuracy mean +/- SD | Delta vs ftSCAR | Delta vs ADIW |",
        "|---:|---|---:|---:|---:|",
    ]
    for rank, row in enumerate(ranked, 1):
        lines.append(
            "| {} | {} | {:.3f} +/- {:.3f} | {:+.3f} | {:+.3f} |".format(
                rank,
                row["method"],
                row["mean_accuracy"],
                row["std_accuracy"],
                row["delta_vs_pooled"],
                row["delta_vs_adiw"],
            )
        )
    lines.extend(["", "## Per-seed accuracy", ""])
    methods = [row["method"] for row in summaries]
    lines.append("| Seed | " + " | ".join(methods) + " |")
    lines.append("|---:|" + "|".join(["---:"] * len(methods)) + "|")
    parsed = [
        [float(value) for value in row["seed_accuracies"].split(";")]
        for row in summaries
    ]
    for seed in range(summaries[0]["num_seeds"]):
        lines.append(
            "| {} | {} |".format(
                seed,
                " | ".join("{:.3f}".format(values[seed]) for values in parsed),
            )
        )
    report_path = os.path.join(output_dir, "SCAR_METHOD_COMPARISON.md")
    with open(report_path, "w") as stream:
        stream.write("\n".join(lines) + "\n")
    print("\n".join(lines))


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-csv", required=True)
    parser.add_argument("--result-dir", required=True)
    parser.add_argument("--output-dir", default=None)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> List[Dict[str, Any]]:
    args = parse_args(argv)
    summaries = summarize(args.baseline_csv, args.result_dir)
    write_report(summaries, args.output_dir or args.result_dir)
    return summaries


if __name__ == "__main__":
    main()
