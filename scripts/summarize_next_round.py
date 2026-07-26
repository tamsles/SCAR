"""Summarize matched next-round runs and generate diagnostic figures."""

import argparse
import csv
import glob
import json
import math
import os
import re
from collections import defaultdict
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


T_CRITICAL_95 = {
    1: 12.706,
    2: 4.303,
    3: 3.182,
    4: 2.776,
    5: 2.571,
    6: 2.447,
    7: 2.365,
    8: 2.306,
    9: 2.262,
    10: 2.228,
    11: 2.201,
    12: 2.179,
    13: 2.160,
    14: 2.145,
    15: 2.131,
    16: 2.120,
    17: 2.110,
    18: 2.101,
    19: 2.093,
}


def read_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as stream:
        return json.load(stream)


def read_csv(path: str) -> List[Dict[str, str]]:
    if not os.path.isfile(path):
        return []
    with open(path, "r", encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def write_csv(
    path: str,
    rows: Sequence[Dict[str, Any]],
    fields: Sequence[str],
) -> None:
    complete_fields = list(fields)
    for row in rows:
        for key in row:
            if key not in complete_fields:
                complete_fields.append(key)
    with open(path, "w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=complete_fields)
        writer.writeheader()
        writer.writerows(rows)


def collect_runs(results_root: str) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    successful = []
    failed = []
    for path in glob.glob(
        os.path.join(results_root, "**", "summary.json"), recursive=True
    ):
        summary = read_json(path)
        summary["_run_dir"] = os.path.dirname(path)
        if summary.get("status") == "ok":
            successful.append(summary)
        else:
            failed.append(summary)
    return successful, failed


def _group_key(run: Dict[str, Any]) -> Tuple[str, str]:
    return str(run["dataset"]), str(run["setting"])


def summarize_by_setting(runs: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    grouped = defaultdict(list)
    for run in runs:
        grouped[(_group_key(run), run["method"])].append(run)
    rows = []
    for ((dataset, setting), method), values in sorted(grouped.items()):
        values = sorted(values, key=lambda item: int(item["seed"]))
        accuracy = np.asarray(
            [float(item["last10_target_accuracy"]) for item in values],
            dtype=np.float64,
        )
        rows.append(
            {
                "dataset": dataset,
                "setting": setting,
                "method": method,
                "num_seeds": len(values),
                "mean_target_accuracy": float(accuracy.mean()),
                "population_sd_target_accuracy": float(accuracy.std()),
                "median_target_accuracy": float(np.median(accuracy)),
                "seed_accuracies": ";".join(
                    "{:.8f}".format(value) for value in accuracy
                ),
                "split_hashes": ";".join(
                    sorted(set(str(item["split_hash"]) for item in values))
                ),
            }
        )
    return rows


def _t_critical(sample_count: int) -> float:
    degrees = sample_count - 1
    return T_CRITICAL_95.get(degrees, 1.96)


def _bootstrap_ci(
    differences: np.ndarray, samples: int = 10000
) -> Tuple[float, float]:
    random_state = np.random.RandomState(23000)
    indices = random_state.randint(
        0, differences.size, size=(samples, differences.size)
    )
    means = differences[indices].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def paired_comparisons(
    runs: Sequence[Dict[str, Any]], meaningful_gain: float
) -> List[Dict[str, Any]]:
    by_setting = defaultdict(lambda: defaultdict(dict))
    for run in runs:
        by_setting[_group_key(run)][run["method"]][int(run["seed"])] = run
    rows = []
    for (dataset, setting), methods in sorted(by_setting.items()):
        if "multihead" not in methods:
            continue
        reference = methods["multihead"]
        for comparator, comparator_runs in sorted(methods.items()):
            if comparator == "multihead":
                continue
            seeds = sorted(set(reference) & set(comparator_runs))
            if len(seeds) < 2:
                continue
            differences = np.asarray(
                [
                    float(reference[seed]["last10_target_accuracy"])
                    - float(
                        comparator_runs[seed]["last10_target_accuracy"]
                    )
                    for seed in seeds
                ],
                dtype=np.float64,
            )
            mean = float(differences.mean())
            sample_sd = float(differences.std(ddof=1))
            half_width = _t_critical(len(seeds)) * sample_sd / math.sqrt(
                len(seeds)
            )
            bootstrap_low, bootstrap_high = _bootstrap_ci(differences)
            distinguishable = mean - half_width > 0 or mean + half_width < 0
            rows.append(
                {
                    "dataset": dataset,
                    "setting": setting,
                    "reference": "multihead",
                    "comparator": comparator,
                    "num_paired_seeds": len(seeds),
                    "paired_mean_difference_pp": mean,
                    "paired_sample_sd_pp": sample_sd,
                    "ci95_lower_pp": mean - half_width,
                    "ci95_upper_pp": mean + half_width,
                    "median_paired_difference_pp": float(
                        np.median(differences)
                    ),
                    "wins": int((differences > 0).sum()),
                    "ties": int((differences == 0).sum()),
                    "losses": int((differences < 0).sum()),
                    "cohens_dz": None
                    if sample_sd == 0
                    else mean / sample_sd,
                    "bootstrap_ci95_lower_pp": bootstrap_low,
                    "bootstrap_ci95_upper_pp": bootstrap_high,
                    "statistical_status": (
                        "statistically_distinguishable"
                        if distinguishable
                        else "statistically_uncertain"
                    ),
                    "practical_status": (
                        "practically_meaningful"
                        if abs(mean) >= meaningful_gain
                        else "below_minimum_meaningful_gain"
                    ),
                    "paired_seeds": ";".join(str(seed) for seed in seeds),
                    "paired_differences_pp": ";".join(
                        "{:.8f}".format(value) for value in differences
                    ),
                }
            )
    return rows


def ratio_diagnostic_rows(runs: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows = []
    for run in runs:
        raw = run.get("raw_ratio") or {}
        normalized = run.get("normalized_ratio") or {}
        validation = run.get("ratio_validation") or {}
        rows.append(
            {
                "dataset": run["dataset"],
                "setting": run["setting"],
                "method": run["method"],
                "seed": run["seed"],
                "raw_mean": raw.get("mean"),
                "raw_std": raw.get("std"),
                "lower_clipping_rate": raw.get("lower_clipping_rate"),
                "upper_clipping_rate": raw.get("upper_clipping_rate"),
                "raw_ess": raw.get("ess"),
                "normalized_mean": normalized.get("mean"),
                "normalized_std": normalized.get("std"),
                "normalized_ess": normalized.get("ess"),
                "classifier_used_weight_mean": run.get(
                    "classifier_used_weight_mean"
                ),
                "classifier_used_weight_std": run.get(
                    "classifier_used_weight_std"
                ),
                "classifier_used_weight_ess": run.get(
                    "classifier_used_weight_ess"
                ),
                "discriminator_validation_accuracy": validation.get(
                    "discriminator_accuracy"
                ),
                "target_accuracy": run.get("last10_target_accuracy"),
            }
        )
    return rows


def _flat_rows(runs: Sequence[Dict[str, Any]], kind: str) -> List[Dict[str, Any]]:
    rows = []
    for run in runs:
        base = {
            "dataset": run["dataset"],
            "setting": run["setting"],
            "method": run["method"],
            "seed": run["seed"],
        }
        if kind == "gate":
            value = run.get("gate_summary")
            if value:
                row = dict(base)
                row.update(value)
                rows.append(row)
        elif kind == "risk":
            row = dict(base)
            row.update(
                {
                    "final_target_accuracy": run.get(
                        "final_target_accuracy"
                    ),
                    "last10_target_accuracy": run.get(
                        "last10_target_accuracy"
                    ),
                    "final_target_ovr_risk": run.get(
                        "target_ovr_risk"
                    ),
                    "last10_target_ovr_risk": run.get(
                        "last10_target_ovr_risk"
                    ),
                    "minimum_target_risk": run.get(
                        "minimum_target_risk"
                    ),
                    "minimum_target_risk_epoch": run.get(
                        "minimum_target_risk_epoch"
                    ),
                    "best_target_accuracy": run.get(
                        "best_target_accuracy"
                    ),
                    "best_target_accuracy_epoch": run.get(
                        "best_target_accuracy_epoch"
                    ),
                    "epoch_risk_accuracy_pearson": run.get(
                        "risk_accuracy_pearson"
                    ),
                    "epoch_risk_accuracy_spearman": run.get(
                        "risk_accuracy_spearman"
                    ),
                }
            )
            rows.append(row)
        elif kind == "runtime":
            row = dict(base)
            row.update(
                {
                    "ratio_parameter_count": run.get(
                        "ratio_parameter_count"
                    ),
                    "classifier_parameter_count": run.get(
                        "classifier_parameter_count"
                    ),
                    "ratio_train_seconds": run.get(
                        "ratio_train_seconds"
                    ),
                    "classifier_train_seconds": run.get(
                        "classifier_train_seconds"
                    ),
                    "target_accuracy": run.get(
                        "last10_target_accuracy"
                    ),
                }
            )
            rows.append(row)
    return rows


def _setting_context(runs: Sequence[Dict[str, Any]]) -> str:
    datasets = sorted(set(str(run["dataset"]) for run in runs))
    settings = sorted(set(str(run["setting"]) for run in runs))
    return "{} | {} setting(s)".format(
        ",".join(datasets), len(settings)
    )


def _representative_run(
    runs: Sequence[Dict[str, Any]], method: Optional[str] = None
) -> Optional[Dict[str, Any]]:
    selected = [
        run for run in runs if method is None or run["method"] == method
    ]
    return sorted(
        selected,
        key=lambda item: (
            str(item["setting"]),
            int(item["seed"]),
        ),
    )[0] if selected else None


def _save_empty_plot(plt: Any, path: str, title: str, message: str) -> None:
    figure, axis = plt.subplots(figsize=(8, 4.5))
    axis.axis("off")
    axis.set_title(title)
    axis.text(0.5, 0.5, message, ha="center", va="center")
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


def generate_plots(
    output_dir: str,
    runs: Sequence[Dict[str, Any]],
    summary_rows: Sequence[Dict[str, Any]],
    paired_rows: Sequence[Dict[str, Any]],
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plot_dir = os.path.join(output_dir, "figures")
    if not os.path.isdir(plot_dir):
        os.makedirs(plot_dir)
    context = _setting_context(runs)

    labels = [
        "{}\n{}".format(row["method"], row["setting"].split("_")[1])
        for row in summary_rows
    ]
    values = [float(row["mean_target_accuracy"]) for row in summary_rows]
    figure, axis = plt.subplots(figsize=(max(8, len(labels) * 0.45), 5))
    axis.bar(np.arange(len(values)), values)
    axis.set_xticks(np.arange(len(labels)))
    axis.set_xticklabels(labels, rotation=70, ha="right", fontsize=7)
    axis.set_ylabel("Target accuracy (%)")
    axis.set_title("Target accuracy | {}".format(context))
    figure.tight_layout()
    figure.savefig(os.path.join(plot_dir, "01_target_accuracy.png"), dpi=150)
    plt.close(figure)

    if paired_rows:
        labels = [
            "{} | {}".format(row["comparator"], row["setting"].split("_")[1])
            for row in paired_rows
        ]
        means = np.asarray(
            [float(row["paired_mean_difference_pp"]) for row in paired_rows]
        )
        lower = np.asarray(
            [float(row["ci95_lower_pp"]) for row in paired_rows]
        )
        upper = np.asarray(
            [float(row["ci95_upper_pp"]) for row in paired_rows]
        )
        figure, axis = plt.subplots(
            figsize=(max(8, len(labels) * 0.45), 5)
        )
        axis.errorbar(
            np.arange(len(means)),
            means,
            yerr=np.vstack((means - lower, upper - means)),
            fmt="o",
            capsize=3,
        )
        axis.axhline(0.0, color="black", linewidth=1)
        axis.set_xticks(np.arange(len(labels)))
        axis.set_xticklabels(labels, rotation=70, ha="right", fontsize=7)
        axis.set_ylabel("Multihead paired difference (pp)")
        axis.set_title("Paired differences, 95% t CI | {}".format(context))
        figure.tight_layout()
        figure.savefig(
            os.path.join(plot_dir, "02_paired_difference.png"), dpi=150
        )
        plt.close(figure)
    else:
        _save_empty_plot(
            plt,
            os.path.join(plot_dir, "02_paired_difference.png"),
            "Paired differences | {}".format(context),
            "Need at least two paired seeds including multihead.",
        )

    def plot_saved_hist(filename: str, csv_name: str, title: str) -> None:
        representative = _representative_run(runs, "multihead")
        rows = [] if representative is None else read_csv(
            os.path.join(representative["_run_dir"], csv_name)
        )
        if not rows:
            _save_empty_plot(
                plt, os.path.join(plot_dir, filename), title, "No ratio rows."
            )
            return
        final_epoch = max(int(row["epoch"]) for row in rows)
        row = next(row for row in rows if int(row["epoch"]) == final_epoch)
        counts = np.asarray(
            [float(value) for value in row["histogram"].split(";")],
            dtype=np.float64,
        )
        low = float(row["histogram_min"])
        high = float(row["histogram_max"])
        centers = np.linspace(low, high, len(counts), endpoint=False)
        width = (high - low) / max(len(counts), 1)
        figure, axis = plt.subplots(figsize=(8, 4.5))
        axis.bar(centers, counts, width=width)
        axis.set_xlabel("Ratio")
        axis.set_ylabel("Count")
        axis.set_title(
            "{} | {} | seed {}".format(
                title,
                representative["setting"],
                representative["seed"],
            )
        )
        figure.tight_layout()
        figure.savefig(os.path.join(plot_dir, filename), dpi=150)
        plt.close(figure)

    plot_saved_hist(
        "03_raw_ratio_histogram.png",
        "ratio_raw_stats.csv",
        "Raw ratio histogram (representative branch)",
    )
    plot_saved_hist(
        "04_normalized_ratio_histogram.png",
        "ratio_normalized_stats.csv",
        "Normalized ratio histogram (representative branch)",
    )

    def method_scatter(
        filename: str, field: str, ylabel: str, title: str
    ) -> None:
        applicable = [
            run for run in runs if run.get(field) not in (None, "")
        ]
        if not applicable:
            _save_empty_plot(
                plt, os.path.join(plot_dir, filename), title, "No values."
            )
            return
        figure, axis = plt.subplots(figsize=(8, 4.5))
        axis.scatter(
            np.arange(len(applicable)),
            [float(run[field]) for run in applicable],
        )
        axis.set_xticks(np.arange(len(applicable)))
        axis.set_xticklabels(
            [str(run["method"]) for run in applicable],
            rotation=60,
            ha="right",
        )
        axis.set_ylabel(ylabel)
        axis.set_title("{} | {}".format(title, context))
        figure.tight_layout()
        figure.savefig(os.path.join(plot_dir, filename), dpi=150)
        plt.close(figure)

    for run in runs:
        raw = run.get("raw_ratio") or {}
        normalized = run.get("normalized_ratio") or {}
        run["_total_clipping_rate"] = (
            None
            if raw.get("lower_clipping_rate") is None
            else float(raw.get("lower_clipping_rate", 0.0))
            + float(raw.get("upper_clipping_rate", 0.0))
        )
        run["_normalized_ess"] = normalized.get("ess")
    method_scatter(
        "05_clipping_rate.png",
        "_total_clipping_rate",
        "Clipping rate",
        "Final ratio clipping rate",
    )
    method_scatter(
        "06_ess.png",
        "_normalized_ess",
        "ESS",
        "Final normalized effective sample size",
    )

    representative = _representative_run(runs, "multihead")
    discriminator_rows = [] if representative is None else read_csv(
        os.path.join(
            representative["_run_dir"], "discriminator_metrics.csv"
        )
    )
    if discriminator_rows:
        final_epoch = max(int(row["epoch"]) for row in discriminator_rows)
        row = next(
            row
            for row in discriminator_rows
            if int(row["epoch"]) == final_epoch
            and row["split"] == "validation"
        )
        source = np.asarray(
            [float(value) for value in row["source_probability_histogram"].split(";")]
        )
        target = np.asarray(
            [float(value) for value in row["target_probability_histogram"].split(";")]
        )
        x = np.arange(len(source))
        figure, axis = plt.subplots(figsize=(8, 4.5))
        axis.bar(x - 0.2, source, width=0.4, label="source")
        axis.bar(x + 0.2, target, width=0.4, label="target")
        axis.legend()
        axis.set_xlabel("Predicted target-probability bin")
        axis.set_title(
            "Discriminator probability histogram | {}".format(
                representative["setting"]
            )
        )
        figure.tight_layout()
        figure.savefig(
            os.path.join(
                plot_dir, "07_discriminator_probability_histogram.png"
            ),
            dpi=150,
        )
        plt.close(figure)
    else:
        _save_empty_plot(
            plt,
            os.path.join(
                plot_dir, "07_discriminator_probability_histogram.png"
            ),
            "Discriminator probability histogram",
            "No discriminator rows.",
        )

    epoch_rows = [] if representative is None else read_csv(
        os.path.join(representative["_run_dir"], "epoch_metrics.csv")
    )
    if epoch_rows:
        figure, left_axis = plt.subplots(figsize=(8, 4.5))
        epochs = [int(row["epoch"]) for row in epoch_rows]
        left_axis.plot(
            epochs,
            [float(row["target_accuracy"]) for row in epoch_rows],
            label="accuracy",
        )
        left_axis.set_ylabel("Target accuracy (%)")
        right_axis = left_axis.twinx()
        right_axis.plot(
            epochs,
            [float(row["target_ovr_risk"]) for row in epoch_rows],
            color="tab:red",
            label="OVR risk",
        )
        right_axis.set_ylabel("Target OVR risk")
        left_axis.set_xlabel("Classifier epoch")
        left_axis.set_title(
            "Risk-accuracy trajectory | {}".format(
                representative["setting"]
            )
        )
        figure.tight_layout()
        figure.savefig(
            os.path.join(plot_dir, "08_risk_accuracy_trajectory.png"),
            dpi=150,
        )
        plt.close(figure)

    fusion = _representative_run(runs, "fusion")
    gate_rows = [] if fusion is None else read_csv(
        os.path.join(fusion["_run_dir"], "gate_metrics.csv")
    )
    if gate_rows:
        final_epoch = max(int(row["epoch"]) for row in gate_rows)
        row = next(row for row in gate_rows if int(row["epoch"]) == final_epoch)
        counts = np.asarray(
            [float(value) for value in row["gate_histogram"].split(";")]
        )
        figure, axis = plt.subplots(figsize=(8, 4.5))
        axis.bar(np.arange(len(counts)), counts)
        axis.set_xlabel("Gate bin")
        axis.set_title("Fusion gate histogram | {}".format(fusion["setting"]))
        figure.tight_layout()
        figure.savefig(
            os.path.join(plot_dir, "09_fusion_gate_histogram.png"), dpi=150
        )
        plt.close(figure)

        final_rows = [
            row for row in gate_rows if int(row["epoch"]) == final_epoch
        ]
        figure, axis = plt.subplots(figsize=(8, 4.5))
        axis.bar(
            ["global", "conditional", "fused"],
            [
                np.mean(
                    [float(row["global_log_ratio_mean"]) for row in final_rows]
                ),
                np.mean(
                    [
                        float(row["conditional_log_ratio_mean"])
                        for row in final_rows
                    ]
                ),
                np.mean(
                    [float(row["fused_log_ratio_mean"]) for row in final_rows]
                ),
            ],
        )
        axis.set_ylabel("Mean log ratio")
        axis.set_title(
            "Global/conditional/fused ratios | {}".format(fusion["setting"])
        )
        figure.tight_layout()
        figure.savefig(
            os.path.join(
                plot_dir, "10_global_conditional_fused_ratio.png"
            ),
            dpi=150,
        )
        plt.close(figure)
    else:
        for filename, title in (
            ("09_fusion_gate_histogram.png", "Fusion gate histogram"),
            (
                "10_global_conditional_fused_ratio.png",
                "Global/conditional/fused ratio",
            ),
        ):
            _save_empty_plot(
                plt, os.path.join(plot_dir, filename), title, "No fusion run."
            )

    def extract_setting_number(setting: str, prefix: str) -> Optional[float]:
        pattern = (
            r"(?:^|_)rotation_(-?[0-9]+(?:\.[0-9]+)?)"
            if prefix == "rotation"
            else r"(?:^|_)tw([0-9]+(?:\.[0-9]+)?)"
        )
        match = re.search(pattern, setting)
        return None if match is None else float(match.group(1))

    def line_by_axis(
        filename: str, prefix: str, xlabel: str, title: str
    ) -> None:
        grouped = defaultdict(list)
        for row in summary_rows:
            value = extract_setting_number(str(row["setting"]), prefix)
            if value is not None:
                grouped[row["method"]].append(
                    (value, float(row["mean_target_accuracy"]))
                )
        figure, axis = plt.subplots(figsize=(8, 4.5))
        for method, points in sorted(grouped.items()):
            points = sorted(points)
            axis.plot(
                [point[0] for point in points],
                [point[1] for point in points],
                marker="o",
                label=method,
            )
        if grouped:
            axis.legend(fontsize=7)
        axis.set_xlabel(xlabel)
        axis.set_ylabel("Target accuracy (%)")
        axis.set_title("{} | {}".format(title, context))
        figure.tight_layout()
        figure.savefig(os.path.join(plot_dir, filename), dpi=150)
        plt.close(figure)

    line_by_axis(
        "11_accuracy_rotation_strength.png",
        "rotation",
        "Rotation (degrees)",
        "Accuracy versus rotation strength",
    )
    line_by_axis(
        "12_accuracy_target_weak_size.png",
        "tw",
        "Target weak sample size",
        "Accuracy versus target weak size",
    )

    applicable = [
        run
        for run in runs
        if run.get("_total_clipping_rate") is not None
        and run.get("last10_target_accuracy") is not None
    ]
    figure, axis = plt.subplots(figsize=(8, 4.5))
    for run in applicable:
        axis.scatter(
            float(run["_total_clipping_rate"]),
            float(run["last10_target_accuracy"]),
            label=run["method"],
        )
    axis.set_xlabel("Clipping rate")
    axis.set_ylabel("Target accuracy (%)")
    axis.set_title("Accuracy versus clipping rate | {}".format(context))
    figure.tight_layout()
    figure.savefig(
        os.path.join(plot_dir, "13_accuracy_clipping_rate.png"), dpi=150
    )
    plt.close(figure)

    applicable = [
        run
        for run in runs
        if run.get("classifier_used_weight_std") is not None
        and run.get("last10_target_accuracy") is not None
    ]
    figure, axis = plt.subplots(figsize=(8, 4.5))
    for run in applicable:
        axis.scatter(
            float(run["classifier_used_weight_std"]),
            float(run["last10_target_accuracy"]),
            label=run["method"],
        )
    axis.set_xlabel("Classifier-used weight SD")
    axis.set_ylabel("Target accuracy (%)")
    axis.set_title("Weight variance versus target accuracy | {}".format(context))
    figure.tight_layout()
    figure.savefig(
        os.path.join(plot_dir, "14_weight_variance_accuracy.png"), dpi=150
    )
    plt.close(figure)


def parse_args(
    argv: Optional[Sequence[str]] = None
) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", default="results/next_round")
    parser.add_argument("--output")
    parser.add_argument(
        "--minimum-meaningful-gain-pp", type=float, default=1.0
    )
    parser.add_argument("--skip-plots", action="store_true")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> Dict[str, Any]:
    args = parse_args(argv)
    output = os.path.abspath(args.output or os.path.join(args.results, "summary"))
    if not os.path.isdir(output):
        os.makedirs(output)
    runs, failed = collect_runs(args.results)
    summary_rows = summarize_by_setting(runs)
    paired_rows = paired_comparisons(
        runs, args.minimum_meaningful_gain_pp
    )
    ratio_rows = ratio_diagnostic_rows(runs)
    gate_rows = _flat_rows(runs, "gate")
    risk_rows = _flat_rows(runs, "risk")
    runtime_rows = _flat_rows(runs, "runtime")
    failed_rows = [
        {
            "dataset": run.get("dataset"),
            "setting": run.get("setting"),
            "method": run.get("method"),
            "seed": run.get("seed"),
            "failure_reason": run.get("failure_reason"),
            "last_stable_classifier_epoch": run.get(
                "last_stable_classifier_epoch"
            ),
            "run_dir": run.get("_run_dir"),
        }
        for run in failed
    ]
    write_csv(
        os.path.join(output, "summary_by_setting.csv"),
        summary_rows,
        (
            "dataset",
            "setting",
            "method",
            "num_seeds",
            "mean_target_accuracy",
            "population_sd_target_accuracy",
        ),
    )
    write_csv(
        os.path.join(output, "paired_comparisons.csv"),
        paired_rows,
        (
            "dataset",
            "setting",
            "reference",
            "comparator",
            "num_paired_seeds",
            "paired_mean_difference_pp",
            "paired_sample_sd_pp",
            "ci95_lower_pp",
            "ci95_upper_pp",
        ),
    )
    write_csv(
        os.path.join(output, "ratio_diagnostics.csv"),
        ratio_rows,
        ("dataset", "setting", "method", "seed"),
    )
    write_csv(
        os.path.join(output, "gate_summary.csv"),
        gate_rows,
        ("dataset", "setting", "method", "seed"),
    )
    write_csv(
        os.path.join(output, "risk_accuracy_analysis.csv"),
        risk_rows,
        ("dataset", "setting", "method", "seed"),
    )
    write_csv(
        os.path.join(output, "runtime_parameter_summary.csv"),
        runtime_rows,
        ("dataset", "setting", "method", "seed"),
    )
    write_csv(
        os.path.join(output, "failed_runs.csv"),
        failed_rows,
        (
            "dataset",
            "setting",
            "method",
            "seed",
            "failure_reason",
            "last_stable_classifier_epoch",
            "run_dir",
        ),
    )
    plots_generated = False
    plot_warning = None
    if not args.skip_plots and runs:
        try:
            generate_plots(output, runs, summary_rows, paired_rows)
            plots_generated = True
        except ImportError as error:
            plot_warning = (
                "Plots require matplotlib; CSV summaries are complete. "
                "Run this same command in an environment with matplotlib: {}"
            ).format(error)
            with open(
                os.path.join(output, "PLOTS_SKIPPED.txt"),
                "w",
                encoding="utf-8",
            ) as stream:
                stream.write(plot_warning + "\n")
    result = {
        "successful_runs": len(runs),
        "failed_runs": len(failed),
        "settings": len(set(_group_key(run) for run in runs)),
        "output": output,
        "plots_generated": plots_generated,
        "plot_warning": plot_warning,
    }
    print(json.dumps(result, sort_keys=True))
    return result


if __name__ == "__main__":
    main()
