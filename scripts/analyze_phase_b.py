"""Validate Phase B runs, select B2 configs, and build reports/figures."""

import argparse
import csv
import glob
import json
import math
import os
import shutil
import sys
from collections import defaultdict
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

REPOSITORY_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPOSITORY_ROOT not in sys.path:
    sys.path.insert(0, REPOSITORY_ROOT)

from src.risk_recovery_metrics import paired_interval, risk_recovery_summary
from src.synthetic_gaussian import (
    estimated_ratios,
    fit_domain_estimators,
    generate_gaussian_shift,
    oracle_branch_ratios,
)


EXPECTED_COUNTS = {
    "B0": 60,
    "B1": 20,
    "B2_grid": 45,
    "B2_calibration": 18,
    "B3": 400,
    "B4": 100,
}


def _json_load(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as stream:
        return json.load(stream)


def _json_dump(path: str, value: Any) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)


def _write_csv(path: str, rows: Sequence[Dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with open(path, "w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _read_csv(path: str) -> List[Dict[str, str]]:
    if not os.path.isfile(path):
        return []
    with open(path, "r", encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def collect_runs(root: str) -> List[Dict[str, Any]]:
    runs = []
    for path in glob.glob(
        os.path.join(root, "**", "summary.json"), recursive=True
    ):
        value = _json_load(path)
        if value.get("experiment") in ("B0", "B1", "B2", "B3", "B4"):
            value["_path"] = path
            value["_run_dir"] = os.path.dirname(path)
            runs.append(value)
    return runs


def _nonfinite_paths(value: Any, prefix: str = "") -> List[str]:
    output = []
    if isinstance(value, dict):
        for key, item in value.items():
            if key.startswith("_"):
                continue
            output.extend(
                _nonfinite_paths(
                    item, "{}.{}".format(prefix, key) if prefix else key
                )
            )
    elif isinstance(value, list):
        for index, item in enumerate(value):
            output.extend(
                _nonfinite_paths(item, "{}[{}]".format(prefix, index))
            )
    elif isinstance(value, float) and not math.isfinite(value):
        output.append(prefix)
    return output


def validate_runs(runs: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    duplicates = defaultdict(list)
    for run in runs:
        key = (
            run.get("experiment"),
            run.get("method"),
            run.get("shift_type"),
            run.get("seed"),
            run.get("lower_clip"),
            run.get("normalization"),
            run.get("calibration"),
        )
        duplicates[key].append(run["_path"])
    duplicate_rows = {
        str(key): values
        for key, values in duplicates.items()
        if len(values) > 1
    }
    paired_failures = []
    groups = defaultdict(list)
    for run in runs:
        experiment = run.get("experiment")
        if experiment == "B0":
            group = (experiment, run.get("seed"))
        elif experiment == "B2":
            group = (
                experiment,
                run.get("seed"),
                run.get("lower_clip"),
                run.get("normalization"),
                run.get("calibration"),
            )
        elif experiment == "B4":
            group = (
                experiment,
                run.get("shift_type"),
                run.get("seed"),
            )
        else:
            continue
        groups[group].append(run)
    for key, values in groups.items():
        split_hashes = {
            value.get("split_hash") for value in values
        }
        if len(split_hashes) != 1:
            paired_failures.append(
                {"group": list(key), "split_hashes": list(split_hashes)}
            )
    nonfinite = {
        run["_path"]: _nonfinite_paths(run)
        for run in runs
        if _nonfinite_paths(run)
    }
    figures = glob.glob(
        os.path.join(
            os.path.dirname(os.path.dirname(runs[0]["_path"]))
            if runs
            else "",
            "**",
            "*.png",
        ),
        recursive=True,
    )
    return {
        "run_count": len(runs),
        "status_failures": [
            run["_path"] for run in runs if run.get("status") != "ok"
        ],
        "target_test_selection_violations": [
            run["_path"]
            for run in runs
            if run.get("target_test_used_for_selection") is not False
        ],
        "duplicate_runs": duplicate_rows,
        "paired_hash_failures": paired_failures,
        "nonfinite_metrics": nonfinite,
        "figure_count": len(figures),
    }


def select_b2_configs(
    runs: Sequence[Dict[str, Any]], b2_root: str
) -> Dict[str, Any]:
    grid = [
        run
        for run in runs
        if run.get("experiment") == "B2"
        and run.get("calibration") == "none"
    ]
    grouped = defaultdict(list)
    for run in grid:
        grouped[
            (float(run["lower_clip"]), str(run["normalization"]))
        ].append(run)
    rankings = []
    for (lower, normalization), values in grouped.items():
        if len(values) < 3:
            continue
        rankings.append(
            {
                "lower_clip": lower,
                "normalization": normalization,
                "num_seeds": len(values),
                "mean_weighted_MMD": float(
                    np.mean([value["weighted_MMD"] for value in values])
                ),
                "mean_auc_distance_from_half": float(
                    np.mean(
                        [
                            abs(value["posthoc_domain_AUC"] - 0.5)
                            for value in values
                        ]
                    )
                ),
                "mean_normalized_ESS": float(
                    np.mean([value["normalized_ESS"] for value in values])
                ),
                "selection_used_target_accuracy": False,
            }
        )
    rankings.sort(
        key=lambda item: (
            item["mean_auc_distance_from_half"],
            item["mean_weighted_MMD"],
            -item["mean_normalized_ESS"],
        )
    )
    selected = [
        {
            "lower_clip": item["lower_clip"],
            "normalization": item["normalization"],
        }
        for item in rankings[:2]
    ]
    output = {
        "selection_rule": (
            "lexicographic: post-hoc AUC distance to 0.5, weighted MMD, "
            "then negative normalized ESS; target class labels/accuracy excluded"
        ),
        "selected_configs": selected,
        "rankings": rankings,
        "target_test_used_for_selection": False,
    }
    _json_dump(os.path.join(b2_root, "selected_configs.json"), output)
    return output


def _mean_rows(
    runs: Sequence[Dict[str, Any]],
    group_fields: Sequence[str],
    metric_fields: Sequence[str],
) -> List[Dict[str, Any]]:
    grouped = defaultdict(list)
    for run in runs:
        grouped[tuple(run.get(field) for field in group_fields)].append(run)
    rows = []
    for key, values in sorted(grouped.items(), key=lambda item: str(item[0])):
        row = dict(zip(group_fields, key))
        row["num_runs"] = len(values)
        for metric in metric_fields:
            present = [
                float(value[metric])
                for value in values
                if value.get(metric) is not None
            ]
            row["mean_{}".format(metric)] = (
                float(np.mean(present)) if present else None
            )
        rows.append(row)
    return rows


def _paired_method_rows(
    runs: Sequence[Dict[str, Any]],
    reference: str,
    comparators: Sequence[str],
    setting_field: str = "shift_type",
) -> List[Dict[str, Any]]:
    by_setting = defaultdict(lambda: defaultdict(dict))
    for run in runs:
        by_setting[run.get(setting_field)][run["method"]][
            int(run["seed"])
        ] = run
    rows = []
    for setting, methods in sorted(by_setting.items()):
        if reference not in methods:
            continue
        for comparator in comparators:
            if comparator not in methods:
                continue
            seeds = sorted(
                set(methods[reference]) & set(methods[comparator])
            )
            if not seeds:
                continue
            differences = [
                methods[reference][seed]["last10_target_accuracy"]
                - methods[comparator][seed]["last10_target_accuracy"]
                for seed in seeds
            ]
            interval = paired_interval(differences)
            rows.append(
                {
                    "setting": setting,
                    "reference": reference,
                    "comparator": comparator,
                    "seeds": ";".join(str(seed) for seed in seeds),
                    **interval,
                }
            )
    return rows


def checkpoint_risk_recovery(
    runs: Sequence[Dict[str, Any]], experiment: str
) -> List[Dict[str, Any]]:
    selected_runs = [
        run for run in runs if run.get("experiment") == experiment
    ]
    output = []
    for estimator in ("estimated_weak_risk", "estimated_iw_risk"):
        truth = []
        estimate = []
        for run in selected_runs:
            for row in _read_csv(
                os.path.join(run["_run_dir"], "epoch_metrics.csv")
            ):
                if row.get("true_target_risk") and row.get(estimator):
                    truth.append(float(row["true_target_risk"]))
                    estimate.append(float(row[estimator]))
        if truth:
            output.append(
                {
                    "experiment": experiment,
                    "estimator": estimator,
                    **risk_recovery_summary(truth, estimate),
                }
            )
    return output


def b0_report(
    runs: Sequence[Dict[str, Any]], report_path: str
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    b0 = [run for run in runs if run.get("experiment") == "B0"]
    comparisons = [
        ("adiw_original", "adiw_shuffle"),
        ("adiw_original", "adiw_mean"),
        ("adiw_original", "adiw_ones"),
        ("adiw_ones", "current_no_iw"),
        ("matched_no_iw", "current_no_iw"),
    ]
    by_method = defaultdict(dict)
    for run in b0:
        by_method[run["method"]][int(run["seed"])] = run
    paired_rows = []
    for reference, comparator in comparisons:
        seeds = sorted(
            set(by_method[reference]) & set(by_method[comparator])
        )
        if not seeds:
            continue
        interval = paired_interval(
            [
                by_method[reference][seed]["last10_target_accuracy"]
                - by_method[comparator][seed]["last10_target_accuracy"]
                for seed in seeds
            ]
        )
        paired_rows.append(
            {
                "reference": reference,
                "comparator": comparator,
                "seeds": ";".join(str(seed) for seed in seeds),
                **interval,
            }
        )
    summaries = _mean_rows(
        b0,
        ["method"],
        [
            "last10_target_accuracy",
            "used_weight_std",
            "normalized_ESS",
            "weighted_MMD",
            "posthoc_domain_AUC",
        ],
    )
    recovery_rows = checkpoint_risk_recovery(b0, "B0")
    lines = [
        "# Phase B0: ADIW causal decomposition",
        "",
        "Primary metric: mean target accuracy over the last 10 classifier "
        "epochs. Target test labels were not used for selection.",
        "",
        "## Method means",
        "",
        "| Method | n | Accuracy (%) | Used-weight SD | normalized ESS | weighted MMD | post-hoc AUC |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summaries:
        lines.append(
            "| {method} | {num_runs} | {mean_last10_target_accuracy:.3f} | "
            "{mean_used_weight_std:.4f} | {mean_normalized_ESS:.4f} | "
            "{mean_weighted_MMD:.4f} | {mean_posthoc_domain_AUC:.4f} |".format(
                **row
            )
        )
    lines.extend(
        [
            "",
            "## Paired causal contrasts",
            "",
            "| Contrast | mean pp | t 95% CI | bootstrap 95% CI |",
            "|---|---:|---:|---:|",
        ]
    )
    for row in paired_rows:
        lines.append(
            "| {reference} − {comparator} | {mean_difference:.3f} | "
            "[{t_ci95_lower:.3f}, {t_ci95_upper:.3f}] | "
            "[{bootstrap_ci95_lower:.3f}, {bootstrap_ci95_upper:.3f}] |".format(
                **row
            )
        )
    lines.extend(
        [
            "",
            "Results are causal only for the preregistered within-seed "
            "interventions. Fewer than 10 paired seeds is a smoke/provisional "
            "result.",
        ]
    )
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as stream:
        stream.write("\n".join(lines) + "\n")
    return paired_rows, recovery_rows


def b1_report(runs: Sequence[Dict[str, Any]], report_path: str) -> None:
    b1 = [run for run in runs if run.get("experiment") == "B1"]
    rows = _mean_rows(
        b1,
        ["method"],
        [
            "weighted_mmd",
            "posthoc_domain_auc",
            "posthoc_domain_accuracy",
        ],
    )
    lines = [
        "# Phase B1: ratio correctness audit",
        "",
        "Domain convention: source=0, target=1; neural ratios use "
        "`D/(1-D) × pi_source/pi_target`. ADIW uses KMM and has no "
        "discriminator probability/odds.",
        "",
        "| Method | n | weighted MMD | post-hoc AUC | post-hoc accuracy |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {method} | {num_runs} | {mean_weighted_mmd:.4f} | "
            "{mean_posthoc_domain_auc:.4f} | "
            "{mean_posthoc_domain_accuracy:.4f} |".format(**row)
        )
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as stream:
        stream.write("\n".join(lines) + "\n")


def b2_report(
    runs: Sequence[Dict[str, Any]],
    report_path: str,
    selected: Optional[Dict[str, Any]],
) -> None:
    b2 = [run for run in runs if run.get("experiment") == "B2"]
    rows = _mean_rows(
        b2,
        ["lower_clip", "normalization", "calibration"],
        [
            "last10_target_accuracy",
            "raw_lower_clip_rate",
            "normalized_ESS",
            "used_weight_CV",
            "weighted_MMD",
            "posthoc_domain_AUC",
        ],
    )
    lines = [
        "# Phase B2: frozen-score post-processing ablation",
        "",
        "Each seed trains one ratio estimator; all classifier configurations "
        "reuse its frozen logits and the same classifier/batch seeds.",
        "",
    ]
    if selected:
        lines.append(
            "Calibration candidates were selected without target class labels: "
            "`{}`.".format(selected["selected_configs"])
        )
        lines.append("")
    lines.extend(
        [
            "| lower | normalization | calibration | n | accuracy | lower clip rate | nESS | CV | MMD | AUC |",
            "|---:|---|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in rows:
        lines.append(
            "| {lower_clip:g} | {normalization} | {calibration} | "
            "{num_runs} | {mean_last10_target_accuracy:.3f} | "
            "{mean_raw_lower_clip_rate:.4f} | {mean_normalized_ESS:.4f} | "
            "{mean_used_weight_CV:.4f} | {mean_weighted_MMD:.4f} | "
            "{mean_posthoc_domain_AUC:.4f} |".format(**row)
        )
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as stream:
        stream.write("\n".join(lines) + "\n")


def b3_report(
    runs: Sequence[Dict[str, Any]], report_path: str
) -> Dict[str, Any]:
    b3 = [run for run in runs if run.get("experiment") == "B3"]
    rows = _mean_rows(
        b3,
        ["shift_type", "method"],
        [
            "last10_target_accuracy",
            "ratio_mse",
            "log_ratio_mse",
            "ratio_oracle_correlation",
            "weighted_MMD",
            "true_target_risk",
            "excess_risk_relative_to_oracle",
        ],
    )
    lines = [
        "# Phase B3: Gaussian ground-truth benchmark",
        "",
        "Oracle methods are diagnostic upper bounds and are not formal weakly "
        "supervised methods.",
        "",
        "| Shift | Method | n | Accuracy | ratio MSE | log-ratio MSE | corr. | MMD | excess risk |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {shift_type} | {method} | {num_runs} | "
            "{mean_last10_target_accuracy:.3f} | {mean_ratio_mse:.4f} | "
            "{mean_log_ratio_mse:.4f} | "
            "{mean_ratio_oracle_correlation:.3f} | "
            "{mean_weighted_MMD:.4f} | "
            "{mean_excess_risk_relative_to_oracle:.4f} |".format(**row)
        )
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as stream:
        stream.write("\n".join(lines) + "\n")
    hetero = {
        row["method"]: row
        for row in rows
        if row["shift_type"] == "heterogeneous_conditional_shift"
    }
    estimated_delta = None
    oracle_delta = None
    if (
        "conditional_estimated_ratio" in hetero
        and "global_estimated_ratio" in hetero
    ):
        estimated_delta = (
            hetero["conditional_estimated_ratio"][
                "mean_last10_target_accuracy"
            ]
            - hetero["global_estimated_ratio"][
                "mean_last10_target_accuracy"
            ]
        )
    if (
        "oracle_conditional_ratio" in hetero
        and "oracle_global_ratio" in hetero
    ):
        oracle_delta = (
            hetero["oracle_conditional_ratio"][
                "mean_last10_target_accuracy"
            ]
            - hetero["oracle_global_ratio"]["mean_last10_target_accuracy"]
        )
    decision = {
        "run_B4": bool(
            (estimated_delta is not None and estimated_delta > 1.0)
            or (oracle_delta is not None and oracle_delta > 1.0)
        ),
        "estimated_conditional_minus_global_pp": estimated_delta,
        "oracle_conditional_minus_global_pp": oracle_delta,
        "rule": (
            "run B4 if heterogeneous Gaussian conditional estimated or "
            "oracle accuracy exceeds the corresponding global method by >1 pp"
        ),
    }
    return decision


def b4_report(
    runs: Sequence[Dict[str, Any]], report_path: str
) -> List[Dict[str, Any]]:
    b4 = [run for run in runs if run.get("experiment") == "B4"]
    paired = []
    for conditional in ("unified", "multihead"):
        paired.extend(
            _paired_method_rows(
                b4,
                reference=conditional,
                comparators=["adiw_original"],
            )
        )
    lines = [
        "# Phase B4: class-heterogeneous MNIST shifts",
        "",
        "A practical conditional advantage requires paired mean >1 pp and "
        "the 95% t-CI lower bound >0 in a heterogeneous setting.",
        "",
        "| Setting | Conditional | mean Δ pp | t 95% CI | bootstrap 95% CI | criterion met |",
        "|---|---|---:|---:|---:|---|",
    ]
    for row in paired:
        heterogeneous = str(row["setting"]).startswith(("H2", "H3", "H4"))
        criterion = (
            heterogeneous
            and row["mean_difference"] > 1.0
            and row["t_ci95_lower"] > 0.0
        )
        lines.append(
            "| {setting} | {reference} | {mean_difference:.3f} | "
            "[{t_ci95_lower:.3f}, {t_ci95_upper:.3f}] | "
            "[{bootstrap_ci95_lower:.3f}, {bootstrap_ci95_upper:.3f}] | "
            "{criterion} |".format(criterion=criterion, **row)
        )
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as stream:
        stream.write("\n".join(lines) + "\n")
    return paired


def generate_figures(
    runs: Sequence[Dict[str, Any]], figure_dir: str
) -> List[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(figure_dir, exist_ok=True)
    paths = []
    b0 = [run for run in runs if run.get("experiment") == "B0"]
    if b0:
        rows = _mean_rows(
            b0, ["method"], ["last10_target_accuracy"]
        )
        plt.figure(figsize=(8, 4))
        plt.bar(
            [row["method"] for row in rows],
            [row["mean_last10_target_accuracy"] for row in rows],
        )
        plt.xticks(rotation=30, ha="right")
        plt.ylabel("Last-10 target accuracy (%)")
        plt.tight_layout()
        path = os.path.join(figure_dir, "b0_accuracy.png")
        plt.savefig(path, dpi=150)
        plt.close()
        paths.append(path)
    b2 = [
        run
        for run in runs
        if run.get("experiment") == "B2"
        and run.get("calibration") == "none"
    ]
    if b2:
        rows = _mean_rows(
            b2,
            ["lower_clip", "normalization"],
            ["last10_target_accuracy", "normalized_ESS"],
        )
        plt.figure(figsize=(7, 4))
        for normalization in sorted(
            {str(row["normalization"]) for row in rows}
        ):
            selected = [
                row for row in rows if row["normalization"] == normalization
            ]
            selected.sort(key=lambda row: row["lower_clip"])
            plt.plot(
                [row["lower_clip"] for row in selected],
                [row["mean_last10_target_accuracy"] for row in selected],
                marker="o",
                label=normalization,
            )
        plt.xscale("symlog", linthresh=1.0e-6)
        plt.xlabel("Lower clip")
        plt.ylabel("Target accuracy (%)")
        plt.legend()
        plt.tight_layout()
        path = os.path.join(figure_dir, "b2_grid_accuracy.png")
        plt.savefig(path, dpi=150)
        plt.close()
        paths.append(path)
    b3 = [run for run in runs if run.get("experiment") == "B3"]
    if b3:
        rows = _mean_rows(
            b3,
            ["shift_type", "method"],
            ["last10_target_accuracy", "ratio_mse"],
        )
        plt.figure(figsize=(10, 5))
        methods = sorted({row["method"] for row in rows})
        settings = sorted({row["shift_type"] for row in rows})
        x = np.arange(len(settings))
        width = 0.8 / max(len(methods), 1)
        for index, method in enumerate(methods):
            values = {
                row["shift_type"]: row["mean_last10_target_accuracy"]
                for row in rows
                if row["method"] == method
            }
            plt.bar(
                x + index * width,
                [values.get(setting, 0.0) for setting in settings],
                width=width,
                label=method,
            )
        plt.xticks(
            x + width * (len(methods) - 1) / 2,
            settings,
            rotation=20,
            ha="right",
        )
        plt.ylabel("Accuracy (%)")
        plt.legend(fontsize=7)
        plt.tight_layout()
        path = os.path.join(figure_dir, "b3_accuracy_across_shifts.png")
        plt.savefig(path, dpi=150)
        plt.close()
        paths.append(path)

        data = generate_gaussian_shift(
            "heterogeneous_conditional_shift",
            seed=0,
            source_size=800,
            target_weak_size=800,
            target_test_size=800,
        )
        source = data["source"]
        target = data["target_weak"]
        plt.figure(figsize=(6, 5))
        plt.scatter(
            source["x"][:, 0],
            source["x"][:, 1],
            s=5,
            alpha=0.35,
            label="source",
        )
        plt.scatter(
            target["x"][:, 0],
            target["x"][:, 1],
            s=5,
            alpha=0.35,
            label="target",
        )
        plt.legend()
        plt.title("Heterogeneous Gaussian source/target")
        plt.tight_layout()
        path = os.path.join(figure_dir, "b3_gaussian_contours.png")
        plt.savefig(path, dpi=150)
        plt.close()
        paths.append(path)

        oracle = oracle_branch_ratios(
            source["x"], source["bar_y"], data["parameters"]
        )
        plt.figure(figsize=(6, 5))
        plt.scatter(
            source["x"][:, 0],
            source["x"][:, 1],
            c=oracle.mean(dim=1),
            s=9,
            cmap="viridis",
        )
        plt.colorbar(label="Oracle conditional ratio")
        plt.tight_layout()
        path = os.path.join(figure_dir, "b3_oracle_ratio_heatmap.png")
        plt.savefig(path, dpi=150)
        plt.close()
        paths.append(path)

        plt.figure(figsize=(6, 5))
        weights = oracle.mean(dim=1).numpy()
        plt.scatter(
            source["x"][:, 0],
            source["x"][:, 1],
            s=np.clip(weights * 8, 2, 40),
            alpha=0.35,
            label="weighted source",
        )
        plt.scatter(
            target["x"][:, 0],
            target["x"][:, 1],
            s=5,
            alpha=0.3,
            label="target",
        )
        plt.legend()
        plt.tight_layout()
        path = os.path.join(
            figure_dir, "b3_weighted_source_vs_target.png"
        )
        plt.savefig(path, dpi=150)
        plt.close()
        paths.append(path)

        estimated = [
            run
            for run in b3
            if run["method"] == "conditional_estimated_ratio"
            and run["shift_type"] == "heterogeneous_conditional_shift"
        ]
        if estimated:
            plt.figure(figsize=(6, 5))
            plt.scatter(
                [run["ratio_mse"] for run in estimated],
                [run["last10_target_accuracy"] for run in estimated],
            )
            plt.xlabel("Conditional ratio MSE")
            plt.ylabel("Target accuracy (%)")
            plt.tight_layout()
            path = os.path.join(
                figure_dir, "b3_ratio_error_vs_accuracy.png"
            )
            plt.savefig(path, dpi=150)
            plt.close()
            paths.append(path)
    for path in paths:
        image = plt.imread(path)
        if image.size == 0:
            raise ValueError("generated figure is empty: {}".format(path))
    return paths


def _copy_report(path: str, output_report_dir: str) -> None:
    os.makedirs(output_report_dir, exist_ok=True)
    shutil.copy2(path, os.path.join(output_report_dir, os.path.basename(path)))


def parse_args(
    argv: Optional[Sequence[str]] = None,
) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-root", default="outputs/phase_b/runs")
    parser.add_argument("--output-root", default="outputs/phase_b")
    parser.add_argument("--reports-root", default="reports")
    parser.add_argument("--select-b2", action="store_true")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> Dict[str, Any]:
    args = parse_args(argv)
    runs = collect_runs(args.runs_root)
    validation = validate_runs(runs)
    _json_dump(
        os.path.join(args.output_root, "csv", "validation.json"),
        validation,
    )
    flat = [
        {key: value for key, value in run.items() if not key.startswith("_")}
        for run in runs
    ]
    _write_csv(os.path.join(args.output_root, "csv", "all_runs.csv"), flat)
    b2_root = os.path.join(args.runs_root, "B2")
    selected = None
    if args.select_b2 or os.path.isfile(
        os.path.join(b2_root, "selected_configs.json")
    ):
        selected = select_b2_configs(runs, b2_root)
    report_paths = {
        "B0": os.path.join(
            args.reports_root, "phase_b0_adiw_causal_decomposition.md"
        ),
        "B1": os.path.join(args.reports_root, "phase_b1_ratio_audit.md"),
        "B2": os.path.join(
            args.reports_root, "phase_b2_postprocessing_ablation.md"
        ),
        "B3": os.path.join(
            args.reports_root, "phase_b3_gaussian_ground_truth.md"
        ),
        "B4": os.path.join(
            args.reports_root, "phase_b4_heterogeneous_mnist.md"
        ),
    }
    paired_b0, risk_rows = b0_report(runs, report_paths["B0"])
    b1_report(runs, report_paths["B1"])
    b2_report(runs, report_paths["B2"], selected)
    decision = b3_report(runs, report_paths["B3"])
    paired_b4 = b4_report(runs, report_paths["B4"])
    _json_dump(
        os.path.join(args.output_root, "b4_decision.json"), decision
    )
    risk_rows.extend(checkpoint_risk_recovery(runs, "B3"))
    risk_rows.extend(checkpoint_risk_recovery(runs, "B4"))
    _write_csv(
        os.path.join(args.output_root, "csv", "paired_comparisons.csv"),
        paired_b0 + paired_b4,
    )
    _write_csv(
        os.path.join(args.output_root, "csv", "risk_recovery.csv"),
        risk_rows,
    )
    figure_paths = generate_figures(
        runs, os.path.join(args.output_root, "figures")
    )
    validation["figure_count"] = len(figure_paths)
    validation["figures_opened_successfully"] = True
    _json_dump(
        os.path.join(args.output_root, "csv", "validation.json"),
        validation,
    )
    for path in report_paths.values():
        _copy_report(path, os.path.join(args.output_root, "reports"))
    complete = (
        len([run for run in runs if run.get("experiment") == "B0"]) >= 60
        and len([run for run in runs if run.get("experiment") == "B3"]) >= 400
    )
    summary_path = os.path.join(args.reports_root, "phase_b_summary.md")
    lines = [
        "# Next-Round Phase B summary",
        "",
        "**Evidence status:** {}.".format(
            "formal matrix complete"
            if complete
            else "provisional; formal matrices incomplete"
        ),
        "",
        "- B0 causal decomposition: see `phase_b0_adiw_causal_decomposition.md`.",
        "- B1 ratio correctness: see `phase_b1_ratio_audit.md`.",
        "- B2 frozen post-processing: see `phase_b2_postprocessing_ablation.md`.",
        "- B3 ground-truth benchmark: see `phase_b3_gaussian_ground_truth.md`.",
        "- B4 decision: `{}`; rule and deltas are in `b4_decision.json`.".format(
            decision["run_B4"]
        ),
        "",
        "Conclusions are classified in the stage reports as data-supported, "
        "statistically uncertain, oracle-only, or not yet identifiable. "
        "Target test labels were used only for offline evaluation.",
    ]
    os.makedirs(args.reports_root, exist_ok=True)
    with open(summary_path, "w", encoding="utf-8") as stream:
        stream.write("\n".join(lines) + "\n")
    _copy_report(summary_path, os.path.join(args.output_root, "reports"))
    return {
        "runs": len(runs),
        "validation": validation,
        "b4_decision": decision,
        "figures": figure_paths,
        "reports": list(report_paths.values()) + [summary_path],
    }


if __name__ == "__main__":
    main()
