"""Build a validated-data report artifact from the completed Wisteria runs."""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List


ARCHITECTURES = ("unified", "separate", "multihead", "fusion")
SEEDS = tuple(range(5))
T_CRITICAL_DF4_95 = 2.776445105


def mean(values: Iterable[float]) -> float:
    values = list(values)
    return sum(values) / len(values)


def std(values: Iterable[float], sample: bool = False) -> float:
    values = list(values)
    center = mean(values)
    denominator = len(values) - 1 if sample else len(values)
    return math.sqrt(sum((value - center) ** 2 for value in values) / denominator)


def load_runs(input_dir: Path) -> List[Dict[str, Any]]:
    paths = sorted(input_dir.glob("*_ratio_scar_seed*.json"))
    runs = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    expected = {(architecture, seed) for architecture in ARCHITECTURES for seed in SEEDS}
    observed = {(run["ratio_arch"], int(run["seed"])) for run in runs}
    if len(runs) != 20 or observed != expected:
        missing = sorted(expected - observed)
        extra = sorted(observed - expected)
        raise RuntimeError(
            "expected 20 unique architecture/seed runs; missing={}, extra={}".format(
                missing, extra
            )
        )
    settings = {
        (
            run["dataset"],
            run["shift"],
            run["target_rotation"],
            run["source_size"],
            run["target_weak_size"],
            run["ratio_epochs"],
            run["classifier_epochs"],
        )
        for run in runs
    }
    if len(settings) != 1:
        raise RuntimeError("runs do not share one experiment configuration")
    return runs


def final_weight_std(input_dir: Path, architecture: str, seed: int) -> float:
    path = input_dir / "{}_ratio_scar_seed{}_detail.csv".format(architecture, seed)
    with path.open("r", encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != 200:
        raise RuntimeError("{} has {} classifier epochs".format(path, len(rows)))
    return mean(float(row["weight_std"]) for row in rows[-10:])


def architecture_rows(
    runs: List[Dict[str, Any]], input_dir: Path
) -> List[Dict[str, Any]]:
    grouped = defaultdict(list)
    for run in runs:
        grouped[run["ratio_arch"]].append(run)
    rows = []
    for architecture in ARCHITECTURES:
        group = sorted(grouped[architecture], key=lambda run: int(run["seed"]))
        target_accuracy = [float(run["target_accuracy"]) for run in group]
        target_risk = [float(run["target_ovr_risk"]) for run in group]
        source_risk = [float(run["source_ovr_risk"]) for run in group]
        weight_std = [
            final_weight_std(input_dir, architecture, int(run["seed"]))
            for run in group
        ]
        validation = [run["ratio_training"]["validation"] for run in group]
        cache = [run["source_weight_cache"] for run in group]
        rows.append(
            {
                "architecture": architecture,
                "rank": 0,
                "seeds": 5,
                "target_accuracy_mean": mean(target_accuracy),
                "target_accuracy_population_std": std(target_accuracy),
                "target_accuracy_sample_std": std(target_accuracy, sample=True),
                "target_accuracy_min": min(target_accuracy),
                "target_accuracy_max": max(target_accuracy),
                "target_ovr_risk_mean": mean(target_risk),
                "target_ovr_risk_population_std": std(target_risk),
                "source_ovr_risk_mean": mean(source_risk),
                "source_ovr_risk_population_std": std(source_risk),
                "classifier_weight_std_last10_mean": mean(weight_std),
                "source_cache_weight_mean": mean(
                    float(item["mean"]) for item in cache
                ),
                "source_cache_weight_std": mean(float(item["std"]) for item in cache),
                "source_cache_weight_min": min(float(item["min"]) for item in cache),
                "source_cache_weight_max": max(float(item["max"]) for item in cache),
                "ratio_parameters": int(group[0]["ratio_parameter_count"]),
                "ratio_training_seconds_mean": mean(
                    float(run["ratio_training"]["seconds"]) for run in group
                ),
                "classifier_minutes_mean": mean(
                    float(run["classifier_seconds"]) / 60.0 for run in group
                ),
                "discriminator_accuracy_mean": mean(
                    float(item["discriminator_accuracy"]) for item in validation
                ),
                "ratio_validation_mean": mean(
                    float(item["ratio_mean"]) for item in validation
                ),
                "ratio_validation_min": min(
                    float(item["ratio_min"]) for item in validation
                ),
                "ratio_validation_max": max(
                    float(item["ratio_max"]) for item in validation
                ),
                "nan_or_inf_runs": sum(
                    bool(item["has_nan_or_inf"]) for item in validation
                ),
                "skipped_updates": sum(
                    int(value)
                    for item in validation
                    for branch in item["skipped_update_count"]
                    for value in branch
                ),
            }
        )
    for rank, row in enumerate(
        sorted(rows, key=lambda item: item["target_accuracy_mean"], reverse=True),
        start=1,
    ):
        row["rank"] = rank
    return sorted(rows, key=lambda item: item["rank"])


def seed_rows(runs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [
        {
            "architecture": run["ratio_arch"],
            "seed": int(run["seed"]),
            "target_accuracy": float(run["target_accuracy"]),
            "target_ovr_risk": float(run["target_ovr_risk"]),
            "source_ovr_risk": float(run["source_ovr_risk"]),
            "ratio_parameters": int(run["ratio_parameter_count"]),
            "source_accuracy_available": run.get("source_accuracy") is not None,
        }
        for run in sorted(runs, key=lambda item: (item["ratio_arch"], item["seed"]))
    ]


def paired_rows(seed_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    by_architecture = defaultdict(dict)
    for row in seed_data:
        by_architecture[row["architecture"]][row["seed"]] = row["target_accuracy"]
    rows = []
    for comparator in ("unified", "fusion", "separate"):
        differences = [
            by_architecture["multihead"][seed] - by_architecture[comparator][seed]
            for seed in SEEDS
        ]
        average = mean(differences)
        standard_error = std(differences, sample=True) / math.sqrt(len(differences))
        half_width = T_CRITICAL_DF4_95 * standard_error
        rows.append(
            {
                "comparison": "multihead - {}".format(comparator),
                "reference": "multihead",
                "comparator": comparator,
                "paired_mean_difference_pp": average,
                "ci95_lower_pp": average - half_width,
                "ci95_upper_pp": average + half_width,
                "ci_excludes_zero": average - half_width > 0
                or average + half_width < 0,
                "seeds": 5,
            }
        )
    return rows


SYNTHETIC_ABLATION = [
    {
        "setting": "mixed covariate + mild label shift",
        "method": "inverse-frequency branches",
        "log_ratio_rmse": 0.2153,
        "population_std": 0.0124,
        "rank": 1,
    },
    {
        "setting": "mixed covariate + mild label shift",
        "method": "conditioned / balanced",
        "log_ratio_rmse": 0.2230,
        "population_std": 0.0110,
        "rank": 3,
    },
    {
        "setting": "mixed covariate + mild label shift",
        "method": "multihead",
        "log_ratio_rmse": 0.2316,
        "population_std": 0.0120,
        "rank": 4,
    },
    {
        "setting": "mixed covariate + mild label shift",
        "method": "marginal ratio",
        "log_ratio_rmse": 0.2523,
        "population_std": 0.0112,
        "rank": 5,
    },
    {
        "setting": "mixed covariate + mild label shift",
        "method": "unit weights",
        "log_ratio_rmse": 1.1120,
        "population_std": 0.0082,
        "rank": 10,
    },
    {
        "setting": "heterogeneous class-specific shift",
        "method": "multihead",
        "log_ratio_rmse": 0.3125,
        "population_std": 0.0264,
        "rank": 1,
    },
    {
        "setting": "heterogeneous class-specific shift",
        "method": "conditioned / balanced",
        "log_ratio_rmse": 0.4184,
        "population_std": 0.0314,
        "rank": 3,
    },
    {
        "setting": "heterogeneous class-specific shift",
        "method": "marginal ratio",
        "log_ratio_rmse": 0.4287,
        "population_std": 0.0379,
        "rank": 5,
    },
    {
        "setting": "heterogeneous class-specific shift",
        "method": "unit weights",
        "log_ratio_rmse": 0.5750,
        "population_std": 0.0038,
        "rank": 9,
    },
]


HISTORICAL_BASELINES = [
    {"method": "Multihead-Ratio-SCAR", "target_accuracy_mean": 60.994, "std": 1.854},
    {"method": "ADIW-SCAR", "target_accuracy_mean": 60.978, "std": 2.438},
    {
        "method": "Conditioned-Ratio-SCAR",
        "target_accuracy_mean": 60.654,
        "std": 2.845,
    },
    {"method": "ftSCAR / pooled SCAR", "target_accuracy_mean": 60.309, "std": 1.678},
    {"method": "trainSCAR", "target_accuracy_mean": 54.298, "std": 0.858},
    {"method": "testSCAR", "target_accuracy_mean": 38.821, "std": 2.437},
]


def source_specs(generated_at: str) -> List[Dict[str, Any]]:
    packaged_sql = (
        "outputs/wisteria_full_results_20260725/report/report_sources.sql"
    )
    return [
        {
            "id": "current-full",
            "label": (
                "SQL snapshot derived from 20 Wisteria JSON files and "
                "20 classifier traces"
            ),
            "path": packaged_sql,
        },
        {
            "id": "ablation-report",
            "label": (
                "SQL snapshot of selected synthetic ablations and historical "
                "SCAR comparison from ABLATION_RESULTS.md"
            ),
            "path": packaged_sql,
        },
        {
            "id": "experiment-code",
            "label": "Experiment and risk implementation",
            "path": "experiments/scarce_ratio_comparison.py",
        },
    ]


def build_artifact(
    runs: List[Dict[str, Any]], input_dir: Path
) -> Dict[str, Any]:
    generated_at = dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat()
    architecture = architecture_rows(runs, input_dir)
    seeds = seed_rows(runs)
    paired = paired_rows(seeds)
    by_name = {row["architecture"]: row for row in architecture}
    paired_by_name = {row["comparator"]: row for row in paired}
    growth_efficiency = [
        {
            "reference": "multihead",
            "comparator": "separate",
            "parameter_reduction_rate": (
                1.0
                - by_name["multihead"]["ratio_parameters"]
                / by_name["separate"]["ratio_parameters"]
            ),
            "multihead_ratio_parameters": by_name["multihead"]["ratio_parameters"],
            "separate_ratio_parameters": by_name["separate"]["ratio_parameters"],
            "ratio_time_reduction_rate": (
                1.0
                - by_name["multihead"]["ratio_training_seconds_mean"]
                / by_name["separate"]["ratio_training_seconds_mean"]
            ),
            "multihead_ratio_seconds": by_name["multihead"][
                "ratio_training_seconds_mean"
            ],
            "separate_ratio_seconds": by_name["separate"][
                "ratio_training_seconds_mean"
            ],
        }
    ]
    sources = source_specs(generated_at)
    title = "Conditional Density Ratio 多分类 SCAR 实验总结"

    manifest = {
        "version": 1,
        "surface": "report",
        "title": title,
        "description": (
            "Wisteria MNIST rotation-30 comparison of unified, separate, "
            "multihead, and global-conditional fusion density-ratio estimators."
        ),
        "generatedAt": generated_at,
        "sources": sources,
        "cards": [
            {
                "id": "best-accuracy",
                "dataset": "architecture_summary",
                "sourceId": "current-full",
                "filter": {"architecture": "multihead"},
                "description": "Five-seed mean of the final ten target-test epochs.",
                "metrics": [
                    {
                        "label": "Multihead target accuracy (%)",
                        "field": "target_accuracy_mean",
                        "format": "number",
                    },
                    {
                        "label": "Population SD (pp)",
                        "field": "target_accuracy_population_std",
                        "format": "number",
                    },
                ],
            },
            {
                "id": "separate-gap",
                "dataset": "paired_comparisons",
                "sourceId": "current-full",
                "filter": {"comparator": "separate"},
                "description": "Seed-paired multihead minus separate accuracy.",
                "metrics": [
                    {
                        "label": "Mean paired gap (pp)",
                        "field": "paired_mean_difference_pp",
                        "format": "number",
                        "signed": True,
                    },
                    {
                        "label": "95% CI lower",
                        "field": "ci95_lower_pp",
                        "format": "number",
                        "signed": True,
                    },
                    {
                        "label": "95% CI upper",
                        "field": "ci95_upper_pp",
                        "format": "number",
                        "signed": True,
                    },
                ],
            },
            {
                "id": "completion",
                "dataset": "run_completion",
                "sourceId": "current-full",
                "description": "All planned architecture/seed combinations completed.",
                "metrics": [
                    {
                        "label": "Successful runs",
                        "field": "successful_runs",
                        "format": "number",
                    },
                    {
                        "label": "Failed runs",
                        "field": "failed_runs",
                        "format": "number",
                    },
                ],
            },
            {
                "id": "parameter-reduction",
                "dataset": "growth_efficiency",
                "sourceId": "current-full",
                "description": (
                    "Multihead ratio-estimator parameter reduction relative "
                    "to 20 independent separate branches."
                ),
                "metrics": [
                    {
                        "label": "Ratio parameter reduction",
                        "field": "parameter_reduction_rate",
                        "format": "percent",
                    },
                    {
                        "label": "Multihead parameters",
                        "field": "multihead_ratio_parameters",
                        "format": "compact",
                    },
                    {
                        "label": "Separate parameters",
                        "field": "separate_ratio_parameters",
                        "format": "compact",
                    },
                ],
            },
            {
                "id": "ratio-time-reduction",
                "dataset": "growth_efficiency",
                "sourceId": "current-full",
                "description": (
                    "Mean ratio-pretraining time reduction relative to "
                    "separate across the same five seeds."
                ),
                "metrics": [
                    {
                        "label": "Ratio train-time reduction",
                        "field": "ratio_time_reduction_rate",
                        "format": "percent",
                    },
                    {
                        "label": "Multihead seconds",
                        "field": "multihead_ratio_seconds",
                        "format": "number",
                    },
                    {
                        "label": "Separate seconds",
                        "field": "separate_ratio_seconds",
                        "format": "number",
                    },
                ],
            },
        ],
        "charts": [
            {
                "id": "target-accuracy",
                "title": "Target accuracy by ratio architecture",
                "subtitle": (
                    "MNIST rotation 30°, five seeds; bars show final-10-epoch "
                    "means in percentage points."
                ),
                "showDescription": True,
                "intent": "comparison",
                "question": (
                    "Which conditional-ratio architecture has the highest "
                    "five-seed target accuracy?"
                ),
                "rationale": (
                    "A sorted horizontal bar chart makes the four discrete "
                    "architecture means directly comparable without implying a trend."
                ),
                "comparisonContext": {
                    "grain": "ratio architecture",
                    "unit": "percentage points",
                    "denominator": "10,000 MNIST target-test examples per epoch",
                    "normalization": "mean of epochs 191-200, then mean over five seeds",
                    "baseline": "same data, classifier, optimizer, and epoch counts",
                },
                "type": "horizontalBar",
                "dataset": "architecture_summary",
                "sourceId": "current-full",
                "encodings": {
                    "x": {
                        "field": "architecture",
                        "type": "nominal",
                        "label": "Ratio architecture",
                    },
                    "y": {
                        "field": "target_accuracy_mean",
                        "type": "quantitative",
                        "label": "Target accuracy",
                        "unit": "%",
                    },
                    "tooltip": [
                        {"field": "target_accuracy_population_std", "label": "Population SD"},
                        {"field": "target_accuracy_min", "label": "Minimum seed"},
                        {"field": "target_accuracy_max", "label": "Maximum seed"},
                        {"field": "rank", "label": "Rank"},
                        {"field": "ratio_parameters", "label": "Ratio parameters"},
                    ],
                },
                "valueFormat": "number",
                "unit": "%",
                "layout": "full",
                "palette": {"kind": "sequential", "name": "blue"},
                "labels": {"values": "all"},
                "settings": {
                    "orientation": "horizontal",
                    "sort": "descending",
                    "showValues": True,
                    "categoryLabelPolicy": "wrap",
                },
                "surface": {"surface": "card", "viewMode": "both"},
            },
            {
                "id": "paired-accuracy-gains",
                "title": "Multihead paired target-accuracy differences",
                "subtitle": (
                    "Positive values favor multihead; seed-matched mean "
                    "differences in percentage points, n=5."
                ),
                "showDescription": True,
                "intent": "comparison",
                "question": (
                    "How much target accuracy does multihead gain over each "
                    "alternative architecture on matched seeds?"
                ),
                "rationale": (
                    "A zero-referenced horizontal delta bar makes effect "
                    "direction and magnitude explicit across the exhaustive "
                    "set of three alternative architectures."
                ),
                "comparisonContext": {
                    "grain": "seed-paired architecture comparison",
                    "unit": "percentage points",
                    "denominator": "five matched random seeds",
                    "normalization": "multihead accuracy minus comparator accuracy",
                    "baseline": "zero means no paired accuracy difference",
                },
                "type": "horizontalBar",
                "dataset": "paired_comparisons",
                "sourceId": "current-full",
                "encodings": {
                    "x": {
                        "field": "comparator",
                        "type": "nominal",
                        "label": "Comparator",
                    },
                    "y": {
                        "field": "paired_mean_difference_pp",
                        "type": "quantitative",
                        "label": "Multihead gain",
                        "unit": "pp",
                    },
                    "tooltip": [
                        {"field": "ci95_lower_pp", "label": "95% CI lower"},
                        {"field": "ci95_upper_pp", "label": "95% CI upper"},
                        {"field": "ci_excludes_zero", "label": "CI excludes zero"},
                        {"field": "seeds", "label": "Matched seeds"},
                    ],
                },
                "valueFormat": "number",
                "unit": "pp",
                "layout": "full",
                "palette": {"kind": "sequential", "name": "blue"},
                "labels": {"values": "all"},
                "referenceLines": [
                    {
                        "axis": "y",
                        "color": "neutral",
                        "label": "No difference",
                        "lineStyle": "solid",
                        "value": 0,
                    }
                ],
                "settings": {
                    "orientation": "horizontal",
                    "sort": "descending",
                    "showValues": True,
                    "categoryLabelPolicy": "wrap",
                },
                "surface": {"surface": "card", "viewMode": "both"},
            },
        ],
        "tables": [
            {
                "id": "architecture-table",
                "title": "Architecture-level metrics",
                "subtitle": (
                    "Five-seed aggregate; SD is population SD and weight spread "
                    "is averaged over the final ten classifier epochs."
                ),
                "showDescription": True,
                "dataset": "architecture_summary",
                "sourceId": "current-full",
                "layout": "full",
                "density": "spacious",
                "defaultSort": {
                    "field": "target_accuracy_mean",
                    "direction": "desc",
                },
                "columns": [
                    {"field": "rank", "label": "Rank", "format": "number"},
                    {"field": "architecture", "label": "Architecture", "type": "text"},
                    {
                        "field": "target_accuracy_mean",
                        "label": "Target accuracy (%)",
                        "format": "number",
                    },
                    {
                        "field": "target_accuracy_population_std",
                        "label": "Accuracy SD (pp)",
                        "format": "number",
                    },
                    {
                        "field": "target_ovr_risk_mean",
                        "label": "Target OVR risk",
                        "format": "number",
                    },
                    {
                        "field": "classifier_weight_std_last10_mean",
                        "label": "Final weight SD",
                        "format": "number",
                    },
                    {
                        "field": "ratio_parameters",
                        "label": "Ratio parameters",
                        "format": "compact",
                    },
                    {
                        "field": "ratio_training_seconds_mean",
                        "label": "Ratio train sec",
                        "format": "number",
                    },
                ],
            },
            {
                "id": "paired-table",
                "title": "Seed-paired accuracy differences",
                "subtitle": (
                    "Multihead minus comparator, percentage points; two-sided "
                    "95% t interval with df=4."
                ),
                "showDescription": True,
                "dataset": "paired_comparisons",
                "sourceId": "current-full",
                "layout": "full",
                "density": "spacious",
                "defaultSort": {
                    "field": "paired_mean_difference_pp",
                    "direction": "desc",
                },
                "columns": [
                    {"field": "comparison", "label": "Comparison", "type": "text"},
                    {
                        "field": "paired_mean_difference_pp",
                        "label": "Mean difference (pp)",
                        "format": "number",
                        "movement": True,
                    },
                    {
                        "field": "ci95_lower_pp",
                        "label": "95% CI lower",
                        "format": "number",
                    },
                    {
                        "field": "ci95_upper_pp",
                        "label": "95% CI upper",
                        "format": "number",
                    },
                    {
                        "field": "ci_excludes_zero",
                        "label": "CI excludes zero",
                        "type": "text",
                    },
                ],
            },
            {
                "id": "synthetic-table",
                "title": "Synthetic conditional-ratio ablations",
                "subtitle": (
                    "Selected trainable and reference methods from the prior "
                    "Wisteria ablation; lower log-ratio RMSE is better."
                ),
                "showDescription": True,
                "dataset": "synthetic_ablation",
                "sourceId": "ablation-report",
                "layout": "full",
                "density": "spacious",
                "defaultSort": {"field": "setting", "direction": "asc"},
                "columns": [
                    {"field": "setting", "label": "Shift setting", "type": "text"},
                    {"field": "method", "label": "Method", "type": "text"},
                    {
                        "field": "log_ratio_rmse",
                        "label": "Log-ratio RMSE",
                        "format": "number",
                    },
                    {
                        "field": "population_std",
                        "label": "Population SD",
                        "format": "number",
                    },
                ],
            },
            {
                "id": "historical-table",
                "title": "Previous matched SCAR comparison",
                "subtitle": (
                    "Historical Wisteria run, MNIST rotation 30°, five seeds; "
                    "shown as context rather than a pooled test with the new run."
                ),
                "showDescription": True,
                "dataset": "historical_baselines",
                "sourceId": "ablation-report",
                "layout": "full",
                "density": "spacious",
                "defaultSort": {
                    "field": "target_accuracy_mean",
                    "direction": "desc",
                },
                "columns": [
                    {"field": "method", "label": "Method", "type": "text"},
                    {
                        "field": "target_accuracy_mean",
                        "label": "Target accuracy (%)",
                        "format": "number",
                    },
                    {
                        "field": "std",
                        "label": "Population SD (pp)",
                        "format": "number",
                    },
                ],
            },
        ],
        "blocks": [
            {"id": "title", "type": "markdown", "body": "# " + title},
            {
                "id": "technical-summary",
                "type": "markdown",
                "body": (
                    "## 技术结论\n\n"
                    "**核心正式实验已完整跑通，但结论应定为“multihead 是当前首选，"
                    "尚未证明其稳定优于 unified/fusion”。** 20 个计划任务全部成功。"
                    "Multihead 的五种子 target accuracy 最高，为 "
                    "{:.3f} ± {:.3f}%，但相对 unified 的配对差值仅 "
                    "{:+.3f} pp，95% CI 跨过 0；相对 fusion 也同样不确定。"
                    "Separate 比 multihead 低 {:.3f} pp，配对区间不含 0，"
                    "并以约 9.30M 参数换来了明显更差的分类结果。\n\n"
                    "更重要的诊断是：domain discriminator 几乎完全可分，"
                    "shared 模型的 source ratio cache 几乎全部贴近 0.01 下界，"
                    "branch normalization 后接近单位权重。因此本轮更像是在比较"
                    "饱和估计器的稳定性，而不是证明精细 conditional ratio "
                    "带来了可靠的域适配增益。"
                ).format(
                    by_name["multihead"]["target_accuracy_mean"],
                    by_name["multihead"]["target_accuracy_population_std"],
                    paired_by_name["unified"]["paired_mean_difference_pp"],
                    paired_by_name["separate"]["paired_mean_difference_pp"],
                ),
                "sourceId": "current-full",
            },
            {
                "id": "headline-metrics",
                "type": "metric-strip",
                "cardIds": [
                    "best-accuracy",
                    "separate-gap",
                    "parameter-reduction",
                    "ratio-time-reduction",
                    "completion",
                ],
            },
            {
                "id": "core-growth",
                "type": "markdown",
                "body": (
                    "## 核心增长点集中在“相对 separate 的效果与效率”\n\n"
                    "**可确认的增长是：multihead 相对 separate 提升 "
                    "{:+.3f} pp target accuracy，同时减少 {:.1f}% ratio 参数，"
                    "ratio 预训练平均耗时下降 {:.1f}%。** 准确率差的 95% CI 为 "
                    "[{:+.3f}, {:+.3f}] pp，不跨 0。\n\n"
                    "相对 unified 和 fusion 的均值增长只有 {:+.3f} pp 和 "
                    "{:+.3f} pp，且置信区间跨 0，因此图中将它们表示为"
                    "“观察到的正增量”，而不是已确认优势。三条比较已经覆盖"
                    "当前正式实验的全部替代架构；没有通过增加不可比的历史"
                    "基线来人为扩充图表。"
                ).format(
                    paired_by_name["separate"]["paired_mean_difference_pp"],
                    growth_efficiency[0]["parameter_reduction_rate"] * 100.0,
                    growth_efficiency[0]["ratio_time_reduction_rate"] * 100.0,
                    paired_by_name["separate"]["ci95_lower_pp"],
                    paired_by_name["separate"]["ci95_upper_pp"],
                    paired_by_name["unified"]["paired_mean_difference_pp"],
                    paired_by_name["fusion"]["paired_mean_difference_pp"],
                ),
                "sourceId": "current-full",
            },
            {
                "id": "paired-growth-chart-block",
                "type": "chart",
                "chartId": "paired-accuracy-gains",
            },
            {
                "id": "accuracy-finding",
                "type": "markdown",
                "body": (
                    "## Multihead 数值第一，但与 unified/fusion 尚未拉开\n\n"
                    "Multihead、unified、fusion 的平均准确率分别为 "
                    "{:.3f}%、{:.3f}% 和 {:.3f}%，三者只相差 0.785 pp。"
                    "这三种共享表示架构的种子波动与均值差异处于同一量级，"
                    "所以不能仅按均值排序宣称 multihead 已显著胜出。"
                    "Separate 的 {:.3f}% 则明显落后，并且不同种子都没有超过 "
                    "multihead。"
                ).format(
                    by_name["multihead"]["target_accuracy_mean"],
                    by_name["unified"]["target_accuracy_mean"],
                    by_name["fusion"]["target_accuracy_mean"],
                    by_name["separate"]["target_accuracy_mean"],
                ),
                "sourceId": "current-full",
            },
            {
                "id": "accuracy-chart-block",
                "type": "chart",
                "chartId": "target-accuracy",
            },
            {
                "id": "accuracy-table-block",
                "type": "table",
                "tableId": "architecture-table",
            },
            {
                "id": "paired-finding",
                "type": "markdown",
                "body": (
                    "## 配对检验只清楚地区分了 separate\n\n"
                    "按相同 seed 配对后，multihead 相对 unified 的差值为 "
                    "{:+.3f} pp（95% CI [{:+.3f}, {:+.3f}]），相对 fusion "
                    "为 {:+.3f} pp（[{:+.3f}, {:+.3f}]）；两者均跨过 0。"
                    "相对 separate 的差值为 {:+.3f} pp"
                    "（[{:+.3f}, {:+.3f}]），区间不含 0。样本只有五个 seed，"
                    "该区间用于表达不确定性，不应被当作充分的最终显著性证据。"
                ).format(
                    paired_by_name["unified"]["paired_mean_difference_pp"],
                    paired_by_name["unified"]["ci95_lower_pp"],
                    paired_by_name["unified"]["ci95_upper_pp"],
                    paired_by_name["fusion"]["paired_mean_difference_pp"],
                    paired_by_name["fusion"]["ci95_lower_pp"],
                    paired_by_name["fusion"]["ci95_upper_pp"],
                    paired_by_name["separate"]["paired_mean_difference_pp"],
                    paired_by_name["separate"]["ci95_lower_pp"],
                    paired_by_name["separate"]["ci95_upper_pp"],
                ),
                "sourceId": "current-full",
            },
            {"id": "paired-table-block", "type": "table", "tableId": "paired-table"},
            {
                "id": "ratio-diagnosis",
                "type": "markdown",
                "body": (
                    "## Ratio 饱和削弱了 shared 架构之间的可辨识差异\n\n"
                    "Unified、multihead、fusion 的 discriminator accuracy "
                    "接近 1.0，validation ratio 覆盖 0.01–20 裁剪边界。"
                    "三者的 source cache weight 均值约为 0.0100；经 classifier "
                    "阶段的 conditional-branch self-normalization 后，最后十轮"
                    "权重标准差分别仅 {:.4f}、{:.4f}、{:.4f}。这使 shared "
                    "模型实际接近单位加权。\n\n"
                    "Separate 的最后十轮权重标准差达到 {:.3f}，同时参数量约为 "
                    "{:.2f}M（shared 模型约 0.47M）。它保留了更强的权重变化，"
                    "但该变化没有转化为更高准确率，反而表现出高方差 ratio "
                    "对分类器的潜在伤害。"
                ).format(
                    by_name["unified"]["classifier_weight_std_last10_mean"],
                    by_name["multihead"]["classifier_weight_std_last10_mean"],
                    by_name["fusion"]["classifier_weight_std_last10_mean"],
                    by_name["separate"]["classifier_weight_std_last10_mean"],
                    by_name["separate"]["ratio_parameters"] / 1_000_000.0,
                ),
                "sourceId": "current-full",
            },
            {
                "id": "risk-interpretation",
                "type": "markdown",
                "body": (
                    "## 最低 OVR risk 并未对应最高 accuracy\n\n"
                    "Separate 的平均 target OVR risk 最低（{:.4f}），但 target "
                    "accuracy 也是最低；multihead 的 accuracy 最高，target OVR "
                    "risk 却最高（{:.4f}）。因此当前实现中的最终 epoch OVR risk "
                    "不能被直接当作架构排名代理，应把 target accuracy 作为主要"
                    "分类指标，并单独检查风险估计、校准和训练轨迹。"
                ).format(
                    by_name["separate"]["target_ovr_risk_mean"],
                    by_name["multihead"]["target_ovr_risk_mean"],
                ),
                "sourceId": "current-full",
            },
            {
                "id": "scope",
                "type": "markdown",
                "body": (
                    "## 比较范围与指标定义\n\n"
                    "- 数据：MNIST，source 59,000、target weak-label 1,000，"
                    "target test 10,000。\n"
                    "- Shift：目标图像旋转 30°；普通标签只用于最终评估。\n"
                    "- 设计：四种 ratio architecture × seeds 0–4；ratio 10 epochs，"
                    "classifier 200 epochs。\n"
                    "- Target accuracy：每个 run 的 classifier epochs 191–200 "
                    "测试准确率均值，再对五个 seeds 汇总。\n"
                    "- Accuracy SD：报告采用五个 seeds 的 population SD；配对 "
                    "95% CI 使用 sample SD 和 df=4 的 t 区间。\n"
                    "- Risk：source/target OVR risk 为第 200 epoch 输出，"
                    "不是跨最后十轮平均。"
                ),
                "sourceId": "experiment-code",
            },
            {
                "id": "method",
                "type": "markdown",
                "body": (
                    "## 架构差异与训练方法\n\n"
                    "**Unified** 保留原有共享条件估计器；**separate** 为每个 "
                    "(k,b) 使用独立模型；**multihead** 共享 ratio encoder，"
                    "保留 2q 个标量 head；**fusion** 同时学习 global 与 "
                    "conditional log-ratio，并由可学习 gate 在 log space 融合。"
                    "所有模型都只用 x、complementary label 和 domain 构造条件"
                    "分支；source classifier risk 对 y_bar[k]=b 严格选取 r[k,b]，"
                    "随后按 conditional branch clip 并 self-normalize。"
                ),
                "sourceId": "experiment-code",
            },
            {
                "id": "synthetic-finding",
                "type": "markdown",
                "body": (
                    "## 合成实验支持“按 shift 选择共享方式”，而非单一架构通吃\n\n"
                    "在 mixed covariate + mild label shift 下，conditioned/balanced "
                    "log-ratio RMSE 为 0.2230，multihead 为 0.2316，均远优于"
                    "单位权重 1.1120，但 inverse-frequency/no-balancing 略好，"
                    "说明 branch balancing 不是在所有场景都有净收益。"
                    "在 heterogeneous class-specific shift 下，multihead RMSE "
                    "0.3125，明显优于 conditioned 0.4184 和 marginal 0.4287；"
                    "此时独立输出 head 的灵活性更有价值。"
                ),
                "sourceId": "ablation-report",
            },
            {
                "id": "synthetic-table-block",
                "type": "table",
                "tableId": "synthetic-table",
            },
            {
                "id": "historical-context",
                "type": "markdown",
                "body": (
                    "## 旧基线说明：ratio 方法没有证实稳定超过 ADIW/pooled SCAR\n\n"
                    "此前匹配的 Wisteria 比较中，Multihead-Ratio-SCAR 为 "
                    "60.994 ± 1.854%，ADIW-SCAR 为 60.978 ± 2.438%，"
                    "ftSCAR/pooled 为 60.309 ± 1.678%；配对检验均未显示可靠的"
                    "小数点级改进。最新 multihead 均值更高，但本轮 ratio clip、"
                    "归一化与实现版本不同，未同步重跑 ADIW/pooled，因此不能把"
                    "两轮均值直接合并为新的优势证据。"
                ),
                "sourceId": "ablation-report",
            },
            {
                "id": "historical-table-block",
                "type": "table",
                "tableId": "historical-table",
            },
            {
                "id": "limitations",
                "type": "markdown",
                "body": (
                    "## 结论可信度：可分享，但必须带上限制\n\n"
                    "1. **只有五个 seeds。** shared 架构之间低于 1 pp 的差异仍"
                    "高度不确定。\n"
                    "2. **Support separation 很强。** discriminator 接近 100%，"
                    "ratio 触及裁剪上下界，削弱对真实 conditional-ratio 质量的"
                    "判断。\n"
                    "3. **缺少当前图像实验的 oracle ratio。** 无法计算真实 "
                    "ratio MSE、log-ratio MSE 或 Spearman correlation。\n"
                    "4. **记录不完整。** 本轮 JSON 未持久化 source accuracy；"
                    "fusion gate 统计虽有模型接口和单元测试，但未写入该桥接实验"
                    "结果。\n"
                    "5. **只有 MNIST rotation-30。** 尚不能外推到 FashionMNIST、"
                    "CIFAR-10、input-output-relation shift 或 support shift。"
                ),
            },
            {
                "id": "next-steps",
                "type": "markdown",
                "body": (
                    "## 下一轮优先级\n\n"
                    "1. 在同一代码版本和 clip/normalization 配置下重跑 "
                    "no-IW、ftSCAR/pooled、ADIW 与四种 ratio 架构，形成真正匹配"
                    "的七方法表。\n"
                    "2. 将 source accuracy、fusion gate mean/std/histogram、"
                    "global/conditional/fused ratio 差异写入每个 run 的 JSON。\n"
                    "3. 增加 15–30 个 seeds，或至少为 shared 架构差异预先定义"
                    "最小有意义差异并做 power analysis。\n"
                    "4. 降低 domain 可分性或扫 ratio clip、rotation strength、"
                    "target weak size，确认结论不是裁剪饱和的产物。\n"
                    "5. 扩展到 FashionMNIST/CIFAR-10 与两类仓库 shift；"
                    "multihead 可作为默认候选，unified/fusion 作为近似同档备选，"
                    "separate 暂不作为主方法。"
                ),
            },
            {
                "id": "further-questions",
                "type": "markdown",
                "body": (
                    "## 仍需回答的问题\n\n"
                    "- Fusion gate 在饱和场景中究竟偏向 global 还是 conditional？\n"
                    "- 解除或放宽裁剪后，multihead 的小幅优势是否保留，还是由"
                    "当前归一化造成？\n"
                    "- Separate 的性能下降主要来自参数量、优化困难，还是"
                    "高方差 importance weights？\n"
                    "- Target OVR risk 与 accuracy 排名相反，是风险估计校准问题，"
                    "还是 accuracy 对当前 objective 的非单调响应？"
                ),
            },
        ],
    }
    snapshot = {
        "version": 1,
        "generatedAt": generated_at,
        "status": "ready",
        "datasets": {
            "architecture_summary": architecture,
            "seed_results": seeds,
            "paired_comparisons": paired,
            "growth_efficiency": growth_efficiency,
            "run_completion": [
                {
                    "successful_runs": 20,
                    "failed_runs": 0,
                    "expected_runs": 20,
                    "unique_architecture_seed_pairs": 20,
                }
            ],
            "synthetic_ablation": SYNTHETIC_ABLATION,
            "historical_baselines": HISTORICAL_BASELINES,
        },
        "accessIssues": [],
    }
    return {
        "surface": "report",
        "manifest": manifest,
        "snapshot": snapshot,
        "sources": sources,
        "package_info": {
            "analysis_script": "scripts/build_wisteria_experiment_report.py",
            "chart_map": [
                {
                    "section": "Multihead 数值第一，但与 unified/fusion 尚未拉开",
                    "question": "Which architecture has the highest target accuracy?",
                    "family": "Comparison & Ranking",
                    "type": "horizontalBar",
                    "fields": ["architecture", "target_accuracy_mean"],
                    "takeaway": "Multihead ranks first; shared architectures remain close.",
                    "palette_policy": "single-root preferred",
                },
                {
                    "section": "核心增长点集中在相对 separate 的效果与效率",
                    "question": (
                        "How much target accuracy does multihead gain over "
                        "each alternative architecture?"
                    ),
                    "family": "Uncertainty & Benchmark",
                    "type": "horizontalBar",
                    "fields": ["comparator", "paired_mean_difference_pp"],
                    "takeaway": (
                        "Only the gain over separate has a 95% interval "
                        "that excludes zero."
                    ),
                    "palette_policy": "single-root preferred with zero reference",
                    "data_sufficiency_note": (
                        "Three rows are intentional and exhaustive: unified, "
                        "fusion, and separate are all alternative architectures "
                        "in the current matched experiment."
                    ),
                }
            ],
            "omitted_metrics": [
                "source_accuracy: source_evaluation was absent in the SCARCE bridge",
                "fusion gate statistics: not persisted by scarce_ratio_comparison",
                "oracle ratio metrics: unavailable for this image-domain experiment",
            ],
        },
    }


def write_source_notes(path: Path) -> None:
    path.write_text(
        """# Report source notes

- Audience: technical.
- Delivery mode: portable HTML because the desktop MCP artifact renderer was
  unavailable in this run.
- Primary source: 20 Wisteria result JSON files and 20 classifier detail CSV
  files under `outputs/wisteria_full_results_20260725/`.
- Supporting source: `ABLATION_RESULTS.md`.
- Chart map: one sorted horizontal bar chart of architecture-level target
  accuracy, plus one exhaustive three-row paired-difference chart for
  multihead versus every alternative architecture. Historical baselines are
  excluded from the growth chart because they come from a different run.
- Population SD is retained for consistency with the existing repository
  report. Paired confidence intervals use sample SD and a two-sided t interval.
- Omitted metrics: source accuracy, current-run fusion gate statistics, and
  oracle image-domain ratio metrics were not persisted or available.
""",
        encoding="utf-8",
    )


def sql_literal(value: Any) -> str:
    if value is None:
        return "NULL"
    if isinstance(value, bool):
        return "TRUE" if value else "FALSE"
    if isinstance(value, (int, float)):
        if not math.isfinite(float(value)):
            raise ValueError("non-finite values cannot be packaged as SQL")
        return repr(value)
    return "'" + str(value).replace("'", "''") + "'"


def sql_select(dataset: str, rows: List[Dict[str, Any]]) -> str:
    columns = list(rows[0])
    selects = []
    for row in rows:
        values = [
            "{} AS {}".format(sql_literal(row.get(column)), column)
            for column in columns
        ]
        selects.append("SELECT " + ", ".join(values))
    return (
        "-- Dataset: {dataset}\n"
        "-- Materialized from the reviewed rows saved in artifact.json.\n"
        "{query};\n"
    ).format(dataset=dataset, query="\nUNION ALL\n".join(selects))


def write_source_sql(path: Path, artifact: Dict[str, Any]) -> None:
    datasets = artifact["snapshot"]["datasets"]
    sections = [
        "-- Reproducible SQL representation of the bounded report snapshot.",
        "-- Primary raw inputs: outputs/wisteria_full_results_20260725/*_seed*.json",
        "-- and outputs/wisteria_full_results_20260725/*_detail.csv.",
        "-- Historical inputs: ABLATION_RESULTS.md.",
        "",
    ]
    for dataset in (
        "architecture_summary",
        "paired_comparisons",
        "growth_efficiency",
        "run_completion",
        "synthetic_ablation",
        "historical_baselines",
    ):
        sections.append(sql_select(dataset, datasets[dataset]))
    path.write_text("\n".join(sections), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-dir",
        default="outputs/wisteria_full_results_20260725",
        type=Path,
    )
    parser.add_argument(
        "--artifact",
        default=(
            "outputs/wisteria_full_results_20260725/report/artifact.json"
        ),
        type=Path,
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    runs = load_runs(args.input_dir)
    artifact = build_artifact(runs, args.input_dir)
    args.artifact.parent.mkdir(parents=True, exist_ok=True)
    args.artifact.write_text(
        json.dumps(artifact, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    write_source_sql(args.artifact.parent / "report_sources.sql", artifact)
    write_source_notes(args.artifact.parent / "source_notes.md")
    print(
        "wrote {} with {} completed runs".format(
            args.artifact, artifact["snapshot"]["datasets"]["run_completion"][0]["successful_runs"]
        )
    )


if __name__ == "__main__":
    main()
