#!/usr/bin/env python3
"""Validate and aggregate the paired IWPU versus traditional-2step grid."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, stdev
from typing import Any, Iterable, Mapping, Sequence

from scipy import stats


PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))

from iwpu.config import load_config


METHODS = ("iwpu", "two_step")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Paired significance/equivalence analysis for IWPU and 2step."
    )
    parser.add_argument("--config", type=Path, default=PROJECT / "configs" / "paper.yaml")
    parser.add_argument("--result-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--equivalence-margin",
        type=float,
        default=0.01,
        help="Two-one-sided-test margin in absolute accuracy (default: 0.01).",
    )
    return parser


def result_path(
    root: Path, task: str, method: str, target_pos: int, target_unl: int, seed: int
) -> Path:
    return root / task / method / f"p{target_pos}_u{target_unl}_seed{seed}.json"


def load_accuracy(path: Path) -> tuple[float, Mapping[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if payload.get("status") != "complete":
        raise ValueError(f"{path}: status is {payload.get('status')!r}, not complete")
    value = payload.get("test_accuracy")
    if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise ValueError(f"{path}: invalid test_accuracy {value!r}")
    return float(value), payload


def paired_rows(config: Mapping[str, Any], root: Path) -> list[dict[str, Any]]:
    tasks = [
        task
        for task, definition in config["tasks"].items()
        if definition.get("paper_scope") == "main"
    ]
    sizes = [
        (int(item["positive"]), int(item["unlabeled"]))
        for item in config["data_protocol"]["target_train_sizes"]
    ]
    seeds = [int(seed) for seed in config["repetitions"]["seeds"]]
    rows: list[dict[str, Any]] = []
    for task in tasks:
        for target_pos, target_unl in sizes:
            for seed in seeds:
                scores: dict[str, float] = {}
                payloads: dict[str, Mapping[str, Any]] = {}
                for method in METHODS:
                    scores[method], payloads[method] = load_accuracy(
                        result_path(root, task, method, target_pos, target_unl, seed)
                    )
                    if payloads[method].get("config_method") != method:
                        raise ValueError(
                            f"method mismatch in {task}/{method}/p{target_pos}_u{target_unl}_seed{seed}"
                        )
                rows.append(
                    {
                        "task": task,
                        "target_pos": target_pos,
                        "target_unl": target_unl,
                        "seed": seed,
                        "iwpu": scores["iwpu"],
                        "two_step": scores["two_step"],
                        "difference_iwpu_minus_two_step": (
                            scores["iwpu"] - scores["two_step"]
                        ),
                    }
                )
    expected = len(tasks) * len(sizes) * len(seeds)
    if len(rows) != expected:
        raise RuntimeError(f"expected {expected} pairs, constructed {len(rows)}")
    return rows


def _tost(differences: Sequence[float], margin: float) -> tuple[float, float, bool]:
    if len(differences) < 2:
        return math.nan, math.nan, False
    average = mean(differences)
    standard_error = stdev(differences) / math.sqrt(len(differences))
    if standard_error == 0:
        lower_p = 0.0 if average > -margin else 1.0
        upper_p = 0.0 if average < margin else 1.0
    else:
        degrees = len(differences) - 1
        lower_t = (average + margin) / standard_error
        upper_t = (average - margin) / standard_error
        lower_p = float(stats.t.sf(lower_t, degrees))
        upper_p = float(stats.t.cdf(upper_t, degrees))
    return lower_p, upper_p, max(lower_p, upper_p) < 0.05


def summarize(
    label: str, rows: Sequence[Mapping[str, Any]], margin: float
) -> dict[str, Any]:
    iwpu = [float(row["iwpu"]) for row in rows]
    two_step = [float(row["two_step"]) for row in rows]
    differences = [float(row["difference_iwpu_minus_two_step"]) for row in rows]
    n = len(differences)
    diff_mean = mean(differences)
    diff_sd = stdev(differences) if n > 1 else 0.0
    diff_se = diff_sd / math.sqrt(n) if n else math.nan
    critical = float(stats.t.ppf(0.975, n - 1)) if n > 1 else math.nan
    paired = stats.ttest_rel(iwpu, two_step) if n > 1 else None
    lower_p, upper_p, equivalent = _tost(differences, margin)
    return {
        "group": label,
        "n_pairs": n,
        "iwpu_mean": mean(iwpu),
        "two_step_mean": mean(two_step),
        "mean_difference_iwpu_minus_two_step": diff_mean,
        "difference_sd": diff_sd,
        "difference_ci95_low": diff_mean - critical * diff_se if n > 1 else math.nan,
        "difference_ci95_high": diff_mean + critical * diff_se if n > 1 else math.nan,
        "paired_t_pvalue": float(paired.pvalue) if paired is not None else math.nan,
        "cohen_dz": diff_mean / diff_sd if diff_sd > 0 else math.nan,
        "iwpu_wins": sum(value > 0 for value in differences),
        "ties": sum(value == 0 for value in differences),
        "two_step_wins": sum(value < 0 for value in differences),
        "equivalence_margin": margin,
        "tost_lower_pvalue": lower_p,
        "tost_upper_pvalue": upper_p,
        "equivalent_at_5pct": equivalent,
    }


def group_summaries(rows: Sequence[Mapping[str, Any]], margin: float) -> list[dict[str, Any]]:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    grouped["ALL"] = list(rows)
    for row in rows:
        task = str(row["task"])
        grouped[task].append(row)
        grouped[f"{task}/p{row['target_pos']}_u{row['target_unl']}"].append(row)
    ordered = ["ALL"]
    ordered.extend(sorted(key for key in grouped if "/" not in key and key != "ALL"))
    ordered.extend(sorted(key for key in grouped if "/" in key))
    return [summarize(key, grouped[key], margin) for key in ordered]


def write_csv(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    materialized = list(rows)
    if not materialized:
        raise ValueError("cannot write an empty CSV")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(materialized[0]))
        writer.writeheader()
        writer.writerows(materialized)


def markdown_report(summaries: Sequence[Mapping[str, Any]], pair_count: int) -> str:
    task_rows = [row for row in summaries if "/" not in str(row["group"])]
    lines = [
        "# IWPU 与传统 two-step 配对比较",
        "",
        f"完整性：{pair_count} 个配对单元、{pair_count * 2} 个完成结果。",
        "差值定义为 `IWPU - two_step`；等价性检验使用双单侧 TOST。",
        "",
        "| 任务 | n | IWPU | two-step | 差值 | 95% CI | paired p | ±margin 等价 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in task_rows:
        lines.append(
            "| {group} | {n_pairs} | {iwpu_mean:.4f} | {two_step_mean:.4f} | "
            "{mean_difference_iwpu_minus_two_step:+.4f} | "
            "[{difference_ci95_low:+.4f}, {difference_ci95_high:+.4f}] | "
            "{paired_t_pvalue:.3g} | {equivalent_at_5pct} |".format(**row)
        )
    return "\n".join(lines) + "\n"


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not 0 < args.equivalence_margin < 1:
        raise SystemExit("--equivalence-margin must lie in (0, 1)")
    config = load_config(args.config)
    rows = paired_rows(config, args.result_root)
    summaries = group_summaries(rows, args.equivalence_margin)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "paired_cells.csv", rows)
    write_csv(args.output_dir / "paired_summary.csv", summaries)
    payload = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "result_root": str(args.result_root.resolve()),
        "expected_result_cells": len(rows) * 2,
        "complete_result_cells": len(rows) * 2,
        "pair_count": len(rows),
        "methods": list(METHODS),
        "equivalence_margin": args.equivalence_margin,
        "summaries": summaries,
    }
    (args.output_dir / "integrity_and_statistics.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (args.output_dir / "report.md").write_text(
        markdown_report(summaries, len(rows)), encoding="utf-8"
    )
    print(json.dumps(payload["summaries"][0], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
