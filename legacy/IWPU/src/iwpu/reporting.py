from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


PAPER_METHOD_ORDER = [
    "ours",
    "tepu",
    "trpu",
    "ftpu",
    "mtpu",
    "mtspu",
    "dapu",
    "udapu",
    "giw",
]
ABLATION_METHOD_ORDER = [
    "two_step",
    "without_importance_weight_correction",
    "without_classifier_loss_correction",
    "without_both_corrections",
]

PAPER_DATASET_NAMES = {
    "fashion_mnist": "fmnist",
}
PAPER_SHIFT_NAMES = {
    "input_output_relation": "io",
}
MAIN_TABLE_DATASETS = {"mnist", "fmnist", "cifar10", "diabetes"}


def normalize_paper_keys(frame: pd.DataFrame) -> pd.DataFrame:
    """Return a copy whose dataset/shift labels match the paper CSVs."""

    normalized = frame.copy()
    if "dataset" in normalized:
        normalized["dataset"] = normalized["dataset"].replace(PAPER_DATASET_NAMES)
    if "shift" in normalized:
        normalized["shift"] = normalized["shift"].replace(PAPER_SHIFT_NAMES)
    return normalized


def read_results(root: str | Path) -> pd.DataFrame:
    records = []
    for path in sorted(Path(root).rglob("*.json")):
        with path.open("r", encoding="utf-8") as handle:
            record = json.load(handle)
        if record.get("status") == "complete" and "test_accuracy" in record:
            record["result_path"] = str(path)
            records.append(record)
    if not records:
        raise ValueError(f"No completed result JSON files found below {root}")
    return pd.json_normalize(records)


def aggregate_main_table(frame: pd.DataFrame) -> pd.DataFrame:
    """Match Table 1/5: average over target sizes and ten random seeds."""
    required = {"dataset", "shift", "method", "test_accuracy", "seed", "target_pos"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Result records are missing columns: {sorted(missing)}")
    main = normalize_paper_keys(frame)
    main = main.loc[
        main["dataset"].isin(MAIN_TABLE_DATASETS)
        & main["method"].isin(PAPER_METHOD_ORDER)
    ]
    if main.empty:
        raise ValueError("No completed main-table task results were supplied")
    grouped = (
        main.groupby(["dataset", "shift", "method"], dropna=False)["test_accuracy"]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    grouped["method"] = pd.Categorical(
        grouped["method"], PAPER_METHOD_ORDER, ordered=True
    )
    return grouped.sort_values(["dataset", "shift", "method"])


def aggregate_foodstamp_table(frame: pd.DataFrame) -> pd.DataFrame:
    """Match appendix Table 7: aggregate the FoodStamp task by method."""

    required = {"dataset", "method", "test_accuracy"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Result records are missing columns: {sorted(missing)}")
    appendix = normalize_paper_keys(frame)
    appendix = appendix.loc[
        (appendix["dataset"] == "foodstamp")
        & appendix["method"].isin(PAPER_METHOD_ORDER)
    ]
    if appendix.empty:
        return pd.DataFrame(columns=["method", "mean", "std", "count"])
    grouped = (
        appendix.groupby("method", dropna=False)["test_accuracy"]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    grouped["method"] = pd.Categorical(
        grouped["method"], PAPER_METHOD_ORDER, ordered=True
    )
    return grouped.sort_values("method")


def aggregate_ablation_table(frame: pd.DataFrame) -> pd.DataFrame:
    """Aggregate Table 2 ablations over target sizes and random seeds."""

    required = {"dataset", "shift", "method", "test_accuracy"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Result records are missing columns: {sorted(missing)}")
    ablations = normalize_paper_keys(frame)
    ablations = ablations.loc[
        ablations["dataset"].isin(MAIN_TABLE_DATASETS)
        & ablations["method"].isin(ABLATION_METHOD_ORDER)
    ]
    if ablations.empty:
        return pd.DataFrame(
            columns=["dataset", "shift", "method", "mean", "std", "count"]
        )
    grouped = (
        ablations.groupby(["dataset", "shift", "method"])["test_accuracy"]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    grouped["method"] = pd.Categorical(
        grouped["method"], ABLATION_METHOD_ORDER, ordered=True
    )
    return grouped.sort_values(["dataset", "shift", "method"])


def completeness_table(
    frame: pd.DataFrame,
    expected_seeds: int = 10,
    expected_target_sizes: int = 3,
) -> pd.DataFrame:
    expected = expected_seeds * expected_target_sizes
    normalized = normalize_paper_keys(frame)
    counts = (
        normalized.groupby(["dataset", "shift", "method"])
        .size()
        .rename("completed")
        .reset_index()
    )
    counts["expected"] = expected
    counts["missing"] = (expected - counts["completed"]).clip(lower=0)
    return counts
