import pandas as pd

from iwpu.reporting import (
    aggregate_ablation_table,
    aggregate_foodstamp_table,
    aggregate_main_table,
    completeness_table,
)


def test_aggregate_main_table() -> None:
    frame = pd.DataFrame(
        [
            {"dataset": "mnist", "shift": "io", "method": "ours", "seed": 0, "target_pos": 10, "test_accuracy": 0.7},
            {"dataset": "mnist", "shift": "io", "method": "ours", "seed": 1, "target_pos": 10, "test_accuracy": 0.9},
        ]
    )
    result = aggregate_main_table(frame)
    assert result.iloc[0]["mean"] == 0.8
    assert result.iloc[0]["count"] == 2


def test_completeness_counts_size_seed_cells() -> None:
    frame = pd.DataFrame(
        [{"dataset": "mnist", "shift": "io", "method": "ours"}] * 4
    )
    result = completeness_table(frame, expected_seeds=2, expected_target_sizes=3)
    assert result.iloc[0]["missing"] == 2


def test_main_table_normalizes_keys_and_excludes_appendix() -> None:
    frame = pd.DataFrame(
        [
            {
                "dataset": "fashion_mnist",
                "shift": "input_output_relation",
                "method": "ours",
                "seed": 0,
                "target_pos": 10,
                "test_accuracy": 0.8,
            },
            {
                "dataset": "foodstamp",
                "shift": "geographic_region",
                "method": "ours",
                "seed": 0,
                "target_pos": 10,
                "test_accuracy": 0.7,
            },
        ]
    )
    result = aggregate_main_table(frame)
    assert result.iloc[0]["dataset"] == "fmnist"
    assert result.iloc[0]["shift"] == "io"
    assert len(result) == 1


def test_main_table_excludes_ablation_methods() -> None:
    frame = pd.DataFrame(
        [
            {"dataset": "mnist", "shift": "io", "method": "ours", "seed": 0, "target_pos": 10, "test_accuracy": 0.8},
            {"dataset": "mnist", "shift": "io", "method": "two_step", "seed": 0, "target_pos": 10, "test_accuracy": 0.7},
        ]
    )
    result = aggregate_main_table(frame)
    assert list(result["method"].astype(str)) == ["ours"]


def test_ablation_table_excludes_main_methods() -> None:
    frame = pd.DataFrame(
        [
            {"dataset": "mnist", "shift": "io", "method": "ours", "test_accuracy": 0.8},
            {"dataset": "mnist", "shift": "io", "method": "two_step", "test_accuracy": 0.6},
            {"dataset": "mnist", "shift": "io", "method": "two_step", "test_accuracy": 0.8},
        ]
    )
    result = aggregate_ablation_table(frame)
    assert len(result) == 1
    assert result.iloc[0]["method"] == "two_step"
    assert result.iloc[0]["mean"] == 0.7


def test_foodstamp_has_separate_appendix_aggregate() -> None:
    frame = pd.DataFrame(
        [
            {"dataset": "foodstamp", "method": "ours", "test_accuracy": 0.6},
            {"dataset": "foodstamp", "method": "ours", "test_accuracy": 0.8},
            {"dataset": "mnist", "method": "ours", "test_accuracy": 0.9},
        ]
    )
    result = aggregate_foodstamp_table(frame)
    assert result.iloc[0]["mean"] == 0.7
    assert result.iloc[0]["count"] == 2
