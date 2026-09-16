from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any


MAIN_METHODS = {
    "iwpu",
    "tepu",
    "trpu",
    "ftpu",
    "mtpu",
    "mtspu",
    "dapu",
    "udapu",
    "giw",
}
ABLATION_METHODS = {
    "two_step",
    "without_importance_weight_correction",
    "without_classifier_loss_correction",
    "without_both_corrections",
}
TARGET_SIZES = {(10, 50), (20, 100), (40, 200)}
SEEDS = set(range(10))


def expected_keys() -> set[tuple[str, str, int, int, int]]:
    keys: set[tuple[str, str, int, int, int]] = set()
    for task in ("diabetes", "foodstamp"):
        for method in MAIN_METHODS:
            for target_pos, target_unl in TARGET_SIZES:
                for seed in SEEDS:
                    keys.add((task, method, target_pos, target_unl, seed))
    for method in ABLATION_METHODS:
        for target_pos, target_unl in TARGET_SIZES:
            for seed in SEEDS:
                keys.add(("diabetes", method, target_pos, target_unl, seed))
    return keys


def finite_probability(value: Any) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(float(value)) and 0 <= value <= 1


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate the corrected TableShift rerun.")
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    records: list[dict[str, Any]] = []
    parse_errors: list[str] = []
    for path in sorted(args.results.rglob("*.json")):
        if "manifests" in path.parts:
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            parse_errors.append(f"{path}: {type(error).__name__}: {error}")
            continue
        payload["_path"] = str(path)
        records.append(payload)

    keys: list[tuple[str, str, int, int, int]] = []
    invalid_status: list[str] = []
    invalid_accuracy: list[str] = []
    invalid_preprocessing: list[str] = []
    candidate_failures: list[str] = []
    invalid_test_protocol: list[str] = []
    config_hashes: Counter[str | None] = Counter()
    counts_by_task_method: Counter[tuple[str, str]] = Counter()

    for record in records:
        path = record["_path"]
        method = record.get("config_method")
        key = (
            record.get("task"),
            method,
            record.get("target_pos"),
            record.get("target_unl"),
            record.get("seed"),
        )
        keys.append(key)  # type: ignore[arg-type]
        counts_by_task_method[(str(key[0]), str(key[1]))] += 1
        config_hashes[record.get("config_file_sha256")] += 1

        if record.get("status") != "complete":
            invalid_status.append(path)
        if not finite_probability(record.get("test_accuracy")):
            invalid_accuracy.append(path)

        preprocessing = record.get("bundle_metadata", {}).get("preprocessing", {})
        if not (
            preprocessing.get("additional_scaling") == "none"
            and preprocessing.get("transform_fit_rows") == 0
            and preprocessing.get("held_out_target_covariates_used_to_fit_transform") is False
        ):
            invalid_preprocessing.append(path)

        failed_candidates = [
            candidate
            for candidate in record.get("tuning", {}).get("candidates", [])
            if candidate.get("status") != "complete"
        ]
        if failed_candidates:
            candidate_failures.append(path)

        test_evaluation = record.get("test_evaluation", {})
        tuning = record.get("tuning", {})
        if not (
            test_evaluation.get("performed_after_hyperparameter_selection") is True
            and test_evaluation.get("evaluation_count") == 1
            and tuning.get("candidate_test_metrics_computed") is False
        ):
            invalid_test_protocol.append(path)

    key_counts = Counter(keys)
    duplicate_keys = [list(key) for key, count in key_counts.items() if count != 1]
    actual_keys = set(keys)
    expected = expected_keys()
    missing_keys = [list(key) for key in sorted(expected - actual_keys)]
    extra_keys = [list(key) for key in sorted(actual_keys - expected)]

    checks = {
        "expected_records": len(expected),
        "actual_records": len(records),
        "parse_error_count": len(parse_errors),
        "invalid_status_count": len(invalid_status),
        "invalid_accuracy_count": len(invalid_accuracy),
        "invalid_preprocessing_count": len(invalid_preprocessing),
        "candidate_failure_record_count": len(candidate_failures),
        "invalid_test_protocol_count": len(invalid_test_protocol),
        "duplicate_key_count": len(duplicate_keys),
        "missing_key_count": len(missing_keys),
        "extra_key_count": len(extra_keys),
        "unique_config_hash_count": len(config_hashes),
    }
    passed = (
        checks["actual_records"] == checks["expected_records"]
        and all(value == 0 for name, value in checks.items() if name.endswith("_count") and name != "unique_config_hash_count")
        and checks["unique_config_hash_count"] == 1
    )
    report = {
        "passed": passed,
        "checks": checks,
        "counts_by_task_method": {
            f"{task}/{method}": count
            for (task, method), count in sorted(counts_by_task_method.items())
        },
        "config_hashes": dict(config_hashes),
        "examples": {
            "parse_errors": parse_errors[:10],
            "invalid_status": invalid_status[:10],
            "invalid_accuracy": invalid_accuracy[:10],
            "invalid_preprocessing": invalid_preprocessing[:10],
            "candidate_failures": candidate_failures[:10],
            "invalid_test_protocol": invalid_test_protocol[:10],
            "duplicate_keys": duplicate_keys[:10],
            "missing_keys": missing_keys[:10],
            "extra_keys": extra_keys[:10],
        },
    }

    rendered = json.dumps(report, indent=2, sort_keys=True)
    print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
