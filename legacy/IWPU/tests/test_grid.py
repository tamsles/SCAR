from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import pytest


PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))
sys.path.insert(0, str(PROJECT / "scripts"))

from iwpu.config import load_config
from run_experiment import (
    _setting_values,
    canonical_method,
    dataset_id_for_task,
    method_execution,
    run_job,
    train_setting_tuning,
    tuning_combinations,
)
from run_paper_grid import enumerate_jobs, manifest_payload, select_smoke_jobs


@pytest.fixture(scope="module")
def paper_config() -> dict:
    return load_config(PROJECT / "configs" / "paper.yaml")


def test_full_grid_has_2160_unique_jobs(paper_config: dict, tmp_path: Path) -> None:
    jobs = enumerate_jobs(paper_config, tmp_path / "results")
    assert len(jobs) == 8 * 9 * 3 * 10 == 2160
    assert len({job["job_id"] for job in jobs}) == len(jobs)
    assert len({job["result_path"] for job in jobs}) == len(jobs)

    assert Counter(job["task"] for job in jobs) == {
        task: 9 * 3 * 10 for task in paper_config["tasks"]
    }
    assert Counter(job["method"] for job in jobs) == {
        method: 8 * 3 * 10 for method in paper_config["methods"]
    }
    assert Counter(job["seed"] for job in jobs) == {
        seed: 8 * 9 * 3 for seed in paper_config["repetitions"]["seeds"]
    }


def test_default_manifest_describes_every_pending_job(
    paper_config: dict, tmp_path: Path
) -> None:
    jobs = enumerate_jobs(paper_config, tmp_path / "results")
    manifest = manifest_payload(jobs, mode="dry-run", full_job_count=len(jobs))
    assert manifest["mode"] == "dry-run"
    assert manifest["job_count"] == 2160
    assert manifest["full_job_count"] == 2160
    assert manifest["status_counts"] == {"pending": 2160}


def test_optional_ablation_grid_preserves_main_grid(
    paper_config: dict, tmp_path: Path
) -> None:
    main = enumerate_jobs(paper_config, tmp_path / "results")
    with_ablations = enumerate_jobs(
        paper_config, tmp_path / "results", include_ablations=True
    )
    assert len(main) == 2160
    assert len(with_ablations) == 2160 + (7 * 4 * 3 * 10) == 3000
    assert {job["job_id"] for job in main}.issubset(
        {job["job_id"] for job in with_ablations}
    )

    ablation_ids = set(paper_config["ablations"])
    ablation_jobs = [job for job in with_ablations if job["method"] in ablation_ids]
    assert Counter(job["method"] for job in ablation_jobs) == {
        method: 7 * 3 * 10 for method in ablation_ids
    }
    assert all(job["task"] != "foodstamp" for job in ablation_jobs)


def test_smoke_selection_is_exact(paper_config: dict, tmp_path: Path) -> None:
    jobs = enumerate_jobs(paper_config, tmp_path / "results")
    smoke = select_smoke_jobs(jobs)
    assert smoke == [
        {
            "job_id": "mnist_io__iwpu__p10_u50__seed0",
            "task": "mnist_io",
            "method": "iwpu",
            "target_pos": 10,
            "target_unl": 50,
            "seed": 0,
            "result_path": str(
                tmp_path / "results" / "mnist_io" / "iwpu" / "p10_u50_seed0.json"
            ),
        }
    ]


@pytest.mark.parametrize(
    ("method", "task", "expected"),
    [
        ("iwpu", "mnist_io", 15),
        ("ours", "mnist_io", 15),
        ("iwpu", "foodstamp", 18),
        ("giw", "mnist_io", 15),
        ("giw", "foodstamp", 18),
        ("mtpu", "mnist_io", 5),
        ("mtspu", "foodstamp", 6),
        ("dapu", "mnist_io", 20),
        ("dapu", "foodstamp", 24),
        ("udapu", "mnist_io", 4),
        ("tepu", "mnist_io", 1),
        ("trpu", "mnist_io", 1),
        ("ftpu", "mnist_io", 1),
        ("two_step", "mnist_io", 15),
        ("without_importance_weight_correction", "mnist_io", 15),
        ("without_classifier_loss_correction", "mnist_io", 15),
        ("without_both_corrections", "mnist_io", 15),
    ],
)
def test_method_specific_tuning_grid_sizes(
    paper_config: dict, method: str, task: str, expected: int
) -> None:
    combinations = tuning_combinations(paper_config, method, task)
    assert len(combinations) == expected
    assert len({tuple(sorted(combo.items())) for combo in combinations}) == expected


def test_iwpu_alias_is_canonical_config_id(paper_config: dict) -> None:
    assert canonical_method("iwpu", paper_config) == "iwpu"
    assert canonical_method("ours", paper_config) == "iwpu"
    assert canonical_method("IW-PU", paper_config) == "iwpu"
    assert canonical_method("2step", paper_config) == "two_step"
    assert (
        canonical_method("w/o IW-C", paper_config)
        == "without_importance_weight_correction"
    )


@pytest.mark.parametrize(
    ("method", "expected"),
    [
        ("iwpu", ("iwpu", True, True)),
        ("two_step", ("two_step", True, True)),
        ("without_importance_weight_correction", ("iwpu", False, True)),
        ("without_classifier_loss_correction", ("iwpu", True, False)),
        ("without_both_corrections", ("iwpu", False, False)),
    ],
)
def test_ablation_trainer_routing_and_correction_flags(
    paper_config: dict, method: str, expected: tuple[str, bool, bool]
) -> None:
    assert method_execution(method, paper_config) == expected


def test_lambda_grid_key_maps_to_train_settings_name() -> None:
    values = train_setting_tuning({"lambda_mmd": 0.01})
    assert values == {"alpha": 0.5, "beta": 0.5, "mmd_lambda": 0.01}


def test_tableshift_tasks_resolve_to_model_dataset_ids(paper_config: dict) -> None:
    assert dataset_id_for_task(paper_config, "diabetes") == "diabetes"
    assert dataset_id_for_task(paper_config, "foodstamp") == "foodstamp"


def test_explicit_zero_patience_is_not_replaced_by_default(
    paper_config: dict,
) -> None:
    bundle = SimpleNamespace(
        input_dim=4,
        metadata={"dataset": "mnist", "pi_src": 0.5, "pi_tgt": 0.5},
    )
    values = _setting_values(
        paper_config,
        bundle,
        "mnist_io",
        0,
        "cpu",
        {"alpha": 0.5, "beta": 0.5},
        1,
        1,
        0,
        True,
        True,
    )
    assert values["patience"] == 0
    assert values["deterministic_algorithms"] is True


def test_paper_batch_sizes_are_domain_totals_and_define_40_steps(
    paper_config: dict,
) -> None:
    bundle = SimpleNamespace(
        input_dim=4,
        metadata={"dataset": "mnist", "pi_src": 0.5, "pi_tgt": 0.5},
    )
    values = _setting_values(
        paper_config,
        bundle,
        "mnist_io",
        0,
        "cpu",
        {"alpha": 0.5, "beta": 0.5},
        1,
        None,
        0,
        True,
        True,
    )
    assert values["source_batch_size"] == 256
    assert values["target_batch_size"] == 256
    assert values["steps_per_epoch"] == 40


def test_run_job_snapshots_inputs_and_tests_only_selected_candidate(
    paper_config: dict,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import iwpu.datasets as datasets_module
    import iwpu.trainers as trainers_module
    import iwpu.utils as utils_module

    bundle_metadata = {
        "dataset": "mnist",
        "pi_src": 0.5,
        "pi_tgt": 0.5,
        "sample_indices": {"src_pos": [7, 11], "test_x": [101, 103]},
    }
    bundle = SimpleNamespace(input_dim=4, metadata=bundle_metadata)
    train_calls: list[dict] = []
    evaluation_calls: list[object] = []
    seed_calls: list[tuple[int, bool]] = []

    def fake_loader(*args: object, **kwargs: object) -> SimpleNamespace:
        return bundle

    def fake_train_one(
        candidate_bundle: object,
        method: str,
        settings: object,
        *,
        evaluate_test: bool = True,
    ) -> dict:
        index = len(train_calls)
        train_calls.append(
            {
                "bundle": candidate_bundle,
                "method": method,
                "settings": settings,
                "evaluate_test": evaluate_test,
            }
        )
        return {
            "model": f"candidate-{index}",
            "best_val_risk": float(100 - index),
            "epochs": 1,
            "runtime": 0.01,
            "history": [],
        }

    def fake_evaluate(model: object, candidate_bundle: object, device: object) -> float:
        evaluation_calls.append(model)
        assert candidate_bundle is bundle
        assert str(device) == "cpu"
        return 0.875

    def fake_seed(seed: int, deterministic: bool = True) -> None:
        seed_calls.append((seed, deterministic))

    monkeypatch.setattr(datasets_module, "load_pu_dataset", fake_loader)
    monkeypatch.setattr(trainers_module, "train_one", fake_train_one)
    monkeypatch.setattr(trainers_module, "evaluate_test_accuracy", fake_evaluate)
    monkeypatch.setattr(utils_module, "seed_everything", fake_seed)
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

    output = tmp_path / "result.json"
    record = run_job(
        config=paper_config,
        config_path=PROJECT / "configs" / "paper.yaml",
        task="mnist_io",
        method="iwpu",
        target_pos=10,
        target_unl=50,
        seed=0,
        device="cpu",
        data_root=tmp_path / "data",
        output_path=output,
        download=False,
        max_epochs=1,
        steps_per_epoch=1,
        patience=0,
    )

    persisted = json.loads(output.read_text(encoding="utf-8"))
    assert persisted["status"] == record["status"] == "complete"
    assert persisted["test_accuracy"] == record["test_accuracy"]
    assert record["config_snapshot"] == paper_config
    assert len(record["config_file_sha256"]) == 64
    assert record["bundle_metadata"] == bundle_metadata
    assert record["resolved_config"]["best_train_settings"]["patience"] == 0
    assert record["resolved_config"]["runtime"]["deterministic_algorithms"] is True
    assert all(call["evaluate_test"] is False for call in train_calls)
    assert all(
        "test_accuracy" not in candidate
        for candidate in record["tuning"]["candidates"]
    )
    assert record["tuning"]["candidate_test_metrics_computed"] is False
    assert record["test_evaluation"]["evaluation_count"] == 1
    assert evaluation_calls == [f"candidate-{len(train_calls) - 1}"]
    assert record["test_accuracy"] == 0.875
    assert record["trainer_result"]["test_accuracy"] == 0.875
    assert seed_calls and all(call == (0, True) for call in seed_calls)
