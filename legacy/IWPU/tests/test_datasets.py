from __future__ import annotations

import json

import numpy as np
import pytest
import torch

from iwpu.datasets import (
    IMAGE_TASKS,
    build_image_pu_bundle,
    build_tabular_pu_bundle,
    load_pu_dataset,
    load_tabular_npz,
)


EXPECTED_CLASS_MAPS = {
    "mnist_io": {
        "source_positive": [0, 2, 5, 7, 9],
        "source_negative": [1, 3, 4, 6, 8],
        "target_positive": [1, 3, 5, 7, 9],
        "target_negative": [0, 2, 4, 6, 8],
    },
    "mnist_support": {
        "source_positive": [1, 3, 5],
        "source_negative": [0, 2, 4],
        "target_positive": [5, 7, 9],
        "target_negative": [4, 6, 8],
    },
    "fmnist_io": {
        "source_positive": [0, 2, 7, 8, 9],
        "source_negative": [1, 3, 4, 5, 6],
        "target_positive": [1, 5, 7, 8, 9],
        "target_negative": [0, 2, 3, 4, 6],
    },
    "fmnist_support": {
        "source_positive": [1, 5, 7],
        "source_negative": [0, 2, 3],
        "target_positive": [7, 8, 9],
        "target_negative": [3, 4, 6],
    },
    "cifar10_io": {
        "source_positive": [2, 3, 8, 9],
        "source_negative": [0, 1, 4, 5, 6, 7],
        "target_positive": [0, 1, 8, 9],
        "target_negative": [2, 3, 4, 5, 6, 7],
    },
    "cifar10_support": {
        "source_positive": [0, 1, 8],
        "source_negative": [2, 3, 4, 5],
        "target_positive": [1, 8, 9],
        "target_negative": [4, 5, 6, 7],
    },
}


def _multiclass_raw(per_class: int = 40) -> tuple[np.ndarray, np.ndarray]:
    labels = np.repeat(np.arange(10, dtype=np.int64), per_class)
    # Two uint8 features make selected rows easy to compare while exercising
    # the exact same 1/255 path as raw torchvision pixels.
    values = np.arange(labels.size * 2, dtype=np.int64).reshape(labels.size, 2)
    return (values % 256).astype(np.uint8), labels


def _small_image_bundle(task: str, seed: int = 7):
    train_x, train_y = _multiclass_raw(40)
    test_x, test_y = _multiclass_raw(20)
    bundle = build_image_pu_bundle(
        train_x,
        train_y,
        test_x,
        test_y,
        task=task,
        target_pos=4,
        target_unl=6,
        seed=seed,
        source_pos=6,
        source_unl=8,
        val_pos=2,
        val_unl=4,
        test_pos=6,
        test_neg=6,
    )
    return bundle, train_y, test_y


@pytest.mark.parametrize("task", sorted(EXPECTED_CLASS_MAPS))
def test_exact_paper_class_maps_and_balanced_unlabeled_sets(task: str) -> None:
    bundle, train_labels, test_labels = _small_image_bundle(task)
    expected = EXPECTED_CLASS_MAPS[task]
    assert bundle.metadata["class_maps"] == expected

    records = bundle.metadata["sample_indices"]
    source_positive_idx = np.asarray(records["src_pos"]["indices"])
    target_positive_idx = np.asarray(records["tgt_pos"]["indices"])
    assert set(train_labels[source_positive_idx]) <= set(expected["source_positive"])
    assert set(train_labels[target_positive_idx]) <= set(expected["target_positive"])

    for key, positive_classes in (
        ("src_unl", expected["source_positive"]),
        ("tgt_unl", expected["target_positive"]),
        ("val_unl", expected["target_positive"]),
    ):
        indices = np.asarray(records[key]["indices"])
        positive_count = np.isin(train_labels[indices], positive_classes).sum()
        assert positive_count * 2 == indices.size

    eval_idx = np.asarray(records["test_x"]["indices"])
    expected_test_y = np.where(
        np.isin(test_labels[eval_idx], expected["target_positive"]), 1.0, -1.0
    )
    np.testing.assert_array_equal(bundle.test_y.numpy(), expected_test_y)


def test_all_train_allocations_are_globally_disjoint_and_test_is_separate() -> None:
    bundle, _, _ = _small_image_bundle("mnist_io")
    records = bundle.metadata["sample_indices"]
    train_keys = ("src_pos", "src_unl", "tgt_pos", "tgt_unl", "val_pos", "val_unl")
    train_indices = [
        (key, set(records[key]["indices"]))
        for key in train_keys
    ]
    for position, (left_name, left) in enumerate(train_indices):
        for right_name, right in train_indices[position + 1 :]:
            assert left.isdisjoint(right), f"{left_name} overlaps {right_name}"
    assert all(records[key]["raw_split"] == "official_train" for key in train_keys)
    assert records["test_x"]["raw_split"] == "official_test"


def test_sampling_is_deterministic_and_seed_sensitive() -> None:
    first, _, _ = _small_image_bundle("cifar10_support", seed=11)
    repeated, _, _ = _small_image_bundle("cifar10_support", seed=11)
    changed, _, _ = _small_image_bundle("cifar10_support", seed=12)
    assert first.metadata["sample_indices"] == repeated.metadata["sample_indices"]
    torch.testing.assert_close(first.src_pos, repeated.src_pos)
    assert first.metadata["sample_indices"] != changed.metadata["sample_indices"]


@pytest.mark.parametrize(
    ("task", "train_shape", "expected_shape", "input_dim"),
    [
        ("mnist_io", (10, 28, 28), (1, 28, 28), 784),
        ("fmnist_io", (10, 28, 28), (1, 28, 28), 784),
        ("cifar10_io", (10, 32, 32, 3), (3, 32, 32), 3072),
    ],
)
def test_torchvision_layouts_are_nchw_float32_unit_scaled(
    task: str,
    train_shape: tuple[int, ...],
    expected_shape: tuple[int, ...],
    input_dim: int,
) -> None:
    # Give every class enough independent examples for tiny test counts.
    train_labels = np.repeat(np.arange(10), 8)
    test_labels = np.repeat(np.arange(10), 4)
    train_x = np.full((train_labels.size, *train_shape[1:]), 255, dtype=np.uint8)
    test_x = np.zeros((test_labels.size, *train_shape[1:]), dtype=np.uint8)
    bundle = build_image_pu_bundle(
        train_x,
        train_labels,
        test_x,
        test_labels,
        task=task,
        target_pos=2,
        target_unl=2,
        seed=3,
        source_pos=2,
        source_unl=2,
        val_pos=1,
        val_unl=2,
        test_pos=2,
        test_neg=2,
    )
    assert bundle.src_pos.shape[1:] == expected_shape
    assert bundle.input_dim == input_dim
    assert bundle.src_pos.dtype == torch.float32
    assert bundle.src_pos.min().item() == 1.0
    assert bundle.src_pos.max().item() == 1.0
    assert bundle.test_x.min().item() == 0.0
    assert bundle.test_x.max().item() == 0.0


def test_tabular_builder_is_balanced_disjoint_and_preserves_preprocessed_values() -> None:
    source_y = np.tile(np.array([0, 1], dtype=np.int8), 40)
    target_y = np.tile(np.array([-1, 1], dtype=np.int8), 50)
    source_x = np.column_stack(
        (np.arange(source_y.size), np.full(source_y.size, -3.0))
    )
    target_x = np.column_stack(
        (np.arange(target_y.size) + 100.0, np.full(target_y.size, -3.0))
    )
    bundle = build_tabular_pu_bundle(
        source_x,
        source_y,
        target_x,
        target_y,
        task="diabetes",
        target_pos=4,
        target_unl=6,
        seed=5,
        source_pos=6,
        source_unl=8,
        val_pos=2,
        val_unl=4,
        test_pos=6,
        test_neg=6,
    )
    for tensor in (
        bundle.src_pos,
        bundle.src_unl,
        bundle.tgt_pos,
        bundle.tgt_unl,
        bundle.val_pos,
        bundle.val_unl,
        bundle.test_x,
    ):
        assert tensor.dtype == torch.float32
    assert bundle.input_dim == 2
    assert bundle.metadata["preprocessing"] == {
        "dtype": "float32",
        "upstream_transform": "tableshift_repository_preprocessing",
        "additional_scaling": "none",
        "transform_fit_rows": 0,
        "held_out_target_covariates_used_to_fit_transform": False,
    }
    assert set(bundle.test_y.tolist()) == {-1.0, 1.0}
    assert int((bundle.test_y == 1).sum()) == int((bundle.test_y == -1).sum()) == 6

    records = bundle.metadata["sample_indices"]
    source_sets = [set(records[key]["indices"]) for key in ("src_pos", "src_unl")]
    target_sets = [
        set(records[key]["indices"])
        for key in ("tgt_pos", "tgt_unl", "val_pos", "val_unl", "test_x")
    ]
    assert source_sets[0].isdisjoint(source_sets[1])
    for position, left in enumerate(target_sets):
        assert all(left.isdisjoint(right) for right in target_sets[position + 1 :])

    tensor_sources = {
        "src_pos": (bundle.src_pos, source_x),
        "src_unl": (bundle.src_unl, source_x),
        "tgt_pos": (bundle.tgt_pos, target_x),
        "tgt_unl": (bundle.tgt_unl, target_x),
        "val_pos": (bundle.val_pos, target_x),
        "val_unl": (bundle.val_unl, target_x),
        "test_x": (bundle.test_x, target_x),
    }
    for name, (tensor, original) in tensor_sources.items():
        indices = records[name]["indices"]
        torch.testing.assert_close(
            tensor, torch.as_tensor(original[indices], dtype=torch.float32)
        )


def test_npz_loader_preserves_export_metadata(tmp_path) -> None:
    # load_tabular_npz intentionally uses the paper's fixed split sizes.  Keep
    # this metadata test realistic enough to satisfy those disjoint draws.
    source_y = np.tile(np.array([0, 1], dtype=np.int8), 3_500)
    target_y = np.tile(np.array([0, 1], dtype=np.int8), 2_600)
    source_x = np.arange(source_y.size * 3, dtype=np.float32).reshape(-1, 3)
    target_x = np.arange(target_y.size * 3, dtype=np.float32).reshape(-1, 3)
    path = tmp_path / "brfss_diabetes.npz"
    np.savez_compressed(
        path,
        source_x=source_x,
        source_y=source_y,
        target_x=target_x,
        target_y=target_y,
        metadata=np.asarray(json.dumps({"tableshift_version": "test-version"})),
    )
    bundle = load_tabular_npz(
        path, target_pos=4, target_unl=6, seed=9, task="diabetes"
    )
    assert bundle.metadata["npz_metadata"]["tableshift_version"] == "test-version"
    assert bundle.metadata["npz_path"] == str(path.resolve())


def test_unified_tabular_entry_uses_exact_contract(monkeypatch, tmp_path) -> None:
    sentinel = object()
    captured = {}

    def fake_loader(path, *, target_pos, target_unl, seed, task):
        captured.update(
            path=path,
            target_pos=target_pos,
            target_unl=target_unl,
            seed=seed,
            task=task,
        )
        return sentinel

    monkeypatch.setattr("iwpu.datasets.load_tabular_npz", fake_loader)
    result = load_pu_dataset(
        "foodstamp",
        tmp_path,
        10,
        50,
        2,
        tabular_npz=tmp_path,
        download=False,
    )
    assert result is sentinel
    assert captured["path"] == tmp_path / "acsfoodstamps.npz"
    assert captured["target_pos"] == 10
    assert captured["target_unl"] == 50
    assert captured["seed"] == 2
    assert captured["task"] == "foodstamp"


def test_odd_unlabeled_size_and_insufficient_pool_fail_clearly() -> None:
    train_x, train_y = _multiclass_raw(4)
    test_x, test_y = _multiclass_raw(2)
    with pytest.raises(ValueError, match="must be even"):
        build_image_pu_bundle(
            train_x,
            train_y,
            test_x,
            test_y,
            task="mnist_io",
            target_pos=2,
            target_unl=3,
            seed=0,
            source_pos=2,
            source_unl=2,
            val_pos=1,
            val_unl=2,
            test_pos=1,
            test_neg=1,
        )
    with pytest.raises(ValueError, match="not enough unused"):
        build_image_pu_bundle(
            train_x,
            train_y,
            test_x,
            test_y,
            task="mnist_io",
            target_pos=100,
            target_unl=2,
            seed=0,
            source_pos=2,
            source_unl=2,
            val_pos=1,
            val_unl=2,
            test_pos=1,
            test_neg=1,
        )


def test_task_registry_matches_the_six_image_tasks() -> None:
    assert set(IMAGE_TASKS) == set(EXPECTED_CLASS_MAPS)
