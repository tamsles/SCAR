"""Dataset construction for the IW-PU reproduction.

The paper specifies the binary class maps and the sizes of every PU split, but
does not publish its split-generation code.  This module makes the executable
assumptions in ``configs/paper.yaml`` explicit:

* image PU/validation examples come only from the official training split;
* the final image evaluation set comes only from the official test split;
* every raw example is used at most once within its official/domain split;
* each unlabeled set is sampled with class prior 0.5; and
* image pixels are scaled to ``[0, 1]`` without augmentation or further
  normalization.

Torchvision and TableShift are deliberately optional.  Torchvision is imported
only by :func:`load_pu_dataset` for an image task, while TableShift data are
consumed through the portable NPZ format produced by
``scripts/export_tableshift.py``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch


PAPER_SOURCE_POS = 1_000
PAPER_SOURCE_UNL = 5_000
PAPER_VAL_POS = 20
PAPER_VAL_UNL = 100
PAPER_TEST_POS = 2_500
PAPER_TEST_NEG = 2_500
PAPER_CLASS_PRIOR = 0.5


@dataclass
class PUDataBundle:
    """All tensors needed by an IW-PU experiment.

    Positive tensors contain positive examples only.  Unlabeled tensors contain
    an exactly balanced, shuffled mixture of positive and negative examples.
    Labels are intentionally exposed only for the final evaluation set, where
    they use the paper's signed convention ``{-1, +1}``.
    """

    src_pos: torch.Tensor
    src_unl: torch.Tensor
    tgt_pos: torch.Tensor
    tgt_unl: torch.Tensor
    val_pos: torch.Tensor
    val_unl: torch.Tensor
    test_x: torch.Tensor
    test_y: torch.Tensor
    input_dim: int
    metadata: dict[str, Any]


@dataclass(frozen=True)
class _ClassMap:
    dataset: str
    shift: str
    source_positive: tuple[int, ...]
    source_negative: tuple[int, ...]
    target_positive: tuple[int, ...]
    target_negative: tuple[int, ...]


# Exact expansions of the rules in Section 5.1 of the paper.
IMAGE_TASKS: dict[str, _ClassMap] = {
    "mnist_io": _ClassMap(
        "mnist",
        "io",
        (0, 2, 5, 7, 9),
        (1, 3, 4, 6, 8),
        (1, 3, 5, 7, 9),
        (0, 2, 4, 6, 8),
    ),
    "mnist_support": _ClassMap(
        "mnist", "support", (1, 3, 5), (0, 2, 4), (5, 7, 9), (4, 6, 8)
    ),
    "fmnist_io": _ClassMap(
        "fashion_mnist",
        "io",
        (0, 2, 7, 8, 9),
        (1, 3, 4, 5, 6),
        (1, 5, 7, 8, 9),
        (0, 2, 3, 4, 6),
    ),
    "fmnist_support": _ClassMap(
        "fashion_mnist", "support", (1, 5, 7), (0, 2, 3), (7, 8, 9), (3, 4, 6)
    ),
    "cifar10_io": _ClassMap(
        "cifar10",
        "io",
        (2, 3, 8, 9),
        (0, 1, 4, 5, 6, 7),
        (0, 1, 8, 9),
        (2, 3, 4, 5, 6, 7),
    ),
    "cifar10_support": _ClassMap(
        "cifar10", "support", (0, 1, 8), (2, 3, 4, 5), (1, 8, 9), (4, 5, 6, 7)
    ),
}


TABULAR_TASKS: dict[str, dict[str, str]] = {
    "diabetes": {
        "dataset": "diabetes",
        "tableshift_id": "brfss_diabetes",
        "shift": "race",
    },
    "foodstamp": {
        "dataset": "foodstamp",
        "tableshift_id": "acsfoodstamps",
        "shift": "geographic_region",
    },
}


def _positive_integer(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an integer")
    value = int(value)
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    return value


def _validate_counts(
    *,
    source_pos: int,
    source_unl: int,
    target_pos: int,
    target_unl: int,
    val_pos: int,
    val_unl: int,
    test_pos: int,
    test_neg: int,
) -> dict[str, int]:
    counts = {
        "source_positive": _positive_integer(source_pos, "source_pos"),
        "source_unlabeled": _positive_integer(source_unl, "source_unl"),
        "target_positive": _positive_integer(target_pos, "target_pos"),
        "target_unlabeled": _positive_integer(target_unl, "target_unl"),
        "validation_positive": _positive_integer(val_pos, "val_pos"),
        "validation_unlabeled": _positive_integer(val_unl, "val_unl"),
        "test_positive": _positive_integer(test_pos, "test_pos"),
        "test_negative": _positive_integer(test_neg, "test_neg"),
    }
    for key in ("source_unlabeled", "target_unlabeled", "validation_unlabeled"):
        if counts[key] % 2:
            raise ValueError(f"{key} must be even to realize class prior 0.5 exactly")
    return counts


class _WithoutReplacementSampler:
    """Seeded sampler that records and enforces global non-overlap."""

    def __init__(self, labels: np.ndarray, rng: np.random.Generator, namespace: str):
        labels = np.asarray(labels).reshape(-1)
        if labels.ndim != 1:
            raise ValueError("labels must be one-dimensional")
        self.labels = labels
        self.rng = rng
        self.namespace = namespace
        self.available = np.ones(labels.shape[0], dtype=bool)

    def draw(self, classes: Sequence[int] | np.ndarray, size: int, name: str) -> np.ndarray:
        mask = self.available & np.isin(self.labels, np.asarray(classes))
        candidates = np.flatnonzero(mask)
        if candidates.size < size:
            raise ValueError(
                f"not enough unused {self.namespace} examples for {name}: "
                f"requested {size}, available {candidates.size}, classes={list(classes)}"
            )
        selected = self.rng.choice(candidates, size=size, replace=False)
        self.available[selected] = False
        return selected.astype(np.int64, copy=False)

    def draw_balanced(
        self,
        positive_classes: Sequence[int] | np.ndarray,
        negative_classes: Sequence[int] | np.ndarray,
        size: int,
        name: str,
    ) -> np.ndarray:
        half = size // 2
        positive = self.draw(positive_classes, half, f"{name}/positive")
        negative = self.draw(negative_classes, half, f"{name}/negative")
        joined = np.concatenate((positive, negative))
        return joined[self.rng.permutation(joined.size)]


def _as_numpy(value: Any, *, name: str) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    result = np.asarray(value)
    if result.ndim == 0:
        raise ValueError(f"{name} must have a sample dimension")
    return result


def _validate_xy(x: Any, y: Any, *, name: str) -> tuple[np.ndarray, np.ndarray]:
    x_array = _as_numpy(x, name=f"{name}_x")
    y_array = _as_numpy(y, name=f"{name}_y").reshape(-1)
    if x_array.shape[0] != y_array.shape[0]:
        raise ValueError(
            f"{name} feature/label length mismatch: {x_array.shape[0]} != {y_array.shape[0]}"
        )
    if x_array.shape[0] == 0:
        raise ValueError(f"{name} arrays must not be empty")
    return x_array, y_array


def _image_rows(raw_x: np.ndarray, indices: np.ndarray, dataset: str) -> torch.Tensor:
    """Copy selected raw rows, canonicalize to NCHW, and unit-scale pixels."""

    rows = np.array(raw_x[indices], copy=True)
    if dataset in {"mnist", "fashion_mnist"} and rows.ndim == 3:
        rows = rows[:, None, :, :]
    elif dataset == "cifar10" and rows.ndim == 4 and rows.shape[-1] == 3:
        rows = np.transpose(rows, (0, 3, 1, 2))

    tensor = torch.from_numpy(np.ascontiguousarray(rows))
    if tensor.dtype == torch.uint8:
        tensor = tensor.to(torch.float32).div_(255.0)
    else:
        tensor = tensor.to(torch.float32)
        if not torch.isfinite(tensor).all():
            raise ValueError("image arrays contain NaN or infinite values")
        if tensor.numel() and (tensor.min().item() < 0.0 or tensor.max().item() > 1.0):
            raise ValueError(
                "floating-point image arrays must already lie in [0, 1]; "
                "pass raw uint8 pixels to apply 1/255 scaling"
            )
    return tensor


def _index_record(raw_split: str, indices: np.ndarray) -> dict[str, Any]:
    return {"raw_split": raw_split, "indices": indices.tolist()}


def _class_map_metadata(class_map: _ClassMap) -> dict[str, list[int]]:
    return {
        "source_positive": list(class_map.source_positive),
        "source_negative": list(class_map.source_negative),
        "target_positive": list(class_map.target_positive),
        "target_negative": list(class_map.target_negative),
    }


def build_image_pu_bundle(
    train_x: Any,
    train_labels: Any,
    test_x: Any,
    test_labels: Any,
    *,
    task: str,
    target_pos: int,
    target_unl: int,
    seed: int,
    source_pos: int = PAPER_SOURCE_POS,
    source_unl: int = PAPER_SOURCE_UNL,
    val_pos: int = PAPER_VAL_POS,
    val_unl: int = PAPER_VAL_UNL,
    test_pos: int = PAPER_TEST_POS,
    test_neg: int = PAPER_TEST_NEG,
) -> PUDataBundle:
    """Construct an image bundle from already-loaded official raw splits.

    This public helper is also the no-download seam used by the test suite.
    ``train_x`` and ``test_x`` follow torchvision's raw layouts: ``NHW`` for
    MNIST/FMNIST and ``NHWC`` for CIFAR10.
    """

    if task not in IMAGE_TASKS:
        raise ValueError(f"unsupported image task {task!r}; expected one of {sorted(IMAGE_TASKS)}")
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)):
        raise TypeError("seed must be an integer")
    seed = int(seed)
    counts = _validate_counts(
        source_pos=source_pos,
        source_unl=source_unl,
        target_pos=target_pos,
        target_unl=target_unl,
        val_pos=val_pos,
        val_unl=val_unl,
        test_pos=test_pos,
        test_neg=test_neg,
    )
    train_x_array, train_y_array = _validate_xy(train_x, train_labels, name="train")
    test_x_array, test_y_array = _validate_xy(test_x, test_labels, name="test")
    class_map = IMAGE_TASKS[task]

    seed_sequence = np.random.SeedSequence(seed)
    train_seed, test_seed = seed_sequence.spawn(2)
    train_sampler = _WithoutReplacementSampler(
        train_y_array, np.random.default_rng(train_seed), "official-train"
    )
    eval_sampler = _WithoutReplacementSampler(
        test_y_array, np.random.default_rng(test_seed), "official-test"
    )

    src_pos_idx = train_sampler.draw(
        class_map.source_positive, counts["source_positive"], "source-positive"
    )
    src_unl_idx = train_sampler.draw_balanced(
        class_map.source_positive,
        class_map.source_negative,
        counts["source_unlabeled"],
        "source-unlabeled",
    )
    tgt_pos_idx = train_sampler.draw(
        class_map.target_positive, counts["target_positive"], "target-positive"
    )
    tgt_unl_idx = train_sampler.draw_balanced(
        class_map.target_positive,
        class_map.target_negative,
        counts["target_unlabeled"],
        "target-unlabeled",
    )
    val_pos_idx = train_sampler.draw(
        class_map.target_positive, counts["validation_positive"], "validation-positive"
    )
    val_unl_idx = train_sampler.draw_balanced(
        class_map.target_positive,
        class_map.target_negative,
        counts["validation_unlabeled"],
        "validation-unlabeled",
    )

    eval_pos_idx = eval_sampler.draw(
        class_map.target_positive, counts["test_positive"], "test-positive"
    )
    eval_neg_idx = eval_sampler.draw(
        class_map.target_negative, counts["test_negative"], "test-negative"
    )
    eval_indices = np.concatenate((eval_pos_idx, eval_neg_idx))
    eval_labels = np.concatenate(
        (
            np.ones(eval_pos_idx.size, dtype=np.float32),
            -np.ones(eval_neg_idx.size, dtype=np.float32),
        )
    )
    eval_order = eval_sampler.rng.permutation(eval_indices.size)
    eval_indices = eval_indices[eval_order]
    eval_labels = eval_labels[eval_order]

    rows = lambda raw, idx: _image_rows(raw, idx, class_map.dataset)
    src_pos_tensor = rows(train_x_array, src_pos_idx)
    input_dim = int(src_pos_tensor[0].numel())
    metadata: dict[str, Any] = {
        "task": task,
        "dataset": class_map.dataset,
        "shift": class_map.shift,
        "seed": seed,
        "pi_src": PAPER_CLASS_PRIOR,
        "pi_tgt": PAPER_CLASS_PRIOR,
        "pi_tr": PAPER_CLASS_PRIOR,
        "pi_te": PAPER_CLASS_PRIOR,
        "counts": counts,
        "class_maps": _class_map_metadata(class_map),
        "sampling": {
            "replacement": False,
            "unlabeled_class_prior": PAPER_CLASS_PRIOR,
            "globally_disjoint_within_raw_split": True,
        },
        "raw_split_policy": {
            "pu_and_validation": "official_train",
            "evaluation": "official_test",
        },
        "preprocessing": {
            "dtype": "float32",
            "pixel_range": [0.0, 1.0],
            "pixel_scaling": "divide_uint8_by_255",
            "augmentation": "none",
            "additional_normalization": "none",
        },
        "sample_indices": {
            "src_pos": _index_record("official_train", src_pos_idx),
            "src_unl": _index_record("official_train", src_unl_idx),
            "tgt_pos": _index_record("official_train", tgt_pos_idx),
            "tgt_unl": _index_record("official_train", tgt_unl_idx),
            "val_pos": _index_record("official_train", val_pos_idx),
            "val_unl": _index_record("official_train", val_unl_idx),
            "test_x": _index_record("official_test", eval_indices),
        },
    }
    return PUDataBundle(
        src_pos=src_pos_tensor,
        src_unl=rows(train_x_array, src_unl_idx),
        tgt_pos=rows(train_x_array, tgt_pos_idx),
        tgt_unl=rows(train_x_array, tgt_unl_idx),
        val_pos=rows(train_x_array, val_pos_idx),
        val_unl=rows(train_x_array, val_unl_idx),
        test_x=rows(test_x_array, eval_indices),
        test_y=torch.from_numpy(eval_labels),
        input_dim=input_dim,
        metadata=metadata,
    )


def _binary_labels(value: Any, *, name: str) -> np.ndarray:
    labels = _as_numpy(value, name=name).reshape(-1)
    if labels.dtype == np.bool_:
        return labels.astype(np.int8)
    if not np.issubdtype(labels.dtype, np.number):
        raise ValueError(f"{name} must use boolean, {{0,1}}, or {{-1,+1}} labels")
    if not np.isfinite(labels).all():
        raise ValueError(f"{name} contains NaN or infinite values")
    unique = set(np.unique(labels).tolist())
    if unique <= {0, 1} and unique:
        return labels.astype(np.int8)
    if unique <= {-1, 1} and unique:
        return (labels > 0).astype(np.int8)
    raise ValueError(
        f"{name} must use boolean, {{0,1}}, or {{-1,+1}} labels; found {sorted(unique)}"
    )


def _tabular_preprocessed_float32(
    source_x: np.ndarray, target_x: np.ndarray
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Validate and preserve TableShift's repository-preprocessed features.

    The NPZ exporter already applies the fitted TableShift transformer.  An
    earlier version fitted an additional source+target min-max transform here,
    before the experiment's target test rows were allocated.  Besides being
    absent from the paper, that made held-out target covariates influence the
    inputs seen during training.  Preserve the exported feature values instead.
    """

    if source_x.ndim != 2 or target_x.ndim != 2:
        raise ValueError("tabular feature arrays must be two-dimensional")
    if source_x.shape[1] != target_x.shape[1]:
        raise ValueError(
            f"source/target feature dimensions differ: {source_x.shape[1]} != {target_x.shape[1]}"
        )
    source = np.ascontiguousarray(source_x, dtype=np.float32)
    target = np.ascontiguousarray(target_x, dtype=np.float32)
    if not np.isfinite(source).all() or not np.isfinite(target).all():
        raise ValueError("tabular feature arrays contain NaN or infinite values")
    return source, target, {
        "dtype": "float32",
        "upstream_transform": "tableshift_repository_preprocessing",
        "additional_scaling": "none",
        "transform_fit_rows": 0,
        "held_out_target_covariates_used_to_fit_transform": False,
    }


def _tensor_rows(array: np.ndarray, indices: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(np.ascontiguousarray(array[indices])).to(torch.float32)


def build_tabular_pu_bundle(
    source_x: Any,
    source_y: Any,
    target_x: Any,
    target_y: Any,
    *,
    target_pos: int,
    target_unl: int,
    seed: int,
    task: str = "diabetes",
    source_pos: int = PAPER_SOURCE_POS,
    source_unl: int = PAPER_SOURCE_UNL,
    val_pos: int = PAPER_VAL_POS,
    val_unl: int = PAPER_VAL_UNL,
    test_pos: int = PAPER_TEST_POS,
    test_neg: int = PAPER_TEST_NEG,
    source_metadata: Mapping[str, Any] | None = None,
) -> PUDataBundle:
    """Construct a PU bundle from explicit source- and target-domain arrays."""

    if task not in TABULAR_TASKS:
        raise ValueError(f"unsupported tabular task {task!r}; expected one of {sorted(TABULAR_TASKS)}")
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)):
        raise TypeError("seed must be an integer")
    seed = int(seed)
    counts = _validate_counts(
        source_pos=source_pos,
        source_unl=source_unl,
        target_pos=target_pos,
        target_unl=target_unl,
        val_pos=val_pos,
        val_unl=val_unl,
        test_pos=test_pos,
        test_neg=test_neg,
    )
    source_x_array, source_y_array = _validate_xy(source_x, source_y, name="source")
    target_x_array, target_y_array = _validate_xy(target_x, target_y, name="target")
    source_y_binary = _binary_labels(source_y_array, name="source_y")
    target_y_binary = _binary_labels(target_y_array, name="target_y")
    source_x_scaled, target_x_scaled, preprocessing = _tabular_preprocessed_float32(
        source_x_array, target_x_array
    )

    source_seed, target_seed = np.random.SeedSequence(seed).spawn(2)
    source_sampler = _WithoutReplacementSampler(
        source_y_binary, np.random.default_rng(source_seed), "source-domain"
    )
    target_sampler = _WithoutReplacementSampler(
        target_y_binary, np.random.default_rng(target_seed), "target-domain"
    )

    src_pos_idx = source_sampler.draw((1,), counts["source_positive"], "source-positive")
    src_unl_idx = source_sampler.draw_balanced(
        (1,), (0,), counts["source_unlabeled"], "source-unlabeled"
    )

    # Evaluation is allocated first so it remains fixed for a given seed when
    # comparing the paper's three target-PU sizes.
    test_pos_idx = target_sampler.draw((1,), counts["test_positive"], "test-positive")
    test_neg_idx = target_sampler.draw((0,), counts["test_negative"], "test-negative")
    test_indices = np.concatenate((test_pos_idx, test_neg_idx))
    test_labels = np.concatenate(
        (
            np.ones(test_pos_idx.size, dtype=np.float32),
            -np.ones(test_neg_idx.size, dtype=np.float32),
        )
    )
    test_order = target_sampler.rng.permutation(test_indices.size)
    test_indices = test_indices[test_order]
    test_labels = test_labels[test_order]

    tgt_pos_idx = target_sampler.draw((1,), counts["target_positive"], "target-positive")
    tgt_unl_idx = target_sampler.draw_balanced(
        (1,), (0,), counts["target_unlabeled"], "target-unlabeled"
    )
    val_pos_idx = target_sampler.draw((1,), counts["validation_positive"], "validation-positive")
    val_unl_idx = target_sampler.draw_balanced(
        (1,), (0,), counts["validation_unlabeled"], "validation-unlabeled"
    )

    task_info = TABULAR_TASKS[task]
    metadata: dict[str, Any] = {
        "task": task,
        "dataset": task_info["dataset"],
        "tableshift_id": task_info["tableshift_id"],
        "shift": task_info["shift"],
        "seed": seed,
        "pi_src": PAPER_CLASS_PRIOR,
        "pi_tgt": PAPER_CLASS_PRIOR,
        "pi_tr": PAPER_CLASS_PRIOR,
        "pi_te": PAPER_CLASS_PRIOR,
        "counts": counts,
        "sampling": {
            "replacement": False,
            "unlabeled_class_prior": PAPER_CLASS_PRIOR,
            "globally_disjoint_within_domain": True,
        },
        "preprocessing": preprocessing,
        "sample_indices": {
            "src_pos": _index_record("source_domain", src_pos_idx),
            "src_unl": _index_record("source_domain", src_unl_idx),
            "tgt_pos": _index_record("target_domain", tgt_pos_idx),
            "tgt_unl": _index_record("target_domain", tgt_unl_idx),
            "val_pos": _index_record("target_domain", val_pos_idx),
            "val_unl": _index_record("target_domain", val_unl_idx),
            "test_x": _index_record("target_domain", test_indices),
        },
    }
    if source_metadata:
        metadata["npz_metadata"] = dict(source_metadata)

    return PUDataBundle(
        src_pos=_tensor_rows(source_x_scaled, src_pos_idx),
        src_unl=_tensor_rows(source_x_scaled, src_unl_idx),
        tgt_pos=_tensor_rows(target_x_scaled, tgt_pos_idx),
        tgt_unl=_tensor_rows(target_x_scaled, tgt_unl_idx),
        val_pos=_tensor_rows(target_x_scaled, val_pos_idx),
        val_unl=_tensor_rows(target_x_scaled, val_unl_idx),
        test_x=_tensor_rows(target_x_scaled, test_indices),
        test_y=torch.from_numpy(test_labels),
        input_dim=int(source_x_scaled.shape[1]),
        metadata=metadata,
    )


def _decode_npz_metadata(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    array = np.asarray(value)
    if array.size != 1:
        raise ValueError("NPZ metadata must be a scalar JSON string")
    scalar = array.reshape(()).item()
    if isinstance(scalar, bytes):
        scalar = scalar.decode("utf-8")
    if not isinstance(scalar, str):
        raise ValueError("NPZ metadata must be a scalar JSON string")
    try:
        decoded = json.loads(scalar)
    except json.JSONDecodeError as error:
        raise ValueError("NPZ metadata is not valid JSON") from error
    if not isinstance(decoded, dict):
        raise ValueError("NPZ metadata JSON must contain an object")
    return decoded


def load_tabular_npz(
    path: str | Path,
    *,
    target_pos: int,
    target_unl: int,
    seed: int,
    task: str = "diabetes",
) -> PUDataBundle:
    """Load the portable TableShift export and construct paper-sized splits."""

    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"tabular NPZ not found: {path}")
    required = {"source_x", "source_y", "target_x", "target_y"}
    try:
        with np.load(path, allow_pickle=False) as archive:
            missing = required.difference(archive.files)
            if missing:
                raise ValueError(f"{path} is missing required arrays: {sorted(missing)}")
            metadata = _decode_npz_metadata(archive["metadata"] if "metadata" in archive else None)
            bundle = build_tabular_pu_bundle(
                archive["source_x"],
                archive["source_y"],
                archive["target_x"],
                archive["target_y"],
                target_pos=target_pos,
                target_unl=target_unl,
                seed=seed,
                task=task,
                source_metadata=metadata,
            )
    except (OSError, ValueError) as error:
        if isinstance(error, ValueError):
            raise
        raise ValueError(f"could not read tabular NPZ {path}: {error}") from error
    bundle.metadata["npz_path"] = str(path.resolve())
    return bundle


def _load_torchvision_raw(
    dataset: str, root: str | Path, *, download: bool
) -> tuple[Any, Any, Any, Any]:
    try:
        from torchvision.datasets import CIFAR10, FashionMNIST, MNIST
    except ImportError as error:
        raise RuntimeError(
            "torchvision is required for image tasks; install a torchvision "
            "version compatible with the installed PyTorch"
        ) from error

    constructors = {
        "mnist": MNIST,
        "fashion_mnist": FashionMNIST,
        "cifar10": CIFAR10,
    }
    constructor = constructors[dataset]
    try:
        train = constructor(root=str(root), train=True, download=download)
        test = constructor(root=str(root), train=False, download=download)
    except Exception as error:
        raise RuntimeError(
            f"failed to load torchvision {dataset} under {root}: {error}. "
            "Set download=True or place the official files in that root."
        ) from error
    return train.data, train.targets, test.data, test.targets


def _resolve_tabular_path(tabular_npz: str | Path, task: str) -> Path:
    path = Path(tabular_npz)
    if path.is_file() or path.suffix.lower() == ".npz":
        return path
    tableshift_id = TABULAR_TASKS[task]["tableshift_id"]
    candidates = (path / f"{tableshift_id}.npz", path / f"{task}.npz")
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return candidates[0]


def load_pu_dataset(
    task: str,
    data_root: str | Path,
    target_pos: int,
    target_unl: int,
    seed: int,
    tabular_npz: str | Path | None = None,
    download: bool = True,
) -> PUDataBundle:
    """Unified loader used by experiment runners.

    ``task`` is one of the exact task IDs in ``configs/paper.yaml``.  Image
    tasks load their official torchvision train/test splits.  Tabular tasks
    require ``tabular_npz`` produced by ``scripts/export_tableshift.py``; it may
    name either the file itself or its containing directory.
    """

    if not isinstance(task, str):
        raise TypeError("task must be a string")
    task = task.strip().lower()
    if task in IMAGE_TASKS:
        train_x, train_y, test_x, test_y = _load_torchvision_raw(
            IMAGE_TASKS[task].dataset, data_root, download=download
        )
        return build_image_pu_bundle(
            train_x,
            train_y,
            test_x,
            test_y,
            task=task,
            target_pos=target_pos,
            target_unl=target_unl,
            seed=seed,
        )
    if task in TABULAR_TASKS:
        if tabular_npz is None:
            expected = Path(data_root) / f"{TABULAR_TASKS[task]['tableshift_id']}.npz"
            raise ValueError(
                f"tabular task {task!r} requires tabular_npz; export it with "
                f"scripts/export_tableshift.py (expected path such as {expected})"
            )
        return load_tabular_npz(
            _resolve_tabular_path(tabular_npz, task),
            target_pos=target_pos,
            target_unl=target_unl,
            seed=seed,
            task=task,
        )
    choices = sorted((*IMAGE_TASKS.keys(), *TABULAR_TASKS.keys()))
    raise ValueError(f"unsupported task {task!r}; expected one of {choices}")


__all__ = [
    "IMAGE_TASKS",
    "PAPER_CLASS_PRIOR",
    "PAPER_SOURCE_POS",
    "PAPER_SOURCE_UNL",
    "PAPER_TEST_NEG",
    "PAPER_TEST_POS",
    "PAPER_VAL_POS",
    "PAPER_VAL_UNL",
    "PUDataBundle",
    "TABULAR_TASKS",
    "build_image_pu_bundle",
    "build_tabular_pu_bundle",
    "load_pu_dataset",
    "load_tabular_npz",
]
