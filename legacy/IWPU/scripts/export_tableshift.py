#!/usr/bin/env python3
"""Export the two paper TableShift tasks to portable NPZ domain pools.

TableShift is an optional, comparatively heavy dependency and is therefore not
imported by the ``iwpu`` package itself.  Run this script in an environment
where TableShift is installed (or point ``--tableshift-source`` at a checkout),
then pass the resulting NPZ to ``load_pu_dataset(..., tabular_npz=...)``.

The export pools all repository-preprocessed in-domain splits as ``source_*``
and all repository-preprocessed OOD splits as ``target_*``.  The IW-PU loader
later draws its mutually disjoint PU, validation, and evaluation samples from
these pools with the paper's fixed counts and class prior.
"""

from __future__ import annotations

import argparse
import gc
import importlib.metadata
import json
import logging
import os
import platform
import subprocess
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np

try:
    import resource
except ImportError:  # pragma: no cover - only relevant when inspecting on Windows
    resource = None


ENGINEERING_COMMIT = "fca9429814703a07e3902d005d46563a207b7f0a"
TASKS = {
    "brfss_diabetes": {
        "paper_task": "diabetes",
        "expected_feature_dim": 142,
        "source_domain": "PRACE1=1 (white non-Hispanic)",
        "target_domain": "PRACE1 in {2,3,4,5,6} (other race/ethnicity groups)",
    },
    "acsfoodstamps": {
        "paper_task": "foodstamp",
        "expected_feature_dim": 239,
        "source_domain": "DIVISION != 06 (all other census divisions)",
        "target_domain": "DIVISION=06 (East South Central)",
    },
}
SOURCE_SPLITS = ("train", "validation", "id_test")
TARGET_SPLITS = ("ood_validation", "ood_test")


def _peak_rss_gib() -> float:
    """Return peak resident memory on Linux in GiB."""

    # Linux reports ru_maxrss in KiB.  The exporter is only executed on
    # Wisteria Linux nodes; keeping this local also avoids an extra dependency.
    if resource is None:
        return float("nan")
    return float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) / (1024.0**2)


def _install_memory_safe_tableshift_patches(*, transform_chunk_rows: int) -> None:
    """Remove two TableShift peak-memory paths without changing row semantics.

    The pinned TableShift implementation retains all columns from all four
    BRFSS years before selecting the model inputs.  It also transforms every
    row into one dense object array before casting to compact numeric dtypes.
    This patch selects the same input columns per year and applies the already
    fitted transformer in row chunks.  Feature fitting, split logic, values,
    column order, and final dtypes remain those of the pinned repository.
    """

    if transform_chunk_rows <= 0:
        raise ValueError("--transform-chunk-rows must be positive")

    import pandas as pd
    from tableshift.core import utils as tableshift_utils
    from tableshift.core.data_source import BRFSSDataSource
    from tableshift.core.features import Preprocessor
    from tableshift.datasets.brfss import _BRFSS_INPUT_FEATURES, align_brfss_features

    def load_brfss_reduced(self):
        frames = []
        input_features = list(_BRFSS_INPUT_FEATURES)
        for url in self.resources:
            zip_filename = tableshift_utils.basename_from_url(url)
            xpt_filename = zip_filename.replace("XPT.zip", ".XPT")
            xpt_path = os.path.join(self.cache_dir, xpt_filename)
            if not os.path.exists(xpt_path):
                import zipfile

                zip_path = os.path.join(self.cache_dir, zip_filename)
                logging.debug("unzipping %s", zip_path)
                with zipfile.ZipFile(zip_path, "r") as archive:
                    archive.extractall(self.cache_dir)
                # The CDC archive member has a trailing space in its name.
                os.rename(xpt_path + " ", xpt_path)

            frame = tableshift_utils.read_xpt(xpt_path)
            frame = align_brfss_features(frame)
            missing = [name for name in input_features if name not in frame.columns]
            if missing:
                raise RuntimeError(
                    f"{xpt_filename} lacks {len(missing)} aligned BRFSS inputs; "
                    f"first missing columns: {missing[:5]}"
                )
            # Select while only one year's full-width frame is resident.
            reduced = frame.loc[:, input_features].copy()
            frames.append(reduced)
            del frame, reduced
            gc.collect()
            print(
                f"memory-safe BRFSS read: {xpt_filename}; "
                f"peak_rss={_peak_rss_gib():.2f} GiB",
                flush=True,
            )
        return pd.concat(frames, axis=0, ignore_index=True, copy=False)

    def fit_transform_chunked(
        self,
        data,
        train_idxs,
        domain_label_colname=None,
        target_colname=None,
        passthrough_columns=None,
    ):
        logging.info("transforming columns in memory-safe chunks")
        if self.config.passthrough_columns == "all":
            logging.info("passthrough is 'all'; data will not be preprocessed")
            return data

        passthrough_columns = self.get_passthrough_columns(
            data,
            passthrough_columns,
            # The engineering pin omits this argument in fit_transform even
            # though it later expects the unsplit domain column by name.  That
            # one-line upstream bug one-hot encodes DIVISION and then raises a
            # KeyError.  Preserve the column as get_passthrough_columns was
            # explicitly designed to do, then apply the repository's label
            # encoder below.
            domain_label_colname=domain_label_colname,
            target_colname=target_colname,
        )
        dtypes_in = data.dtypes.to_dict()
        post_transform_cast_dtypes = (
            {
                column: dtypes_in[column]
                for column in passthrough_columns
                if column != domain_label_colname
            }
            if passthrough_columns
            else None
        )
        self._check_inputs(data)
        self.fit_feature_transformer(
            data,
            train_idxs,
            passthrough_columns,
        )
        columns = self.feature_transformer.get_feature_names_out()

        parts = []
        total = len(data)
        for start in range(0, total, transform_chunk_rows):
            stop = min(start + transform_chunk_rows, total)
            values = self.feature_transformer.transform(data.iloc[start:stop])
            transformed_part = pd.DataFrame(values, columns=columns)
            transformed_part = self._post_transform(
                transformed_part,
                cast_dtypes=post_transform_cast_dtypes,
            )
            parts.append(transformed_part)
            del values, transformed_part
            if start == 0 or stop == total or (start // transform_chunk_rows) % 10 == 0:
                print(
                    f"memory-safe transform: {stop}/{total} rows; "
                    f"peak_rss={_peak_rss_gib():.2f} GiB",
                    flush=True,
                )
        transformed = pd.concat(parts, axis=0, ignore_index=True, copy=False)
        del parts
        gc.collect()

        if domain_label_colname:
            transformed.loc[:, domain_label_colname] = self.fit_transform_domain_labels(
                transformed.loc[:, domain_label_colname]
            )
        self._post_transform_summary(transformed)
        logging.info("transforming columns complete.")
        return transformed

    BRFSSDataSource._load_data = load_brfss_reduced
    Preprocessor.fit_transform = fit_transform_chunked
    print(
        "installed memory-safe TableShift patches "
        f"(transform_chunk_rows={transform_chunk_rows})",
        flush=True,
    )


def _distribution_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _source_commit(source: Path | None) -> str | None:
    if source is None:
        return None
    try:
        completed = subprocess.run(
            ["git", "-C", str(source.resolve()), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return completed.stdout.strip() or None


def _import_tableshift(source: Path | None):
    if source is not None:
        source = source.expanduser().resolve()
        if not (source / "tableshift").is_dir():
            raise RuntimeError(
                f"--tableshift-source must point to a TableShift checkout; "
                f"no 'tableshift' package directory exists under {source}"
            )
        sys.path.insert(0, str(source))
    try:
        import tableshift
        from tableshift import get_dataset
    except ImportError as error:
        raise RuntimeError(
            "TableShift is not installed. Install the repository version used "
            "by this reproduction, for example:\n"
            f"  pip install 'git+https://github.com/mlfoundations/tableshift.git@{ENGINEERING_COMMIT}'\n"
            "or pass --tableshift-source /path/to/tableshift. The core IW-PU "
            "package intentionally does not require TableShift."
        ) from error
    return tableshift, get_dataset


def _available_splits(dataset: Any) -> set[str]:
    splits = getattr(dataset, "splits", None)
    if isinstance(splits, dict):
        return set(splits)
    if splits is not None:
        try:
            return set(splits)
        except TypeError:
            pass
    raise RuntimeError(
        "this TableShift dataset object does not expose its split names through "
        "'.splits'; the installed TableShift API is incompatible with this exporter"
    )


def _binary_labels(values: Any, *, task: str, split: str) -> np.ndarray:
    labels = np.asarray(values).reshape(-1)
    if labels.dtype == np.bool_:
        return labels.astype(np.int8)
    if not np.issubdtype(labels.dtype, np.number) or not np.isfinite(labels).all():
        raise RuntimeError(
            f"{task}/{split} labels are not finite binary numeric values "
            f"(dtype={labels.dtype})"
        )
    unique = set(np.unique(labels).tolist())
    if unique <= {0, 1} and unique:
        return labels.astype(np.int8)
    if unique <= {-1, 1} and unique:
        return (labels > 0).astype(np.int8)
    raise RuntimeError(
        f"{task}/{split} expected boolean, {{0,1}}, or {{-1,+1}} labels; "
        f"found {sorted(unique)}"
    )


def _numeric_features(values: Any, *, task: str, split: str) -> np.ndarray:
    try:
        array = np.asarray(values, dtype=np.float32)
    except (TypeError, ValueError) as error:
        raise RuntimeError(
            f"{task}/{split} contains non-numeric features after TableShift "
            "preprocessing; use the repository's default preprocessor"
        ) from error
    if array.ndim != 2:
        raise RuntimeError(
            f"{task}/{split} features must be a 2-D matrix, found shape {array.shape}"
        )
    if not np.isfinite(array).all():
        raise RuntimeError(
            f"{task}/{split} contains NaN or infinite features after preprocessing"
        )
    return array


def _read_split(dataset: Any, task: str, split: str):
    try:
        result = dataset.get_pandas(split)
    except Exception as error:
        raise RuntimeError(f"TableShift failed while reading {task}/{split}: {error}") from error
    if not isinstance(result, tuple) or len(result) < 2:
        raise RuntimeError(
            f"TableShift get_pandas({split!r}) returned an incompatible value; "
            "expected at least (X, y)"
        )
    frame, labels = result[:2]
    feature_names = list(getattr(frame, "columns", []))
    features = _numeric_features(frame, task=task, split=split)
    if not feature_names:
        feature_names = [f"feature_{index}" for index in range(features.shape[1])]
    if len(feature_names) != features.shape[1]:
        raise RuntimeError(
            f"{task}/{split} feature-name count does not match matrix width"
        )
    binary = _binary_labels(labels, task=task, split=split)
    if features.shape[0] != binary.shape[0]:
        raise RuntimeError(
            f"{task}/{split} feature/label length mismatch: "
            f"{features.shape[0]} != {binary.shape[0]}"
        )
    return features, binary, feature_names


def _pool_splits(dataset: Any, task: str, requested: Iterable[str]):
    available = _available_splits(dataset)
    selected = [name for name in requested if name in available]
    if not selected:
        raise RuntimeError(
            f"{task} has none of the required splits {tuple(requested)}; "
            f"available splits are {sorted(available)}"
        )
    feature_parts: list[np.ndarray] = []
    label_parts: list[np.ndarray] = []
    names: list[str] | None = None
    split_rows: dict[str, int] = {}
    for split in selected:
        features, labels, current_names = _read_split(dataset, task, split)
        if names is None:
            names = current_names
        elif current_names != names:
            raise RuntimeError(
                f"feature columns differ between TableShift splits for {task}; "
                f"mismatch first observed in {split}"
            )
        feature_parts.append(features)
        label_parts.append(labels)
        split_rows[split] = int(labels.size)
    assert names is not None
    return (
        np.concatenate(feature_parts, axis=0),
        np.concatenate(label_parts, axis=0),
        names,
        split_rows,
    )


def _domain_metadata(dataset: Any) -> dict[str, Any]:
    splitter = getattr(dataset, "splitter", None)
    return {
        "domain_split_variable": getattr(dataset, "domain_split_varname", None),
        "domain_split_id_values": getattr(splitter, "domain_split_id_values", None),
        "domain_split_ood_values": getattr(splitter, "domain_split_ood_values", None),
        "domain_split_threshold": getattr(splitter, "domain_split_gt_thresh", None),
    }


def _json_default(value: Any):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"cannot encode {type(value).__name__} in metadata JSON")


def export_task(
    task: str,
    *,
    output_dir: Path,
    cache_dir: Path,
    use_cached: bool,
    strict_feature_dim: bool,
    tableshift_module: Any,
    get_dataset: Any,
    transform_chunk_rows: int,
    tableshift_source: Path | None = None,
) -> Path:
    if task not in TASKS:
        raise ValueError(f"unsupported task {task!r}; expected one of {sorted(TASKS)}")
    try:
        dataset = get_dataset(task, cache_dir=str(cache_dir), use_cached=use_cached)
    except Exception as error:
        raise RuntimeError(
            f"could not initialize TableShift task {task!r}: {error}. "
            "If raw public data are unavailable, check network access and the "
            "cache directory; if using a prepared cache, pass --use-cached."
        ) from error

    source_x, source_y, source_names, source_rows = _pool_splits(
        dataset, task, SOURCE_SPLITS
    )
    target_x, target_y, target_names, target_rows = _pool_splits(
        dataset, task, TARGET_SPLITS
    )
    if source_names != target_names:
        raise RuntimeError(f"source and target feature columns differ for {task}")
    expected_dim = int(TASKS[task]["expected_feature_dim"])
    actual_dim = int(source_x.shape[1])
    feature_dim_matches = actual_dim == expected_dim
    if not feature_dim_matches:
        message = (
            f"{task} has {actual_dim} processed features, while the paper reports "
            f"{expected_dim}. This normally indicates a TableShift version or "
            "preprocessing mismatch."
        )
        if strict_feature_dim:
            raise RuntimeError(message)
        warnings.warn(message, RuntimeWarning, stacklevel=2)

    metadata = {
        "schema_version": 1,
        "exporter": "scripts/export_tableshift.py",
        "exported_at_utc": datetime.now(timezone.utc).isoformat(),
        "task": task,
        "paper_task": TASKS[task]["paper_task"],
        "source_domain_description": TASKS[task]["source_domain"],
        "target_domain_description": TASKS[task]["target_domain"],
        "source_splits": source_rows,
        "target_splits": target_rows,
        "source_rows": int(source_y.size),
        "target_rows": int(target_y.size),
        "source_positive_rows": int(source_y.sum()),
        "target_positive_rows": int(target_y.sum()),
        "feature_names": source_names,
        "feature_dimension": actual_dim,
        "paper_reported_feature_dimension": expected_dim,
        "feature_dimension_matches_paper": feature_dim_matches,
        "tableshift_version": _distribution_version("tableshift"),
        "tableshift_module_file": str(Path(tableshift_module.__file__).resolve()),
        "tableshift_engineering_commit_expected": ENGINEERING_COMMIT,
        "tableshift_source_commit_actual": (
            os.environ.get("IWPU_TABLESHIFT_SOURCE_COMMIT")
            or _source_commit(tableshift_source)
        ),
        "tableshift_exact_paper_commit": None,
        "tableshift_exact_paper_commit_status": "not_reported_by_paper",
        "tableshift_container_reference": os.environ.get(
            "IWPU_TABLESHIFT_CONTAINER_REFERENCE"
        ),
        "tableshift_container_sha256": os.environ.get(
            "IWPU_TABLESHIFT_CONTAINER_SHA256"
        ),
        "numpy_version": np.__version__,
        "pandas_version": _distribution_version("pandas"),
        "scikit_learn_version": _distribution_version("scikit-learn"),
        "python_version": platform.python_version(),
        "repository_preprocessing_used": True,
        "memory_safe_tableshift_patches": True,
        "tableshift_domain_label_passthrough_fix": True,
        "transform_chunk_rows": transform_chunk_rows,
        "export_peak_rss_gib_before_write": _peak_rss_gib(),
        **_domain_metadata(dataset),
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    destination = output_dir / f"{task}.npz"
    temporary = destination.with_suffix(".npz.tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(
            handle,
            source_x=source_x.astype(np.float32, copy=False),
            source_y=source_y.astype(np.int8, copy=False),
            target_x=target_x.astype(np.float32, copy=False),
            target_y=target_y.astype(np.int8, copy=False),
            metadata=np.asarray(
                json.dumps(metadata, sort_keys=True, default=_json_default)
            ),
        )
    temporary.replace(destination)
    return destination


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--task",
        nargs="+",
        choices=sorted((*TASKS.keys(), "all")),
        default=["all"],
        help="TableShift task(s) to export (default: both)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/tableshift"),
        help="directory for brfss_diabetes.npz and/or acsfoodstamps.npz",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path("data/tableshift_cache"),
        help="TableShift raw/preprocessed cache directory",
    )
    parser.add_argument(
        "--tableshift-source",
        type=Path,
        default=None,
        help="optional path to a TableShift source checkout",
    )
    parser.add_argument(
        "--use-cached",
        action="store_true",
        help="ask TableShift to load a previously prepared sharded cache",
    )
    parser.add_argument(
        "--strict-feature-dim",
        action="store_true",
        help="fail if processed width differs from the paper's 142/239",
    )
    parser.add_argument(
        "--transform-chunk-rows",
        type=int,
        default=20_000,
        help="rows per TableShift dense transform block (default: 20000)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        tableshift_module, get_dataset = _import_tableshift(args.tableshift_source)
        _install_memory_safe_tableshift_patches(
            transform_chunk_rows=args.transform_chunk_rows
        )
        selected = list(TASKS) if "all" in args.task else list(dict.fromkeys(args.task))
        for task in selected:
            destination = export_task(
                task,
                output_dir=args.output_dir,
                cache_dir=args.cache_dir,
                use_cached=args.use_cached,
                strict_feature_dim=args.strict_feature_dim,
                tableshift_module=tableshift_module,
                get_dataset=get_dataset,
                transform_chunk_rows=args.transform_chunk_rows,
                tableshift_source=args.tableshift_source,
            )
            print(f"exported {task}: {destination}")
    except (RuntimeError, ValueError, OSError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
