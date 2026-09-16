from __future__ import annotations

import argparse
import dataclasses
import hashlib
import importlib.metadata
import itertools
import json
import math
import os
import platform
import re
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence


PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))

from iwpu.config import load_config


METHOD_ALIASES = {
    "ours": "iwpu",
    "iw-pu": "iwpu",
    "iw_pu": "iwpu",
    "iwpu": "iwpu",
    "2step": "two_step",
    "two-step": "two_step",
    "two_step": "two_step",
    "w/o iw-c": "without_importance_weight_correction",
    "wo_iw_c": "without_importance_weight_correction",
    "without_iw_c": "without_importance_weight_correction",
    "w/o cl-c": "without_classifier_loss_correction",
    "wo_cl_c": "without_classifier_loss_correction",
    "without_cl_c": "without_classifier_loss_correction",
    "w/o iwcl-c": "without_both_corrections",
    "wo_iwcl_c": "without_both_corrections",
    "without_iwcl_c": "without_both_corrections",
}
ARTIFACT_KEYS = {
    "model",
    "state_dict",
    "model_state_dict",
    "optimizer",
    "optimizer_state",
    "optimizer_state_dict",
}
VALIDATION_RISK_KEYS = (
    "best_val_risk",
    "target_validation_pu_risk",
    "target_val_pu_risk",
    "validation_pu_risk",
    "val_pu_risk",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def canonical_method(method: str, config: Mapping[str, Any] | None = None) -> str:
    """Return the config/trainer method ID, accepting the paper label ``ours``."""
    normalized = method.strip().lower()
    canonical = METHOD_ALIASES.get(normalized, normalized)
    available_ids: dict[str, Any] = {}
    if config is not None:
        available_ids.update(config.get("methods", {}))
        available_ids.update(config.get("ablations", {}))
    if config is not None and canonical not in available_ids:
        available = ", ".join(available_ids)
        raise ValueError(f"Unknown method {method!r}; available config IDs: {available}")
    return canonical


def method_definition(config: Mapping[str, Any], method: str) -> Mapping[str, Any]:
    method = canonical_method(method, config)
    if method in config.get("methods", {}):
        return config["methods"][method]
    return config["ablations"][method]


def method_execution(method: str, config: Mapping[str, Any]) -> tuple[str, bool, bool]:
    """Map a config method/ablation to trainer ID and correction switches."""
    method = canonical_method(method, config)
    if method == "two_step":
        return "two_step", True, True
    if method == "without_importance_weight_correction":
        return "iwpu", False, True
    if method == "without_classifier_loss_correction":
        return "iwpu", True, False
    if method == "without_both_corrections":
        return "iwpu", False, False
    return method, True, True


def reporting_method_id(method: str) -> str:
    """Use the existing reporting module's paper label for the proposed method."""
    return "ours" if method == "iwpu" else method


def dataset_id_for_task(config: Mapping[str, Any], task: str) -> str:
    """Resolve the model/report dataset key, including both TableShift tasks."""
    dataset = str(config["tasks"][task]["dataset"])
    return task if dataset == "tableshift" else dataset


def parse_target_size(value: str | Sequence[int]) -> tuple[int, int]:
    """Parse target positive/unlabeled counts from ``10,50``-style input."""
    if isinstance(value, str):
        cleaned = value.strip().strip("()[]")
        pieces = [part for part in re.split(r"[,xX:/\s]+", cleaned) if part]
        if len(pieces) != 2:
            raise argparse.ArgumentTypeError(
                "target size must contain positive and unlabeled counts, e.g. 10,50"
            )
        try:
            positive, unlabeled = (int(piece) for piece in pieces)
        except ValueError as exc:
            raise argparse.ArgumentTypeError("target-size counts must be integers") from exc
    else:
        if len(value) != 2:
            raise ValueError("target size must contain exactly two counts")
        positive, unlabeled = (int(item) for item in value)
    if positive <= 0 or unlabeled <= 0:
        raise argparse.ArgumentTypeError("target-size counts must be positive")
    return positive, unlabeled


def paper_target_sizes(config: Mapping[str, Any]) -> list[tuple[int, int]]:
    return [
        (int(entry["positive"]), int(entry["unlabeled"]))
        for entry in config["data_protocol"]["target_train_sizes"]
    ]


def tuning_combinations(
    config: Mapping[str, Any], method: str, task: str
) -> list[dict[str, float]]:
    """Build the paper's method-specific alpha/beta/MMD Cartesian product."""
    method = canonical_method(method, config)
    if task not in config.get("tasks", {}):
        raise ValueError(f"Unknown task {task!r}")
    if method in config.get("ablations", {}):
        # Table 2 variants use the same alpha/beta selection as the full method.
        tune = ["alpha", "beta"]
    else:
        tune = list(config["methods"][method].get("tune", []))
    if not tune:
        return [{}]

    grids = config["hyperparameters"]["grids"]
    values: list[list[float]] = []
    for name in tune:
        if name == "beta" and task == "foodstamp":
            grid = grids["beta_foodstamp"]
        elif name in {"lambda", "mmd_lambda"}:
            grid = grids["lambda_mmd"]
        else:
            grid = grids[name]
        values.append([float(item) for item in grid])

    return [
        {name: value for name, value in zip(tune, choice, strict=True)}
        for choice in itertools.product(*values)
    ]


def train_setting_tuning(combination: Mapping[str, float]) -> dict[str, float]:
    """Translate config grid keys to the exact TrainSettings field names."""
    return {
        "alpha": float(combination.get("alpha", 0.5)),
        "beta": float(combination.get("beta", 0.5)),
        "mmd_lambda": float(
            combination.get("lambda_mmd", combination.get("mmd_lambda", 1.0))
        ),
    }


def result_path_for_job(
    output_root: str | Path,
    task: str,
    method: str,
    target_pos: int,
    target_unl: int,
    seed: int,
) -> Path:
    method = METHOD_ALIASES.get(method.lower(), method.lower())
    filename = f"p{target_pos}_u{target_unl}_seed{seed}.json"
    return Path(output_root) / task / method / filename


def _jsonable(value: Any) -> Any:
    if dataclasses.is_dataclass(value):
        return _jsonable(dataclasses.asdict(value))
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if hasattr(value, "detach") and hasattr(value, "cpu"):
        detached = value.detach().cpu()
        return detached.item() if getattr(detached, "numel", lambda: 2)() == 1 else detached.tolist()
    if hasattr(value, "tolist"):
        try:
            return _jsonable(value.tolist())
        except (TypeError, ValueError):
            pass
    if hasattr(value, "item"):
        try:
            return value.item()
        except (TypeError, ValueError):
            pass
    return repr(value)


def file_sha256(path: str | Path) -> str | None:
    """Return a config-file digest without making metadata collection fatal."""

    digest = hashlib.sha256()
    try:
        with Path(path).open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
    except OSError:
        return None
    return digest.hexdigest()


def public_training_result(result: Mapping[str, Any]) -> dict[str, Any]:
    """Remove learned model/optimizer objects before serializing trainer output."""
    return {
        str(key): _jsonable(value)
        for key, value in result.items()
        if str(key).lower() not in ARTIFACT_KEYS
    }


def atomic_write_json(path: str | Path, payload: Mapping[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(_jsonable(payload), handle, indent=2, sort_keys=True)
        handle.write("\n")
    temporary.replace(path)


def environment_metadata() -> dict[str, Any]:
    packages: dict[str, str | None] = {}
    for name in ("numpy", "pandas", "scipy", "scikit-learn", "torch", "torchvision"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None

    metadata: dict[str, Any] = {
        "python": platform.python_version(),
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "working_directory": os.getcwd(),
        "packages": packages,
    }
    try:
        import torch

        metadata["torch"] = {
            "version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_version": torch.version.cuda,
            "cudnn_version": torch.backends.cudnn.version(),
            "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
            "device_name": (
                torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
            ),
            "deterministic_algorithms_enabled": (
                torch.are_deterministic_algorithms_enabled()
            ),
            "deterministic_algorithms_warn_only": (
                torch.is_deterministic_algorithms_warn_only_enabled()
                if hasattr(torch, "is_deterministic_algorithms_warn_only_enabled")
                else None
            ),
            "cudnn_deterministic": torch.backends.cudnn.deterministic,
            "cudnn_benchmark": torch.backends.cudnn.benchmark,
            "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        }
    except Exception as exc:  # Environment reporting must not mask a training error.
        metadata["torch_probe_error"] = f"{type(exc).__name__}: {exc}"
    return metadata


def validation_risk(result: Mapping[str, Any]) -> float:
    for key in VALIDATION_RISK_KEYS:
        if key in result:
            risk = float(result[key])
            if math.isfinite(risk):
                return risk
            return math.inf
    raise KeyError(
        "Trainer result has no target validation PU risk; expected one of "
        + ", ".join(VALIDATION_RISK_KEYS)
    )


def runtime_deterministic_algorithms(config: Mapping[str, Any]) -> bool:
    """Read and type-check the resolved strict-determinism switch."""

    value = config.get("runtime", {}).get("deterministic_algorithms", True)
    if not isinstance(value, bool):
        raise ValueError("runtime.deterministic_algorithms must be a boolean")
    return value


def _setting_values(
    config: Mapping[str, Any],
    bundle: Any,
    task: str,
    seed: int,
    device: str,
    combination: Mapping[str, float],
    max_epochs: int | None,
    steps_per_epoch: int | None,
    patience: int | None,
    correct_iw: bool,
    correct_cl: bool,
) -> dict[str, Any]:
    optimization = config["optimization"]
    source_batch_size = int(optimization["batch_sizes"]["source"])
    target_batch_size = int(optimization["batch_sizes"]["target"])
    if steps_per_epoch is None:
        source_unlabeled = int(config["data_protocol"]["source_train"]["unlabeled"])
        # Algorithm 1 reports one total PU batch per domain.  The trainer's
        # explicit local assumption divides it equally between P and U, so an
        # epoch covers the 5,000 source-U observations once in expectation.
        source_unlabeled_batch = source_batch_size - source_batch_size // 2
        steps_per_epoch = math.ceil(source_unlabeled / source_unlabeled_batch)

    metadata = getattr(bundle, "metadata", {}) or {}
    priors = config["data_protocol"]["class_priors"]
    tuning = train_setting_tuning(combination)
    deterministic_algorithms = runtime_deterministic_algorithms(config)
    return {
        "steps_per_epoch": int(steps_per_epoch),
        **tuning,
        "max_epochs": int(
            optimization["max_epochs"] if max_epochs is None else max_epochs
        ),
        "patience": int(
            optimization["early_stopping"]["patience"]
            if patience is None
            else patience
        ),
        "source_batch_size": source_batch_size,
        "target_batch_size": target_batch_size,
        "lr_classifier": float(
            optimization["learning_rates"]["classifier_and_features"]
        ),
        "lr_importance": float(optimization["learning_rates"]["importance_model"]),
        "device": device,
        "seed": int(seed),
        "correct_iw": bool(correct_iw),
        "correct_cl": bool(correct_cl),
        "dataset": metadata.get("dataset", dataset_id_for_task(config, task)),
        "input_dim": int(getattr(bundle, "input_dim")),
        "pi_src": float(metadata.get("pi_src", priors["source_positive"])),
        "pi_tgt": float(metadata.get("pi_tgt", priors["target_positive"])),
        "pretrain_epochs": None,
        "deterministic_algorithms": deterministic_algorithms,
    }


def run_job(
    *,
    config: Mapping[str, Any],
    config_path: str | Path,
    task: str,
    method: str,
    target_pos: int,
    target_unl: int,
    seed: int,
    device: str,
    data_root: str | Path,
    output_path: str | Path,
    tabular_npz: str | Path | None = None,
    download: bool = True,
    max_epochs: int | None = None,
    steps_per_epoch: int | None = None,
    patience: int | None = None,
) -> dict[str, Any]:
    """Train all tuning candidates for one paper cell and persist the best one."""
    method = canonical_method(method, config)
    if task not in config["tasks"]:
        raise ValueError(f"Unknown task {task!r}")
    size = (int(target_pos), int(target_unl))
    if size not in paper_target_sizes(config):
        raise ValueError(f"Target size {size} is not listed in the paper config")

    output_path = Path(output_path)
    config_path = Path(config_path).resolve()
    data_root = Path(data_root).resolve()
    resolved_tabular_npz = (
        str(Path(tabular_npz).resolve()) if tabular_npz is not None else None
    )
    deterministic_algorithms = runtime_deterministic_algorithms(config)
    # This must be set before the first CUDA/cuBLAS operation in this process.
    if deterministic_algorithms:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    started_at = utc_now()
    started_clock = time.perf_counter()
    environment = environment_metadata()
    resolved_config: dict[str, Any] = {
        "schema_version": config.get("schema_version"),
        "task_id": task,
        "task": _jsonable(config["tasks"][task]),
        "method_id": method,
        "method": _jsonable(method_definition(config, method)),
        "target_size": {
            "positive": int(target_pos),
            "unlabeled": int(target_unl),
        },
        "seed": int(seed),
        "requested_device": device,
        "data_root": str(data_root),
        "tabular_npz": resolved_tabular_npz,
        "download": bool(download),
        "runtime": {
            "deterministic_algorithms": deterministic_algorithms,
        },
        "cli_overrides": {
            "max_epochs": max_epochs,
            "steps_per_epoch": steps_per_epoch,
            "patience": patience,
        },
    }
    base_record: dict[str, Any] = {
        "status": "running",
        "config_path": str(config_path),
        "config_file_sha256": file_sha256(config_path),
        "config_snapshot": _jsonable(config),
        "resolved_config": resolved_config,
        "task": task,
        "dataset": dataset_id_for_task(config, task),
        "shift": config["tasks"][task]["shift"],
        "method": reporting_method_id(method),
        "config_method": method,
        "paper_method_label": method_definition(config, method)["paper_label"],
        "target_pos": int(target_pos),
        "target_unl": int(target_unl),
        "seed": int(seed),
        "requested_device": device,
        "data_root": str(data_root),
        "tabular_npz": resolved_tabular_npz,
        "bundle_metadata": None,
        "started_at": started_at,
        "test_accuracy": None,
        "best_config": None,
        "runtime": {},
        "environment": environment,
    }
    atomic_write_json(output_path, base_record)

    try:
        # Training imports are intentionally lazy: grid enumeration and dry runs
        # do not import PyTorch, datasets, or trainer implementations.
        from iwpu.datasets import load_pu_dataset
        from iwpu.trainers import (
            TrainSettings,
            evaluate_test_accuracy,
            train_one,
        )
        from iwpu.utils import resolve_device, seed_everything

        resolved_device = str(resolve_device(device))
        trainer_method, correct_iw, correct_cl = method_execution(method, config)
        seed_everything(seed, deterministic=deterministic_algorithms)
        # Refresh after strict determinism has actually been applied so that
        # the artifact records resolved PyTorch backend state, not only intent.
        base_record["environment"] = environment_metadata()
        bundle = load_pu_dataset(
            task,
            data_root,
            int(target_pos),
            int(target_unl),
            int(seed),
            tabular_npz=resolved_tabular_npz,
            download=download,
        )
        base_record["bundle_metadata"] = _jsonable(
            getattr(bundle, "metadata", {}) or {}
        )

        combinations = tuning_combinations(config, method, task)
        resolved_config.update(
            {
                "resolved_device": resolved_device,
                "dataset": dataset_id_for_task(config, task),
                "trainer_method": trainer_method,
                "corrections": {
                    "importance_weight": correct_iw,
                    "classifier_loss": correct_cl,
                },
                "tuning_candidate_count": len(combinations),
                "tuning_combinations": _jsonable(combinations),
            }
        )
        # Persist the exact sampled indices and resolved runtime state before
        # training, so an interrupted job still leaves a reproducible record.
        atomic_write_json(output_path, base_record)
        candidate_summaries: list[dict[str, Any]] = []
        best: tuple[float, int, Mapping[str, float], Any, Mapping[str, Any]] | None = None

        for index, combination in enumerate(combinations):
            seed_everything(seed, deterministic=deterministic_algorithms)
            values = _setting_values(
                config,
                bundle,
                task,
                seed,
                resolved_device,
                combination,
                max_epochs,
                steps_per_epoch,
                patience,
                correct_iw,
                correct_cl,
            )
            settings = TrainSettings(**values)
            candidate_started = time.perf_counter()
            try:
                result = train_one(
                    bundle,
                    trainer_method,
                    settings,
                    evaluate_test=False,
                )
                risk = validation_risk(result)
                summary = {
                    "index": index,
                    "status": "complete",
                    "tuning": dict(combination),
                    "validation_pu_risk": risk if math.isfinite(risk) else None,
                    "wall_seconds": time.perf_counter() - candidate_started,
                    "peak_cuda_memory_bytes": result.get(
                        "peak_cuda_memory_bytes"
                    ),
                }
                candidate_summaries.append(summary)
                if math.isfinite(risk) and (best is None or risk < best[0]):
                    best = (risk, index, dict(combination), settings, result)
            except Exception as exc:
                candidate_summaries.append(
                    {
                        "index": index,
                        "status": "failed",
                        "tuning": dict(combination),
                        "error": f"{type(exc).__name__}: {exc}",
                        "wall_seconds": time.perf_counter() - candidate_started,
                    }
                )

        if best is None:
            failures = sum(item["status"] == "failed" for item in candidate_summaries)
            raise RuntimeError(
                f"No tuning candidate produced a finite validation risk "
                f"({failures}/{len(candidate_summaries)} candidates failed)"
            )

        best_risk, best_index, best_tuning, best_settings, best_result = best
        if "model" not in best_result:
            raise KeyError("Best trainer result does not contain the selected model")
        # This is the only test-set metric computation in hyperparameter search.
        test_accuracy = evaluate_test_accuracy(
            best_result["model"], bundle, resolved_device
        )
        public_best_result = public_training_result(best_result)
        public_best_result["test_accuracy"] = float(test_accuracy)
        resolved_config.update(
            {
                "best_candidate_index": best_index,
                "best_tuning": _jsonable(best_tuning),
                "best_train_settings": _jsonable(
                    dataclasses.asdict(best_settings)
                ),
            }
        )

        finished_at = utc_now()
        record = {
            **base_record,
            "status": "complete",
            "resolved_device": resolved_device,
            "trainer_method": trainer_method,
            "finished_at": finished_at,
            "test_accuracy": float(test_accuracy),
            "best_validation_pu_risk": best_risk,
            "best_config": {
                "candidate_index": best_index,
                "tuning": best_tuning,
                "train_settings": dataclasses.asdict(best_settings),
            },
            "tuning": {
                "candidate_count": len(combinations),
                "selection_metric": "target_validation_empirical_pu_risk",
                "selection_mode": "min",
                "candidate_test_metrics_computed": False,
                "candidates": candidate_summaries,
            },
            "test_evaluation": {
                "performed_after_hyperparameter_selection": True,
                "candidate_index": best_index,
                "evaluation_count": 1,
            },
            "trainer_result": public_best_result,
            "runtime": {
                "wall_seconds": time.perf_counter() - started_clock,
                "candidate_wall_seconds": [
                    item["wall_seconds"] for item in candidate_summaries
                ],
            },
        }
        atomic_write_json(output_path, record)
        return record
    except Exception as exc:
        failed = {
            **base_record,
            "status": "failed",
            "finished_at": utc_now(),
            "error": {
                "type": type(exc).__name__,
                "message": str(exc),
                "traceback": traceback.format_exc(),
            },
            "runtime": {"wall_seconds": time.perf_counter() - started_clock},
        }
        atomic_write_json(output_path, failed)
        raise


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run one IWPU paper cell, tune on target validation PU risk, and save JSON."
    )
    parser.add_argument("--config", type=Path, default=PROJECT / "configs" / "paper.yaml")
    parser.add_argument("--task", required=True)
    parser.add_argument(
        "--method",
        required=True,
        help="Main method or ablation config ID; paper-style aliases are accepted.",
    )
    parser.add_argument("--target-size", type=parse_target_size)
    parser.add_argument("--target-pos", type=int)
    parser.add_argument("--target-unl", type=int)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--data-root", type=Path, default=PROJECT / "data")
    parser.add_argument("--tabular-npz", type=Path)
    parser.add_argument("--output", type=Path, default=PROJECT / "results")
    parser.add_argument(
        "--download", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--max-epochs", type=int)
    parser.add_argument("--steps-per-epoch", type=int)
    parser.add_argument("--patience", type=int)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = load_config(args.config)
    method = canonical_method(args.method, config)
    if args.target_size is not None:
        if args.target_pos is not None or args.target_unl is not None:
            raise SystemExit(
                "Use either --target-size or --target-pos/--target-unl, not both."
            )
        target_pos, target_unl = args.target_size
    else:
        if args.target_pos is None or args.target_unl is None:
            raise SystemExit(
                "Provide --target-size P,U or both --target-pos P and --target-unl U."
            )
        target_pos, target_unl = parse_target_size((args.target_pos, args.target_unl))
    output_path = (
        args.output
        if args.output.suffix.lower() == ".json"
        else result_path_for_job(
            args.output, args.task, method, target_pos, target_unl, args.seed
        )
    )
    record = run_job(
        config=config,
        config_path=args.config,
        task=args.task,
        method=method,
        target_pos=target_pos,
        target_unl=target_unl,
        seed=args.seed,
        device=args.device,
        data_root=args.data_root,
        output_path=output_path,
        tabular_npz=args.tabular_npz,
        download=args.download,
        max_epochs=args.max_epochs,
        steps_per_epoch=args.steps_per_epoch,
        patience=args.patience,
    )
    print(json.dumps({"status": record["status"], "result": str(output_path)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
