from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence


PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))

from iwpu.config import load_config
from run_experiment import (
    atomic_write_json,
    canonical_method,
    paper_target_sizes,
    result_path_for_job,
    run_job,
    utc_now,
)


def enumerate_jobs(
    config: Mapping[str, Any],
    output_root: str | Path,
    *,
    include_ablations: bool = False,
) -> list[dict[str, Any]]:
    """Enumerate the 2160-cell main grid and, optionally, Table 2 ablations."""
    jobs: list[dict[str, Any]] = []
    sizes = paper_target_sizes(config)
    seeds = [int(seed) for seed in config["repetitions"]["seeds"]]
    for task in config["tasks"]:
        methods = list(config["methods"])
        if include_ablations and config["tasks"][task].get("paper_scope") == "main":
            methods.extend(config.get("ablations", {}))
        for method in methods:
            canonical = canonical_method(method, config)
            for target_pos, target_unl in sizes:
                for seed in seeds:
                    path = result_path_for_job(
                        output_root, task, canonical, target_pos, target_unl, seed
                    )
                    jobs.append(
                        {
                            "job_id": (
                                f"{task}__{canonical}__p{target_pos}_u{target_unl}__seed{seed}"
                            ),
                            "task": task,
                            "method": canonical,
                            "target_pos": target_pos,
                            "target_unl": target_unl,
                            "seed": seed,
                            "result_path": str(path),
                        }
                    )
    return jobs


def select_smoke_jobs(jobs: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    selected = [
        dict(job)
        for job in jobs
        if job["task"] == "mnist_io"
        and job["method"] == "iwpu"
        and job["target_pos"] == 10
        and job["target_unl"] == 50
        and job["seed"] == 0
    ]
    if len(selected) != 1:
        raise ValueError(f"Expected exactly one smoke job, found {len(selected)}")
    return selected


def result_status(path: str | Path) -> str:
    path = Path(path)
    if not path.exists():
        return "pending"
    try:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        return str(payload.get("status", "unknown"))
    except (OSError, json.JSONDecodeError):
        return "invalid"


def manifest_payload(
    jobs: Sequence[Mapping[str, Any]], *, mode: str, full_job_count: int
) -> dict[str, Any]:
    manifested: list[dict[str, Any]] = []
    for job in jobs:
        entry = dict(job)
        entry["status"] = result_status(entry["result_path"])
        manifested.append(entry)
    return {
        "schema_version": 1,
        "created_at": utc_now(),
        "mode": mode,
        "full_job_count": int(full_job_count),
        "job_count": len(manifested),
        "status_counts": dict(Counter(job["status"] for job in manifested)),
        "jobs": manifested,
    }


def parse_tabular_npz(values: Sequence[str]) -> dict[str, Path]:
    mapping: dict[str, Path] = {}
    for value in values:
        if "=" not in value:
            raise argparse.ArgumentTypeError(
                "--tabular-npz must be TASK=PATH, e.g. diabetes=data/diabetes.npz"
            )
        task, raw_path = value.split("=", 1)
        task = task.strip()
        if task not in {"diabetes", "foodstamp"}:
            raise argparse.ArgumentTypeError(
                "--tabular-npz task must be diabetes or foodstamp"
            )
        mapping[task] = Path(raw_path)
    return mapping


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Enumerate or sequentially execute the complete IWPU paper grid."
    )
    parser.add_argument("--config", type=Path, default=PROJECT / "configs" / "paper.yaml")
    parser.add_argument("--output", type=Path, default=PROJECT / "results")
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--data-root", type=Path, default=PROJECT / "data")
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--tabular-npz",
        action="append",
        default=[],
        metavar="TASK=PATH",
        help="Optional preprocessed NPZ path; repeat for diabetes and foodstamp.",
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--execute", action="store_true", help="Run pending jobs sequentially.")
    mode.add_argument(
        "--dry-run",
        action="store_true",
        help="Write the manifest only (this is the default).",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Execute only mnist_io/iwpu/(10,50)/seed0 for 2 epochs x 2 steps.",
    )
    parser.add_argument(
        "--include-ablations",
        action="store_true",
        help=(
            "Add the four Table 2 variants for the seven main tasks. "
            "The default main manifest remains 2160 jobs."
        ),
    )
    parser.add_argument("--force", action="store_true", help="Rerun completed jobs.")
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument(
        "--download", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--max-epochs", type=int)
    parser.add_argument("--steps-per-epoch", type=int)
    parser.add_argument("--patience", type=int)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.smoke and args.dry_run:
        parser.error("--smoke executes its selected job and cannot be combined with --dry-run")

    config = load_config(args.config)
    main_jobs = enumerate_jobs(config, args.output)
    full_jobs = enumerate_jobs(
        config, args.output, include_ablations=args.include_ablations
    )
    jobs = select_smoke_jobs(full_jobs) if args.smoke else full_jobs
    execute = bool(args.execute or args.smoke)
    mode = "smoke" if args.smoke else ("execute" if execute else "dry-run")
    manifest_path = args.manifest or args.output / "grid_manifest.json"

    manifest = manifest_payload(jobs, mode=mode, full_job_count=len(full_jobs))
    manifest["main_job_count"] = len(main_jobs)
    manifest["include_ablations"] = bool(args.include_ablations)
    atomic_write_json(manifest_path, manifest)
    if not execute:
        print(
            json.dumps(
                {
                    "mode": "dry-run",
                    "jobs": len(jobs),
                    "manifest": str(manifest_path),
                }
            )
        )
        return 0

    tabular_paths = parse_tabular_npz(args.tabular_npz)
    completed = skipped = failed = 0
    try:
        for job in jobs:
            path = Path(job["result_path"])
            if not args.force and result_status(path) == "complete":
                skipped += 1
                continue

            max_epochs = 2 if args.smoke else args.max_epochs
            steps_per_epoch = 2 if args.smoke else args.steps_per_epoch
            patience = 2 if args.smoke and args.patience is None else args.patience
            try:
                run_job(
                    config=config,
                    config_path=args.config,
                    task=job["task"],
                    method=job["method"],
                    target_pos=job["target_pos"],
                    target_unl=job["target_unl"],
                    seed=job["seed"],
                    device=args.device,
                    data_root=args.data_root,
                    output_path=path,
                    tabular_npz=tabular_paths.get(job["task"]),
                    download=args.download,
                    max_epochs=max_epochs,
                    steps_per_epoch=steps_per_epoch,
                    patience=patience,
                )
                completed += 1
            except Exception as exc:
                failed += 1
                print(f"FAILED {job['job_id']}: {type(exc).__name__}: {exc}", file=sys.stderr)
                if args.fail_fast:
                    raise
    finally:
        final_manifest = manifest_payload(jobs, mode=mode, full_job_count=len(full_jobs))
        final_manifest["main_job_count"] = len(main_jobs)
        final_manifest["include_ablations"] = bool(args.include_ablations)
        final_manifest["execution_summary"] = {
            "completed_this_invocation": completed,
            "skipped_complete": skipped,
            "failed_this_invocation": failed,
        }
        atomic_write_json(manifest_path, final_manifest)

    print(
        json.dumps(
            {
                "mode": mode,
                "completed": completed,
                "skipped": skipped,
                "failed": failed,
                "manifest": str(manifest_path),
            }
        )
    )
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
