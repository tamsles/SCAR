#!/usr/bin/env python3
"""Execute one deterministic, resumable shard of the paper experiment grid.

This entry point is intentionally separate from ``scripts/run_paper_grid.py``:
the latter is useful for enumeration and small sequential runs, while a Wisteria
job must own only a bounded, deterministic subset of the 2,160 result cells.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import sys
import time
from collections import Counter
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence, TextIO


PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))
sys.path.insert(0, str(PROJECT / "scripts"))

from iwpu.config import load_config
from run_experiment import atomic_write_json, run_job, utc_now
from run_paper_grid import enumerate_jobs, parse_tabular_npz, result_status


EXIT_NEEDS_RESUME = 75


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run one interleaved Wisteria shard of the IWPU paper grid."
    )
    parser.add_argument("--config", type=Path, default=PROJECT / "configs" / "paper.yaml")
    parser.add_argument("--output", type=Path, default=PROJECT / "results" / "paper_grid")
    parser.add_argument("--data-root", type=Path, default=PROJECT / "data")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--shard-id", type=int, required=True)
    parser.add_argument("--num-shards", type=int, required=True)
    parser.add_argument(
        "--mode",
        choices=("main", "ablations", "comparison", "all"),
        default="main",
        help=(
            "main=2,160 cells; ablations=840 cells; comparison=420 paired "
            "IWPU/2step cells on the seven main tasks; all=3,000 cells."
        ),
    )
    parser.add_argument(
        "--task",
        action="append",
        default=[],
        help="Restrict to a task ID; repeat this option. The default is all tasks.",
    )
    parser.add_argument(
        "--tabular-npz",
        action="append",
        default=[],
        metavar="TASK=PATH",
        help="Preprocessed diabetes/foodstamp NPZ; repeat for both tasks.",
    )
    parser.add_argument("--force", action="store_true", help="Rerun complete cells too.")
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument(
        "--max-wall-seconds",
        type=float,
        default=0.0,
        help=(
            "Stop between cells after this many seconds and return 75. "
            "Use a value below the PJM elapsed-time limit. Zero disables the guard."
        ),
    )
    parser.add_argument("--manifest", type=Path)
    return parser


def select_grid_jobs(
    config: Mapping[str, Any],
    output: Path,
    *,
    mode: str,
    tasks: Sequence[str],
) -> list[dict[str, Any]]:
    include_ablations = mode in {"ablations", "comparison", "all"}
    jobs = enumerate_jobs(config, output, include_ablations=include_ablations)
    if mode == "ablations":
        ablation_ids = set(config.get("ablations", {}))
        jobs = [job for job in jobs if job["method"] in ablation_ids]
    elif mode == "comparison":
        comparison_methods = {"iwpu", "two_step"}
        main_tasks = {
            task
            for task, definition in config["tasks"].items()
            if definition.get("paper_scope") == "main"
        }
        jobs = [
            job
            for job in jobs
            if job["method"] in comparison_methods and job["task"] in main_tasks
        ]

    if tasks:
        unknown = sorted(set(tasks) - set(config["tasks"]))
        if unknown:
            raise ValueError(f"Unknown task IDs: {', '.join(unknown)}")
        selected_tasks = set(tasks)
        jobs = [job for job in jobs if job["task"] in selected_tasks]

    return [dict(job, grid_index=index) for index, job in enumerate(jobs)]


def lock_path_for_job(output: Path, job: Mapping[str, Any]) -> Path:
    result_name = Path(str(job["result_path"])).name
    return output / ".locks" / str(job["task"]) / str(job["method"]) / f"{result_name}.lock"


@contextmanager
def nonblocking_job_lock(path: Path) -> Iterator[tuple[bool, TextIO]]:
    """Hold a Linux advisory lock for one result cell.

    Persistent zero-byte lock files are deliberate. Removing a lock pathname
    while another process has its inode open can create a second, independent
    lock and permit duplicate training.
    """

    path.parent.mkdir(parents=True, exist_ok=True)
    handle = path.open("a+", encoding="utf-8")
    acquired = False
    try:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            acquired = True
        except BlockingIOError:
            pass
        yield acquired, handle
    finally:
        if acquired:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        handle.close()


@contextmanager
def blocking_file_lock(path: Path) -> Iterator[None]:
    """Serialize writers that share an ``atomic_write_json`` temporary path."""

    path.parent.mkdir(parents=True, exist_ok=True)
    handle = path.open("a+", encoding="utf-8")
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        yield
    finally:
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        handle.close()


def status_counts(jobs: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    return dict(Counter(result_status(job["result_path"]) for job in jobs))


def manifest_payload(
    *,
    args: argparse.Namespace,
    all_jobs: Sequence[Mapping[str, Any]],
    shard_jobs: Sequence[Mapping[str, Any]],
    started_at: str,
    started_clock: float,
    summary: Mapping[str, int],
    state: str,
) -> dict[str, Any]:
    jobs: list[dict[str, Any]] = []
    for job in shard_jobs:
        entry = dict(job)
        entry["status"] = result_status(entry["result_path"])
        jobs.append(entry)
    return {
        "schema_version": 1,
        "mode": args.mode,
        "state": state,
        "started_at": started_at,
        "updated_at": utc_now(),
        "elapsed_seconds": time.monotonic() - started_clock,
        "shard_id": args.shard_id,
        "num_shards": args.num_shards,
        "force": bool(args.force),
        "task_filter": list(args.task),
        "full_filtered_job_count": len(all_jobs),
        "shard_job_count": len(shard_jobs),
        "shard_status_counts": status_counts(shard_jobs),
        "execution_summary": dict(summary),
        "jobs": jobs,
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.num_shards <= 0:
        raise SystemExit("--num-shards must be positive")
    if not 0 <= args.shard_id < args.num_shards:
        raise SystemExit("--shard-id must satisfy 0 <= shard-id < num-shards")
    if args.max_wall_seconds < 0:
        raise SystemExit("--max-wall-seconds cannot be negative")

    config = load_config(args.config)
    all_jobs = select_grid_jobs(
        config, args.output, mode=args.mode, tasks=args.task
    )
    if not all_jobs:
        raise SystemExit("The requested mode/task filter selects no experiment cells.")
    shard_jobs = [
        job for job in all_jobs if int(job["grid_index"]) % args.num_shards == args.shard_id
    ]
    tabular_paths = parse_tabular_npz(args.tabular_npz)
    manifest = args.manifest or (
        args.output
        / "manifests"
        / args.mode
        / f"shard_{args.shard_id:03d}_of_{args.num_shards:03d}.json"
    )

    started_at = utc_now()
    started_clock = time.monotonic()
    summary: Counter[str] = Counter()
    state = "running"
    manifest_lock = manifest.with_suffix(manifest.suffix + ".lock")

    def write_manifest() -> None:
        with blocking_file_lock(manifest_lock):
            atomic_write_json(
                manifest,
                manifest_payload(
                    args=args,
                    all_jobs=all_jobs,
                    shard_jobs=shard_jobs,
                    started_at=started_at,
                    started_clock=started_clock,
                    summary=summary,
                    state=state,
                ),
            )

    write_manifest()
    try:
        for job in shard_jobs:
            if (
                args.max_wall_seconds
                and time.monotonic() - started_clock >= args.max_wall_seconds
            ):
                state = "needs_resume_wall_guard"
                summary["deferred_by_wall_guard"] += 1
                break

            path = Path(str(job["result_path"]))
            if not args.force and result_status(path) == "complete":
                summary["skipped_complete"] += 1
                continue

            lock_path = lock_path_for_job(args.output, job)
            with nonblocking_job_lock(lock_path) as (acquired, _handle):
                if not acquired:
                    summary["locked_elsewhere"] += 1
                    continue
                # A concurrent worker may have finished between the first status
                # check and this lock acquisition.
                if not args.force and result_status(path) == "complete":
                    summary["skipped_complete"] += 1
                    continue
                try:
                    run_job(
                        config=config,
                        config_path=args.config,
                        task=str(job["task"]),
                        method=str(job["method"]),
                        target_pos=int(job["target_pos"]),
                        target_unl=int(job["target_unl"]),
                        seed=int(job["seed"]),
                        device=args.device,
                        data_root=args.data_root,
                        output_path=path,
                        tabular_npz=tabular_paths.get(str(job["task"])),
                        download=False,
                    )
                    summary["completed_this_invocation"] += 1
                except Exception as exc:
                    summary["failed_this_invocation"] += 1
                    print(
                        f"FAILED {job['job_id']}: {type(exc).__name__}: {exc}",
                        file=sys.stderr,
                        flush=True,
                    )
                    if args.fail_fast:
                        state = "failed_fast"
                        raise
                finally:
                    write_manifest()
    finally:
        final_counts = status_counts(shard_jobs)
        remaining = len(shard_jobs) - final_counts.get("complete", 0)
        summary["remaining_incomplete_at_exit"] = remaining
        if state == "running":
            if summary["failed_this_invocation"]:
                state = "completed_with_failures"
            elif remaining:
                state = "needs_resume_incomplete"
            else:
                state = "complete"
        write_manifest()

    output = {
        "state": state,
        "mode": args.mode,
        "shard_id": args.shard_id,
        "num_shards": args.num_shards,
        "selected_cells": len(shard_jobs),
        "manifest": str(manifest),
        **dict(summary),
    }
    print(json.dumps(output, sort_keys=True), flush=True)
    if state in {"needs_resume_wall_guard", "needs_resume_incomplete"}:
        return EXIT_NEEDS_RESUME
    return 1 if summary["failed_this_invocation"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
