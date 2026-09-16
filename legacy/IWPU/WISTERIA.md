# Wisteria deployment

This project is deployed to Wisteria/BDEC-01 Aquarius. Training and test
execution must happen there, not on the local Windows machine.

Verified deployment coordinates:

- login: `j23000@wisteria.cc.u-tokyo.ac.jp`
- project: `gu15`
- remote root: `/work/04/gu15/j23000/iwpu_reproduction`
- GPU system: Aquarius (NVIDIA A100)

The Wisteria system Python and packaged PyTorch are too old for this project.
`wisteria/bootstrap_env.sh` installs an isolated Python 3.10 environment below
the remote project directory and installs PyTorch 2.2.2 with CUDA 11.8. It does
not change the system installation.

## Bootstrap on the login node

```bash
ssh -i ~/.ssh/id_ed25519 j23000@wisteria.cc.u-tokyo.ac.jp
cd /work/04/gu15/j23000/iwpu_reproduction
bash wisteria/bootstrap_env.sh
```

The bootstrap reuses an existing MNIST copy already present under this account
by creating a symlink at `data/MNIST`; compute nodes therefore do not need
outbound internet access for the smoke run.

## Prefetch all image data on the login node

Formal jobs are deliberately run with `download=False`. Fetch and verify
MNIST, Fashion-MNIST, and CIFAR-10 once from the login node:

```bash
cd /work/04/gu15/j23000/iwpu_reproduction
bash wisteria/prefetch_image_data.sh
```

The script serializes concurrent prefetch attempts, resumes an interrupted
CIFAR-10 archive, verifies its official MD5 checksum, verifies the official
train/test lengths, reopens every dataset with downloads disabled, and writes
`data/.image_data_ready.json`. It refuses to run inside a PJM batch job. No
dataset is downloaded to the local Windows machine.

## Prepare TableShift data

TableShift is isolated from the image environment. On a Wisteria login node,
bootstrap the pinned source checkout and the official TableShift container:

```bash
cd /work/04/gu15/j23000/iwpu_reproduction
bash wisteria/bootstrap_tabular_env.sh
bash wisteria/build_tableshift_sandbox.sh
pjsub wisteria/job_prefetch_acs.pjm.sh
```

The second command expands the read-only SIF once on the login node. The third
prefetches the ACS household files on a compute node with bounded retries,
backoff, and ZIP
validation because the legacy Folktables downloader accepts HTTP error pages as
archives. Aquarius
can then run that directory image with an unprivileged user namespace. Submit
the public-data export to a 56 GiB `share-short` compute allocation:

```bash
pjsub wisteria/job_export_tableshift.pjm.sh
```

Monitor `pjstat` and `logs/tableshift_export_<job-id>.log`, then wait for
`data/tableshift/brfss_diabetes.npz.sha256` and
`data/tableshift/acsfoodstamps.npz.sha256`. The exporter records the pinned
TableShift commit and container digest in each NPZ metadata payload. Its
memory-safe compatibility patch preserves repository preprocessing semantics
while selecting BRFSS input columns one year at a time and applying the fitted
dense transformer in 20,000-row chunks. It also routes the domain-label column
through the pinned preprocessor's existing passthrough/label-encode path; the
engineering pin otherwise one-hot encodes `DIVISION` and then raises a
`KeyError` when it looks up that original name. Do not run this export on the
24 GiB login node.

## Submit the remote-only smoke check

```bash
cd /work/04/gu15/j23000/iwpu_reproduction
pjsub -g gu15 wisteria/job_smoke.pjm.sh
pjstat
```

The smoke job requests one Aquarius GPU in `share-debug`, refuses CPU fallback,
runs the test suite, then runs the `mnist_io`/IWPU cell for seed 0 and target
size `(10, 50)` with 2 epochs x 2 steps per hyperparameter candidate. Logs are
written to `logs/`; results are written to `results/smoke/`.

Do not launch the full grid directly from the login node. The default grid has
2,160 result cells and about 16,440 hyperparameter-candidate trainings before
ablations. Submit bounded batches after checking the smoke result and the
remaining `gu15` allocation.

## Formal grid: interleaved PJM shards

Wisteria's user guide lists bulk-job use as prohibited on Aquarius, and the
published `pjstat --limit` example shows `BULK_ACCEPT`/`BULK_RUN` as `-`.
Therefore the dispatcher does
not use `pjsub --bulk`; it submits a small, bounded set of ordinary PJM jobs and
passes each shard index with `pjsub -x`. This also avoids placing 2,160 cells in
one short job. See the [official Wisteria job FAQ](https://www.cc.u-tokyo.ac.jp/en/faq/wisteria.php)
and [Aquarius queue limits](https://www.cc.u-tokyo.ac.jp/supercomputer/wisteria/service/job.php).

The default layout is eight interleaved shards. Interleaving by global grid
index distributes every task, method, target size, and seed across workers more
evenly than contiguous task blocks.

| Mode | Total cells | Cells per shard with 8 shards |
|---|---:|---:|
| `main` | 2,160 | 270 |
| `ablations` | 840 | 105 |
| `all` | 3,000 | 375 |

The tabular NPZ files are prepared by the separate TableShift workflow. Preview
the exact `pjsub` commands first; dry-run is the default. The paths below are
also the dispatcher defaults, but are shown explicitly for auditability:

```bash
cd /work/04/gu15/j23000/iwpu_reproduction
bash wisteria/submit_grid.sh \
  --mode main \
  --diabetes-npz /work/04/gu15/j23000/iwpu_reproduction/data/tableshift/brfss_diabetes.npz \
  --foodstamp-npz /work/04/gu15/j23000/iwpu_reproduction/data/tableshift/acsfoodstamps.npz
```

After the smoke job, image prefetch, TableShift export, remaining-token check,
and `pjstat --limit` check all pass, the same command can be submitted by adding
`--submit`. The dispatcher changes to `REMOTE_ROOT` before every `pjsub`, so the
worker can reliably resolve the project through `PJM_O_WORKDIR`.

For a lower-risk staged wave, cap scheduler time and leave cleanup margin, for
example `--elapse 12:00:00 --max-wall-seconds 41400`. Shards are resumable and
skip completed cells when submitted again.

If the project has fewer than eight free submission slots, submit a wave with,
for example, `--range 0-3 --submit`, then submit the remaining range later.
The dispatcher enforces an eight-job cap per invocation by default; a larger
partition can still be staged with `--range`. Raising the per-invocation cap
requires an explicit `IWPU_SUBMISSION_CAP=N` after checking live group limits.
The cap is not a scheduler replacement: do not start another wave until
`pjstat` confirms that its jobs fit alongside all other `gu15` work.

Each shard requests one Aquarius GPU in `share` for at most 48 hours. The worker
stops between cells after 47 hours, leaving scheduler margin; exit code 75 means
the same shard needs another pass. One default eight-shard wave therefore
requests at most eight GPUs and has a worst-case reservation ceiling
of 384 GPU-hours per full-duration wave. The published `share` allocation is
also 9 CPU cores and 56 GiB per GPU, so one eight-job wave is bounded at 72 CPU
cores and 448 GiB in aggregate. Actual availability can be lower when other
`gu15` jobs are active.

## Resume and force behavior

Result JSON files are the source of truth. Re-run the same dispatcher command
without `--force` to skip `complete` cells and retry missing, `running`, failed,
or invalid cells. Per-cell advisory locks prevent duplicate training if the
same shard is accidentally active twice. Shard progress manifests are written
under `results/paper_grid/manifests/`, and scheduler logs/status files are under
`logs/`.

Use `--force` only when complete cells must intentionally be recomputed. A cell
is atomically marked `running` before training, so interruption during a forced
rerun remains recoverable by a later non-force pass.
