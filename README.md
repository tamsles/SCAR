# SCAR: Complementary-Label Learning under Distribution Shift

An archive of experiments on complementary-label learning under source/target
distribution shift, including SCARCE baselines, dynamic importance weighting,
conditional density-ratio estimation, and matched diagnostic experiments.
Here, SCAR means **Selected Completely At Random**: a complementary label
indicates a class that an example does **not** belong to.

The repository contains the current research code and an earlier SCARCE working
snapshot in [`legacy/SCARCE`](legacy/SCARCE). It preserves experimental methods
and findings, including negative results; it does not claim that the proposed
conditional estimators consistently outperform the baselines.

## Start here

- **Run a small synthetic example:** follow [Installation and quick start](#installation-and-quick-start).
- **Explore the experimental stages:** see [Experiment map](#experiment-map).
- **Read the results:** begin with the [complete experiment inventory](reports/SCAR_CLL_all_experiments_EN.md)
  and [Phase B synthesis](reports/phase_b_summary.md).
- **Run original SCARCE, DIW or ADIW:** use the [legacy README](legacy/SCARCE/README.md).
- **Check what was archived:** see [EXPERIMENT_ARCHIVE.md](EXPERIMENT_ARCHIVE.md).

## Installation and quick start

Clone the repository using a GitHub account with access to it:

```bash
git clone https://github.com/tamsles/SCAR.git
cd SCAR
python -m venv .venv
```

Activate the environment on Linux/macOS:

```bash
source .venv/bin/activate
```

Or on Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

Install the core dependencies and run the synthetic example on CPU:

```bash
python -m pip install -r requirements.txt
python train.py --config configs/ratio_synthetic.json --device cpu
```

The core requirements are PyTorch, NumPy and Matplotlib. The image-data bridge
and legacy experiments additionally use torchvision, SciPy and pandas:

```bash
python -m pip install torchvision scipy pandas
```

Use mutually compatible PyTorch/torchvision builds for your Python and CUDA
environment. The repository records minimum dependency versions, not a locked
environment; this archive has not been validated against every newer version.
CVXOPT is optional for the legacy DIW solver; its `auto` setting can fall back
to SciPy. Synthetic examples require no downloaded image dataset. Image
experiments use the legacy data loaders and may download datasets on first use.

Unless stated otherwise, run commands from the repository root. Multiline
commands below use Bash continuation syntax; on PowerShell, put each command
on one line or use PowerShell's continuation syntax.

## Repository layout

| Path | Purpose |
| --- | --- |
| `train.py`, `demo_synthetic.py` | Density-ratio training entry point and synthetic demo |
| `models/`, `ratio_estimation/` | Backbones, ratio architectures, training and risk integration |
| `datasets/`, `src/` | Synthetic data, calibration, diagnostics and weight interventions |
| `experiments/`, `scripts/` | Experiment runners, analysis and report generation |
| `configs/` | Synthetic, next-round and Phase B experiment settings |
| `tests/` | Current project's unit and experiment tests |
| `wisteria/` | Historical cluster submission and monitoring scripts |
| `reports/`, `ABLATION_RESULTS.md` | Archived findings and experiment inventories |
| `legacy/SCARCE/` | Earlier SCARCE, domain-shift baselines, DIW and ADIW implementation |

## Experiment map

| Stage | Question or comparison | Main entry point |
| --- | --- | --- |
| Legacy baselines | Original SCARCE, source-only, target-only, pooled SCAR, DIW and ADIW | [Legacy README](legacy/SCARCE/README.md) |
| Synthetic ratio ablations | Conditioning, architecture, prior correction and branch balancing | `scripts/run_ratio_ablation.py`, `experiments/ablation.py` |
| SCARCE bridge | Neural importance weights in the original classifier training path | `experiments/scarce_ratio_comparison.py` |
| Matched next round / Phase A | Matched training conditions, capacity and ratio diagnostics | `scripts/run_next_round.py` |
| B0 | ADIW pipeline and sample-weight causal interventions | `scripts/run_phase_b0.py` |
| B1 | Ratio quality and residual source/target distinguishability | `scripts/run_phase_b1.py` |
| B2 | Frozen-score clipping, normalization and calibration | `scripts/run_phase_b2.py` |
| B3 | Gaussian benchmark with known ground-truth ratios | `scripts/run_phase_b3.py` |
| B4 | Heterogeneous class-specific MNIST shifts | `scripts/run_phase_b4.py` |

For bridge commands, use `--scarce-repo legacy/SCARCE`. For the Bash cluster
launchers, set the bundled legacy path before running the commands below:

```bash
export SCARCE_REPO="$(pwd)/legacy/SCARCE"
```

Review `wisteria/config.sh` and the job templates before submission: project
accounts, module names, resource limits and default paths reflect the original
cluster environment and may need adjustment.

## Results and interpretation

The [complete inventory](reports/SCAR_CLL_all_experiments_EN.md) separates
experimental batches, completed runs and missing comparisons. The historical
name `ftSCAR` refers to pooled source/target training in these experiments;
it must not be interpreted as source pretraining followed by target fine-tuning.

The [Phase B report](reports/phase_b_summary.md) records 637 formal runs across
B0–B4. Its findings include a substantial contribution from the matched ADIW
training pipeline, saturation and clipping problems in the learned ratios, and
no practical conditional-estimator advantage in the tested heterogeneous MNIST
settings. These are archived, setting-specific findings, not newly rerun or
universally applicable conclusions. See the individual reports for uncertainty,
evaluation protocols and oracle-only diagnostics.

## Reproducibility and archive scope

Source code, configurations, tests, cluster scripts and written reports are
included. Downloaded datasets, model checkpoints, raw result directories and
runtime logs are excluded. Reproducing a result requires rerunning its matching
configuration and seeds or supplying the original raw artifacts; the written
reports alone are insufficient to regenerate every table.

Some historical reports and PowerShell report builders retain machine-specific
paths. The legacy Colab instructions refer to a notebook and ZIP bundle that are
not included here. The legacy code snapshot comes from
[wwangwitsel/SCARCE](https://github.com/wwangwitsel/SCARCE) with local experiment
extensions; its DIW path references
[TongtongFANG/DIW](https://github.com/TongtongFANG/DIW). Original documentation is
retained in the legacy directory.

The upload and README update do not constitute a fresh execution of the training
experiments. Run the [tests](#tests) and a smoke experiment in your environment
before launching a full matrix.

## Conditional density-ratio estimator

This repository implements a PyTorch density-ratio estimator for multiclass
complementary-label learning under source/target distribution shift. It trains
the `2q` ratios
`p_target(x | bar_y[k]=b) / p_source(x | bar_y[k]=b)` directly, using only
`x`, the binary complementary-label vector, and the source/target domain.
Ordinary labels are not accepted by the ratio-training API.

Four comparable architectures are available through `ratio_arch`:

- `unified`: the original class/state-conditioned shared estimator.
- `separate`: `2q` private backbones, encoders, heads, optimizer states, and
  warm-start parameter states.
- `multihead`: one shared backbone/ratio encoder and `2q` scalar heads.
- `fusion`: one global ratio, `2q` conditional heads, and `2q` learned
  sample-dependent gates combined geometrically in log space.

## Data and model flow

Source and target loaders return `(x, bar_y)`:

- `x`: `[B, ...]`, image or vector input.
- `bar_y`: `[B, q]`, binary complementary-label indicators.

`LabelConditionedRatioEstimator` runs a caller-supplied backbone once:

```text
x -> shared backbone -> h [B,F]
                       + class embedding [q,C]
                       + state embedding [2,S]
                       -> shared ratio head -> logits [B,q,2]
```

`forward_selected(x, bar_y)` gathers state `bar_y[i,k]` and returns `[B,q]`.
The `multihead` alternative still shares one backbone but directly emits
`[B,q,2]`.

The domain classifier uses source label `d=0` and target label `d=1`. For each
branch,

```text
log r_{k,z}(x) = logit_{k,z}(x) + log(pi_source(k,z)/pi_target(k,z))
r_{k,z}(x)     = exp(clamp(log r_{k,z}(x)))
```

The implementation clips log-ratios and ratios, performs the exponential in
float32, and replaces non-finite results safely. With genuinely balanced
source/target sampling, set `balanced_domain_sampling` to `true`, making the
prior correction exactly one.

## Train

Run the synthetic example:

```bash
python train.py --config configs/ratio_synthetic.json --device auto
```

Run each direct-conditional architecture with the same configuration/data seed:

```bash
python train.py --config configs/conditional_synthetic.json ratio_arch=separate
python train.py --config configs/conditional_synthetic.json ratio_arch=multihead
python train.py --config configs/conditional_synthetic.json ratio_arch=fusion
```

The fair multi-seed architecture/no-IW/oracle comparison is:

```bash
python scripts/run_ratio_ablation.py \
  --config configs/conditional_synthetic.json \
  --output-dir results/conditional_ratio_ablation
```

Or the short demo:

```bash
python demo_synthetic.py
```

The JSON file configures `num_classes`, `feature_dim`, embedding dimensions,
head dimensions, `conditioned|multihead`, branch balancing, support threshold,
weight bounds, prior behavior, ratio clipping, self-normalization, joint
training guard, optimizer, data, and runtime settings.

To reuse an image or project-specific backbone:

```python
from train import build_estimator

backbone = ExistingBackbone(...)  # must return [B, config["feature_dim"]]
ratio_estimator = build_estimator(config, backbone=backbone)
```

If an existing loader returns dictionaries, the trainer supports configurable
`x_key` and `bar_y_key`. Other formats can be adapted in
`ratio_estimation/data.py` without changing the model.

## Importance weights and downstream risk

```python
weights = ratio_estimator.compute_importance_weights(
    x,
    bar_y,
    detach=True,
    clip_min=None,
    clip_max=None,
    self_normalize=False,
)  # [B,q]

classwise_loss = complementary_loss(classifier(x), bar_y)  # [B,q]
final_loss = (weights * classwise_loss).mean()
```

Each class uses its own `r[k,bar_y_k](x)`. Do not reduce weights to `[B]`
before multiplying class-wise loss. `weighted_classwise_risk` provides this
integration and detaches ratios by default. Even if `joint_training=True`, it
requires explicit `allow_ratio_gradients=True` before classifier gradients can
flow into ratios, guarding against degenerate weight shrinkage.

Optional self-normalization uses `scatter_add` to divide each observed
`(k,z)` branch by its own batch mean. It is disabled by default because it
changes the raw importance-weighted objective.

The new conditional experiment configuration enables it by default. Processing
is finite-check, clipping, then independent `(k,b)` source-branch
normalization. It records mean/std/min/max, 5/50/95% quantiles, clipping rate,
ESS, source/target counts, branch objective, and skipped updates. A branch with
fewer than `ratio.min_condition_samples` observations in either domain is
skipped for that iteration. Its parameters remain warm-started; until the first
valid update, classifier weighting falls back to one.

## Branch balancing

- `none`: average all supported selected entries.
- `inverse_frequency`: inverse branch-frequency weights with
  `max_branch_weight` clipping.
- `balanced_bce` (default): average source and target BCE within each branch,
  then average supported branches.

`compute_branch_statistics` returns source/target `[q,2]` counts, configures
prior correction, warns about insufficient support, and excludes unsupported
branches from loss. It never builds branch-specific datasets.

Because `balanced_bce` normalizes source and target separately inside every
branch, its effective discriminator prior is 1:1 and the correction is
automatically one. Count-based prior correction is used for unbalanced
`none`/`inverse_frequency` training unless balanced sampling is declared.

## Tests

```bash
python -m unittest discover -s tests -v
```

Run the legacy test suite separately from its own directory after installing
the additional dependencies:

```bash
cd legacy/SCARCE
python -m unittest discover -s tests -v
cd ../..
```

The two projects use overlapping top-level module names, so keep their test
discovery roots separate. The coverage described below refers to the current
project's suite.

Tests cover shapes, positivity/finite values, identical distributions, a known
conditional shift, one-backbone-call vectorization, gradients, missing
branches, prior correction, self-normalization, batch adaptation, and
class-wise risk integration.

They additionally cover all four architecture smoke runs, exact state
selection, strict conditional masks, independent optimizer/parameter state,
learned fusion gates, ordinary-label leakage traps, empty-branch fallback, and
classifier-to-ratio gradient detachment.

## Ablations and baselines

Run the multi-seed comparison:

```bash
python -m experiments.ablation \
  --config configs/ablation_synthetic.json \
  --output-dir outputs/ablation \
  --device auto \
  --seeds 7,17,29
```

The suite compares the proposed conditioned estimator against `multihead`,
removed class/state conditioning, removed prior correction, all branch
balancing modes, a pooled marginal-ratio baseline, unit weights, and the known
synthetic oracle. It reports conditional log-ratio RMSE/MAE/correlation,
branch-wise feature-moment RMSE, source ratio calibration, ESS, parameter
count, runtime, and discriminator validation metrics. Outputs are
`raw_results.json`, `summary.csv`, and `REPORT.md`.

Setting `class_embedding_dim` or `state_embedding_dim` to zero explicitly
removes that conditioning input for controlled ablations.

## Wisteria

`wisteria/job_ratio_estimator.sh` loads the verified PyTorch/CUDA modules,
requires a visible GPU, runs the complete unit test suite, and starts
`train.py`. Submit from the repository root with:

```bash
bash wisteria/submit.sh
```

Outputs are written to `outputs/wisteria/`; scheduler and training logs go to
the job output and `logs/`.

Submit the three-seed ablation/baseline suite with:

```bash
bash wisteria/submit_ablation.sh
```

It runs both the mixed covariate-shift and heterogeneous branch-shift
scenarios. Reports are written to `outputs/ablation_mixed/REPORT.md` and
`outputs/ablation_branch_shift/REPORT.md`.

### Original SCAR baseline comparison

When the original SCARCE repository is available, train the frozen neural
ratio estimator and feed its `[B,q]` weights into the original
`combined_adiw_scar_loss`:

```bash
python -m experiments.scarce_ratio_comparison \
  --scarce-repo legacy/SCARCE \
  --ratio-arch fusion \
  --dataset mnist \
  --shift rotation \
  --seed 0 \
  --ratio-epochs 10 \
  --classifier-epochs 200 \
  --device cuda
```

The bridge reuses the original MNIST dual-domain loader, MLP feature layer,
SCAR branch risk, classifier, optimizer, and target evaluation. The Wisteria
launcher submits matched seeds 0--4 for the requested architectures:

```bash
bash wisteria/submit_scarce_ratio_comparison.sh
```

To submit selected architectures while keeping the same seed launcher:

```bash
RATIO_ARCHS="separate multihead fusion" \
  bash wisteria/submit_scarce_ratio_comparison.sh
```

### Multi-GPU debugging

Aquarius has eight A100 GPUs per node, while its shared queues accept at most
four GPUs per job. The multi-GPU launcher maps one independent
`(ratio_arch, seed)` task to one GPU and defaults to the complete 20-task
matrix. It requests five four-GPU jobs, so no GPU is reserved without a task:

```bash
SCARCE_REPO="$(pwd)/legacy/SCARCE" \
RUN_MODE=debug \
bash wisteria/submit_scarce_ratio_multigpu.sh
```

Debug mode uses one ratio epoch, two classifier epochs, and 256 target weak
examples. Inspect scheduler and per-task state with:

```bash
bash wisteria/monitor_scarce_multigpu.sh
```

After all 20 `.ok` files are present and no `.failed` file remains, submit the
full 10/200-epoch experiment with the same GPU layout:

```bash
SCARCE_REPO="$(pwd)/legacy/SCARCE" \
RUN_MODE=full \
bash wisteria/submit_scarce_ratio_multigpu.sh
```

Use `GPUS_PER_JOB=1|2|4`, `SEEDS`, or `RATIO_ARCHS` to reduce the matrix.
`DRY_RUN=1` generates the PJM scripts without submitting them.

`experiments/summarize_scarce_methods.py` combines the new JSON records with
the original `trainSCAR`, `testSCAR`, pooled/`ftSCAR`, and ADIW-SCAR table.
The completed Wisteria results and interpretation are recorded in
`ABLATION_RESULTS.md`.

### Matched next-round diagnostics

The next-round runner uses the JSON-compatible YAML configuration in
`configs/next_round.yaml`. It provides matched `no_iw`, `pooled_scar`, `adiw`,
`unified`, `multihead`, `fusion`, `separate_full`, and
`separate_matched_capacity` tasks. List or run one deterministic task with:

```bash
python scripts/run_next_round.py \
  --config configs/next_round.yaml \
  --phase smoke \
  --experiment-mode smoke \
  --list-tasks

python scripts/run_next_round.py \
  --config configs/next_round.yaml \
  --phase smoke \
  --experiment-mode smoke \
  --task-index 0 \
  --scarce-repo legacy/SCARCE \
  --results-root results/next_round_smoke \
  --device cuda
```

Every run writes `config.yaml`, `summary.json`, a checkpoint, classifier and
per-class trajectories, raw/clipped/normalized/classifier-used ratio
statistics, discriminator calibration, and fusion gate diagnostics under:

```text
results/<dataset>/<setting>/<method>/seed_<seed>/
```

Submit the bounded smoke matrix on Wisteria using every task-bearing GPU:

```bash
PHASE=smoke EXPERIMENT_MODE=smoke GPUS_PER_JOB=4 \
  RESULTS_ROOT=results/next_round_smoke \
  bash wisteria/submit_next_round.sh
```

After smoke succeeds, Phase A1 contains 90 diagnostic runs. The first rotation
block is the matched 30-degree six-method baseline; add the five
`A0_separate_standard` tasks to complete the seven-method main table.
`OFFSET` and `MAX_JOBS` keep submissions within the Aquarius project limit:

```bash
PHASE=A1 EXPERIMENT_MODE=diagnostic OFFSET=0 MAX_JOBS=16 \
  RESULTS_ROOT=results/next_round_phase_a \
  bash wisteria/submit_next_round.sh

PHASE=A0_separate_standard EXPERIMENT_MODE=diagnostic \
  RESULTS_ROOT=results/next_round_phase_a \
  bash wisteria/submit_next_round.sh
```

Summarize completed runs and generate the seven requested CSV tables plus 14
diagnostic figures with:

```bash
python scripts/summarize_next_round.py \
  --results results/next_round_phase_a
```

### Next-Round Phase B

Phase B reuses the matched runner and adds causal weight interventions,
frozen-score post-processing, a ground-truth Gaussian benchmark, and
class-specific MNIST rotations. List the formal matrices with:

```bash
python scripts/run_phase_b0.py --experiment-mode full --count  # 60 runs
python scripts/run_phase_b1.py --experiment-mode full --count  # 20 audits
python scripts/run_phase_b2.py --experiment-mode full --count  # 3 seeds / 45 grid runs
python scripts/run_phase_b3.py --experiment-mode full --count  # 20 seeds / 400 runs
python scripts/run_phase_b4.py --experiment-mode full --count  # 100 runs
```

All runners accept the Wisteria task interface. For example:

```bash
RUNNER=scripts/run_phase_b0.py \
CONFIG=configs/phase_b/b0_adiw_causal/config.json \
PHASE=B0 EXPERIMENT_MODE=full \
RESULTS_ROOT=results/phase_b/B0 \
bash wisteria/submit_next_round.sh
```

Validate, select the two B2 calibration candidates without target class
labels, and build the stage reports with:

```bash
python scripts/analyze_phase_b.py \
  --runs-root results/phase_b \
  --output-root outputs/phase_b \
  --select-b2
```

The audited code-path map is in `reports/phase_b_repository_audit.md`.

## Assumptions and limitations

- Ratios are only identifiable for branches with both source and target
  support; unsupported branches are reported and excluded.
- Count-based correction assumes loader counts represent the discriminator's
  effective sampling distribution. Pass explicit priors when a custom sampler
  changes that distribution.
- The bundled dataset is synthetic. Real image/vector loaders should preserve
  the `(x, bar_y)` contract, and a custom backbone must return `[B,F]`.
- MNIST, FashionMNIST, and CIFAR-10 classifier experiments use the existing
  SCARCE bridge (`--dataset mnist|fashionmnist|cifar10`), so that repository
  is bundled at `legacy/SCARCE` and must be selected with `--scarce-repo`.
  Its data module remains responsible
  for the exact input-output-relation and support-shift implementations.
- Source and target are forwarded separately, so each domain batch invokes the
  shared backbone once; there is never a per-class or per-state backbone call.
