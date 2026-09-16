# Run ADIW-SCAR on Wisteria/BDEC-01 Aquarius

This bundle targets the Aquarius data/learning nodes with one NVIDIA A100 per
seed. Each formal run uses 200 epochs and one seed. Seeds `0` through `4` are
submitted as independent jobs so a failed or timed-out job does not discard
the other repetitions.

## Upload and unpack

From the local computer:

```bash
scp SCARCE_ADIW_Wisteria.zip USER@wisteria.cc.u-tokyo.ac.jp:/work/PROJECT/USER/
```

On the Wisteria login node:

```bash
cd /work/PROJECT/USER
unzip SCARCE_ADIW_Wisteria.zip
cd SCARCE
```

## Configure

`wisteria/config.sh` is currently configured for project `gu15`. Change
`PROJECT_GROUP` only when submitting under another Wisteria project.

The defaults `EXTRA_MODULES="cuda/11.1"` and
`PYTORCH_MODULE="pytorch/1.8.1"` were verified with the Wisteria module catalog
on July 14, 2026. Check the installed module names with:

```bash
show_module | grep -i -E 'pytorch|miniconda|cuda'
```

If the default module is not available, set `PYTORCH_MODULE` to the module
reported by `show_module`. For a custom virtual environment, set
`PYTORCH_MODULE=""`, set `EXTRA_MODULES="cuda"`, and set `PYTHON_BIN` to its
absolute Python path. The job verifies NumPy, SciPy, PyTorch, torchvision, and
CUDA before training and stops rather than silently using the CPU.

## Submit

Submit one seed first:

```bash
bash wisteria/submit.sh 0
pjstat
```

After that job starts successfully, submit the remaining seeds:

```bash
bash wisteria/submit.sh 1 2 3 4
```

Calling `bash wisteria/submit.sh` with no seed submits all five seeds. Each job
requests the Aquarius `share` queue, one GPU, and 48 hours. Logs are written to
`logs/`; total and per-epoch CSV files are written to `result/adiw_scar/`.

To submit the two single-domain SCARCE baselines, use:

```bash
bash wisteria/submit_scarce_domains.sh 0
```

Each seed job trains two independent models in sequence: one with only the
train-domain weak data and one with only the test-domain weak data. Their CSV
files are written to `result/domain_scarce/` and their combined job log is
written to `logs/domain_scarce_seed*.log`.

Before a formal run, submit the one-epoch, 64-sample-per-domain GPU smoke job:

```bash
bash wisteria/submit_scarce_domains_smoke.sh 0
```

Submit the unweighted pooled train+test SCAR control with:

```bash
bash wisteria/submit_pooled_scar.sh
```

The archive includes the MNIST raw files because outbound downloads should not
be assumed to work from a compute node.
