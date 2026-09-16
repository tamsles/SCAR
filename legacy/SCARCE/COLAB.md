# Run ADIW-SCAR on Google Colab

The generated `ADIW_SCAR_Colab.ipynb` and `SCARCE_ADIW_Colab.zip` form one
Colab bundle. The archive contains source code and tests only; datasets are
downloaded by torchvision inside Colab.

## Steps

1. Open <https://colab.research.google.com/> and upload
   `ADIW_SCAR_Colab.ipynb`.
2. Select `Runtime > Change runtime type > T4 GPU`.
3. Run the notebook from the first cell. When prompted, upload
   `SCARCE_ADIW_Colab.zip`.
4. Keep `RUN_SMOKE = True` for the first run. This executes the unit tests and
   a one-epoch MNIST experiment.
5. After the smoke test passes, set `RUN_FULL = True` in the formal experiment
   cell. Run one seed at a time and set `seed` to `0`, `1`, `2`, `3`, and `4`
   for five independent runs.

The formal configuration is the ADIW-SCAR fusion experiment, not an exact
reproduction of the original SCARCE paper table. Its outputs are written to
`result/adiw_scar/total` and `result/adiw_scar/detail`.

For long experiments, enable the optional Google Drive cell before training.
It maps `result/adiw_scar` to Drive so the per-epoch CSV survives a Colab
disconnect. A free Colab session may still be too short for all five 200-epoch
runs, so run and preserve each seed separately.
