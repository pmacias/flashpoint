# Flashpoint setup: troubleshooting and the CNN environment

Details behind the short install in [README.md](README.md#setup): how to
recover a broken `flashpoint` env, macOS-specific pitfalls, and the separate
`flashpoint_mps` environment used for the CNN work (notebook 05).

## If something breaks again anyway

Verify what's ACTUALLY loaded, not what pip/conda claim is installed --
their bookkeeping can disagree with reality after a messy install history:

```python
import sys, numpy
print(sys.executable)   # should contain "envs/flashpoint"
print(numpy.__version__)  # should be < 2.4
```

If a version still looks wrong after a "successful" reinstall, don't trust
a second reinstall to fix it -- force-remove the stale directory first:
```bash
rm -rf $(python -c "import site; print(site.getsitepackages()[0])")/<package>*
```
then reinstall via conda-forge.

**Never run `conda uninstall <package>` casually** -- it cascades and
removes everything that depends on it too (this cost us xgboost,
scikit-learn, joblib, and threadpoolctl in one command during setup).
Prefer reinstalling/upgrading over uninstalling when possible.

## macOS gotchas specific to this project

- **xgboost + libomp**: a pip-only xgboost install fails to import with
  `Library not loaded: @rpath/libomp.dylib` -- Apple's toolchain doesn't
  ship OpenMP. Installing xgboost via conda-forge (README "Setup") sidesteps
  this entirely.
- **interpret-core, not interpret**: the full `interpret` package pulls in
  a Dash/Plotly/gevent dashboard stack that tries to compile `gevent`'s C
  extension and fails without working Xcode Command Line Tools.
  `interpret-core` gives the EBM itself with none of that.
- **HDF5 conversion needs its own throwaway env**: the dataset authors'
  `CreateHDF5Dataset.py` has its own older pinned `requirements.txt` that
  conflicts with this project's dependencies. Run it in a separate env
  (`conda create -n wildfirespreadts-convert python=3.10.4`), never inside
  `flashpoint` -- once the HDF5 files exist, that env is never needed again.

## Second environment for the CNN work: `flashpoint_mps`

This project uses **two conda environments**. The anaconda install (and the
`flashpoint` env) is **x86_64 running under Rosetta** on an Apple Silicon
machine, and PyTorch stopped shipping macOS x86_64 builds after 2.2.2 --
there is no MPS-capable torch >= 2.3 for that env, and torch 2.2.2 is
broken against its numpy 2.x (`torch.from_numpy` raises "Numpy is not
available"). Rather than destabilize the working tabular env, the
torch/MPS CNN work (notebook 05+) lives in a separate **native osx-arm64**
env created inside the same conda install via `CONDA_SUBDIR`:

```bash
CONDA_SUBDIR=osx-arm64 conda create -n flashpoint_mps python=3.11 -y
printf 'subdir: osx-arm64\n' > "$(conda info --base)/envs/flashpoint_mps/.condarc"
CONDA_SUBDIR=osx-arm64 CONDA_OVERRIDE_OSX=$(sw_vers -productVersion) \
  conda install -n flashpoint_mps --override-channels -c conda-forge \
  "conda-forge::pytorch" duckdb h5py pandas matplotlib ipykernel -y
cd <repo> && conda run -n flashpoint_mps pip install -e . --no-deps
conda run -n flashpoint_mps python -m ipykernel install --user \
  --name flashpoint_mps --display-name "flashpoint_mps (arm64/MPS)"
```

Three non-obvious flags, all learned the hard way:

- **`CONDA_SUBDIR=osx-arm64` on the `create` too**, or the env gets an
  x86_64 python that can't load arm64 torch; the `.condarc` line pins the
  subdir so later installs into this env can't silently mix architectures.
- **`CONDA_OVERRIDE_OSX=$(sw_vers -productVersion)`**: the Rosetta conda
  reports macOS as "10.16", which fails conda-forge's `__osx >= 11.0`
  requirement on arm64 packages -- without the override the solve fails.
- **`"conda-forge::pytorch"` with `--override-channels`**: defaults'
  pytorch builds for osx-arm64 are compiled WITHOUT MPS (`cpu_openblas`,
  `mps.is_built() == False`); only the conda-forge builds have it, and a
  bare `pytorch` spec no-ops if a defaults build is already installed.

Verify before use: `platform.machine()` must print `arm64`,
`torch.backends.mps.is_available()` must be True, and a float32
`torch.from_numpy(...).to("mps")` round-trip must work (MPS has no
float64 -- keep tensors float32). The env is deliberately lean: **no
xgboost / interpret-core / shap / cartopy** -- notebook 05 reads the
committed EBM/XGBoost comparison numbers from notebook 03's output instead
of recomputing them, so the gevent/Xcode-CLT install pain never enters
this env. Notebooks 01-04 keep using the `flashpoint` kernel; notebook 05+
uses `flashpoint_mps (arm64/MPS)`.
