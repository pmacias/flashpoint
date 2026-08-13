# Flashpoint

Early-state wildfire severity classification: given only the first 1-2 days
of an active fire (extent, weather, cumulative dryness), predict a severity
tier that informs how quickly/heavily to mobilize response -- rather than
waiting for the full trajectory to play out.

Successor to `steam_trajectory`, carrying forward database-first workflow,
gradient boosting + feature analysis, and adding: a second data modality
(early raster stacks), a neural net comparison arm, and an explicit
interpretability comparison (SHAP vs. EBM vs. Grad-CAM).

## Data

[WildfireSpreadTS](https://doi.org/10.5281/zenodo.8006177) (Gerard et al.,
2023) -- 607 full US wildfire event trajectories, 2018-2021, 23 channels
(active fire, weather, fuel, topography) at 24hr resolution. ~50GB of
GeoTIFFs, CC-BY-4.0, converted locally to HDF5 (~92GB) via the authors' own
`CreateHDF5Dataset.py` for fast repeated reads. Both live outside Dropbox
and outside Time Machine's scope (`~/ml_datasets/flashpoint/`) -- large,
reproducible, not worth syncing or backing up.

## Setup

**Install everything numerically-compiled via conda-forge first, then let
pip fill in the pure-Python rest.** Mixing pip and conda for packages like
numpy/xgboost/scikit-learn is what caused most of this project's setup pain
(ABI mismatches, phantom "already satisfied" states, `conda uninstall`
cascading and removing unrelated packages) -- conda-forge keeps their
binaries mutually compatible; pip doesn't know or care.

```bash
conda create -n flashpoint python=3.11 -y
conda activate flashpoint
conda install -c conda-forge "numpy<2.4" xgboost scikit-learn scipy pandas matplotlib -y
pip install -e ".[dev]" --no-deps
```

`--no-deps` on the last line is deliberate -- it installs the `flashpoint`
package itself without pip trying to also resolve numpy/xgboost/etc. and
potentially reinstalling incompatible versions on top of the conda ones.
After this, install the remaining pure-Python packages pip already listed
in `pyproject.toml` (duckdb, h5py, interpret-core, shap, tqdm, jupyter,
ipykernel) -- these have no compiled-binary conflicts, so plain pip is fine
for them specifically.

### If something breaks again anyway

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

### macOS gotchas specific to this project

- **xgboost + libomp**: a pip-only xgboost install fails to import with
  `Library not loaded: @rpath/libomp.dylib` -- Apple's toolchain doesn't
  ship OpenMP. Installing xgboost via conda-forge (above) sidesteps this
  entirely.
- **interpret-core, not interpret**: the full `interpret` package pulls in
  a Dash/Plotly/gevent dashboard stack that tries to compile `gevent`'s C
  extension and fails without working Xcode Command Line Tools.
  `interpret-core` gives the EBM itself with none of that.
- **HDF5 conversion needs its own throwaway env**: the dataset authors'
  `CreateHDF5Dataset.py` has its own older pinned `requirements.txt` that
  conflicts with this project's dependencies. Run it in a separate env
  (`conda create -n wildfirespreadts-convert python=3.10.4`), never inside
  `flashpoint` -- once the HDF5 files exist, that env is never needed again.

## Plan

1. **Data & database** -- stage the archive, convert to HDF5, build the
   DuckDB manifest, derive severity labels from full trajectories
   (peak active-fire extent, not last-day extent -- see
   `01_data_ingestion.ipynb` for why). *Done.*
2. **Feature engineering & GBM/EBM baseline** -- early-window tabular
   features (`02_feature_engineering.ipynb`), XGBoost + SHAP, and an
   Explainable Boosting Machine as a fully-transparent second baseline.
   *Done through several iterations: topography + vegetation features
   added (topography helps, ~+5pp; vegetation doesn't), leave-one-year-out
   is the standard evaluation (`src/flashpoint/evaluation.py`), and the
   EBM consistently outperforms XGBoost. The primary reported result is
   now the BINARY escalation target -- contained (bottom-quartile peak
   extent) vs. escalates -- where the EBM gets LOYO mean accuracy 0.83 /
   macro-F1 0.75 / escalates-recall 0.95, vs. 0.73 / 0.42 / 1.00 for a
   majority-class baseline. The 4-class quartile tiers are kept as a
   secondary result (XGBoost 0.46 accuracy vs. 0.25 chance); adjacent-tier
   separation at the bin boundaries is where most of that error lives.
   NB: the early window is indices [BUFFER_DAYS, BUFFER_DAYS+2) of the
   stored sequence -- the first 4 stored days are pre-ignition padding,
   and slicing from index 0 (an early bug) cost ~5pp everywhere.
   Forecast-channel iteration (targeting the false-alarm problem -- the
   EBM stands down correctly on only ~half of genuinely contained fires):
   next-day GFS aggregates + "easing" deltas from the last window day
   (available at cutoff, not leakage; notebook 02 Steps 2b/9). Outcome is
   split: for the primary EBM it's a null result (macro-F1 0.754 -> 0.748,
   escalates-recall 0.953 -> 0.922, false alarms only -4), so the EBM's
   reported feature set stays base+topo; for XGBoost the same features
   help across the board (accuracy 0.788 -> 0.822, macro-F1 0.718 -> 0.765,
   escalates-recall 0.884 -> 0.911, false alarms 75 -> 67 of 155), making
   xgb+forecast the best false-alarm-limiting variant -- best pooled
   contained-class F1 (0.645) and macro-F1 (0.771) of any model -- though
   still below the EBM's 0.95 escalates-recall, which is why the EBM
   remains primary under the missed-escalation-costs-more doctrine. The
   columns stay in `early_features` for reuse (e.g. as CNN input planes).
   Gotcha for anyone touching these channels: raw GFS
   `forecast_temperature` is stored in CELSIUS while observed GRIDMET
   temps are Kelvin -- `features.py` converts before differencing.*
3. **Neural net exploration** -- tabular MLP (sanity check vs. GBM) and a
   small CNN on early raster stacks, both predicting the same severity
   label, with Grad-CAM/saliency for the CNN.
4. **Stretch: next-day spread prediction** -- single-step U-Net as an
   explicit "future work" extension, not the core deliverable.

## Structure

```
src/flashpoint/
    data_access.py   # HDF5 discovery + reading (channel order, pixel area)
    db.py            # DuckDB schema (events, event_outcomes, early_features)
    labels.py        # severity tier derivation (peak extent across trajectory)
    features.py      # early-window tabular feature engineering
    evaluation.py    # leave-one-year-out CV used by every reported result
notebooks/
    01_data_ingestion.ipynb       # DB build + severity labels
    02_feature_engineering.ipynb  # early features + GBM/EBM baselines
data/                # local cache (duckdb file) -- gitignored
```
