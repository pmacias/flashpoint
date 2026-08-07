# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Early-state wildfire severity classification on WildfireSpreadTS (607 US wildfire
event trajectories, 2018-2021, 23 raster channels at 24hr resolution): given only
the first 1-2 days of an active fire, predict a severity tier derived from the
fire's eventual peak extent. Tabular GBM/EBM baselines now; CNN-on-rasters and an
interpretability comparison (SHAP vs. EBM vs. Grad-CAM) planned. The README's
"Plan" section tracks current status — as of this writing, baselines are trained
but weak (~27-30% on 4 classes vs. 25% chance), and the neural-net arm hasn't
started.

There is no test suite, linter, or build step. Development happens in the
notebooks (`01` → `02` in dependency order, `03` is exploration), run in the
`flashpoint` conda env.

## Environment

Follow README.md "Setup" exactly when creating or repairing the env — the
sequence is deliberate, worked out after ABI-mismatch pain, and the README
documents the recovery procedure. Don't improvise package fixes; in particular
never run `conda uninstall` casually (it cascades).

### Environment gotchas

- **conda-forge first, pip second**: install anything numerically compiled
  (numpy, xgboost, scikit-learn, ...) via conda-forge, then
  `pip install -e ".[dev]" --no-deps` — plain pip breaks ABI compatibility and
  a pip-only xgboost fails on macOS's missing libomp.
- **`interpret-core`, not `interpret`**: the full package drags in a
  Dash/gevent dashboard stack that fails to compile without Xcode CLT.

Raster data lives **outside** the repo at `~/ml_datasets/flashpoint/` (outside
Dropbox and Time Machine deliberately): original GeoTIFFs plus the HDF5
conversion the code actually reads (`<hdf5_dir>/<year>/<fire_name>.hdf5`, dataset
`"data"` of shape `(n_days, 23, H, W)`). If the HDF5 ever needs regenerating, the
authors' conversion script requires its own throwaway env (python 3.10.4) — never
run it inside `flashpoint`.

## Architecture

Database-first pipeline: DuckDB (`data/flashpoint.duckdb`, gitignored) holds
metadata, labels, and engineered features; pixel data stays in HDF5 and is read
on demand, sliced as narrowly as possible (`read_event_window` for the early
days, `read_channel_all_days` for one channel across the trajectory — not full
`(n_days, 23, H, W)` reads).

Three tables, built in notebook order (schema in `src/flashpoint/db.py`):

1. `events` — manifest from `data_access.discover_hdf5_events` (HDF5 attrs only,
   no pixel reads).
2. `event_outcomes` — severity label per event, computed by `labels.py` over the
   **full** trajectory.
3. `early_features` — tabular features from `features.py`, computed **only**
   from the day-1/day-2 cutoff window.

That asymmetry is the project's core integrity constraint: labels may see the
whole trajectory, model inputs may never see past the cutoff day. Any new
feature that reaches into later days leaks the label and invalidates the task.

Domain corrections already baked into the code — preserve them when extending:

- **`CHANNEL_NAMES` in `data_access.py` is the source of truth** for the
  23-channel order (verified against the dataset docs and the authors' own
  code). Index channels through it, never by hardcoded position.
- **The active-fire channel is detection HOUR (0-23), not a 0/1 mask.**
  Binarize with `> 0` (`labels.active_fire_mask`) before counting pixels.
- **Severity comes from PEAK extent, not last-day extent** — events are padded
  with 4 buffer days and 56.5% of them show zero fire on their literal last day
  (`labels.py` docstring has the full argument).
- **Severity bin edges are empirical quartiles** of `max_area_ha`, set in
  notebook 01 after inspecting the distribution — never hardcoded hectare
  thresholds.
- **Feature aggregation must be nan-aware** (`np.nanmean` etc.) — ~1-1.5%
  per-pixel missingness in weather channels otherwise poisons ~8% of events'
  aggregates.
- **Wind direction is encoded as sin AND cos** — the dataset authors' own
  sin-only encoding is a known, unfixed information-loss bug; don't copy it.
- **Train/test splits are year-based, never random** (2019 is a known outlier
  year); notebook 02 also runs leave-one-year-out for a less noisy estimate.

Notebooks do `sys.path.insert(0, "../src")` at the top — keep that line in new
notebooks so they work even when the editable install is stale.
