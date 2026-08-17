# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Early-state wildfire severity classification on WildfireSpreadTS (607 US wildfire
event trajectories, 2018-2021, 23 raster channels at 24hr resolution): given only
the first 1-2 days of an active fire, predict a severity tier derived from the
fire's eventual peak extent. Tabular GBM/EBM baselines now; CNN-on-rasters and an
interpretability comparison (SHAP vs. EBM vs. Grad-CAM) planned. The README's
"Plan" section tracks current status — as of this writing, the primary reported
target is BINARY escalation (contained vs. escalates past bottom-quartile peak
extent; EBM ~0.83 LOYO accuracy / 0.75 macro-F1 vs. 0.73 / 0.42 majority
baseline), the 4-class quartile tiers are secondary (XGBoost leads there), and
the neural-net arm hasn't started.

There is no linter or build step. A pytest suite (`tests/`) covers the pure,
deterministic functions in `labels.py` and `features.py` -- run with
`pytest tests/` in the `flashpoint` env. Most development still happens in
the notebooks (`01` → `03` in dependency order; `02` and `04` are
exploration/diagnostics; `05` is the CNN arm, in `flashpoint_mps`), run in
the `flashpoint` conda env.

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
- **Two environments, split by architecture**: `flashpoint` (x86_64 under
  Rosetta -- the whole anaconda install is Intel) runs the tabular pipeline
  (notebooks 01-04); `flashpoint_mps` (native osx-arm64, created with
  `CONDA_SUBDIR=osx-arm64` inside the same conda install) runs the
  torch/MPS CNN work (notebook 05+). No MPS-capable torch >= 2.3 exists for
  the x86_64 env, so never try to install/upgrade torch there. Installs
  into `flashpoint_mps` need `CONDA_SUBDIR=osx-arm64`,
  `CONDA_OVERRIDE_OSX=$(sw_vers -productVersion)`, and
  `--override-channels -c conda-forge` (defaults' arm64 pytorch lacks MPS).
  Keep it lean: no xgboost/interpret-core/shap/cartopy -- notebook 05 reads
  tabular comparison numbers from notebook 03's output, not by recomputing.
  README "Setup" has the full recipe and the reasons.

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
- **The early window starts at index `BUFFER_DAYS` (4), not 0** — stored
  sequences open with 4 pre-ignition padding days, so the fire's actual day 1
  is index 4 (`data_access.BUFFER_DAYS`). Slicing `[0, cutoff)` measures
  pre-ignition conditions and silently degrades every downstream result.
- **Severity bin edges are empirical quartiles** of `max_area_ha`, set in
  notebook 01 after inspecting the distribution — never hardcoded hectare
  thresholds.
- **Feature aggregation must be nan-aware** (`np.nanmean` etc.) — ~1-1.5%
  per-pixel missingness in weather channels otherwise poisons ~8% of events'
  aggregates.
- **Wind direction is encoded as sin AND cos** — the dataset authors' own
  sin-only encoding is a known, unfixed information-loss bug; don't copy it.
- **Train/test splits are year-based, never random** (2019 is a known outlier
  year); every reported result uses leave-one-year-out via
  `evaluation.leave_one_year_out`, which also reports recall on a designated
  positive class (escalates-recall matters asymmetrically: a missed escalating
  fire is worse than a false alarm).

Notebooks do `sys.path.insert(0, "../src")` at the top — keep that line in new
notebooks so they work even when the editable install is stale.
