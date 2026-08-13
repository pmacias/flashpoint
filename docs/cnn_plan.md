# Status

**Phase 0 and Phase A are complete -- do not touch either conda environment or
re-run notebook 02. Forecast planes (channels 17-21) are stored in the crop set
but excluded from the CNN's default training channels -- Phase A found a null
result for forecast features on the EBM, the CNN's comparison target, so
there's no reason to spend the CNN's limited capacity there.**

---

# Next phase: forecast features first, then the small CNN + Grad-CAM arm

## Context

The binary escalation EBM (0.826 LOYO accuracy / 0.754 macro-F1) fails almost entirely
on false alarms: pooled confusion matrix [[77, 78], [18, 434]] — contained-class recall
is 0.497. The proposed next phase is a small CNN on early raster stacks to feed the
Grad-CAM vs. SHAP/EBM interpretability comparison. This plan answers the four open
questions, then sequences the work: a cheap forecast-feature experiment (Phase A) before
the CNN (Phase B), plus a required environment fix (Phase 0).

## Answers to the four open questions (reasoning first, as requested)

### Q4 first — prioritization, and a correction from the repo

**Yes, do the forecast-channel experiment before the CNN.** But one correction: the
"false-alarm diagnostic flagged forecast channels (day-3+ weather easing)" conclusion
**does not exist in the repo**. The only recorded false-alarm diagnostic is notebook 02
Step 11 (growth headroom: FP ratio 0.305 vs TN 0.299 — negative result, a growth-trend
feature won't fix the FPs). "Forecast", "easing", "day-3" appear nowhere in any notebook,
doc, or commit. What *is* true: the FP exemplar `fire_24462912` had hot/dry early
conditions (erc 91.4, max_temp 310.7 K) on a fire that then stopped — consistent with a
"conditions eased afterward" story — and channels 17–21 (GFS next-day forecasts) are
completely untouched by the current feature set. So the lead is real, just never tested.

It goes first because: (a) it's ~1 session vs. several for the CNN; (b) it's the only
genuinely *new information* available (day-6 forecast at cutoff — everything else is a
re-representation of data the EBM already sees); (c) it targets the FP failure mode
directly; (d) whatever it finds, the tabular baseline should be frozen *before* the
interpretability comparison, so the CNN is compared against the best tabular model, not
a moving target.

Leakage note: forecast channels at day index t hold GFS forecasts for day t+1, *issued
by* day t. Reading them inside the [4, 6) window is information available at cutoff —
the one legitimate look-ahead in the dataset. Reading them at index ≥ 6 would be leakage.

### Q1 — architecture and variable H/W

- **Crop, don't resize.** Resolution is uniform 375 m/px across all 607 events; H/W vary
  only because bounding boxes differ (H 297–356, W 207–311, none square). Resizing would
  destroy physical scale — the one thing a CNN can exploit that aggregates can't (fire
  position relative to terrain). A **fixed 128×128 crop (48×48 km) centered on the
  early-window active-fire union centroid** preserves scale and centers the object of
  interest. Verified: only 69/607 events (11.4%) clip a raster edge (median pad 24 px) —
  NaN-pad, no validity plane needed in v1. **156/607 events (25.7%) have zero fire pixels
  in the early window** → defined fallback: box center (the box is drawn around the event
  footprint, so this is a reasonable prior; not new leakage — tabular whole-box aggregates
  inherit the same geometry).
- **Architecture:** 4 blocks of [Conv3×3 → GroupNorm(8) → ReLU → MaxPool2], channels
  16→32→64→128, global average pooling → Dropout(0.3) → Linear(128, 1). **~110k params**
  (assert < 500k). GroupNorm because batches are small; GAP kills most of the parameter
  count and gives Grad-CAM a clean 8×8 last conv map. Class-weighted BCE
  (pos_weight = n_contained/n_escalates ≈ 0.34 from the train fold — downweights the
  majority "escalates" class, pushing the operating point toward fewer false alarms).
  AdamW (lr 3e-4, wd 1e-2), ReduceLROnPlateau, early stop patience 10, restore best.
- **Augmentation: no rotation/flip** — aspect and wind-direction sin/cos planes encode
  absolute compass bearing (north-facing vs south-facing slopes differ physically in the
  northern hemisphere); flipping without counter-rotating those channels creates
  physically inconsistent inputs, and counter-rotation is exactly the subtle-bug class
  this project avoids. Safe substitutes: store crops at 144×144 and take random ±8 px
  crop-center jitter at train time (pure array slice), center-crop 128 at eval.
- **Model selection inside LOYO:** per fold, validation = random 15% of *training-year*
  events, stratified by year×label, re-drawn per seed. Year discipline protects the
  *test* year; a val split drawn only from training years never touches it. Fixed-year
  val would waste 201 events (2020) or early-stop against outlier 2019's 74 events;
  rotating-year val triples cost. Re-drawing per seed makes the 3-seed std include
  split variance honestly.

### Q2 — channels: curated subset, not all 23

All 23 × 2 days ≈ 46 planes against ~450 training events per fold is asking a small CNN
to do feature selection with no data. Curate by EBM evidence + the hypothesis under test
(fire position relative to terrain). **26 planes stored, 20 used by default:**

| group | planes | n |
|---|---|---|
| static terrain | slope, elevation, aspect_sin, aspect_cos | 4 |
| slow-varying (nanmean over window) | ndvi, pdsi | 2 |
| per-day weather (×2 days) | erc, max_temp, wind_speed, wind_dir_sin, wind_dir_cos, humidity | 12 |
| fire (binarized, per day) | active_fire_d1, active_fire_d2 | 2 |
| forecast (last window day → valid day 6) | fc_precip, fc_wind_speed, fc_wind_dir_sin, fc_wind_dir_cos, fc_temp, fc_humidity | 6 |

Excluded: landcover (16-class one-hot balloons width for nothing at this n), viirs/evi2
(vegetation *hurt* the tabular ablation). Forecast planes stored regardless of Phase A's
outcome (storage is free); included in training only if Phase A shows signal. Static
channels appear once, not per-day.

### Q3 — will it beat 0.826 / 0.754? Honest answer: probably not.

Expected range: LOYO accuracy **0.74–0.81**, macro-F1 **0.62–0.73**, seed std 1–3 pp per
fold. The strongest signal (day-2 extent, EBM importance 0.445) is one number the tabular
model gets for free; the CNN burns capacity rediscovering box-mean weather from noisy
planes, on 450 events. **Still worth building** — the project's stated deliverable is the
interpretability comparison, which only requires a *credible* comparator (pre-registered
bar: within ~5 pp macro-F1 of the EBM), not a winner. The realistic upside lives in the
FP/FN cells: fire-shape/position-vs-terrain patterns no aggregate encodes. Step 8 below
makes the comparison quantitative (CAM mass statistics vs EBM importances), which is the
paper-able output either way.

---

## Phase 0 — Environment fix (prerequisite for Phase B only)

Verified blocker: `flashpoint` env has torch 2.2.2 + numpy 2.3.5 → `torch.from_numpy`
raises `RuntimeError: Numpy is not available` (torch 2.2 predates numpy 2.x ABI). MPS
reports available.

1. `conda install -n flashpoint -c conda-forge "pytorch>=2.3"` (conda-forge, never pip,
   per README rules; package name is `pytorch`).
2. Verify: torch ≥ 2.3, `torch.from_numpy(np.zeros(3))` works, MPS still available,
   xgboost/interpret still import (no `conda uninstall`, watch the solver's plan).
3. `pyproject.toml`: bump `torch>=2.2` → `torch>=2.3` (2.2 breaks under numpy 2.x).
4. README Setup: add pytorch to the conda-forge line + one sentence on upgrading.

## Phase A — Forecast-channel tabular features (~1 session)

**Files: `src/flashpoint/features.py`, `src/flashpoint/db.py`,
`notebooks/02_feature_engineering.ipynb`** (three sync points: dict keys ↔ CREATE TABLE
columns ↔ notebook cell 8's positional insert tuple, all must stay aligned).

1. `features.py` — extend `early_window_stats` with 11 nan-aware features computed from
   **the last window day only** (index 5's forecast is for day 6, the first unseen day;
   index 4's forecast is for day 5, redundant with observations):
   `forecast_precip_mean/max`, `forecast_wind_speed_mean/max`,
   `forecast_wind_dir_sin/cos_mean`, `forecast_temp_mean`, `forecast_humidity_mean`,
   and three "easing deltas": `forecast_temp_delta` (fc_temp − observed max_temp mean),
   `forecast_humidity_delta`, `forecast_wind_speed_delta`. Docstring states the leakage
   argument (forecast issued by day t, valid t+1).
2. `db.py` — append 11 DOUBLE columns to `early_features` in `SCHEMA_SQL`, commented as
   next-day GFS forecast available at cutoff (not leakage).
3. Notebook 02 — cell 8's DROP/recreate + insert tuple widens 22 → 33 values; add
   `FORECAST_COLS`; ablation adds `base+topo+forecast` and `base+topo+deltas_only` rows
   (deltas-only isolates whether "easing" framing vs raw forecast levels carries signal);
   Step 9 reruns binary EBM/XGB on `SHAP_COLS + FORECAST_COLS` with pooled confusion
   matrix side-by-side vs the current [[77,78],[18,434]].
4. One-time sanity checks in the notebook: (a) units — confirm forecast_temperature and
   max_temp are both Kelvin before trusting delta signs; (b) alignment — correlate day-5
   forecast planes against day-6 *observed* weather for a few events (diagnostic only,
   never a feature) to confirm the t → t+1 convention.

**Success:** FPs drop ≥ 10 with escalates-recall ≥ 0.93 → adopt into the reported set,
update README. **Null result is plausible** (GFS is ~25 km native; one day's box-mean is
blunt) → keep DB columns, exclude from SHAP_COLS, record the negative ablation row in
README the way vegetation was.

## Phase B — Small CNN + Grad-CAM comparison

### New module `src/flashpoint/rasters.py`
- `early_fire_center(early_stack)` — union-mask centroid, box-center fallback (documented
  156/607 rate).
- `extract_crop(stack, cy, cx, size)` — NaN-padded at edges (pad and missing weather get
  the same train-mean fill downstream; simple and defensible).
- `build_plane_stack(...)` — the 26-plane assembly above, returns planes + names.
- `build_crop_store(events, out_path, store_size=144)` — one pass over all 607 events via
  `read_event_window(event, BUFFER_DAYS, BUFFER_DAYS + CUTOFF_DAY)` (the only pixel-read
  path, preserving the leakage rule), writes
  `~/ml_datasets/flashpoint/early_crops_144.h5` (~1.3 GB, **outside the Dropbox repo**):
  datasets `crops` (607, 26, 144, 144) f32, `event_id`, `year`; attrs `plane_names`,
  `cutoff_day`, `buffer_days`, fallback/edge-clip counts.

### New module `src/flashpoint/cnn.py`
- `EarlyCropDataset` — RAM-resident store, plane subset by name, per-plane train-fold
  nan-aware mean/std normalization, `nan_to_num(0.0)` after normalization (0 = train
  mean), jitter-crop 128 (train) / center-crop (eval). No flip/rotation (module docstring
  states why). Optional `channel_dropout_p` default 0.
- `SmallFireCNN(n_planes)` — the 4-block/GroupNorm/GAP architecture above, ~110k params.
- `train_one(...)` — AdamW, batch 32, ≤60 epochs, plateau LR, early stop on val loss,
  MPS with CPU fallback.
- `leave_one_year_out_raster(...)` — mirrors `evaluation.leave_one_year_out`'s signature
  and returns the SAME metric columns (`test_year, n_test, accuracy, macro_f1,
  positive_recall`) + `seed`, so comparison tables are a plain concat. Lives in `cnn.py`
  to keep `evaluation.py` torch-free. Seeds (0, 1, 2), report mean±std.
- `grad_cam(model, x)` — hand-rolled (~30 lines): hook on block-4 output (128×8×8),
  backward from the logit, gradient-weighted channel sum, ReLU, upsample to 128. No
  captum dependency.

### New notebook `notebooks/04_cnn_severity.ipynb`
Convention: `sys.path.insert(0, "../src")` first cell, ALL-CAPS constants
(`CROP_SIZE=128, STORE_SIZE=144, JITTER_PX=8, SEEDS=(0,1,2)`), "## Step N" headers.

- Step 0: env verification (Phase 0 checks) — fail loudly before training.
- Step 1: load events/outcomes from DuckDB, derive `escalates`, eligibility filter.
- Step 2: build/load crop store; print fallback-center (156) and edge-clip (69) counts.
- Step 3: sanity viz — 3–4 events, crop box over full raster + fire/slope/erc planes.
- Step 4: rerun binary EBM LOYO (cheap) so the comparison table is self-contained.
- Step 5: CNN LOYO × 3 seeds; per-fold mean±std; pooled confusion per seed; explicit
  FP-count comparison vs the EBM's 78.
- Step 6: ablations — drop the 2 fire planes (is anything spatial being used, or is it
  extent in disguise?); ± forecast planes.
- Step 7: Grad-CAM on notebook 02 Step 10's exact TP/TN/FP/FN exemplar event_ids AND the
  CNN's own out-of-fold exemplars — "same events, two lenses" plus each model's own view.
- Step 8 (the deliverable): quantitative CAM statistics over all 607 events, per
  confusion cell: (a) CAM mass fraction within 5 px (~1.9 km) of early fire pixels vs
  uniform-expectation baseline; (b) per-event Spearman correlation of CAM vs slope plane
  and vs erc plane. Rank fire-proximity / terrain / weather attention against EBM
  importances (fire_extent 0.445, slope 0.277, erc 0.205).
- Step 9: summary markdown — honest comparison table, expected-vs-actual.

### Bookkeeping
- `src/flashpoint/__init__.py` module map + README Structure: add `rasters`, `cnn`,
  notebook 04.
- README Plan item 3: update with results; mark the tabular-MLP sub-item deferred (adds
  little next to the EBM-vs-CNN comparison).

## Verification
- Phase A: notebook 02 runs top-to-bottom; new LOYO/ablation tables print; pooled
  confusion compared against [[77,78],[18,434]]; escalates-recall floor 0.93 checked.
- Phase B: param-count assert (< 500k); train-fold-only normalization stats asserted
  (no test-year events in mean/std computation); crop-store attrs match constants before
  reuse; LOYO table shape identical to tabular for concat; Grad-CAM overlays render for
  exemplars. Wall-clock sanity: crop store ~5–10 min one-time; 4 folds × 3 seeds ≈
  15–30 min on MPS; CAM over 607 events < 1 min.

## Execution order
Phase 0 (5 min) → Phase A end-to-end (decides forecast planes' default inclusion) →
Phase B modules → crop store → notebook 04.
