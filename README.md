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
(active fire, weather, fuel, topography) at 24hr resolution. ~50GB,
CC-BY-4.0. One-time bulk download rather than true remote streaming -- see
project notes on that tradeoff.

## Plan

1. **Data & database** -- stage the archive, build the DuckDB manifest,
   derive severity labels from full trajectories (`notebooks/01_data_ingestion.ipynb`).
2. **Feature engineering & GBM/EBM baseline** -- early-window tabular
   features, XGBoost + SHAP, and an Explainable Boosting Machine as a
   fully-transparent second baseline.
3. **Neural net exploration** -- tabular MLP (sanity check vs. GBM) and a
   small CNN on early raster stacks, both predicting the same severity
   label, with Grad-CAM/saliency for the CNN.
4. **Stretch: next-day spread prediction** -- single-step U-Net as an
   explicit "future work" extension, not the core deliverable.

## Structure

```
src/flashpoint/
    data_access.py   # staging + reading WildfireSpreadTS GeoTIFFs
    db.py            # DuckDB schema (events, daily_rasters, event_outcomes, early_features)
    labels.py        # severity tier derivation from full event trajectories
    features.py      # early-window tabular feature engineering
notebooks/
    01_data_ingestion.ipynb
data/                # local cache (raw GeoTIFFs, duckdb file) -- gitignored
```

## Status

Scaffold only -- `discover_events`, DB schema, label/feature functions are
stubs pending the actual data being staged locally. Channel names/ordering
and the WildfireSpreadTS directory layout are placeholders to confirm
against the dataset documentation once downloaded.
