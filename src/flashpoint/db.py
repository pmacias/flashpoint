"""DuckDB schema for the wildfire triage project.

Mirrors the pattern from steam_trajectory: a compact, portable local database
holding metadata, labels, and engineered features -- while raw per-day raster
data stays on disk (HDF5, converted from the original GeoTIFFs) and is
referenced by path, not duplicated into the DB.
"""

from __future__ import annotations

from pathlib import Path

import duckdb

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS events (
    event_id      VARCHAR PRIMARY KEY,
    year          INTEGER,   -- for year-based CV splits, per the paper's own
                              -- recommendation (yearly distributions vary a lot,
                              -- e.g. 2019 has far fewer/smaller fires than other years)
    hdf5_path     VARCHAR,
    n_days        INTEGER,
    centroid_lon  DOUBLE,
    centroid_lat  DOUBLE
);

-- Eventual outcome + derived severity tier, computed over the FULL event
-- trajectory (not just the early cutoff window).
CREATE TABLE IF NOT EXISTS event_outcomes (
    event_id        VARCHAR PRIMARY KEY REFERENCES events(event_id),
    max_area_ha     DOUBLE,   -- PEAK active-fire extent across the full trajectory,
                              -- not the last day (buffer days often show zero fire)
    duration_days   INTEGER,
    severity_class  INTEGER   -- e.g. 0=contained locally ... 3=major mobilization
);

-- Engineered tabular features computed ONLY from the early cutoff window
-- (day 1-2), used by the GBM/EBM/tabular-NN branch.
CREATE TABLE IF NOT EXISTS early_features (
    event_id                VARCHAR PRIMARY KEY REFERENCES events(event_id),
    cutoff_day               INTEGER,
    fire_extent_ha           DOUBLE,
    wind_speed_mean          DOUBLE,
    wind_speed_max           DOUBLE,
    wind_direction_sin_mean  DOUBLE,
    wind_direction_cos_mean  DOUBLE,
    max_temp_max             DOUBLE,
    min_temp_min             DOUBLE,
    humidity_min             DOUBLE,
    pdsi_mean                DOUBLE,
    erc_mean                 DOUBLE,
    slope_mean               DOUBLE,
    slope_max                DOUBLE,
    aspect_sin_mean          DOUBLE,
    aspect_cos_mean          DOUBLE,
    elevation_mean           DOUBLE,
    ndvi_mean                DOUBLE,
    evi2_mean                DOUBLE,
    viirs_m11_mean           DOUBLE,
    viirs_i1_mean            DOUBLE,
    viirs_i2_mean            DOUBLE
);
"""


def get_connection(db_path: Path) -> duckdb.DuckDBPyConnection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(db_path))
    con.execute(SCHEMA_SQL)
    return con
