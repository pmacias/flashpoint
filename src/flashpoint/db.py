"""DuckDB schema for the wildfire triage project.

Mirrors the pattern from steam_trajectory: a compact, portable local database
holding metadata, labels, and engineered features -- while raw per-day raster
data stays on disk (or remote) and is referenced by path, not duplicated into
the DB.
"""

from __future__ import annotations

from pathlib import Path

import duckdb

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS events (
    event_id      VARCHAR PRIMARY KEY,
    start_date    DATE,
    end_date      DATE,
    n_days        INTEGER,
    centroid_lat  DOUBLE,
    centroid_lon  DOUBLE
);

CREATE TABLE IF NOT EXISTS daily_rasters (
    event_id   VARCHAR REFERENCES events(event_id),
    day_index  INTEGER,
    date       DATE,
    tif_path   VARCHAR,
    PRIMARY KEY (event_id, day_index)
);

-- Eventual outcome + derived severity tier, computed over the FULL event
-- trajectory (not just the early cutoff window).
CREATE TABLE IF NOT EXISTS event_outcomes (
    event_id        VARCHAR PRIMARY KEY REFERENCES events(event_id),
    final_area_ha   DOUBLE,
    duration_days   INTEGER,
    severity_class  INTEGER   -- e.g. 0=contained locally ... 3=major mobilization
);

-- Engineered tabular features computed ONLY from the early cutoff window
-- (day 1-2), used by the GBM/EBM/tabular-NN branch.
CREATE TABLE IF NOT EXISTS early_features (
    event_id        VARCHAR PRIMARY KEY REFERENCES events(event_id),
    cutoff_day      INTEGER,
    fire_extent_ha  DOUBLE,
    wind_speed_mean DOUBLE,
    wind_speed_max  DOUBLE,
    temp_max        DOUBLE,
    humidity_min    DOUBLE,
    drought_index   DOUBLE
    -- extend as feature engineering (week 2) progresses
);
"""


def get_connection(db_path: Path) -> duckdb.DuckDBPyConnection:
    con = duckdb.connect(str(db_path))
    con.execute(SCHEMA_SQL)
    return con
