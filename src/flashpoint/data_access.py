"""Staging and reading WildfireSpreadTS event data.

WildfireSpreadTS (Gerard et al. 2023) ships as a directory of per-event,
per-day GeoTIFF stacks, ~50GB total, hosted on Zenodo (DOI 10.5281/zenodo.8006177).
This module assumes the raw GeoTIFFs have already been downloaded/extracted
to `raw_data_dir` (see notebooks/01_data_ingestion.ipynb for the fetch step)
and focuses on turning that directory tree into something the rest of the
pipeline can enumerate and read lazily.

We deliberately do NOT load full event stacks into memory here -- only
metadata (paths, shapes, dates) goes into the manifest that gets written to
the DuckDB layer. Actual pixel data is read on demand at train time.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import rasterio


# Channel names as documented in the WildfireSpreadTS paper (23 channels:
# active fire / weather / fuel / topography). Fill in exact order once the
# data is staged and you've confirmed it against the dataset documentation --
# treat this list as a starting point, not ground truth.
CHANNEL_NAMES = [
    "prev_fire_mask",
    "wind_speed",
    "wind_direction",
    "min_temp",
    "max_temp",
    "humidity",
    "precipitation",
    "drought_index",
    "ndvi",
    "elevation",
    "slope",
    "aspect",
    # ... remaining channels TBD from dataset docs
]


@dataclass
class EventDay:
    event_id: str
    day_index: int
    date: str
    tif_path: Path


@dataclass
class EventManifestEntry:
    event_id: str
    n_days: int
    start_date: str
    end_date: str
    days: list[EventDay]


def discover_events(raw_data_dir: Path) -> list[EventManifestEntry]:
    """Walk the staged WildfireSpreadTS directory and build per-event manifests.

    TODO: adapt the glob pattern below once the data is staged locally --
    this assumes a layout of `raw_data_dir/<event_id>/<day_index>_<date>.tif`,
    which is a guess pending confirmation against the actual archive layout.
    """
    events: list[EventManifestEntry] = []
    for event_dir in sorted(p for p in raw_data_dir.iterdir() if p.is_dir()):
        day_files = sorted(event_dir.glob("*.tif"))
        if not day_files:
            continue
        days = [
            EventDay(
                event_id=event_dir.name,
                day_index=i,
                date=f.stem.split("_", 1)[-1],
                tif_path=f,
            )
            for i, f in enumerate(day_files)
        ]
        events.append(
            EventManifestEntry(
                event_id=event_dir.name,
                n_days=len(days),
                start_date=days[0].date,
                end_date=days[-1].date,
                days=days,
            )
        )
    return events


def read_day_raster(day: EventDay) -> np.ndarray:
    """Read a single day's multi-channel raster as a (C, H, W) array."""
    with rasterio.open(day.tif_path) as src:
        return src.read()
