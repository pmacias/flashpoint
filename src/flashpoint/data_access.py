"""Staging and reading WildfireSpreadTS event data.

WildfireSpreadTS (Gerard et al. 2023) ships as ~50GB of GeoTIFFs on Zenodo
(DOI 10.5281/zenodo.8006177). We've already converted the full archive to
HDF5 via the authors' own `CreateHDF5Dataset.py`, so this module reads from
that HDF5 layout as the primary path -- much faster than re-reading GeoTIFFs
per event, which was the whole point of the conversion.

HDF5 layout produced by their conversion script:
    <hdf5_dir>/<year>/<fire_name>.hdf5
        dataset "data": shape (n_days, 23, H, W), float32
        attrs: year, fire_name, img_dates (list of "YYYY-MM-DD" strings), lnglat

We deliberately do NOT load full event stacks into memory at manifest-build
time -- only metadata (paths, shapes, dates) goes into the DuckDB layer.
Actual pixel data is read on demand.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np

# Real channel order, confirmed against the WildfireSpreadTS documentation
# (Supplementary Material, "Composition" section) and cross-checked against
# FireSpreadDataset.map_channel_index_to_features(only_base=True) in their
# repo -- this is the RAW GeoTIFF/HDF5 order, before the land-cover one-hot
# expansion and binary-active-fire-mask addition that their training code
# adds downstream. 23 channels, 0-indexed.
CHANNEL_NAMES = [
    "viirs_band_m11",       # 0
    "viirs_band_i2",        # 1
    "viirs_band_i1",        # 2
    "ndvi",                 # 3
    "evi2",                 # 4
    "total_precipitation",  # 5
    "wind_speed",           # 6
    "wind_direction",       # 7  -- degrees; encode as sin AND cos (their own
                             #      code only does sin, see features.py docstring)
    "min_temp",              # 8
    "max_temp",              # 9
    "energy_release_component",  # 10
    "specific_humidity",     # 11
    "slope",                 # 12
    "aspect",                # 13
    "elevation",              # 14
    "pdsi",                   # 15  Palmer Drought Severity Index -- our "cumulative dryness" feature
    "landcover_class",        # 16  categorical, not yet one-hot expanded
    "forecast_total_precipitation",  # 17
    "forecast_wind_speed",           # 18
    "forecast_wind_direction",       # 19
    "forecast_temperature",          # 20
    "forecast_specific_humidity",    # 21
    "active_fire",            # 22  -- detection HOUR (0-23), 0 = no detection.
                               #       Binarize with (channel > 0), don't sum raw values.
]

ACTIVE_FIRE_CHANNEL_IDX = CHANNEL_NAMES.index("active_fire")

# Active fire maps are natively 375m resolution (per the dataset paper);
# everything else is resampled to match. 375m x 375m in hectares:
PIXEL_SIDE_M = 375
PIXEL_AREA_HA = (PIXEL_SIDE_M * PIXEL_SIDE_M) / 10_000


@dataclass
class HDF5Event:
    event_id: str          # fire_name
    year: int
    hdf5_path: Path
    n_days: int
    img_dates: list[str]
    lnglat: tuple[float, float]


def discover_hdf5_events(hdf5_dir: Path) -> list[HDF5Event]:
    """Walk the converted HDF5 directory and build per-event metadata.

    Reads only attrs (cheap) -- does not touch the "data" dataset itself.
    """
    events: list[HDF5Event] = []
    for year_dir in sorted(p for p in hdf5_dir.iterdir() if p.is_dir()):
        for h5_path in sorted(year_dir.glob("*.hdf5")):
            with h5py.File(h5_path, "r") as f:
                dset = f["data"]
                events.append(
                    HDF5Event(
                        event_id=dset.attrs["fire_name"],
                        year=int(dset.attrs["year"]),
                        hdf5_path=h5_path,
                        n_days=dset.shape[0],
                        img_dates=list(dset.attrs["img_dates"]),
                        lnglat=tuple(dset.attrs["lnglat"]),
                    )
                )
    return events


def read_event_stack(event: HDF5Event) -> np.ndarray:
    """Read a full event's (n_days, 23, H, W) array.

    Convenience function for interactive single-event inspection (e.g. in
    paper_questions.ipynb) -- the pipeline itself always reads narrower via
    read_event_window or read_channel_all_days.
    """
    with h5py.File(event.hdf5_path, "r") as f:
        return f["data"][:]


def read_event_window(event: HDF5Event, day_start: int, day_end: int) -> np.ndarray:
    """Read only a day-range slice (e.g. the early cutoff window), avoiding a
    full-event read when only the first day_end days are needed."""
    with h5py.File(event.hdf5_path, "r") as f:
        return f["data"][day_start:day_end]


def read_channel_all_days(event: HDF5Event, channel_idx: int) -> np.ndarray:
    """Read a single channel across ALL days of an event -> (n_days, H, W).

    Avoids reading all 23 channels when only one is needed (e.g. active fire,
    for computing peak extent across the full trajectory).
    """
    with h5py.File(event.hdf5_path, "r") as f:
        return f["data"][:, channel_idx, :, :]
