"""Build fixed-size early-window raster crops for the CNN arm (Phase B).

See docs/cnn_plan.md (Q1/Q2) for the full reasoning; summary here:

- **Crop, don't resize.** All events share 375m/px resolution; only the
  bounding box H/W vary. Resizing would destroy physical scale, which is
  the one thing a CNN can exploit that the tabular aggregates can't (fire
  position relative to terrain). Crops are centered on the early-window
  active-fire union centroid, falling back to the box center when the
  early window shows no fire pixels yet (a real, documented ~26% case --
  not new leakage, since the box itself is drawn around the event's full
  footprint, same geometry the tabular whole-box aggregates already use).
- **26 planes, one pass, from `read_event_window(event, BUFFER_DAYS,
  BUFFER_DAYS + cutoff_day)`** -- the same leakage-respecting read path
  `features.early_window_stats` uses. Static terrain channels are
  nanmean'd over the window (they don't change day to day; nanmean is
  just a cheap way to shrug off a stray missing pixel). Weather channels
  are kept per-day rather than aggregated, since day-to-day change is
  itself a plausible spatial signal for a CNN. Forecast channels use only
  the last window day, per the same t -> t+1 argument as features.py.
- **NaN-padded at edges.** Only ~11% of events clip a raster edge (median
  pad 24px); no separate validity plane in v1. Padding and true missing
  weather pixels both get filled downstream by the training-fold mean
  (see cnn.py), so there's no need to distinguish them here.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from tqdm import tqdm

from flashpoint.data_access import BUFFER_DAYS, CHANNEL_NAMES, HDF5Event, read_event_window
from flashpoint.labels import active_fire_mask

# Plane groups, in the fixed order they're written to the crop store.
# Forecast planes are stored regardless of whether Phase A adopted them for
# the tabular model (storage is free); `default_plane_names` excludes them.
_STATIC_TERRAIN = ["slope", "elevation"]  # aspect handled separately (sin/cos)
_SLOW_VARYING = ["ndvi", "pdsi"]
_PER_DAY_WEATHER = [
    "energy_release_component", "max_temp", "wind_speed", "specific_humidity",
]  # wind_direction handled separately (sin/cos per day)
_FORECAST_LAST_DAY = [
    "forecast_total_precipitation", "forecast_wind_speed", "forecast_temperature",
    "forecast_specific_humidity",
]  # wind direction handled separately (sin/cos)

# Friendly plane-name prefixes matching docs/cnn_plan.md's table.
_WEATHER_SHORT_NAME = {
    "energy_release_component": "erc",
    "max_temp": "max_temp",
    "wind_speed": "wind_speed",
    "specific_humidity": "humidity",
}
_FORECAST_SHORT_NAME = {
    "forecast_total_precipitation": "fc_precip",
    "forecast_wind_speed": "fc_wind_speed",
    "forecast_temperature": "fc_temp",
    "forecast_specific_humidity": "fc_humidity",
}


def _channel(stack: np.ndarray, name: str) -> np.ndarray:
    """Slice a named channel out of a (T, 23, H, W) stack -> (T, H, W)."""
    return stack[:, CHANNEL_NAMES.index(name)]


def early_fire_center(early_stack: np.ndarray) -> tuple[int, int, bool]:
    """Union-mask centroid of active fire across the early window.

    `early_stack` is (T, 23, H, W), the same leakage-respecting window
    `features.early_window_stats` consumes. Returns (cy, cx, used_fallback).
    When the early window shows zero fire pixels (156/607 events, 25.7%,
    per docs/cnn_plan.md), falls back to the raster box center -- a
    reasonable prior since the box is drawn around the event's full
    eventual footprint.
    """
    union = np.zeros(early_stack.shape[-2:], dtype=bool)
    for day in range(early_stack.shape[0]):
        union |= active_fire_mask(early_stack[day])

    h, w = union.shape
    if not union.any():
        return h // 2, w // 2, True

    ys, xs = np.nonzero(union)
    return int(round(ys.mean())), int(round(xs.mean())), False


def extract_crop(stack: np.ndarray, cy: int, cx: int, size: int) -> tuple[np.ndarray, bool]:
    """Crop a (C, H, W) plane stack to (C, size, size), centered at (cy, cx).

    NaN-pads whatever falls outside the source raster. Returns (crop,
    was_edge_clipped) so callers can count how often padding kicked in.
    """
    c, h, w = stack.shape
    half_lo = size // 2
    half_hi = size - half_lo

    row_start, row_end = cy - half_lo, cy + half_hi
    col_start, col_end = cx - half_lo, cx + half_hi

    pad_top = max(0, -row_start)
    pad_left = max(0, -col_start)
    src_row_start, src_row_end = max(row_start, 0), min(row_end, h)
    src_col_start, src_col_end = max(col_start, 0), min(col_end, w)

    was_clipped = (
        row_start < 0 or col_start < 0 or row_end > h or col_end > w
    )

    crop = np.full((c, size, size), np.nan, dtype=np.float32)
    dst_row = slice(pad_top, pad_top + (src_row_end - src_row_start))
    dst_col = slice(pad_left, pad_left + (src_col_end - src_col_start))
    crop[:, dst_row, dst_col] = stack[:, src_row_start:src_row_end, src_col_start:src_col_end]

    return crop, was_clipped


def build_plane_stack(early_stack: np.ndarray) -> tuple[np.ndarray, list[str]]:
    """Assemble the 26-plane stack for one event's early window.

    `early_stack` is (T, 23, H, W) from `read_event_window(event,
    BUFFER_DAYS, BUFFER_DAYS + cutoff_day)` -- T is the number of early-
    window days (2 in the project's default CUTOFF_DAY), and per-day plane
    names below (`_d1`, `_d2`, ...) are 1-indexed over that axis. Returns
    (planes, names) with planes shaped (26, H, W), float32.
    """
    n_days = early_stack.shape[0]
    planes: list[np.ndarray] = []
    names: list[str] = []

    for name in _STATIC_TERRAIN:
        planes.append(np.nanmean(_channel(early_stack, name), axis=0))
        names.append(name)

    aspect_rad = np.deg2rad(_channel(early_stack, "aspect"))
    planes.append(np.nanmean(np.sin(aspect_rad), axis=0)); names.append("aspect_sin")
    planes.append(np.nanmean(np.cos(aspect_rad), axis=0)); names.append("aspect_cos")

    for name in _SLOW_VARYING:
        planes.append(np.nanmean(_channel(early_stack, name), axis=0))
        names.append(name)

    for name in _PER_DAY_WEATHER:
        chan = _channel(early_stack, name)
        short = _WEATHER_SHORT_NAME[name]
        for day in range(n_days):
            planes.append(chan[day].astype(np.float32))
            names.append(f"{short}_d{day + 1}")

    wind_dir_rad = np.deg2rad(_channel(early_stack, "wind_direction"))
    for day in range(n_days):
        planes.append(np.sin(wind_dir_rad[day]).astype(np.float32))
        names.append(f"wind_dir_sin_d{day + 1}")
        planes.append(np.cos(wind_dir_rad[day]).astype(np.float32))
        names.append(f"wind_dir_cos_d{day + 1}")

    for day in range(n_days):
        planes.append(active_fire_mask(early_stack[day]).astype(np.float32))
        names.append(f"active_fire_d{day + 1}")

    for name in _FORECAST_LAST_DAY:
        last = _channel(early_stack, name)[-1].astype(np.float32)
        if name == "forecast_temperature":
            # GFS forecast_temperature is Celsius; observed GRIDMET temps
            # are Kelvin (see features.py docstring) -- convert so this
            # plane is on the same scale as max_temp_d*.
            last = last + 273.15
        planes.append(last)
        names.append(_FORECAST_SHORT_NAME[name])

    fc_wind_dir_rad = np.deg2rad(_channel(early_stack, "forecast_wind_direction")[-1])
    planes.append(np.sin(fc_wind_dir_rad).astype(np.float32)); names.append("fc_wind_dir_sin")
    planes.append(np.cos(fc_wind_dir_rad).astype(np.float32)); names.append("fc_wind_dir_cos")

    return np.stack(planes).astype(np.float32), names


def default_plane_names(names: list[str]) -> list[str]:
    """The 20-of-26 default training subset: all planes except forecast.

    Phase A found forecast features a null result for the EBM (the CNN's
    comparison target), so they're stored but excluded by default -- see
    docs/cnn_plan.md status note. Callers wanting the +forecast ablation
    pass the full `names` list instead.
    """
    return [n for n in names if not n.startswith("fc_")]


@dataclass
class CropStoreStats:
    n_events: int
    n_fallback_center: int
    n_edge_clipped: int


def build_crop_store(
    events: pd.DataFrame,
    out_path: Path,
    store_size: int = 144,
    cutoff_day: int = 2,
) -> CropStoreStats:
    """One pass over `events`, writing the crop store to `out_path`.

    `events` needs columns event_id, year, hdf5_path, n_days, centroid_lon,
    centroid_lat (the `events` table's own schema). The only pixel read per
    event is `read_event_window(event, BUFFER_DAYS, BUFFER_DAYS +
    cutoff_day)` -- preserves the project's early-window leakage rule.
    Writes datasets `crops` (n, 26, store_size, store_size) f32, `event_id`,
    `year`; attrs `plane_names`, `cutoff_day`, `buffer_days`,
    `n_fallback_center`, `n_edge_clipped`.
    """
    crops = []
    event_ids = []
    years = []
    plane_names: list[str] | None = None
    n_fallback = 0
    n_clipped = 0

    for row in tqdm(events.itertuples(), total=len(events)):
        event = HDF5Event(
            event_id=row.event_id, year=int(row.year), hdf5_path=Path(row.hdf5_path),
            n_days=row.n_days, img_dates=[], lnglat=(row.centroid_lon, row.centroid_lat),
        )
        early_stack = read_event_window(event, BUFFER_DAYS, BUFFER_DAYS + cutoff_day)

        cy, cx, used_fallback = early_fire_center(early_stack)
        planes, names = build_plane_stack(early_stack)
        crop, was_clipped = extract_crop(planes, cy, cx, store_size)

        if plane_names is None:
            plane_names = names
        elif plane_names != names:
            raise ValueError(f"plane name mismatch for {event.event_id}")

        crops.append(crop)
        event_ids.append(event.event_id)
        years.append(event.year)
        n_fallback += used_fallback
        n_clipped += was_clipped

    crops_arr = np.stack(crops)  # (n, 26, store_size, store_size)
    assert plane_names is not None

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(out_path, "w") as f:
        f.create_dataset("crops", data=crops_arr)
        f.create_dataset("event_id", data=np.array(event_ids, dtype=h5py.string_dtype()))
        f.create_dataset("year", data=np.array(years, dtype=np.int32))
        f.attrs["plane_names"] = np.array(plane_names, dtype=h5py.string_dtype())
        f.attrs["cutoff_day"] = cutoff_day
        f.attrs["buffer_days"] = BUFFER_DAYS
        f.attrs["store_size"] = store_size
        f.attrs["n_fallback_center"] = n_fallback
        f.attrs["n_edge_clipped"] = n_clipped

    return CropStoreStats(
        n_events=len(events), n_fallback_center=n_fallback, n_edge_clipped=n_clipped,
    )
