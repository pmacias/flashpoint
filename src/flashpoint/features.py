"""Engineer early-state tabular features from the day-1/day-2 cutoff window.

These feed the GBM (XGBoost) and EBM (InterpretML) baselines, and the
tabular-NN comparison point. Everything here must only look at data from
the cutoff window -- reaching into later days would leak the label.

Note on missing values: the dataset documentation reports small (~1-1.5%
per-pixel) missingness in the weather/drought channels used here. Plain
np.mean()/np.max()/np.min() propagate to NaN if even a single pixel in the
whole (T, H, W) window is missing -- across a full image that's likely
enough to poison ~8% of events' aggregates entirely. Use the nan-aware
variants (np.nanmean etc.) so one missing pixel doesn't wipe out an
otherwise-valid feature.

Note on wind direction: the dataset's own training code encodes degree
features with `sin` only, which the authors have flagged (unfixed as of
this writing) as losing information -- two different directions can map to
the same sine value. Since we're building our own feature extraction from
scratch, we encode wind direction with BOTH sin and cos, which fixes that
without needing to touch their code.
"""

from __future__ import annotations

import numpy as np

from flashpoint.data_access import CHANNEL_NAMES, PIXEL_AREA_HA
from flashpoint.labels import active_fire_mask


def _channel(stack: np.ndarray, name: str) -> np.ndarray:
    """Slice a named channel out of a (T, C, H, W) stack -> (T, H, W)."""
    return stack[:, CHANNEL_NAMES.index(name)]


def early_window_stats(early_stack: np.ndarray) -> dict[str, float]:
    """Compute summary stats over the early cutoff window's raster stack.

    `early_stack` is a (T, 23, H, W) array for the fire's actual days
    1..cutoff_day -- i.e. sliced starting at BUFFER_DAYS, not index 0,
    since the stored sequences open with ~4 pre-ignition buffer days
    (e.g. read_event_window(event, BUFFER_DAYS, BUFFER_DAYS + cutoff)).
    """
    wind_speed = _channel(early_stack, "wind_speed")
    wind_dir_deg = _channel(early_stack, "wind_direction")
    max_temp = _channel(early_stack, "max_temp")
    min_temp = _channel(early_stack, "min_temp")
    humidity = _channel(early_stack, "specific_humidity")
    pdsi = _channel(early_stack, "pdsi")  # cumulative dryness -- the "hysteresis" feature
    erc = _channel(early_stack, "energy_release_component")

    slope = _channel(early_stack, "slope")
    aspect_deg = _channel(early_stack, "aspect")
    elevation = _channel(early_stack, "elevation")

    ndvi = _channel(early_stack, "ndvi")
    evi2 = _channel(early_stack, "evi2")
    viirs_m11 = _channel(early_stack, "viirs_band_m11")
    viirs_i1 = _channel(early_stack, "viirs_band_i1")
    viirs_i2 = _channel(early_stack, "viirs_band_i2")

    last_day_fire_mask = active_fire_mask(early_stack[-1])

    wind_dir_rad = np.deg2rad(wind_dir_deg)
    # aspect is a compass direction in degrees, so it gets the same sin+cos
    # treatment as wind direction (a raw mean of 1 deg and 359 deg would be
    # 180 deg -- the opposite slope face)
    aspect_rad = np.deg2rad(aspect_deg)

    return {
        "fire_extent_ha": float(last_day_fire_mask.sum()) * PIXEL_AREA_HA,
        "wind_speed_mean": float(np.nanmean(wind_speed)),
        "wind_speed_max": float(np.nanmax(wind_speed)),
        # sin+cos encoding (not sin-only) so 0 deg and 360 deg map to the
        # same point instead of losing direction information
        "wind_direction_sin_mean": float(np.nanmean(np.sin(wind_dir_rad))),
        "wind_direction_cos_mean": float(np.nanmean(np.cos(wind_dir_rad))),
        "max_temp_max": float(np.nanmax(max_temp)),
        "min_temp_min": float(np.nanmin(min_temp)),
        "humidity_min": float(np.nanmin(humidity)),
        "pdsi_mean": float(np.nanmean(pdsi)),
        "erc_mean": float(np.nanmean(erc)),
        "slope_mean": float(np.nanmean(slope)),
        "slope_max": float(np.nanmax(slope)),
        "aspect_sin_mean": float(np.nanmean(np.sin(aspect_rad))),
        "aspect_cos_mean": float(np.nanmean(np.cos(aspect_rad))),
        "elevation_mean": float(np.nanmean(elevation)),
        "ndvi_mean": float(np.nanmean(ndvi)),
        "evi2_mean": float(np.nanmean(evi2)),
        "viirs_m11_mean": float(np.nanmean(viirs_m11)),
        "viirs_i1_mean": float(np.nanmean(viirs_i1)),
        "viirs_i2_mean": float(np.nanmean(viirs_i2)),
    }
