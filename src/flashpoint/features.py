"""Engineer early-state tabular features from the day-1/day-2 cutoff window.

These feed the GBM (XGBoost) and EBM (InterpretML) baselines, and the
tabular-NN comparison point. Everything here must only look at data from
the cutoff window -- reaching into later days would leak the label.

Note on wind direction: the dataset's own training code encodes degree
features with `sin` only, which the authors have flagged (unfixed as of
this writing) as losing information -- two different directions can map to
the same sine value. Since we're building our own feature extraction from
scratch, we encode wind direction with BOTH sin and cos, which fixes that
without needing to touch their code.
"""

from __future__ import annotations

import numpy as np

from flashpoint.data_access import CHANNEL_NAMES, ACTIVE_FIRE_CHANNEL_IDX, PIXEL_AREA_HA
from flashpoint.labels import active_fire_mask


def _channel(stack: np.ndarray, name: str) -> np.ndarray:
    """Slice a named channel out of a (T, C, H, W) stack -> (T, H, W)."""
    return stack[:, CHANNEL_NAMES.index(name)]


def early_window_stats(early_stack: np.ndarray) -> dict[str, float]:
    """Compute summary stats over the early cutoff window's raster stack.

    `early_stack` is a (T, 23, H, W) array for days 1..cutoff_day only
    (e.g. from data_access.read_event_window).
    """
    wind_speed = _channel(early_stack, "wind_speed")
    wind_dir_deg = _channel(early_stack, "wind_direction")
    max_temp = _channel(early_stack, "max_temp")
    min_temp = _channel(early_stack, "min_temp")
    humidity = _channel(early_stack, "specific_humidity")
    pdsi = _channel(early_stack, "pdsi")  # cumulative dryness -- the "hysteresis" feature
    erc = _channel(early_stack, "energy_release_component")

    last_day_fire_mask = active_fire_mask(early_stack[-1])

    wind_dir_rad = np.deg2rad(wind_dir_deg)

    return {
        "fire_extent_ha": float(last_day_fire_mask.sum()) * PIXEL_AREA_HA,
        "wind_speed_mean": float(wind_speed.mean()),
        "wind_speed_max": float(wind_speed.max()),
        # sin+cos encoding (not sin-only) so 0 deg and 360 deg map to the
        # same point instead of losing direction information
        "wind_direction_sin_mean": float(np.sin(wind_dir_rad).mean()),
        "wind_direction_cos_mean": float(np.cos(wind_dir_rad).mean()),
        "max_temp_max": float(max_temp.max()),
        "min_temp_min": float(min_temp.min()),
        "humidity_min": float(humidity.min()),
        "pdsi_mean": float(pdsi.mean()),
        "erc_mean": float(erc.mean()),
    }
