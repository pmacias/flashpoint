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

Note on forecast channels: at day index t the forecast_* channels hold GFS
forecasts VALID FOR day t+1 but ISSUED BY day t. So the last window day's
forecast planes describe the first day past the cutoff using only
information available at the cutoff -- the one legitimate look-ahead in the
dataset, and the reason these features don't violate the leakage rule.
Only the last window day is used: earlier days' forecasts describe days
inside the window, redundant with the observed channels. The *_delta
features (forecast minus same-day observed) directly encode "conditions
are forecast to ease/worsen after the window". Units caveat: GFS
forecast_temperature is stored in CELSIUS (per-event window means span
~-4 to ~34, verified in notebook 03 Step 2b) while GRIDMET min/max_temp
are Kelvin, so the forecast temp is converted to Kelvin here before the
delta. The products still differ (GFS mean-ish temp vs GRIDMET daily max),
so deltas carry a calibration offset -- fine for tree/EBM models, don't
read them as physics.
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

    # Forecast channels, LAST window day only (see module docstring): day
    # index t holds the GFS forecast valid for t+1, so [-1] describes the
    # first day past the cutoff -- available at cutoff time, not leakage.
    fc_precip = _channel(early_stack, "forecast_total_precipitation")[-1]
    fc_wind_speed = _channel(early_stack, "forecast_wind_speed")[-1]
    fc_wind_dir_rad = np.deg2rad(_channel(early_stack, "forecast_wind_direction")[-1])
    # GFS forecast temp is Celsius; observed GRIDMET temps are Kelvin --
    # convert so forecast_temp_delta compares like with like (see docstring)
    fc_temp = _channel(early_stack, "forecast_temperature")[-1] + 273.15
    fc_humidity = _channel(early_stack, "forecast_specific_humidity")[-1]

    slope = _channel(early_stack, "slope")
    aspect_deg = _channel(early_stack, "aspect")
    elevation = _channel(early_stack, "elevation")

    ndvi = _channel(early_stack, "ndvi")
    evi2 = _channel(early_stack, "evi2")
    viirs_m11 = _channel(early_stack, "viirs_band_m11")
    viirs_i1 = _channel(early_stack, "viirs_band_i1")
    viirs_i2 = _channel(early_stack, "viirs_band_i2")

    last_day_fire_mask = active_fire_mask(early_stack[-1])

    # Same-day observed values for the *_delta features: forecast (valid
    # day t+1) minus observed (day t), so negative temp / positive humidity
    # deltas mean "conditions forecast to ease after the window".
    fc_temp_mean = float(np.nanmean(fc_temp))
    fc_humidity_mean = float(np.nanmean(fc_humidity))
    fc_wind_speed_mean = float(np.nanmean(fc_wind_speed))

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
        "forecast_precip_mean": float(np.nanmean(fc_precip)),
        "forecast_precip_max": float(np.nanmax(fc_precip)),
        "forecast_wind_speed_mean": fc_wind_speed_mean,
        "forecast_wind_speed_max": float(np.nanmax(fc_wind_speed)),
        "forecast_wind_dir_sin_mean": float(np.nanmean(np.sin(fc_wind_dir_rad))),
        "forecast_wind_dir_cos_mean": float(np.nanmean(np.cos(fc_wind_dir_rad))),
        "forecast_temp_mean": fc_temp_mean,
        "forecast_humidity_mean": fc_humidity_mean,
        "forecast_temp_delta": fc_temp_mean - float(np.nanmean(max_temp[-1])),
        "forecast_humidity_delta": fc_humidity_mean - float(np.nanmean(humidity[-1])),
        "forecast_wind_speed_delta": fc_wind_speed_mean - float(np.nanmean(wind_speed[-1])),
    }
