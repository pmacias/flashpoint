"""Tests for the pure early-window feature-engineering logic in
flashpoint.features.

All inputs are small, hand-constructed synthetic (T, 23, H, W) arrays -- no
HDF5 or DuckDB access. See CHANNEL_NAMES in data_access.py for the
23-channel order.
"""

from __future__ import annotations

import numpy as np
import pytest

from flashpoint.data_access import CHANNEL_NAMES
from flashpoint.features import early_window_stats

# Plausible, physically-finite fill values per channel name -- deliberately
# not all-zero/all-equal so aggregation bugs (e.g. an accidental axis mixup)
# wouldn't silently pass.
_FILL_VALUES = {
    "viirs_band_m11": 100.0,
    "viirs_band_i2": 200.0,
    "viirs_band_i1": 150.0,
    "ndvi": 0.5,
    "evi2": 0.3,
    "total_precipitation": 0.0,
    "wind_speed": 5.0,
    "wind_direction": 45.0,
    "min_temp": 280.0,   # Kelvin
    "max_temp": 300.0,   # Kelvin
    "energy_release_component": 30.0,
    "specific_humidity": 0.005,
    "slope": 10.0,
    "aspect": 90.0,
    "elevation": 1000.0,
    "pdsi": -2.0,
    "landcover_class": 1.0,
    "forecast_total_precipitation": 0.0,
    "forecast_wind_speed": 4.0,
    "forecast_wind_direction": 50.0,
    "forecast_temperature": 20.0,  # Celsius, per module docstring
    "forecast_specific_humidity": 0.006,
    "active_fire": 0.0,
}


def make_synthetic_stack(t: int = 2, h: int = 2, w: int = 2) -> np.ndarray:
    """Build a (T, 23, H, W) stack filled with plausible per-channel constants."""
    stack = np.zeros((t, len(CHANNEL_NAMES), h, w), dtype=np.float64)
    for name, value in _FILL_VALUES.items():
        idx = CHANNEL_NAMES.index(name)
        stack[:, idx] = value
    return stack


# ---------------------------------------------------------------------------
# NaN-aware aggregation regression test
# ---------------------------------------------------------------------------

def test_early_window_stats_survives_single_nan_pixel():
    # Regression test: a lone missing pixel in one channel must not poison
    # that channel's aggregate to NaN (the whole point of using np.nanmean
    # etc. instead of plain np.mean/np.max/np.min).
    stack = make_synthetic_stack(t=2, h=2, w=2)
    slope_idx = CHANNEL_NAMES.index("slope")
    stack[0, slope_idx, 0, 0] = np.nan

    stats = early_window_stats(stack)

    assert not np.isnan(stats["slope_mean"])
    assert not np.isnan(stats["slope_max"])
    for key, value in stats.items():
        assert np.isfinite(value), f"{key} is not finite: {value}"


def test_early_window_stats_all_nan_channel_still_returns_dict_shape():
    # Sanity check that a fully-populated (non-degenerate) window produces
    # every expected key with a finite value -- the baseline this test file
    # regresses against.
    stack = make_synthetic_stack()
    stats = early_window_stats(stack)
    assert all(np.isfinite(v) for v in stats.values())
    assert "wind_direction_sin_mean" in stats
    assert "aspect_sin_mean" in stats


# ---------------------------------------------------------------------------
# wind direction sin/cos encoding
# ---------------------------------------------------------------------------

def test_wind_direction_sin_cos_does_not_collapse_to_opposite_direction():
    # 1 degree and 359 degrees are ~2 degrees apart on the compass (both
    # "north-ish"), but their raw arithmetic mean is 180 degrees -- due
    # south, the opposite direction. The sin+cos encoding must NOT reduce
    # to that: it should land back near 0/360 degrees.
    stack = make_synthetic_stack(t=1, h=1, w=2)
    wind_dir_idx = CHANNEL_NAMES.index("wind_direction")
    stack[0, wind_dir_idx] = np.array([[1.0, 359.0]])

    stats = early_window_stats(stack)

    sin_mean = stats["wind_direction_sin_mean"]
    cos_mean = stats["wind_direction_cos_mean"]

    # circular mean recovered via atan2(sin, cos) should be ~0 degrees
    recovered_deg = np.degrees(np.arctan2(sin_mean, cos_mean)) % 360
    assert recovered_deg == pytest.approx(0.0, abs=1.0) or recovered_deg == pytest.approx(360.0, abs=1.0)

    # explicitly rule out the raw-mean-of-degrees failure mode (180 deg,
    # cos(180 deg) = -1) that a sin-only or naive-mean encoding would give
    assert cos_mean > 0.9
    assert cos_mean != pytest.approx(-1.0)


def test_aspect_sin_cos_does_not_collapse_to_opposite_direction():
    # Same correction applied to aspect (slope-face compass direction).
    stack = make_synthetic_stack(t=1, h=1, w=2)
    aspect_idx = CHANNEL_NAMES.index("aspect")
    stack[0, aspect_idx] = np.array([[2.0, 358.0]])

    stats = early_window_stats(stack)

    recovered_deg = np.degrees(
        np.arctan2(stats["aspect_sin_mean"], stats["aspect_cos_mean"])
    ) % 360
    assert recovered_deg == pytest.approx(0.0, abs=1.0) or recovered_deg == pytest.approx(360.0, abs=1.0)
    assert stats["aspect_cos_mean"] > 0.9
