"""Engineer early-state tabular features from the day-1/day-2 cutoff window.

These feed the GBM (XGBoost) and EBM (InterpretML) baselines, and the
tabular-NN comparison point. Everything here must only look at data from
the cutoff window -- reaching into later days would leak the label.
"""

from __future__ import annotations

import numpy as np

from flashpoint.data_access import CHANNEL_NAMES


def early_window_stats(day_rasters: list[np.ndarray]) -> dict[str, float]:
    """Compute summary stats over the early cutoff window's raster stack.

    `day_rasters` is a list of (C, H, W) arrays for days 1..cutoff_day only.
    Returns a flat dict suitable for a DataFrame row.

    TODO: confirm exact channel indices against CHANNEL_NAMES once data is
    staged -- indices below are placeholders matching the draft ordering
    in data_access.CHANNEL_NAMES.
    """
    stacked = np.stack(day_rasters, axis=0)  # (T, C, H, W)

    def ch(name: str) -> np.ndarray:
        idx = CHANNEL_NAMES.index(name)
        return stacked[:, idx]

    return {
        "fire_extent_ha": float(ch("prev_fire_mask")[-1].sum()),
        "wind_speed_mean": float(ch("wind_speed").mean()),
        "wind_speed_max": float(ch("wind_speed").max()),
        "temp_max": float(ch("max_temp").max()),
        "humidity_min": float(ch("humidity").min()),
        "drought_index": float(ch("drought_index")[-1].mean()),
    }
