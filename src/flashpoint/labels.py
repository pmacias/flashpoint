"""Derive severity/triage labels from full event trajectories.

The core idea: look at the WHOLE trajectory of an event (peak active-fire
extent, total duration) to define a severity tier, but only ever feed the
model data from the early cutoff window (day 1-2). This is what makes it a
fair "early decision" task rather than hindsight classification.

Two important corrections vs. the initial stub:

1. The "active fire" channel is NOT a clean 0/1 mask. Per the dataset
   documentation, it stores the HOUR of last detection (0-23) and is
   zero-filled where there was no detection (99.83% of pixels, dataset-wide).
   Always threshold with `> 0` before treating it as a mask -- this matches
   how the dataset's own training code binarizes it
   (`y = (y > 0).long()` in FireSpreadDataset.preprocess_and_augment).

2. "Final outcome" should be PEAK extent across the trajectory, not the
   extent on the literal last day. The dataset pads each event with four
   buffer days before AND after the official GlobFire dates, and those
   buffer days frequently show zero detected fire -- in this dataset,
   56.5% of the 607 events have zero active-fire pixels on their last day.
   Using the last day alone would score a fire that burned thousands of
   hectares and was later contained as "severity 0", which defeats the
   purpose of a severity label entirely.
"""

from __future__ import annotations

import numpy as np

from flashpoint.data_access import ACTIVE_FIRE_CHANNEL_IDX, PIXEL_AREA_HA


def active_fire_mask(day_raster: np.ndarray) -> np.ndarray:
    """Binarize the active-fire channel of a single day's (23, H, W) raster."""
    return day_raster[ACTIVE_FIRE_CHANNEL_IDX] > 0


def max_area_ha(active_fire_all_days: np.ndarray, pixel_area_ha: float = PIXEL_AREA_HA) -> float:
    """Peak active-fire extent across the ENTIRE event trajectory, in hectares.

    `active_fire_all_days` is a (n_days, H, W) array -- just the active-fire
    channel across all days (see data_access.read_channel_all_days), not the
    full 23-channel stack.
    """
    masks = active_fire_all_days > 0
    per_day_pixel_count = masks.reshape(masks.shape[0], -1).sum(axis=1)
    return float(per_day_pixel_count.max()) * pixel_area_ha


def severity_class(peak_ha: float, bin_edges: list[float]) -> int:
    """Bin an event's eventual outcome into a discrete severity tier.

    `bin_edges` should be chosen from the training set's own distribution
    (e.g. quantiles of max_area_ha across all events) once the HDF5 data
    has been scanned -- the paper doesn't report a "peak size" histogram
    directly (it reports duration and per-frame pixel-change stats instead),
    so these edges need to be computed empirically from our own data, not
    guessed. Run this over the full event set first, look at the
    distribution, THEN set bin_edges -- don't ship a placeholder here.
    """
    for i, edge in enumerate(bin_edges):
        if peak_ha <= edge:
            return i
    return len(bin_edges)
