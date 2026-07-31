"""Derive severity/triage labels from full event trajectories.

The core idea: look at the WHOLE trajectory of an event (final burned area,
total duration) to define a severity tier, but only ever feed the model
data from the early cutoff window (day 1-2). This is what makes it a fair
"early decision" task rather than hindsight classification.
"""

from __future__ import annotations

import numpy as np


def final_area_ha(fire_masks: list[np.ndarray], pixel_area_ha: float) -> float:
    """Burned/active-fire area at the last observed day, in hectares.

    `fire_masks` is the list of per-day active-fire mask arrays for one
    event (boolean or 0/1), in day order.
    """
    return float(fire_masks[-1].sum()) * pixel_area_ha


def severity_class(final_ha: float, duration_days: int, bin_edges: list[float]) -> int:
    """Bin an event's eventual outcome into a discrete severity tier.

    `bin_edges` should be chosen from the training set's distribution
    (e.g. quantiles) once real data is staged -- placeholder edges below
    are NOT calibrated to anything yet.
    """
    for i, edge in enumerate(bin_edges):
        if final_ha <= edge:
            return i
    return len(bin_edges)


# Placeholder -- replace with quantile-derived edges from the actual
# event distribution once data is staged (week 1 deliverable).
DEFAULT_BIN_EDGES_HA = [10.0, 100.0, 1000.0]
