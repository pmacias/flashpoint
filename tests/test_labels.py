"""Tests for the pure severity/triage-label functions in flashpoint.labels.

All inputs here are small, hand-constructed synthetic arrays -- no HDF5 or
DuckDB access. See CHANNEL_NAMES in data_access.py for the 23-channel order;
these tests only care about index 22 (active_fire).
"""

from __future__ import annotations

import numpy as np
import pytest

from flashpoint.data_access import ACTIVE_FIRE_CHANNEL_IDX, CHANNEL_NAMES, PIXEL_AREA_HA
from flashpoint.labels import active_fire_mask, max_area_ha, severity_class


# ---------------------------------------------------------------------------
# severity_class
# ---------------------------------------------------------------------------

@pytest.fixture
def bin_edges():
    return [10.0, 50.0, 100.0]


def test_severity_class_below_first_edge(bin_edges):
    assert severity_class(5.0, bin_edges) == 0


def test_severity_class_exactly_at_first_edge(bin_edges):
    # boundary is <=, so a value exactly on an edge belongs to the lower tier
    assert severity_class(10.0, bin_edges) == 0


def test_severity_class_between_first_and_second_edge(bin_edges):
    assert severity_class(30.0, bin_edges) == 1


def test_severity_class_exactly_at_second_edge(bin_edges):
    assert severity_class(50.0, bin_edges) == 1


def test_severity_class_between_second_and_third_edge(bin_edges):
    assert severity_class(75.0, bin_edges) == 2


def test_severity_class_exactly_at_last_edge(bin_edges):
    assert severity_class(100.0, bin_edges) == 2


def test_severity_class_above_last_edge(bin_edges):
    # strictly above every edge -> falls into the final, unbounded tier
    assert severity_class(500.0, bin_edges) == len(bin_edges)


# ---------------------------------------------------------------------------
# active_fire_mask
# ---------------------------------------------------------------------------

@pytest.fixture
def day_raster():
    """A synthetic (23, H, W) single-day raster with a mixed active-fire plane.

    Active-fire channel (index 22) values, laid out on a 2x3 grid:
        [[ 0,  1,  5],
         [23,  0,  0]]
    i.e. detection hours 1, 5, 23 at three pixels, and exact zero (no
    detection) at the other three.
    """
    stack = np.zeros((len(CHANNEL_NAMES), 2, 3), dtype=np.float32)
    stack[ACTIVE_FIRE_CHANNEL_IDX] = np.array([[0, 1, 5], [23, 0, 0]], dtype=np.float32)
    return stack


def test_active_fire_mask_true_only_where_positive(day_raster):
    mask = active_fire_mask(day_raster)
    expected = np.array([[False, True, True], [True, False, False]])
    np.testing.assert_array_equal(mask, expected)


def test_active_fire_mask_zero_is_false_not_true(day_raster):
    # Regression test: the active-fire channel stores detection HOUR (0-23),
    # not a 0/1 mask -- an exact-zero pixel means "no detection" and must
    # binarize to False, not be treated as truthy/hour-zero-detection.
    mask = active_fire_mask(day_raster)
    assert mask[0, 0] == False  # noqa: E712 -- explicit bool check is the point
    assert mask.dtype == np.bool_


# ---------------------------------------------------------------------------
# max_area_ha
# ---------------------------------------------------------------------------

def test_max_area_ha_returns_peak_day_not_sum_or_last_day():
    # 4 days x 5x5 grid. "On" pixel counts per day: 2, 7, 3, 1 -- peak is
    # day index 1 (7 pixels), which is neither the sum (13) nor the last
    # day's count (1).
    h, w = 5, 5
    active_fire_all_days = np.zeros((4, h, w), dtype=np.float32)

    def set_on_pixels(day_idx, n_on):
        flat = active_fire_all_days[day_idx].ravel()
        flat[:n_on] = 12.0  # any positive detection-hour value
        active_fire_all_days[day_idx] = flat.reshape(h, w)

    set_on_pixels(0, 2)
    set_on_pixels(1, 7)
    set_on_pixels(2, 3)
    set_on_pixels(3, 1)

    result = max_area_ha(active_fire_all_days)
    assert result == pytest.approx(7 * PIXEL_AREA_HA)


def test_max_area_ha_custom_pixel_area():
    active_fire_all_days = np.zeros((2, 3, 3), dtype=np.float32)
    active_fire_all_days[0, 0, 0] = 5.0  # 1 "on" pixel on day 0
    active_fire_all_days[1] = 0.0  # nothing on day 1

    result = max_area_ha(active_fire_all_days, pixel_area_ha=2.5)
    assert result == pytest.approx(2.5)
