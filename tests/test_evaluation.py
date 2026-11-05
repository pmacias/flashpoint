"""Tests for flashpoint.evaluation: LOYO out-of-fold probabilities, the
calibration curve computation, and the Brier decomposition.

All inputs here are small, synthetic arrays and frames -- no DuckDB access.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss

from flashpoint.evaluation import (
    brier_decomposition,
    calibration_curve_data,
    leave_one_year_out,
)


@pytest.fixture
def loyo_df():
    rng = np.random.default_rng(0)
    n = 120
    x = rng.normal(size=n)
    return pd.DataFrame({
        "year": np.repeat([2018, 2019, 2020], n // 3),
        "x": x,
        "target": (x + rng.normal(scale=0.5, size=n) > 0).astype(int),
    })


# ---------------------------------------------------------------------------
# leave_one_year_out
# ---------------------------------------------------------------------------

def test_loyo_y_prob_covers_every_event_and_matches_y_pred(loyo_df):
    _, preds = leave_one_year_out(
        loyo_df, ["x"], "target", LogisticRegression,
        positive_class=1, return_predictions=True,
    )
    assert preds.index.sort_values().equals(loyo_df.index)
    assert preds["y_prob"].between(0, 1).all()
    assert ((preds["y_prob"] > 0.5).astype(int) == preds["y_pred"]).all()


def test_loyo_y_prob_is_probability_of_positive_class(loyo_df):
    _, preds_1 = leave_one_year_out(
        loyo_df, ["x"], "target", LogisticRegression,
        positive_class=1, return_predictions=True,
    )
    _, preds_0 = leave_one_year_out(
        loyo_df, ["x"], "target", LogisticRegression,
        positive_class=0, return_predictions=True,
    )
    np.testing.assert_allclose(preds_1["y_prob"] + preds_0["y_prob"], 1.0)


def test_loyo_metrics_unchanged_by_prediction_collection(loyo_df):
    plain = leave_one_year_out(loyo_df, ["x"], "target", LogisticRegression, positive_class=1)
    with_preds, _ = leave_one_year_out(
        loyo_df, ["x"], "target", LogisticRegression,
        positive_class=1, return_predictions=True,
    )
    pd.testing.assert_frame_equal(plain, with_preds)


def test_loyo_omits_y_prob_without_positive_class(loyo_df):
    _, preds = leave_one_year_out(
        loyo_df, ["x"], "target", LogisticRegression, return_predictions=True,
    )
    assert "y_prob" not in preds.columns


# ---------------------------------------------------------------------------
# calibration_curve_data
# ---------------------------------------------------------------------------

def test_calibration_equal_count_bins():
    y_prob = np.linspace(0.01, 0.99, 100)
    y_true = (np.arange(100) % 2).astype(int)
    curve, _ = calibration_curve_data(y_true, y_prob)
    assert curve["n"].tolist() == [20] * 5
    assert curve["bin"].tolist() == list(range(5))
    assert curve["mean_pred"].is_monotonic_increasing


def test_calibration_bin_values_by_hand():
    y_prob = np.array([0.1, 0.1, 0.2, 0.2, 0.8, 0.8, 0.9, 0.9, 0.95, 0.95])
    y_true = np.array([0, 0, 0, 1, 1, 1, 1, 0, 1, 1])
    curve, _ = calibration_curve_data(y_true, y_prob, n_bins=2)
    # the median is 0.8, and both 0.8s stay together in the lower bin
    assert curve["n"].tolist() == [6, 4]
    assert curve["obs_rate"].tolist() == pytest.approx([0.5, 0.75])
    assert curve["mean_pred"].tolist() == pytest.approx([2.2 / 6, 0.925])


def test_calibration_ties_share_a_bin_and_collapse_edges():
    y_prob = np.array([0.2] * 80 + [0.3, 0.4, 0.5, 0.6, 0.7] * 4)
    y_true = np.zeros(100, dtype=int)
    curve, _ = calibration_curve_data(y_true, y_prob)
    assert len(curve) < 5
    assert curve["n"].sum() == 100
    assert curve.loc[0, "n"] == 80
    assert curve.loc[0, "prob_max"] == 0.2


def test_calibration_all_tied_gives_one_bin():
    curve, _ = calibration_curve_data(np.array([0, 1, 1, 1]), np.full(4, 0.75))
    assert curve["n"].tolist() == [4]


def test_calibration_wilson_interval_known_value():
    # 8/10 at 95%: textbook Wilson interval
    y_true = np.array([0, 0] + [1] * 8)
    curve, _ = calibration_curve_data(y_true, np.full(10, 0.5))
    assert curve.loc[0, "ci_lower"] == pytest.approx(0.4902, abs=1e-3)
    assert curve.loc[0, "ci_upper"] == pytest.approx(0.9433, abs=1e-3)


@pytest.mark.parametrize("label", [0, 1])
def test_calibration_interval_brackets_rate_in_pure_bins(label):
    curve, _ = calibration_curve_data(np.full(20, label), np.linspace(0.1, 0.9, 20))
    assert (curve["obs_rate"] - curve["ci_lower"] >= 0).all()
    assert (curve["ci_upper"] - curve["obs_rate"] >= 0).all()
    assert ((curve["ci_upper"] - curve["ci_lower"]) > 0).all()


def test_calibration_brier_matches_definition():
    y_true = np.array([0, 1, 1, 0])
    y_prob = np.array([0.1, 0.9, 0.6, 0.3])
    _, brier = calibration_curve_data(y_true, y_prob, n_bins=2)
    assert brier == pytest.approx(np.mean((y_prob - y_true) ** 2))
    assert brier == pytest.approx(brier_score_loss(y_true, y_prob))


def test_calibration_accepts_float32_probabilities():
    y_true = np.array([0, 1, 1, 0, 1])
    y_prob = np.array([0.1, 0.9, 0.6, 0.3, 0.7], dtype=np.float32)
    curve, brier = calibration_curve_data(y_true, y_prob, n_bins=2)
    assert curve["n"].sum() == 5
    assert np.isfinite(brier)


def test_calibration_rejects_non_binary_labels():
    with pytest.raises(ValueError, match="0/1"):
        calibration_curve_data(np.array([0, 1, 2]), np.array([0.1, 0.5, 0.9]))


# ---------------------------------------------------------------------------
# brier_decomposition
# ---------------------------------------------------------------------------

def _identity_sum(terms):
    return terms["rel"] - terms["res"] + terms["unc"] + terms["wbv"] - terms["wbc"]


@pytest.mark.parametrize("n_bins", [1, 2, 5, 10, 50])
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_brier_decomposition_identity_continuous_probabilities(n_bins, seed):
    rng = np.random.default_rng(seed)
    y_prob = rng.beta(2, 1, size=500)
    y_true = (rng.random(500) < y_prob ** 2).astype(int)
    terms = brier_decomposition(y_true, y_prob, n_bins=n_bins)
    assert _identity_sum(terms) == pytest.approx(terms["brier"], rel=1e-12)
    assert terms["brier"] == pytest.approx(brier_score_loss(y_true, y_prob), rel=1e-12)


def test_brier_decomposition_identity_when_ties_collapse_bins():
    y_prob = np.array([0.2] * 80 + [0.3, 0.4, 0.5, 0.6, 0.7] * 4)
    y_true = np.array(([0, 0, 0, 1] * 25))
    terms = brier_decomposition(y_true, y_prob)
    assert _identity_sum(terms) == pytest.approx(terms["brier"], rel=1e-12)


def test_brier_decomposition_terms_by_hand():
    # two bins of two events: (0.2, 0.4) with outcomes (0, 1), (0.6, 0.8) with (1, 1)
    y_prob = np.array([0.2, 0.4, 0.6, 0.8])
    y_true = np.array([0, 1, 1, 1])
    terms = brier_decomposition(y_true, y_prob, n_bins=2)
    assert terms["rel"] == pytest.approx((0.2 ** 2 + 0.3 ** 2) / 2)
    assert terms["res"] == pytest.approx(0.25 ** 2)
    assert terms["unc"] == pytest.approx(0.75 * 0.25)
    assert terms["wbv"] == pytest.approx(0.02 / 2)
    assert terms["wbc"] == pytest.approx(2 * 0.1 / 4)


def test_brier_decomposition_uses_calibration_curve_bins():
    rng = np.random.default_rng(3)
    y_prob = rng.random(200)
    y_true = (rng.random(200) < y_prob).astype(int)
    curve, brier = calibration_curve_data(y_true, y_prob)
    terms = brier_decomposition(y_true, y_prob)
    weights = curve["n"] / curve["n"].sum()
    assert terms["brier"] == brier
    assert terms["rel"] == pytest.approx(
        (weights * (curve["mean_pred"] - curve["obs_rate"]) ** 2).sum()
    )
    assert terms["res"] == pytest.approx(
        (weights * (curve["obs_rate"] - y_true.mean()) ** 2).sum()
    )


def test_brier_decomposition_constant_base_rate_forecast_has_zero_skill():
    y_true = np.array([0, 1, 1, 1])
    terms = brier_decomposition(y_true, np.full(4, 0.75))
    assert terms["brier"] == pytest.approx(terms["unc"])
    assert terms["bss"] == pytest.approx(0.0)
    assert terms["rel"] == pytest.approx(0.0)
    assert terms["res"] == pytest.approx(0.0)


def test_brier_decomposition_single_class_has_undefined_skill():
    terms = brier_decomposition(np.ones(4, dtype=int), np.array([0.5, 0.6, 0.7, 0.8]))
    assert terms["unc"] == 0
    assert np.isnan(terms["bss"])
    assert _identity_sum(terms) == pytest.approx(terms["brier"])


def test_brier_decomposition_rejects_non_binary_labels():
    with pytest.raises(ValueError, match="0/1"):
        brier_decomposition(np.array([0, 1, 2]), np.array([0.1, 0.5, 0.9]))
