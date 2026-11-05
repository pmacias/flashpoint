"""Leave-one-year-out evaluation.

Year-based splits are the project's evaluation backbone: yearly fire
distributions vary a lot (2019 is a known outlier -- fewer, smaller fires),
so a random shuffle would leak year-level distribution information into
training. Notebook 02 previously duplicated this loop per experiment; it
lives here so every result is computed by the same code.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd
from scipy.stats import binomtest
from sklearn.metrics import accuracy_score, brier_score_loss, f1_score, recall_score


def leave_one_year_out(
    df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
    model_factory: Callable,
    positive_class: int | None = None,
    return_predictions: bool = False,
) -> pd.DataFrame | tuple[pd.DataFrame, pd.DataFrame]:
    """Fit and score a fresh model per held-out year.

    `df` must carry a "year" column alongside `feature_cols` and `target_col`.
    For each year, a new estimator from `model_factory()` (anything with the
    sklearn fit/predict API) is trained on all other years and scored on the
    held-out one.

    Returns a per-year DataFrame with test_year, n_test, accuracy, macro_f1,
    and -- when `positive_class` is given -- positive_recall: recall on that
    class alone. For the binary escalation target this is the "how many
    escalating fires did we catch" number, which matters asymmetrically:
    missing an escalating fire is a worse outcome than a false alarm, so it
    is reported independently of the symmetric macro-F1 average.

    With `return_predictions=True`, also returns a second DataFrame of
    per-event out-of-fold predictions (columns test_year, y_true, y_pred),
    indexed like `df`. Because the year folds are disjoint and cover every
    event exactly once, these pool into one true confusion matrix over the
    whole dataset -- not an average of per-fold matrices. When `positive_class`
    is given and the model has `predict_proba`, the frame also carries y_prob,
    the out-of-fold predicted probability of that class. It is collected
    alongside `predict`, which still drives every metric and y_pred above.
    """
    rows = []
    pred_frames = []
    for test_year in sorted(df["year"].unique()):
        tr = df[df["year"] != test_year]
        te = df[df["year"] == test_year]
        model = model_factory()
        model.fit(tr[feature_cols], tr[target_col])
        preds = model.predict(te[feature_cols])
        row = {
            "test_year": test_year,
            "n_test": len(te),
            "accuracy": accuracy_score(te[target_col], preds),
            "macro_f1": f1_score(te[target_col], preds, average="macro"),
        }
        if positive_class is not None:
            row["positive_recall"] = recall_score(
                te[target_col], preds, labels=[positive_class], average=None
            )[0]
        rows.append(row)
        if return_predictions:
            pred_frame = pd.DataFrame(
                {"test_year": test_year, "y_true": te[target_col].to_numpy(), "y_pred": preds},
                index=te.index,
            )
            if positive_class is not None and hasattr(model, "predict_proba"):
                positive_idx = list(model.classes_).index(positive_class)
                pred_frame["y_prob"] = model.predict_proba(te[feature_cols])[:, positive_idx]
            pred_frames.append(pred_frame)
    metrics = pd.DataFrame(rows)
    if return_predictions:
        return metrics, pd.concat(pred_frames)
    return metrics


def _validate_binary_forecast(y_true, y_prob) -> tuple[np.ndarray, np.ndarray]:
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob, dtype=float)
    if not np.isin(y_true, [0, 1]).all():
        raise ValueError("y_true must contain only 0/1 labels")
    return y_true.astype(int), y_prob


def _quantile_bin_index(y_prob: np.ndarray, n_bins: int) -> np.ndarray:
    """Per-event quantile bin labels; tied probabilities always share a bin."""
    quantile_edges = np.quantile(y_prob, np.linspace(0, 1, n_bins + 1))
    return np.searchsorted(np.unique(quantile_edges[1:-1]), y_prob, side="left")


def calibration_curve_data(
    y_true,
    y_prob,
    n_bins: int = 5,
    confidence: float = 0.95,
) -> tuple[pd.DataFrame, float]:
    """Quantile-binned reliability curve and Brier score for a binary target.

    `y_true` holds 0/1 labels and `y_prob` the predicted probability of class
    1. Quantile bins hold roughly equal event counts, so every point on the
    curve rests on a similar sample even when predictions pile up near 1.
    Tied probabilities always share a bin; if ties collapse quantile edges,
    fewer than `n_bins` rows come back rather than an empty or split bin.

    Returns a DataFrame with one row per bin (bin, n, prob_min, prob_max,
    mean_pred, obs_rate, ci_lower, ci_upper) and the Brier score over all
    events. The interval is a Wilson binomial CI on obs_rate: unlike the
    normal approximation it stays inside [0, 1] and has nonzero width when a
    bin is all one class, which is likely in the top bin here. It covers
    sampling error in the observed rate only, treating each event as an
    independent draw.
    """
    y_true, y_prob = _validate_binary_forecast(y_true, y_prob)
    brier = float(brier_score_loss(y_true, y_prob))
    bin_idx = _quantile_bin_index(y_prob, n_bins)

    rows = []
    for i, b in enumerate(np.unique(bin_idx)):
        in_bin = bin_idx == b
        n = int(in_bin.sum())
        n_positive = int(y_true[in_bin].sum())
        ci = binomtest(n_positive, n).proportion_ci(confidence, method="wilson")
        rows.append({
            "bin": i,
            "n": n,
            "prob_min": y_prob[in_bin].min(),
            "prob_max": y_prob[in_bin].max(),
            "mean_pred": y_prob[in_bin].mean(),
            "obs_rate": n_positive / n,
            "ci_lower": ci.low,
            "ci_upper": ci.high,
        })
    return pd.DataFrame(rows), brier


def brier_decomposition(y_true, y_prob, n_bins: int = 5) -> dict[str, float]:
    """Murphy decomposition of the Brier score, on calibration_curve_data's bins.

    Same inputs and same quantile bins as `calibration_curve_data`, so the
    terms describe the reliability curve it returns. Binning continuous
    probabilities leaves a remainder that Murphy's three terms don't capture,
    so this adds the two within-bin terms of Stephenson, Coelho & Jolliffe
    (2008), which make the identity exact:

        brier = rel - res + unc + wbv - wbc

    With N events, bins k of size n_k, mean forecast f_k and observed rate o_k,
    and overall rate o:

    - rel: reliability, sum n_k (f_k - o_k)^2 / N. Lower is better.
    - res: resolution, sum n_k (o_k - o)^2 / N. Higher is better.
    - unc: uncertainty, o (1 - o). A property of the outcomes, the same for
      every model, and the Brier score of a constant forecast at o.
    - wbv: within-bin variance of the forecasts, sum_k sum_i (f_ki - f_k)^2 / N.
    - wbc: within-bin covariance, 2 sum_k sum_i (f_ki - f_k)(o_ki - o_k) / N.
    - brier: the Brier score, identical to calibration_curve_data's.
    - bss: Brier skill score against that constant forecast, 1 - brier / unc.
      NaN when unc is 0 (only one class present).

    Because bins are quantiles of each model's own predictions, rel and res
    depend on `n_bins` and aren't computed on a common partition across models.
    """
    y_true, y_prob = _validate_binary_forecast(y_true, y_prob)
    bin_idx = _quantile_bin_index(y_prob, n_bins)
    n_total = len(y_true)
    base_rate = y_true.mean()

    rel = res = wbv = wbc = 0.0
    for b in np.unique(bin_idx):
        in_bin = bin_idx == b
        forecasts, outcomes = y_prob[in_bin], y_true[in_bin]
        mean_pred, obs_rate = forecasts.mean(), outcomes.mean()
        rel += in_bin.sum() * (mean_pred - obs_rate) ** 2
        res += in_bin.sum() * (obs_rate - base_rate) ** 2
        wbv += ((forecasts - mean_pred) ** 2).sum()
        wbc += 2 * ((forecasts - mean_pred) * (outcomes - obs_rate)).sum()

    brier = float(brier_score_loss(y_true, y_prob))
    unc = float(base_rate * (1 - base_rate))
    return {
        "brier": brier,
        "rel": float(rel / n_total),
        "res": float(res / n_total),
        "unc": unc,
        "wbv": float(wbv / n_total),
        "wbc": float(wbc / n_total),
        "bss": 1 - brier / unc if unc > 0 else float("nan"),
    }
