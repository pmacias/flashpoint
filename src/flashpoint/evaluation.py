"""Leave-one-year-out evaluation.

Year-based splits are the project's evaluation backbone: yearly fire
distributions vary a lot (2019 is a known outlier -- fewer, smaller fires),
so a random shuffle would leak year-level distribution information into
training. Notebook 02 previously duplicated this loop per experiment; it
lives here so every result is computed by the same code.
"""

from __future__ import annotations

from typing import Callable

import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, recall_score


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
    whole dataset -- not an average of per-fold matrices.
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
            pred_frames.append(pd.DataFrame(
                {"test_year": test_year, "y_true": te[target_col].to_numpy(), "y_pred": preds},
                index=te.index,
            ))
    metrics = pd.DataFrame(rows)
    if return_predictions:
        return metrics, pd.concat(pred_frames)
    return metrics
