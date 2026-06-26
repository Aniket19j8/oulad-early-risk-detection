"""Probability calibration helpers.

A RandomForest's `predict_proba` is a discriminative score, not necessarily a
true probability. If an advisor reads "0.6 risk" as "60% chance", the number
should mean that. These helpers measure calibration (reliability curve, Brier,
ECE) and build a calibrated wrapper around the trained pipeline.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss


def reliability_curve(
    y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10
) -> pd.DataFrame:
    """Bin predictions and compare mean predicted prob vs observed frequency."""
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(y_prob, bins) - 1, 0, n_bins - 1)

    rows = []
    for b in range(n_bins):
        mask = idx == b
        if not mask.any():
            continue
        rows.append(
            {
                "bin": b,
                "bin_lower": bins[b],
                "bin_upper": bins[b + 1],
                "n": int(mask.sum()),
                "mean_predicted": float(y_prob[mask].mean()),
                "observed_frequency": float(y_true[mask].mean()),
            }
        )
    return pd.DataFrame(rows)


def expected_calibration_error(
    y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10
) -> float:
    """Weighted average gap between confidence and accuracy across bins (ECE)."""
    curve = reliability_curve(y_true, y_prob, n_bins=n_bins)
    if curve.empty:
        return float("nan")
    weights = curve["n"] / curve["n"].sum()
    gaps = (curve["mean_predicted"] - curve["observed_frequency"]).abs()
    return float((weights * gaps).sum())


def calibration_report(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> dict:
    """Headline calibration metrics."""
    return {
        "brier": float(brier_score_loss(y_true, y_prob)),
        "ece": expected_calibration_error(y_true, y_prob, n_bins=n_bins),
    }


def build_calibrated_model(
    estimator, X_train, y_train, method: str = "isotonic", cv: int = 5
):
    """Cross-validated probability calibration around a *fresh* estimator.

    Pass an unfitted pipeline (same architecture as the deployed model).
    CalibratedClassifierCV refits it across `cv` folds and learns the calibration
    map on each held-out fold, so the calibration is not fit on data the model
    has already seen. The test set stays untouched for honest evaluation.
    """
    from sklearn.calibration import CalibratedClassifierCV

    calibrated = CalibratedClassifierCV(estimator, method=method, cv=cv)
    calibrated.fit(X_train, y_train)
    return calibrated
