"""Group fairness metrics for the at-risk classifier.

The key question for an early-warning system is not just "is it accurate?" but
"does it miss at-risk students more often for some groups than others?" A high
false-negative rate (FNR) for, say, students with a declared disability would
mean the system silently denies them the very intervention it exists to trigger.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _safe_div(a: float, b: float) -> float:
    return float(a) / float(b) if b else np.nan


def group_metrics(
    df: pd.DataFrame,
    group_col: str,
    y_true: str = "at_risk",
    y_pred: str = "predicted_at_risk",
    y_prob: str = "risk_probability",
    min_group_size: int = 30,
) -> pd.DataFrame:
    """Per-group classification metrics.

    Returns one row per group value with:
      - n, base_rate (actual at-risk rate)
      - selection_rate (predicted at-risk rate)  -> demographic parity view
      - tpr / recall, fnr (missed at-risk), fpr, tnr
      - precision, accuracy
      - mean_predicted_prob
    Groups smaller than `min_group_size` are dropped to avoid noisy rates.
    """
    rows = []
    for value, g in df.groupby(group_col, dropna=False):
        n = len(g)
        if n < min_group_size:
            continue
        yt = g[y_true].to_numpy()
        yp = g[y_pred].to_numpy()

        tp = int(((yt == 1) & (yp == 1)).sum())
        fn = int(((yt == 1) & (yp == 0)).sum())
        fp = int(((yt == 0) & (yp == 1)).sum())
        tn = int(((yt == 0) & (yp == 0)).sum())

        rows.append(
            {
                "group": value if pd.notna(value) else "(missing)",
                "n": n,
                "base_rate": float(yt.mean()),
                "selection_rate": float(yp.mean()),
                "tpr_recall": _safe_div(tp, tp + fn),
                "fnr_missed": _safe_div(fn, tp + fn),
                "fpr": _safe_div(fp, fp + tn),
                "precision": _safe_div(tp, tp + fp),
                "accuracy": _safe_div(tp + tn, n),
                "mean_predicted_prob": float(g[y_prob].mean()),
            }
        )

    out = pd.DataFrame(rows).sort_values("group").reset_index(drop=True)
    return out


def disparities(metrics: pd.DataFrame, reference: str | None = None) -> dict:
    """Summarize fairness gaps across groups for a metrics table.

    `reference` selects a baseline group for ratios (e.g. the majority group);
    if None, ratios are computed against the most favorable group.
    Returns a dict of headline disparities.
    """
    m = metrics.set_index("group")

    def ratio(col: str, higher_is_better: bool) -> dict:
        vals = m[col]
        if reference is not None and reference in vals.index:
            base = vals[reference]
        else:
            base = vals.max() if higher_is_better else vals.min()
        return {
            "min": float(vals.min()),
            "max": float(vals.max()),
            "gap": float(vals.max() - vals.min()),
            "worst_group": str(vals.idxmin() if higher_is_better else vals.idxmax()),
            "ratio_to_reference": float((vals / base).replace([np.inf, -np.inf], np.nan).min()),
        }

    return {
        "selection_rate": ratio("selection_rate", higher_is_better=True),
        "tpr_recall": ratio("tpr_recall", higher_is_better=True),
        "fnr_missed": ratio("fnr_missed", higher_is_better=False),
        "precision": ratio("precision", higher_is_better=True),
    }
