"""Cost-aware threshold optimization.

The default 0.5 (or the project's 0.6 High tier) cutoff is arbitrary unless you
state what a mistake costs. For an early-warning system the asymmetry is real:
missing an at-risk student (a false negative — no outreach happens) is usually
far more costly than a false alarm (a false positive — a quick, cheap check-in).

`cost_curve` sweeps thresholds and computes total expected cost given a cost
ratio, so the chosen threshold is a defensible decision, not a default.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score


def cost_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    cost_fn: float = 3.0,
    cost_fp: float = 1.0,
    step: float = 0.01,
) -> pd.DataFrame:
    """Sweep thresholds and report counts, P/R/F1, and total cost at each.

    cost_fn : relative cost of a false negative (missed at-risk student)
    cost_fp : relative cost of a false positive (unnecessary outreach)
    """
    thresholds = np.round(np.arange(0.05, 0.96, step), 4)
    yt = np.asarray(y_true)
    rows = []
    for t in thresholds:
        yp = (y_prob >= t).astype(int)
        tp = int(((yt == 1) & (yp == 1)).sum())
        fn = int(((yt == 1) & (yp == 0)).sum())
        fp = int(((yt == 0) & (yp == 1)).sum())
        tn = int(((yt == 0) & (yp == 0)).sum())
        total_cost = cost_fn * fn + cost_fp * fp
        rows.append(
            {
                "threshold": float(t),
                "tp": tp, "fn": fn, "fp": fp, "tn": tn,
                "precision": precision_score(yt, yp, zero_division=0),
                "recall": recall_score(yt, yp, zero_division=0),
                "f1": f1_score(yt, yp, zero_division=0),
                "flagged_rate": float(yp.mean()),
                "total_cost": float(total_cost),
            }
        )
    return pd.DataFrame(rows)


def optimal_threshold(curve: pd.DataFrame, objective: str = "total_cost") -> dict:
    """Pick the best threshold from a cost_curve frame.

    objective="total_cost" -> minimize cost; objective="f1" -> maximize F1.
    """
    if objective == "total_cost":
        best = curve.loc[curve["total_cost"].idxmin()]
    elif objective == "f1":
        best = curve.loc[curve["f1"].idxmax()]
    else:
        raise ValueError(f"Unknown objective: {objective}")
    return best.to_dict()
