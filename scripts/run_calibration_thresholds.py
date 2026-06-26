#!/usr/bin/env python
"""Calibration + cost-aware threshold selection.

Two senior-level modeling moves on the trained Random Forest:

1. CALIBRATION — is a predicted "0.6" really a 60% chance? We measure the
   reliability curve, Brier score, and ECE, then fit a calibrated wrapper
   (isotonic, cross-validated) and show the improvement. Saved as
   `models/random_forest_calibrated.joblib`.

2. THRESHOLDS — the 0.5/0.6 cutoff is arbitrary unless mistakes have costs.
   Missing an at-risk student (FN) is treated as 3x costlier than a needless
   check-in (FP). We sweep thresholds, find the cost-minimizing one, and compare
   it to the F1-optimal cutoff and the project defaults.

Re-creates notebook 03's exact train/test split so the test set matches.

Outputs:
  models/random_forest_calibrated.joblib
  outputs/calibration_curve.csv
  outputs/calibration_report.csv
  outputs/threshold_analysis.csv
  outputs/figures/calibration_reliability.png
  outputs/figures/threshold_cost.png
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib

matplotlib.use("Agg")
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from src import config
from src.calibration import (
    build_calibrated_model,
    calibration_report,
    reliability_curve,
)
from src.features import load_features
from src.model import load_model
from src.thresholds import cost_curve, optimal_threshold

COST_FN = 3.0  # missing an at-risk student
COST_FP = 1.0  # an unnecessary check-in


def _rebuild_pipeline() -> Pipeline:
    """Same architecture/params as notebook 03, for cross-validated calibration."""
    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), config.CAT_FEATURES),
            ("num", SimpleImputer(strategy="median"), config.NUM_FEATURES),
        ]
    )
    model = RandomForestClassifier(
        n_estimators=200, max_depth=12, class_weight="balanced",
        random_state=config.RANDOM_STATE, n_jobs=-1,
    )
    return Pipeline([("prep", preprocessor), ("model", model)])


def main() -> int:
    config.ensure_dirs()
    df = load_features()
    X = df[config.FEATURE_COLS]
    y = df[config.TARGET_COL]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=config.RANDOM_STATE,
    )
    y_test_arr = y_test.to_numpy()

    # ---- 1. Calibration -----------------------------------------------------
    print("Scoring test set with the deployed (uncalibrated) model...")
    base = load_model()
    prob_uncal = base.predict_proba(X_test)[:, 1]
    rep_uncal = calibration_report(y_test_arr, prob_uncal)
    print(f"  uncalibrated: Brier={rep_uncal['brier']:.4f}  ECE={rep_uncal['ece']:.4f}")

    print("Fitting cross-validated isotonic calibration (this retrains a few RFs)...")
    calibrated = build_calibrated_model(
        _rebuild_pipeline(), X_train, y_train, method="isotonic"
    )
    prob_cal = calibrated.predict_proba(X_test)[:, 1]
    rep_cal = calibration_report(y_test_arr, prob_cal)
    print(f"  calibrated:   Brier={rep_cal['brier']:.4f}  ECE={rep_cal['ece']:.4f}")

    joblib.dump(calibrated, config.CALIBRATED_MODEL_PATH)
    print(f"  saved calibrated model -> {config.CALIBRATED_MODEL_PATH}")

    curve_uncal = reliability_curve(y_test_arr, prob_uncal).assign(model="uncalibrated")
    curve_cal = reliability_curve(y_test_arr, prob_cal).assign(model="calibrated")
    pd.concat([curve_uncal, curve_cal], ignore_index=True).round(4).to_csv(
        config.OUTPUTS / "calibration_curve.csv", index=False
    )
    pd.DataFrame([
        {"model": "uncalibrated", **rep_uncal},
        {"model": "calibrated", **rep_cal},
    ]).round(4).to_csv(config.OUTPUTS / "calibration_report.csv", index=False)

    # Reliability figure
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0, 1], [0, 1], ls="--", color="gray", label="perfect calibration")
    ax.plot(curve_uncal["mean_predicted"], curve_uncal["observed_frequency"],
            marker="o", color="crimson",
            label=f"uncalibrated (ECE={rep_uncal['ece']:.3f})")
    ax.plot(curve_cal["mean_predicted"], curve_cal["observed_frequency"],
            marker="s", color="seagreen",
            label=f"calibrated (ECE={rep_cal['ece']:.3f})")
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Observed at-risk frequency")
    ax.set_title("Reliability diagram (test set)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(config.FIGURES / "calibration_reliability.png", dpi=130)
    plt.close(fig)
    print(f"  figure -> {config.FIGURES / 'calibration_reliability.png'}")

    # ---- 2. Cost-aware thresholds (on the deployed model) -------------------
    print(f"\nSweeping thresholds (FN cost = {COST_FN}x FP cost)...")
    curve = cost_curve(y_test_arr, prob_uncal, cost_fn=COST_FN, cost_fp=COST_FP)
    curve.round(4).to_csv(config.OUTPUTS / "threshold_analysis.csv", index=False)

    best_cost = optimal_threshold(curve, "total_cost")
    best_f1 = optimal_threshold(curve, "f1")

    def at(t):
        r = curve.iloc[(curve["threshold"] - t).abs().idxmin()]
        return r

    print("\nThreshold comparison:")
    print(f"  {'threshold':>10} {'precision':>10} {'recall':>9} {'f1':>7} {'flagged':>9} {'cost':>10}")
    for label, t in [("cost-optimal", best_cost["threshold"]),
                     ("F1-optimal", best_f1["threshold"]),
                     ("default 0.50", 0.50),
                     ("High tier 0.60", 0.60)]:
        r = at(t)
        print(f"  {label:>14} {r['threshold']:.2f}  "
              f"{r['precision']:.3f}    {r['recall']:.3f}  {r['f1']:.3f}  "
              f"{r['flagged_rate']:.3f}   {r['total_cost']:.0f}")

    # Cost + P/R figure
    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax1.plot(curve["threshold"], curve["total_cost"], color="black", label="total cost")
    ax1.axvline(best_cost["threshold"], color="crimson", ls="--",
                label=f"cost-optimal = {best_cost['threshold']:.2f}")
    ax1.axvline(0.6, color="steelblue", ls=":", label="current High tier = 0.60")
    ax1.set_xlabel("Decision threshold")
    ax1.set_ylabel(f"Total cost  (FN={COST_FN}x, FP={COST_FP}x)")
    ax2 = ax1.twinx()
    ax2.plot(curve["threshold"], curve["precision"], color="seagreen", alpha=0.7, label="precision")
    ax2.plot(curve["threshold"], curve["recall"], color="darkorange", alpha=0.7, label="recall")
    ax2.set_ylabel("Precision / Recall")
    ax1.set_title("Cost-aware threshold selection")
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper center", fontsize=8)
    fig.tight_layout()
    fig.savefig(config.FIGURES / "threshold_cost.png", dpi=130)
    plt.close(fig)
    print(f"\n  figure -> {config.FIGURES / 'threshold_cost.png'}")

    print(f"\nReasoning: with FN {COST_FN}x costlier than FP, the cost-minimizing "
          f"threshold is {best_cost['threshold']:.2f} "
          f"(recall {at(best_cost['threshold'])['recall']:.1%}). The current 0.60 "
          f"High-tier cutoff favors precision; lowering it toward the cost-optimal "
          f"point catches more at-risk students at the price of more check-ins.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
