#!/usr/bin/env python
"""Fairness audit: does the early-warning model fail some student groups?

For an intervention-triggering model, the most important fairness metric is the
false-negative rate (FNR) — the share of genuinely at-risk students the model
*misses*. A higher FNR for a protected group means that group is
disproportionately denied the outreach the system exists to provide.

We audit across `disability` and `imd_band` (deprivation index) primarily, with
`age_band` and `gender` as supporting cuts.

Outputs:
  outputs/fairness_by_<group>.csv          (per-group metrics)
  outputs/fairness_summary.csv             (headline disparities per attribute)
  outputs/figures/fairness_<group>.png     (FNR + selection-rate bars)
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from src import config, fairness
from src.features import load_features
from src.model import add_risk_columns, load_model

GROUPS = ["disability", "imd_band", "age_band", "gender"]


def _plot_group(metrics: pd.DataFrame, group_col: str) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))

    axes[0].bar(metrics["group"].astype(str), metrics["fnr_missed"], color="crimson")
    axes[0].set_title(f"Missed at-risk rate (FNR) by {group_col}")
    axes[0].set_ylabel("FNR  (lower = better)")
    axes[0].tick_params(axis="x", rotation=45)

    axes[1].bar(metrics["group"].astype(str), metrics["selection_rate"], color="steelblue")
    axes[1].bar(metrics["group"].astype(str), metrics["base_rate"],
                color="black", alpha=0.0, edgecolor="black", linewidth=1.5,
                label="actual at-risk rate")
    axes[1].set_title(f"Flag rate vs actual risk by {group_col}")
    axes[1].set_ylabel("Rate")
    axes[1].tick_params(axis="x", rotation=45)
    axes[1].legend()

    fig.tight_layout()
    out = config.FIGURES / f"fairness_{group_col}.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out


def main() -> int:
    config.ensure_dirs()
    print("Loading features + model, scoring full cohort...")
    df = load_features()
    model = load_model()
    scored = add_risk_columns(model, df)

    summary_rows = []
    for group_col in GROUPS:
        metrics = fairness.group_metrics(scored, group_col)
        out_csv = config.OUTPUTS / f"fairness_by_{group_col}.csv"
        metrics.round(4).to_csv(out_csv, index=False)
        print(f"\n=== {group_col} ===")
        print(metrics.round(3).to_string(index=False))

        disp = fairness.disparities(metrics)
        for metric_name, vals in disp.items():
            summary_rows.append({
                "attribute": group_col,
                "metric": metric_name,
                "min": round(vals["min"], 4),
                "max": round(vals["max"], 4),
                "gap": round(vals["gap"], 4),
                "worst_group": vals["worst_group"],
            })

        fig = _plot_group(metrics, group_col)
        print(f"  figure -> {fig}")

    summary = pd.DataFrame(summary_rows)
    summary_path = config.OUTPUTS / "fairness_summary.csv"
    summary.to_csv(summary_path, index=False)
    print(f"\nSaved disparity summary -> {summary_path}")

    # Headline callouts for the two priority attributes
    print("\n--- HEADLINE (priority attributes) ---")
    for attr in ["disability", "imd_band"]:
        sub = summary[(summary["attribute"] == attr) & (summary["metric"] == "fnr_missed")]
        if not sub.empty:
            r = sub.iloc[0]
            print(f"{attr}: FNR gap = {r['gap']:.1%} (worst: '{r['worst_group']}' "
                  f"misses {r['max']:.1%} of at-risk students)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
