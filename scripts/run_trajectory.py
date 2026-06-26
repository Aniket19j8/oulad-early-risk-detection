#!/usr/bin/env python
"""Weekly risk trajectory: how early-warning risk evolves over weeks 1-3.

We rebuild the early-window features at successive cutoffs (day 7, 14, 21) and
score each snapshot with the trained model. This turns a single day-21 snapshot
into a *trajectory* per student, so you can see whether risk is rising, falling,
or stable — which is far more actionable than one number.

Honest caveat (state it in the interview): the model is trained on the 21-day
window and applied to the shorter cumulative windows. So weeks 1-2 are an
"apply the early-warning model at an earlier checkpoint" view, not a separately
trained weekly model. It answers "how early could we have flagged this student?"

Outputs:
  outputs/tableau_risk_trajectory.csv          (long: one row per student-week)
  outputs/tableau_risk_trajectory_summary.csv  (mean risk by week x actual outcome)
  outputs/figures/trajectory_mean_risk.png
  outputs/figures/trajectory_samples.png
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src import config
from src.features import build_early_features, load_raw_tables
from src.model import load_model, score

WEEK_LABEL = {7: "Week 1 (d7)", 14: "Week 2 (d14)", 21: "Week 3 (d21)"}


def main() -> int:
    config.ensure_dirs()
    print("Loading raw OULAD tables (studentVle is large, this takes a moment)...")
    tables = load_raw_tables()
    model = load_model()

    snapshots = []
    for window in config.TRAJECTORY_WINDOWS:
        print(f"  building + scoring features at day {window} ...")
        feats = build_early_features(tables, window_days=window)
        feats = feats.copy()
        feats["risk_probability"] = score(model, feats)
        feats["risk_tier"] = feats["risk_probability"].map(config.risk_tier)
        feats["window_day"] = window
        feats["week_label"] = WEEK_LABEL[window]
        snapshots.append(
            feats[config.ID_COLS + ["final_result", "at_risk", "window_day",
                                    "week_label", "risk_probability", "risk_tier"]]
        )

    traj = pd.concat(snapshots, ignore_index=True)

    # Wide view + trajectory direction (week1 -> week3 delta)
    wide = traj.pivot_table(
        index=config.ID_COLS + ["final_result", "at_risk"],
        columns="window_day", values="risk_probability",
    ).reset_index()
    wide["risk_delta_w1_w3"] = wide[21] - wide[7]
    wide["trajectory"] = pd.cut(
        wide["risk_delta_w1_w3"],
        bins=[-np.inf, -0.05, 0.05, np.inf],
        labels=["falling", "stable", "rising"],
    )
    traj = traj.merge(
        wide[config.ID_COLS + ["risk_delta_w1_w3", "trajectory"]],
        on=config.ID_COLS, how="left",
    )

    traj_path = config.OUTPUTS / "tableau_risk_trajectory.csv"
    traj.to_csv(traj_path, index=False)
    print(f"Saved {len(traj):,} rows -> {traj_path}")

    # Summary: mean risk by week split by eventual outcome
    summary = (
        traj.groupby(["window_day", "week_label", "at_risk"], as_index=False)
        .agg(mean_risk=("risk_probability", "mean"),
             n=("id_student", "count"))
    )
    summary_path = config.OUTPUTS / "tableau_risk_trajectory_summary.csv"
    summary.to_csv(summary_path, index=False)
    print(f"Saved summary -> {summary_path}")

    # Tier migration counts (how High/Med/Low populations grow week over week)
    tier_counts = (
        traj.groupby(["window_day", "risk_tier"], as_index=False)
        .agg(n=("id_student", "count"))
    )
    tier_counts.to_csv(config.OUTPUTS / "tableau_risk_trajectory_tiers.csv", index=False)

    # ---- Figure 1: mean risk by week, separated by eventual outcome ----------
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for outcome, label, color in [(1, "Eventually at-risk", "crimson"),
                                  (0, "Eventually OK", "seagreen")]:
        s = summary[summary["at_risk"] == outcome].sort_values("window_day")
        ax.plot(s["window_day"], s["mean_risk"], marker="o", label=label, color=color)
    ax.set_xticks(config.TRAJECTORY_WINDOWS)
    ax.set_xlabel("Days into course (cumulative early window)")
    ax.set_ylabel("Mean predicted risk")
    ax.set_title("Risk trajectory: the gap opens early")
    ax.legend()
    fig.tight_layout()
    fig1 = config.FIGURES / "trajectory_mean_risk.png"
    fig.savefig(fig1, dpi=130)
    plt.close(fig)
    print(f"Saved figure -> {fig1}")

    # ---- Figure 2: a sample of individual student trajectories ----------------
    rng = np.random.default_rng(config.RANDOM_STATE)
    sample_ids = (wide.sample(min(40, len(wide)), random_state=config.RANDOM_STATE))
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for _, r in sample_ids.iterrows():
        ys = [r[7], r[14], r[21]]
        color = "crimson" if r["at_risk"] == 1 else "seagreen"
        ax.plot(config.TRAJECTORY_WINDOWS, ys, color=color, alpha=0.35, marker="o", ms=3)
    ax.axhline(config.RISK_HIGH, ls="--", color="black", lw=0.8, label=f"High tier ({config.RISK_HIGH})")
    ax.set_xticks(config.TRAJECTORY_WINDOWS)
    ax.set_xlabel("Days into course")
    ax.set_ylabel("Predicted risk")
    ax.set_title("Individual risk trajectories (sample of 40)\nred = eventually at-risk, green = OK")
    ax.legend()
    fig.tight_layout()
    fig2 = config.FIGURES / "trajectory_samples.png"
    fig.savefig(fig2, dpi=130)
    plt.close(fig)
    print(f"Saved figure -> {fig2}")

    # Console takeaways
    print("\nTrajectory mix (week1 -> week3):")
    print(wide["trajectory"].value_counts())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
