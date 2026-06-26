"""Early-window feature engineering (mirrors notebook 02), parameterized by cutoff.

The original notebook hard-codes a 21-day window. Here the cutoff is a parameter
so the same logic can build features at successive checkpoints (day 7, 14, 21),
which is what powers the weekly risk-trajectory analysis.

`build_early_features(tables, window_days=21)` reproduces
`data/processed/student_features_early.csv` exactly when window_days == 21.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd

from . import config

KEYS = config.ID_COLS

DEMO_COLS = [
    "gender", "region", "highest_education", "imd_band",
    "age_band", "num_of_prev_attempts", "studied_credits", "disability",
]

_ZERO_FILL = [
    "early_clicks", "early_active_days", "early_unique_resources",
    "vle_diversity", "early_n_assessments", "early_late_count",
    "early_submissions", "days_since_last_login",
]


def load_raw_tables(raw_dir: Path | None = None) -> Dict[str, pd.DataFrame]:
    """Load the OULAD CSVs once so feature builds at multiple windows can reuse them.

    `studentVle.csv` is ~450 MB, so loading it a single time and filtering per
    window is far cheaper than re-reading it for every checkpoint.
    """
    raw_dir = Path(raw_dir) if raw_dir is not None else config.RAW
    return {
        "student_info": pd.read_csv(raw_dir / "studentInfo.csv"),
        "student_assessment": pd.read_csv(raw_dir / "studentAssessment.csv"),
        "assessments": pd.read_csv(raw_dir / "assessments.csv"),
        "student_vle": pd.read_csv(raw_dir / "studentVle.csv"),
        "vle": pd.read_csv(raw_dir / "vle.csv"),
    }


def _vle_features(student_vle: pd.DataFrame, vle: pd.DataFrame, window_days: int) -> pd.DataFrame:
    vle_early = student_vle[student_vle["date"] <= window_days].copy()

    agg = (
        vle_early.groupby(KEYS, as_index=False)
        .agg(
            early_clicks=("sum_click", "sum"),
            early_active_days=("date", "nunique"),
            early_first_activity=("date", "min"),
            early_last_activity=("date", "max"),
            early_unique_resources=("id_site", "nunique"),
        )
    )
    agg["days_since_last_login"] = window_days - agg["early_last_activity"]
    agg.loc[agg["early_clicks"] == 0, "days_since_last_login"] = window_days

    labeled = vle_early.merge(vle[["id_site", "activity_type"]], on="id_site", how="left")
    diversity = (
        labeled.groupby(KEYS, as_index=False)
        .agg(vle_diversity=("activity_type", "nunique"))
    )
    agg = agg.merge(diversity, on=KEYS, how="left")
    agg["vle_diversity"] = agg["vle_diversity"].fillna(0)
    return agg


def _assessment_features(
    student_assessment: pd.DataFrame, assessments: pd.DataFrame, window_days: int
) -> pd.DataFrame:
    sa = student_assessment.merge(assessments, on="id_assessment", how="inner")

    sa_early = sa[sa["date"] <= window_days].copy()
    sa_early["late"] = np.where(
        sa_early["date"].notna(),
        sa_early["date_submitted"] > sa_early["date"],
        np.nan,
    )
    assess_agg = (
        sa_early.groupby(KEYS, as_index=False)
        .agg(
            early_mean_score=("score", "mean"),
            early_n_assessments=("score", "count"),
            early_late_count=("late", "sum"),
        )
    )

    sa_submitted_early = sa[sa["date_submitted"] <= window_days]
    submit_agg = (
        sa_submitted_early.groupby(KEYS, as_index=False)
        .agg(
            early_submissions=("score", "count"),
            early_submit_mean_score=("score", "mean"),
        )
    )
    return assess_agg.merge(submit_agg, on=KEYS, how="left")


def build_early_features(
    tables: Dict[str, pd.DataFrame], window_days: int = config.EARLY_WINDOW_DAYS
) -> pd.DataFrame:
    """Build the early-window feature table for a given cutoff day.

    Returns a frame with ID columns, `final_result`, `at_risk`, demographics,
    and all engineered features — the same schema notebook 03 trains on.
    """
    student_info = tables["student_info"].copy()
    student_info["at_risk"] = (
        student_info["final_result"].isin(["Withdrawn", "Fail"]).astype(int)
    )

    vle_agg = _vle_features(tables["student_vle"], tables["vle"], window_days)
    assess_agg = _assessment_features(
        tables["student_assessment"], tables["assessments"], window_days
    )

    df = (
        student_info[KEYS + ["final_result", "at_risk"] + DEMO_COLS]
        .merge(vle_agg, on=KEYS, how="left")
        .merge(assess_agg, on=KEYS, how="left")
    )

    df[_ZERO_FILL] = df[_ZERO_FILL].fillna(0)
    df.loc[df["early_clicks"] == 0, ["early_first_activity", "early_last_activity"]] = np.nan
    return df


def load_features(path: Path | None = None) -> pd.DataFrame:
    """Load the precomputed day-21 feature table from disk."""
    path = Path(path) if path is not None else config.FEATURES_PATH
    return pd.read_csv(path)


def feature_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Select just the model input columns (in the order the model expects)."""
    return df[config.FEATURE_COLS]
