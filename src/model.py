"""Model loading, scoring, and per-student SHAP explanation helpers."""

from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import joblib
import numpy as np
import pandas as pd

from . import config
from .features import feature_matrix


def load_model(path: Path | None = None):
    """Load the saved sklearn Pipeline (ColumnTransformer + classifier)."""
    path = Path(path) if path is not None else config.DEFAULT_MODEL_PATH
    if not path.exists():
        raise FileNotFoundError(
            f"Model not found at {path}. Run notebook 03 to regenerate it."
        )
    return joblib.load(path)


def score(pipe, df: pd.DataFrame) -> np.ndarray:
    """Return at-risk probabilities for a feature frame."""
    return pipe.predict_proba(feature_matrix(df))[:, 1]


def add_risk_columns(pipe, df: pd.DataFrame) -> pd.DataFrame:
    """Append risk_probability / risk_tier / predicted_at_risk to a copy of df."""
    out = df.copy()
    out["risk_probability"] = score(pipe, out)
    out["risk_tier"] = out["risk_probability"].map(config.risk_tier)
    out["predicted_at_risk"] = (out["risk_probability"] >= 0.5).astype(int)
    return out


def transformed_feature_names(pipe) -> List[str]:
    """Feature names after the ColumnTransformer (one-hot expanded)."""
    prep = pipe.named_steps["prep"]
    cat_encoder = prep.named_transformers_["cat"]
    encoded_cat = list(cat_encoder.get_feature_names_out(config.CAT_FEATURES))
    return encoded_cat + config.NUM_FEATURES


def top_shap_drivers(
    pipe, row: pd.DataFrame, top_n: int = 6
) -> pd.DataFrame:
    """Top SHAP drivers (toward / away from at-risk) for a single student row.

    Returns a frame with columns: feature, shap_value, direction. Only works for
    tree models (RandomForest / XGBoost) wrapped in the standard pipeline.
    """
    import shap

    prep = pipe.named_steps["prep"]
    model = pipe.named_steps["model"]

    x = prep.transform(feature_matrix(row))
    names = transformed_feature_names(pipe)

    explainer = shap.TreeExplainer(model)
    raw = explainer.shap_values(x)

    sv = np.asarray(raw.values if hasattr(raw, "values") else raw)
    if sv.ndim == 3:          # (n, features, classes)
        sv = sv[:, :, 1]
    sv = sv[0]                # single row

    drivers = (
        pd.DataFrame({"feature": names, "shap_value": sv})
        .assign(abs_val=lambda d: d["shap_value"].abs())
        .sort_values("abs_val", ascending=False)
        .head(top_n)
        .drop(columns="abs_val")
        .reset_index(drop=True)
    )
    drivers["direction"] = np.where(
        drivers["shap_value"] >= 0, "increases risk", "lowers risk"
    )
    return drivers


def find_student(
    df: pd.DataFrame, student_id: int, module: str | None = None, presentation: str | None = None
) -> pd.DataFrame:
    """Return all rows matching a student id (optionally a specific enrollment)."""
    mask = df["id_student"] == student_id
    if module is not None:
        mask &= df["code_module"] == module
    if presentation is not None:
        mask &= df["code_presentation"] == presentation
    return df[mask].copy()
