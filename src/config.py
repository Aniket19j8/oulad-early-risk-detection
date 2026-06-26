"""Central configuration: paths, feature lists, and risk constants.

Keeping these in one place means the notebooks, the CLI, the analysis scripts,
and the Streamlit app all agree on exactly which columns the model expects and
how risk tiers are defined. If notebook 02/03 change, change them here too.
"""

from __future__ import annotations

from pathlib import Path

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW = PROJECT_ROOT / "data" / "raw"
PROCESSED = PROJECT_ROOT / "data" / "processed"
MODELS = PROJECT_ROOT / "models"
OUTPUTS = PROJECT_ROOT / "outputs"
FIGURES = OUTPUTS / "figures"

DEFAULT_MODEL_PATH = MODELS / "random_forest.joblib"
CALIBRATED_MODEL_PATH = MODELS / "random_forest_calibrated.joblib"
FEATURES_PATH = PROCESSED / "student_features_early.csv"

# --------------------------------------------------------------------------- #
# Identifiers / target
# --------------------------------------------------------------------------- #
ID_COLS = ["code_module", "code_presentation", "id_student"]
TARGET_COL = "at_risk"

# --------------------------------------------------------------------------- #
# Features (must match notebook 03 exactly)
# --------------------------------------------------------------------------- #
CAT_FEATURES = [
    "gender", "region", "highest_education", "imd_band", "age_band", "disability",
]
NUM_FEATURES = [
    "num_of_prev_attempts", "studied_credits",
    "early_clicks", "early_active_days", "early_unique_resources",
    "days_since_last_login", "vle_diversity",
    "early_mean_score", "early_n_assessments", "early_late_count",
    "early_submissions", "early_submit_mean_score",
]
FEATURE_COLS = CAT_FEATURES + NUM_FEATURES

# Numeric features a student/advisor could plausibly influence — used by the
# intervention simulator. Demographics and prior attempts are intentionally
# excluded (you cannot "intervene" on someone's age band).
ACTIONABLE_FEATURES = [
    "early_clicks", "early_active_days", "early_unique_resources",
    "vle_diversity", "early_submissions", "early_n_assessments",
    "early_mean_score", "early_submit_mean_score", "days_since_last_login",
]

# --------------------------------------------------------------------------- #
# Risk tiers / windows
# --------------------------------------------------------------------------- #
RISK_HIGH = 0.6
RISK_MEDIUM = 0.4

EARLY_WINDOW_DAYS = 21              # the production cutoff (end of week 3)
TRAJECTORY_WINDOWS = [7, 14, 21]    # week 1 / week 2 / week 3 checkpoints

RANDOM_STATE = 42


def risk_tier(prob: float) -> str:
    """Map a probability to a High / Medium / Low tier."""
    if prob >= RISK_HIGH:
        return "High"
    if prob >= RISK_MEDIUM:
        return "Medium"
    return "Low"


def ensure_dirs() -> None:
    """Create output directories if they do not yet exist."""
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    MODELS.mkdir(parents=True, exist_ok=True)
