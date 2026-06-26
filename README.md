# Early At-Risk Student Detection (OULAD)

A learning analytics project built around the [Open University Learning Analytics Dataset (OULAD)](https://analyse.kmi.open.ac.uk/open_dataset). The goal is simple: flag students who may struggle early in a course, explain what is driving that risk, and layer in a small NLP pipeline for learner feedback themes (including accessibility).

This started as a portfolio piece aligned with how online learning teams actually work — EDA, feature engineering, modeling, explainability, text classification, and a dashboard stakeholders can open without running Python.

**Live dashboard:** [Student Risk & Accessibility Dashboard on Tableau Public](https://public.tableau.com/app/profile/aniket.d6701/viz/OULADEarlyRiskAccessibility/StudentRiskAccessibilityDashboard)

**Deployed Link:** (https://oulad-early-risk-detection-crdu25wslmdjs6xhyqpsaq.streamlit.app/)

---

## What this project does

1. **Merges OULAD tables** and defines at-risk students as those who withdrew or failed.
2. **Engineers early-window features** using only the first three weeks of activity (`date <= 21`) so predictions are usable in practice, not after the fact.
3. **Trains and compares classifiers** (Logistic Regression, Random Forest, XGBoost) and picks a winner.
4. **Explains predictions with SHAP** so you can see which behaviors matter most.
5. **Classifies synthetic learner feedback** with a tuned Hugging Face zero-shot pipeline (with fallbacks when the model cannot download).
6. **Exports flat CSVs** for Tableau and scores the full student cohort for the dashboard.

---

## Results at a glance

| Area | Outcome |
|------|---------|
| Dataset | 32,593 enrollments across 7 modules |
| At-risk rate | ~52.8% (Withdrawn + Fail) |
| Best model | Random Forest |
| ROC-AUC (test) | 0.820 |
| Precision / Recall (at-risk) | 0.809 / 0.659 |
| Top early signals (SHAP) | `early_clicks`, `early_active_days`, `early_unique_resources` |
| NLP (synthetic QA) | ~68% accuracy with hybrid zero-shot + keyword assist |
| Calibration (ECE) | 0.042 → 0.015 after isotonic calibration |
| Cost-optimal threshold | ~0.29 (recall ~93%) when FN is 3× costlier than FP |
| Fairness | Audited across disability / IMD / age / gender (FNR-focused) |
| Dashboard | Full cohort risk tiers + SHAP drivers + accessibility heatmap |

Risk tiers used in exports and Tableau:

- **High** — probability ≥ 0.6  
- **Medium** — 0.4 to 0.6  
- **Low** — below 0.4  

---

## Dashboard

The Tableau workbook has three views:

1. **Risk tiers by module** — how High / Medium / Low risk is distributed across course modules (AAA–GGG).
2. **SHAP risk drivers** — which early features push the model toward an at-risk prediction.
3. **Accessibility heatmap** — NLP-derived accessibility flag rates by module and week (from synthetic feedback).

Built from the `tableau_*.csv` files in `outputs/` after running notebook 05.

---

## Project structure

```
ed_project/
├── data/
│   ├── raw/              # OULAD CSVs (not in repo — download separately)
│   └── processed/        # merged + early-window feature tables
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_model_training.ipynb
│   ├── 04_nlp_pipeline.ipynb
│   └── 05_dashboard_prep.ipynb
├── src/                  # reusable library (used by CLI, scripts, app)
│   ├── config.py         # paths, feature lists, risk constants
│   ├── features.py       # window-parameterized feature engineering
│   ├── model.py          # load / score / per-student SHAP drivers
│   ├── fairness.py       # group fairness metrics
│   ├── calibration.py    # reliability curve, ECE, calibrated model
│   └── thresholds.py     # cost-aware threshold sweep
├── scripts/              # reproducible analyses (headless, write to outputs/)
│   ├── run_trajectory.py
│   ├── run_fairness.py
│   └── run_calibration_thresholds.py
├── app/
│   └── streamlit_app.py  # intervention impact simulator
├── predict.py            # CLI: score one student + explain why
├── models/               # trained + calibrated models (generated)
├── outputs/              # predictions, SHAP, NLP, Tableau exports, figures/
├── requirements.txt
└── README.md
```

---

## Setup

Requires **Python 3.11+** (3.12 recommended). Windows commands below; adjust paths if needed.

```powershell
cd d:\ed_project
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

Select the `.venv` kernel in Jupyter before running notebooks.

If Hugging Face downloads fail (firewall, antivirus, SSL), allow Python through your firewall or run notebook 04 on a network where `huggingface.co` is reachable. The notebook includes a keyword fallback so the pipeline still completes.

---

## Data

1. Download OULAD from https://analyse.kmi.open.ac.uk/open_dataset  
2. Place these files in `data/raw/`:
   - `studentInfo.csv`
   - `studentAssessment.csv`
   - `studentVle.csv`
   - `vle.csv`
   - `assessments.csv`

`synthetic_feedback.csv` is included in the repo for the NLP demo (60 rows, 5 themes).

---

## How to reproduce

Run the notebooks in order:

| Notebook | What it does |
|----------|----------------|
| `01_eda.ipynb` | Load, merge, explore OULAD; save `student_merged.csv` |
| `02_feature_engineering.ipynb` | Build weeks 1–3 features; save `student_features_early.csv` |
| `03_model_training.ipynb` | Train models, SHAP, save `random_forest.joblib` and test predictions |
| `04_nlp_pipeline.ipynb` | Classify synthetic feedback; export NLP CSVs |
| `05_dashboard_prep.ipynb` | Score all 32,593 students; export `tableau_*.csv` files |

After notebook 05, the main Tableau files are:

- `outputs/tableau_student_risk_full.csv`
- `outputs/tableau_risk_by_module.csv`
- `outputs/tableau_shap_drivers.csv`
- `outputs/tableau_accessibility_heatmap.csv`

---

## Advanced analyses (beyond the notebooks)

The notebooks train and explain the model; these add the rigor that turns it
from a prototype into something defensible. Everything below reuses the `src/`
library, so the logic is shared (and testable) rather than copy-pasted.

### 1. Score & explain a single student (CLI)

```powershell
python predict.py --list-high 10            # find high-risk ids to try
python predict.py --student-id 636749       # risk + top SHAP drivers
python predict.py --student-id 636749 --calibrated
```

### 2. Weekly risk trajectory

```powershell
python scripts/run_trajectory.py
```

Rebuilds the early-window features at **day 7 / 14 / 21** and scores each
snapshot, so risk becomes a *trajectory*, not a single number. Finding: the gap
between eventually-at-risk and eventually-OK students is visible by **week 1**
and widens — and because early snapshots have less activity, the model
over-flags early then settles (most students' risk *falls* week 1 → week 3).
Outputs `outputs/tableau_risk_trajectory*.csv` and two figures in
`outputs/figures/`.

> Caveat (stated honestly): the model is trained on the 21-day window and
> applied at earlier checkpoints. It answers *"how early could we have flagged
> this student?"*, not *"here is a separate weekly model."*

### 3. Fairness audit

```powershell
python scripts/run_fairness.py
```

Audits the model across `disability`, `imd_band`, `age_band`, and `gender`,
focusing on **false-negative rate (FNR)** — the share of genuinely at-risk
students the model *misses*, since a missed student gets no outreach. Key
findings on this data:

- **Disability:** students with a declared disability are *not* under-served on
  recall (FNR 19% vs 31% for non-disabled); they are flagged at a higher rate,
  driven by a genuinely higher base at-risk rate.
- **IMD band:** the real disparity is the **missing-IMD group** (FNR ~43%) and
  the least-deprived band — a data-quality fairness issue worth flagging.

Outputs `outputs/fairness_by_*.csv`, `outputs/fairness_summary.csv`, and bar
charts in `outputs/figures/`.

### 4. Calibration + cost-aware thresholds

```powershell
python scripts/run_calibration_thresholds.py
```

- **Calibration:** measures whether a predicted "0.6" really means 60%. Isotonic
  (cross-validated) calibration cuts **ECE 0.042 → 0.015**; saves
  `models/random_forest_calibrated.joblib`.
- **Thresholds:** with a missed at-risk student treated as **3× costlier** than
  an unnecessary check-in, the cost-minimizing threshold is **~0.29** (recall
  ~93%), vs the current 0.60 High-tier cutoff that misses ~47% of at-risk
  students. Makes the threshold a decision, not a default.

Outputs `outputs/calibration_*.csv`, `outputs/threshold_analysis.csv`, and
figures in `outputs/figures/`.

### 5. Intervention impact simulator (interactive)

```powershell
streamlit run app/streamlit_app.py
```

Pick a student, drag sliders for the behaviors an advisor could influence
(clicks, active days, submissions, scores), and watch predicted risk and the
risk tier update live, with SHAP drivers for the modified profile. Demographics
are intentionally fixed — you can't intervene on someone's age band.

---

## Honest limitations

This is a portfolio prototype, not a production system.

- **Early window only** — features from the first 21 days; later behavior is ignored by design.
- **NLP feedback is synthetic** — 60 demo rows, not real student comments. The hybrid classifier (~68% on that set) is a pipeline proof, not a validated production model.
- **Accessibility flags are NLP-derived** — useful for dashboarding, not a substitute for disability services records or WCAG audits.
- **OULAD is historical** — patterns may not transfer directly to another institution without retraining.
- **Model artifact** — `models/random_forest.joblib` is gitignored; run notebook 03 to regenerate.
- **Trajectory uses the day-21 model at earlier checkpoints** — it shows how early a student *could* be flagged, not a separately trained weekly model.
- **Fairness audit is a starting point** — group FNR/selection-rate gaps are surfaced, not yet mitigated (e.g. via reweighting or group-specific thresholds).

---

## Tech stack

Python, pandas, scikit-learn, XGBoost, SHAP, Hugging Face Transformers (DistilBERT-MNLI), Jupyter, Streamlit, Plotly, Tableau Public

---

## Author

Aniket — learning analytics portfolio project.

If you use this work, please cite the [OULAD dataset](https://analyse.kmi.open.ac.uk/open_dataset) and link back to the Tableau dashboard above.
