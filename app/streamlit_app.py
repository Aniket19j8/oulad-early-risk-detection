"""Intervention Impact Simulator — interactive what-if tool for at-risk risk.

Pick a student, then drag sliders for the behaviors an advisor could actually
influence (clicks, active days, submissions, scores...) and watch the model's
risk prediction update live. It turns a static SHAP chart into a question an
advisor cares about: "if this student re-engaged, how much would risk drop?"

Run:
    streamlit run app/streamlit_app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src import config
from src.features import load_features
from src.model import find_student, load_model, score, top_shap_drivers

st.set_page_config(page_title="At-Risk Intervention Simulator", page_icon="🎯", layout="wide")


@st.cache_data
def get_features() -> pd.DataFrame:
    return load_features()


@st.cache_resource
def get_model(calibrated: bool):
    path = config.CALIBRATED_MODEL_PATH if calibrated else config.DEFAULT_MODEL_PATH
    if calibrated and not path.exists():
        path = config.DEFAULT_MODEL_PATH
    return load_model(path)


@st.cache_resource
def get_base_model():
    return load_model(config.DEFAULT_MODEL_PATH)


def feature_ranges(df: pd.DataFrame) -> dict:
    """Sensible slider bounds per actionable feature (cohort-derived)."""
    bounds = {}
    for f in config.ACTIONABLE_FEATURES:
        if f in ("early_mean_score", "early_submit_mean_score"):
            bounds[f] = (0.0, 100.0, 1.0)
        elif f == "days_since_last_login":
            bounds[f] = (0.0, float(config.EARLY_WINDOW_DAYS), 1.0)
        else:
            hi = float(df[f].quantile(0.99))
            hi = max(hi, 1.0)
            step = 1.0 if hi > 20 else 0.5
            bounds[f] = (0.0, float(round(hi)), step)
    return bounds


def gauge(prob: float, title: str) -> go.Figure:
    tier = config.risk_tier(prob)
    color = {"High": "crimson", "Medium": "darkorange", "Low": "seagreen"}[tier]
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=prob * 100,
        number={"suffix": "%"},
        title={"text": f"{title}<br><span style='font-size:0.8em'>{tier} tier</span>"},
        gauge={
            "axis": {"range": [0, 100]},
            "bar": {"color": color},
            "steps": [
                {"range": [0, config.RISK_MEDIUM * 100], "color": "#e8f5e9"},
                {"range": [config.RISK_MEDIUM * 100, config.RISK_HIGH * 100], "color": "#fff3e0"},
                {"range": [config.RISK_HIGH * 100, 100], "color": "#ffebee"},
            ],
        },
    ))
    fig.update_layout(height=280, margin=dict(l=20, r=20, t=70, b=10))
    return fig


def main() -> None:
    df = get_features()

    st.title("🎯 At-Risk Intervention Impact Simulator")
    st.caption(
        "Adjust the behaviors an advisor could influence and watch predicted risk "
        "respond. Demographics and prior attempts are intentionally fixed — you "
        "can't intervene on someone's age band."
    )

    # ---- Sidebar: pick a student ----
    with st.sidebar:
        st.header("Select a student")
        use_calibrated = st.toggle("Use calibrated model", value=False,
                                   help="Calibrated probabilities read as true likelihoods.")
        model = get_model(use_calibrated)

        scored = df.copy()
        scored["risk_probability"] = score(model, scored)
        scored["risk_tier"] = scored["risk_probability"].map(config.risk_tier)

        module = st.selectbox("Module", ["(any)"] + sorted(df["code_module"].unique()))
        tier_filter = st.selectbox("Risk tier", ["(any)", "High", "Medium", "Low"])

        pool = scored
        if module != "(any)":
            pool = pool[pool["code_module"] == module]
        if tier_filter != "(any)":
            pool = pool[pool["risk_tier"] == tier_filter]
        pool = pool.sort_values("risk_probability", ascending=False)

        if pool.empty:
            st.warning("No students match that filter.")
            st.stop()

        options = pool.head(300).apply(
            lambda r: f"{int(r['id_student'])} — {r['code_module']}/{r['code_presentation']} "
                      f"({r['risk_probability']:.0%}, {r['final_result']})",
            axis=1,
        ).tolist()
        choice = st.selectbox("Student", options)
        chosen_id = int(choice.split(" — ")[0])
        chosen_mod = choice.split("— ")[1].split("/")[0]
        chosen_pres = choice.split("/")[1].split(" ")[0]

    row = find_student(df, chosen_id, chosen_mod, chosen_pres)
    if row.empty:
        st.error("Could not load that student.")
        st.stop()
    row = row.iloc[[0]].copy()

    baseline_prob = float(score(model, row)[0])

    # ---- Sliders for actionable features ----
    st.subheader("Simulate an intervention")
    bounds = feature_ranges(df)
    medians = df[config.ACTIONABLE_FEATURES].median(numeric_only=True)

    modified = row.copy()
    cols = st.columns(3)
    for i, feat in enumerate(config.ACTIONABLE_FEATURES):
        lo, hi, step = bounds[feat]
        current = row.iloc[0][feat]
        if pd.isna(current):
            current = float(medians.get(feat, lo))
        current = float(min(max(current, lo), hi))
        with cols[i % 3]:
            modified.iloc[0, modified.columns.get_loc(feat)] = st.slider(
                feat.replace("_", " "), min_value=lo, max_value=hi,
                value=current, step=step,
            )

    new_prob = float(score(model, modified)[0])

    # ---- Results ----
    st.divider()
    g1, g2, g3 = st.columns([1, 1, 1])
    with g1:
        st.plotly_chart(gauge(baseline_prob, "Current risk"), width="stretch")
    with g2:
        st.plotly_chart(gauge(new_prob, "Simulated risk"), width="stretch")
    with g3:
        delta = new_prob - baseline_prob
        st.metric("Risk change", f"{new_prob:.1%}", f"{delta:+.1%}",
                  delta_color="inverse")
        old_tier = config.risk_tier(baseline_prob)
        new_tier = config.risk_tier(new_prob)
        if new_tier != old_tier:
            st.success(f"Tier moves **{old_tier} → {new_tier}**")
        else:
            st.info(f"Still **{new_tier}** tier")
        st.caption(f"Student {chosen_id} · {chosen_mod}/{chosen_pres} · "
                   f"actual outcome: {row.iloc[0]['final_result']}")

    # ---- SHAP drivers (from base tree model) ----
    st.subheader("What's driving this prediction?")
    try:
        drivers = top_shap_drivers(get_base_model(), modified, top_n=8)
        drivers = drivers.sort_values("shap_value")
        fig = go.Figure(go.Bar(
            x=drivers["shap_value"],
            y=[f.replace("_", " ") for f in drivers["feature"]],
            orientation="h",
            marker_color=["crimson" if v >= 0 else "seagreen" for v in drivers["shap_value"]],
        ))
        fig.update_layout(
            height=360, margin=dict(l=10, r=10, t=10, b=10),
            xaxis_title="SHAP value (right = increases risk)",
        )
        st.plotly_chart(fig, width="stretch")
    except Exception as exc:
        st.caption(f"SHAP explanation unavailable: {exc}")


if __name__ == "__main__":
    main()
