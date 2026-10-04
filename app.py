"""Employee Attrition Analytics - interactive Streamlit dashboard.

Run locally:  streamlit run app.py
Data is SYNTHETIC (see src/generate_data.py); no real employees are represented.
"""
import json
import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
import attrition_lib as al  # noqa: E402

st.set_page_config(page_title="Employee Attrition Analytics", page_icon="📉", layout="wide")

ACCENT = "#2a6fdb"
MUTED = "#9aa5b1"
BAD = "#d9534f"


@st.cache_data
def load():
    raw = pd.read_csv(ROOT / "data" / "employees.csv")
    clean, quality = al.clean(raw)
    return al.add_derived(clean), quality


@st.cache_data
def full_model(_df: pd.DataFrame) -> pd.DataFrame:
    return al.poisson_rate_model(_df)


def fmt_p(p: float) -> str:
    return "<0.001" if p < 0.001 else f"{p:.3f}"


df_all, quality = load()

# ------------------------------------------------------------------ sidebar filters
st.sidebar.header("Filters")
depts = st.sidebar.multiselect("Department", sorted(df_all["department"].unique()), default=[])
levels = st.sidebar.multiselect("Job level", sorted(df_all["job_level"].unique()), default=[])
perf = st.sidebar.multiselect("Performance group", sorted(df_all["performance_group"].unique()), default=[])
st.sidebar.caption("Leave a filter empty to include everyone.")

df = df_all.copy()
if depts:
    df = df[df["department"].isin(depts)]
if levels:
    df = df[df["job_level"].isin(levels)]
if perf:
    df = df[df["performance_group"].isin(perf)]

st.title("Employee Attrition Analytics")
st.caption(
    f"Synthetic workforce data · observation window {al.WINDOW_START:%d %b %Y} to {al.SNAPSHOT:%d %b %Y} · "
    "attrition rate = exits per 100 person-years"
)

if len(df) < 30:
    st.warning("Fewer than 30 employees match these filters - results would be unreliable. Widen the filters.")
    st.stop()

monthly = al.monthly_attrition(df)
early = al.early_tenure(df)
exits = int(df["attrited"].sum())
rate = exits / df["exposure_years"].sum() * 100
t12 = monthly["annual_attrition_pct_t12m"].dropna()
t12_now = float(t12.iloc[-1]) if len(t12) else float("nan")

k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Employees in view", f"{len(df):,}")
k2.metric("Exits in window", f"{exits:,}")
k3.metric("Annual attrition (window)", f"{rate:.1f}%")
k4.metric("Annual attrition (last 12 mo)", f"{t12_now:.1f}%")
k5.metric("Left within 12 months of hire", f"{early['exit_12m_pct'].iloc[0]:.1f}%")

tab_trend, tab_seg, tab_tenure, tab_drivers, tab_notes = st.tabs(
    ["Trend", "Segments", "Tenure", "Drivers", "Data & caveats"]
)

# ------------------------------------------------------------------------ trend
with tab_trend:
    c1, c2 = st.columns(2)
    fig = px.bar(monthly, x="month", y="monthly_attrition_pct", color_discrete_sequence=[ACCENT])
    fig.update_layout(title="Monthly attrition rate (%)", yaxis_title="% of avg headcount", xaxis_title=None)
    c1.plotly_chart(fig, use_container_width=True)

    fig = go.Figure(
        go.Scatter(x=monthly["month"], y=monthly["annual_attrition_pct_t12m"], mode="lines", line=dict(color=BAD, width=3))
    )
    fig.update_layout(title="Annual attrition, trailing 12 months (%)", yaxis_title="%", xaxis_title=None)
    c2.plotly_chart(fig, use_container_width=True)

    hc = go.Figure()
    hc.add_trace(go.Scatter(x=monthly["month"], y=monthly["headcount_end"], name="Headcount", line=dict(color=ACCENT)))
    hc.add_trace(go.Bar(x=monthly["month"], y=monthly["hires"], name="Hires", marker_color=MUTED))
    hc.add_trace(go.Bar(x=monthly["month"], y=monthly["exits"], name="Exits", marker_color=BAD))
    hc.update_layout(title="Headcount, hires and exits", barmode="group", xaxis_title=None)
    st.plotly_chart(hc, use_container_width=True)

# ---------------------------------------------------------------------- segments
with tab_seg:
    dimension = st.radio(
        "Break attrition down by",
        ["department", "role", "comp_band", "hours_band", "performance_group", "job_level"],
        format_func=lambda s: {
            "department": "Department",
            "role": "Role",
            "comp_band": "Pay position (compa-ratio)",
            "hours_band": "Weekly hours",
            "performance_group": "Performance",
            "job_level": "Job level",
        }[s],
        horizontal=True,
    )
    tbl = al.rate_table(df, dimension)
    order = {"comp_band": al.COMP_BANDS, "hours_band": al.HOURS_BANDS}.get(dimension)
    if order:
        tbl[dimension] = pd.Categorical(tbl[dimension], categories=order, ordered=True)
        tbl = tbl.sort_values(dimension)
    elif dimension in ("department", "role"):
        tbl = tbl.sort_values("annual_attrition_pct", ascending=True)
    tbl[dimension] = tbl[dimension].astype(str)
    horizontal = dimension in ("department", "role")
    fig = go.Figure(
        go.Bar(
            x=tbl["annual_attrition_pct"] if horizontal else tbl[dimension],
            y=tbl[dimension] if horizontal else tbl["annual_attrition_pct"],
            orientation="h" if horizontal else "v",
            marker_color=ACCENT,
            error_x=dict(
                type="data", symmetric=False,
                array=tbl["ci_high_pct"] - tbl["annual_attrition_pct"],
                arrayminus=tbl["annual_attrition_pct"] - tbl["ci_low_pct"],
            ) if horizontal else None,
            error_y=None if horizontal else dict(
                type="data", symmetric=False,
                array=tbl["ci_high_pct"] - tbl["annual_attrition_pct"],
                arrayminus=tbl["annual_attrition_pct"] - tbl["ci_low_pct"],
            ),
        )
    )
    fig.add_vline(x=rate, line_dash="dash", line_color=MUTED) if horizontal else fig.add_hline(
        y=rate, line_dash="dash", line_color=MUTED
    )
    fig.update_layout(
        title="Annual attrition rate (%) with 95% confidence interval - dashed line = overall",
        height=max(380, 28 * len(tbl) + 160) if horizontal else 420,
    )
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(
        tbl.rename(
            columns={
                "employees": "Employees",
                "exits": "Exits",
                "person_years": "Person-years",
                "annual_attrition_pct": "Annual attrition %",
                "ci_low_pct": "CI low %",
                "ci_high_pct": "CI high %",
            }
        ).drop(columns=["share_exited_pct"]),
        hide_index=True,
        use_container_width=True,
    )
    st.caption("Wide intervals mean small groups - treat those differences cautiously.")

# ------------------------------------------------------------------------ tenure
with tab_tenure:
    t = al.tenure_rate_table(df)
    t["tenure_band"] = t["tenure_band"].astype(str)
    c1, c2 = st.columns(2)
    fig = px.bar(t, x="tenure_band", y="annual_attrition_pct", color_discrete_sequence=[ACCENT])
    fig.update_layout(title="Annual attrition rate by tenure band (%)", xaxis_title=None, yaxis_title="%")
    c1.plotly_chart(fig, use_container_width=True)

    e = al.early_tenure(df, "department").sort_values("exit_12m_pct")
    fig = go.Figure(go.Bar(x=e["exit_12m_pct"], y=e["department"], orientation="h", marker_color=BAD))
    fig.update_layout(title="Left within 12 months of hire, by department (%)", xaxis_title="%", yaxis_title=None)
    c2.plotly_chart(fig, use_container_width=True)
    st.caption(
        "Tenure is time-varying: each person counts toward a band only for the time they spent in it. "
        "Early-tenure figures use hires made inside the window (with enough follow-up) to avoid survivor bias."
    )

# ----------------------------------------------------------------------- drivers
with tab_drivers:
    st.subheader("What is associated with leaving?")
    st.write(
        "Rate ratios from a Poisson regression with person-years exposure, fitted on **all employees** "
        "(filters do not apply here). A ratio of 1.30 = 30% higher exit rate, other factors held equal; "
        "below 1 = lower exit rate."
    )
    model = full_model(df_all).copy()
    labels = {
        "compa_ratio_per_0.1": "Pay: +0.1 compa-ratio",
        "weekly_hours_per_5h": "Workload: +5 weekly hours",
        "perf_low_1_2": "Low performer (vs 3-4)",
        "perf_top_5": "Top performer (vs 3-4)",
        "perf_unrated": "Unrated (vs 3-4)",
        "job_level": "Job level: +1",
    }
    model["label"] = model["term"].map(labels).fillna(
        model["term"].str.replace("department_", "Dept: ", regex=False).str.replace("tenure_", "Tenure ", regex=False)
    )
    model = model.sort_values("rate_ratio")
    fig = go.Figure(
        go.Scatter(
            x=model["rate_ratio"], y=model["label"], mode="markers",
            marker=dict(size=9, color=[BAD if s else MUTED for s in model["significant_at_5pct"]]),
            error_x=dict(
                type="data", symmetric=False,
                array=model["ci_high"] - model["rate_ratio"],
                arrayminus=model["rate_ratio"] - model["ci_low"],
            ),
        )
    )
    fig.add_vline(x=1, line_dash="dash", line_color=MUTED)
    fig.update_layout(
        title="Attrition rate ratio (95% CI) - red = statistically significant at 5%",
        xaxis_title="Rate ratio", yaxis_title=None, height=620,
    )
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Departments are compared with Engineering; tenure bands with 5+ years.")

    shown = model[["label", "rate_ratio", "ci_low", "ci_high", "p_value"]].copy()
    shown["p_value"] = shown["p_value"].map(fmt_p)
    st.dataframe(shown.rename(columns={"label": "Factor", "rate_ratio": "Rate ratio", "ci_low": "CI low", "ci_high": "CI high", "p_value": "p-value"}), hide_index=True, use_container_width=True)

    st.subheader("Simple association tests")
    tests = al.association_tests(df_all).copy()
    tests["p_value"] = tests["p_value"].map(fmt_p)
    st.dataframe(tests, hide_index=True, use_container_width=True)

# ------------------------------------------------------------------------- notes
with tab_notes:
    st.subheader("Data quality checks")
    st.json(quality)
    st.subheader("Caveats")
    st.markdown(
        """
- **The data is synthetic.** It was generated to look like a typical HR extract, so the patterns shown demonstrate the method and are not findings about any real company.
- Associations are **not proof of cause**: pay, workload and tenure are related, and unmeasured factors (manager quality, personal reasons) are missing.
- Exits include all leavers; voluntary and involuntary exits are not separated.
- Employees who left before the window start are not in the data, so only hires from 2023 onward are used for early-tenure measures.
- Privacy: no names, ages, gender or contact details are stored - only an anonymous employee ID.
        """
    )
