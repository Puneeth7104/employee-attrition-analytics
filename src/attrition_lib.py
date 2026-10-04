"""Shared helpers: cleaning, derived metrics, rates, tests and regression.

Used by src/run_analysis.py (batch report) and app.py (Streamlit dashboard).
Only needs pandas, numpy and scipy.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

WINDOW_START = pd.Timestamp("2023-01-01")
SNAPSHOT = pd.Timestamp("2026-09-30")

TENURE_BANDS = ["0-6 months", "6-12 months", "1-2 years", "2-5 years", "5+ years"]
COMP_BANDS = ["Below 0.90", "0.90-1.00", "1.00-1.10", "Above 1.10"]
HOURS_BANDS = ["Under 40h", "40-45h", "45-50h", "50h+"]


# --------------------------------------------------------------------------- cleaning
def clean(raw: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Basic data-quality checks. Returns (clean_df, quality_report)."""
    report: dict = {"rows_raw": int(len(raw))}
    df = raw.copy()

    report["duplicate_rows_removed"] = int(df.duplicated().sum())
    df = df.drop_duplicates().reset_index(drop=True)
    report["duplicate_employee_ids_after_dedup"] = int(df["employee_id"].duplicated().sum())

    df["hire_date"] = pd.to_datetime(df["hire_date"])
    df["exit_date"] = pd.to_datetime(df["exit_date"])

    report["missing_values"] = {c: int(v) for c, v in df.isna().sum().items() if v and c != "exit_date"}
    report["exit_date_missing_for_active"] = int(df["exit_date"].isna().sum())

    bad_dates = (df["exit_date"].notna()) & (df["exit_date"] < df["hire_date"])
    report["exit_before_hire"] = int(bad_dates.sum())
    report["attrited_flag_vs_exit_date_mismatch"] = int(
        ((df["exit_date"].notna()).astype(int) != df["attrited"]).sum()
    )
    report["negative_or_zero_tenure"] = int((df["tenure_months"] <= 0).sum())
    report["compa_ratio_out_of_range"] = int(((df["compa_ratio"] < 0.5) | (df["compa_ratio"] > 1.6)).sum())
    report["weekly_hours_out_of_range"] = int(((df["avg_weekly_hours"] < 20) | (df["avg_weekly_hours"] > 80)).sum())

    # Treatments: keep rows, flag/impute so no one is silently dropped.
    df["performance_missing"] = df["performance_rating"].isna()
    df["hours_missing"] = df["avg_weekly_hours"].isna()
    df["avg_weekly_hours"] = df["avg_weekly_hours"].fillna(df["avg_weekly_hours"].median())
    df["overtime_hours_month"] = df["overtime_hours_month"].fillna(df["overtime_hours_month"].median())

    report["rows_clean"] = int(len(df))
    report["personal_data_fields"] = "none (only anonymous employee_id; no name, age, gender, address or contact info)"
    return df, report


# ---------------------------------------------------------------------------- derived
def add_derived(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["hire_date"] = pd.to_datetime(df["hire_date"])
    df["exit_date"] = pd.to_datetime(df["exit_date"])

    df["comp_band"] = pd.cut(
        df["compa_ratio"], bins=[-np.inf, 0.90, 1.00, 1.10, np.inf], labels=COMP_BANDS, right=False
    )
    df["hours_band"] = pd.cut(
        df["avg_weekly_hours"], bins=[-np.inf, 40, 45, 50, np.inf], labels=HOURS_BANDS, right=False
    )
    df["performance_group"] = np.select(
        [df["performance_rating"].isna(), df["performance_rating"] <= 2, df["performance_rating"] == 5],
        ["Unrated", "Low (1-2)", "Top (5)"],
        default="Solid (3-4)",
    )

    # Exposure inside the observation window (person-years), used for fair rates.
    start = df["hire_date"].where(df["hire_date"] > WINDOW_START, WINDOW_START)
    end = df["exit_date"].fillna(SNAPSHOT)
    df["exposure_years"] = ((end - start).dt.total_seconds() / 86400 / 365.25).clip(lower=1 / 365.25)

    # Early-tenure flags (only meaningful for hires inside the window, see early_tenure()).
    days_to_exit = (df["exit_date"] - df["hire_date"]).dt.days
    df["exit_within_90d"] = (days_to_exit <= 90) & df["exit_date"].notna()
    df["exit_within_12m"] = (days_to_exit <= 365) & df["exit_date"].notna()
    return df


# ------------------------------------------------------------------------------ rates
def rate_table(df: pd.DataFrame, by: str, exposure: str = "exposure_years", event: str = "attrited") -> pd.DataFrame:
    """Exits, headcount and annualised attrition rate per group.

    annualised rate = exits / person-years of exposure.  This avoids the bias of
    comparing raw % exited when people were observed for different lengths of time.
    """
    g = df.groupby(by, observed=True)
    out = pd.DataFrame(
        {
            "employees": g.size(),
            "exits": g[event].sum(),
            "person_years": g[exposure].sum().round(1),
        }
    )
    out["annual_attrition_pct"] = (out["exits"] / out["person_years"] * 100).round(1)
    out["share_exited_pct"] = (out["exits"] / out["employees"] * 100).round(1)
    # 95% Poisson (exact) CI on the annualised rate
    lo = stats.chi2.ppf(0.025, 2 * out["exits"]) / 2
    hi = stats.chi2.ppf(0.975, 2 * (out["exits"] + 1)) / 2
    out["ci_low_pct"] = (np.nan_to_num(lo) / out["person_years"] * 100).round(1)
    out["ci_high_pct"] = (hi / out["person_years"] * 100).round(1)
    return out.reset_index()


TENURE_EDGES_MONTHS = [0, 6, 12, 24, 60, np.inf]
_DAYS_PER_MONTH = 30.4375


def person_periods(df: pd.DataFrame) -> pd.DataFrame:
    """Split each person's observed time into tenure bands (episode splitting).

    A person contributes person-years to the band they were *in* at the time, and an
    exit is counted in the band where it happened. This avoids the bias of assigning
    people to a band using their final tenure (which would pile early leavers into the
    shortest bands). Returns one row per person per band with tenure_band, seg_years
    and seg_exit.
    """
    obs_start = df["hire_date"].where(df["hire_date"] > WINDOW_START, WINDOW_START)
    obs_end = df["exit_date"].fillna(SNAPSHOT)
    frames = []
    for k, label in enumerate(TENURE_BANDS):
        lo = TENURE_EDGES_MONTHS[k]
        hi = TENURE_EDGES_MONTHS[k + 1]
        band_start = df["hire_date"] + pd.to_timedelta(lo * _DAYS_PER_MONTH, unit="D")
        if np.isinf(hi):
            band_end = pd.Series(pd.Timestamp.max.floor("D"), index=df.index)
        else:
            band_end = df["hire_date"] + pd.to_timedelta(hi * _DAYS_PER_MONTH, unit="D")
        seg_start = pd.concat([obs_start, band_start], axis=1).max(axis=1)
        seg_end = pd.concat([obs_end, band_end], axis=1).min(axis=1)
        years = (seg_end - seg_start).dt.total_seconds() / 86400 / 365.25
        in_band = years > 0
        exit_here = (df["attrited"] == 1) & (obs_end > band_start) & (obs_end <= band_end)
        part = df.loc[in_band].copy()
        part["tenure_band"] = label
        part["seg_years"] = years[in_band]
        part["seg_exit"] = exit_here[in_band].astype(int)
        frames.append(part)
    out = pd.concat(frames, ignore_index=True)
    out["tenure_band"] = pd.Categorical(out["tenure_band"], categories=TENURE_BANDS, ordered=True)
    return out


def tenure_rate_table(df: pd.DataFrame) -> pd.DataFrame:
    return rate_table(person_periods(df), "tenure_band", exposure="seg_years", event="seg_exit")


def monthly_attrition(df: pd.DataFrame) -> pd.DataFrame:
    """Monthly attrition rate = exits in month / average headcount, plus trailing-12m."""
    rows = []
    for m in pd.date_range(WINDOW_START, SNAPSHOT, freq="MS"):
        m_end = m + pd.offsets.MonthEnd(0)
        hc_start = ((df["hire_date"] < m) & (df["exit_date"].isna() | (df["exit_date"] >= m))).sum()
        hc_end = ((df["hire_date"] <= m_end) & (df["exit_date"].isna() | (df["exit_date"] > m_end))).sum()
        exits = ((df["exit_date"] >= m) & (df["exit_date"] <= m_end)).sum()
        hires = ((df["hire_date"] >= m) & (df["hire_date"] <= m_end)).sum()
        avg_hc = (hc_start + hc_end) / 2
        rows.append(
            {
                "month": m,
                "headcount_start": int(hc_start),
                "headcount_end": int(hc_end),
                "hires": int(hires),
                "exits": int(exits),
                "monthly_attrition_pct": round(exits / avg_hc * 100, 2) if avg_hc else np.nan,
            }
        )
    out = pd.DataFrame(rows)
    out["avg_headcount"] = (out["headcount_start"] + out["headcount_end"]) / 2
    out["trailing_12m_exits"] = out["exits"].rolling(12).sum()
    out["trailing_12m_avg_headcount"] = out["avg_headcount"].rolling(12).mean()
    out["annual_attrition_pct_t12m"] = (
        out["trailing_12m_exits"] / out["trailing_12m_avg_headcount"] * 100
    ).round(1)
    return out.drop(columns=["avg_headcount", "trailing_12m_exits", "trailing_12m_avg_headcount"])


def early_tenure(df: pd.DataFrame, by: str | None = None) -> pd.DataFrame:
    """Early-tenure attrition among hires made inside the window.

    Only hires on/after WINDOW_START are used: earlier hires who left before the window
    are not in the data (survivor bias). 12-month figures additionally need 12 months of
    follow-up, so hires after SNAPSHOT-12m are excluded for that metric.
    """
    cohort = df[df["hire_date"] >= WINDOW_START].copy()
    c90 = cohort[cohort["hire_date"] <= SNAPSHOT - pd.Timedelta(days=90)]
    c12 = cohort[cohort["hire_date"] <= SNAPSHOT - pd.Timedelta(days=365)]

    def agg(frame, col, name):
        if by is None:
            return pd.DataFrame({name: [frame[col].mean() * 100], "n_" + name: [len(frame)]}, index=["All"])
        g = frame.groupby(by, observed=True)[col]
        return pd.DataFrame({name: g.mean() * 100, "n_" + name: g.size()})

    out = agg(c90, "exit_within_90d", "exit_90d_pct").join(
        agg(c12, "exit_within_12m", "exit_12m_pct"), how="outer"
    )
    return out.round(1).reset_index().rename(columns={"index": by or "group"})


# ----------------------------------------------------------------------------- tests
def cramers_v(table: pd.DataFrame) -> float:
    chi2 = stats.chi2_contingency(table, correction=False)[0]
    n = table.to_numpy().sum()
    k = min(table.shape) - 1
    return float(np.sqrt(chi2 / (n * k))) if k > 0 else np.nan


def association_tests(df: pd.DataFrame) -> pd.DataFrame:
    """Chi-square (categorical) and Mann-Whitney (numeric) tests of association with exit."""
    rows = []
    for col in ["department", "comp_band", "hours_band", "performance_group", "job_level"]:
        table = pd.crosstab(df[col], df["attrited"])
        chi2, p, dof, _ = stats.chi2_contingency(table)
        rows.append(
            {
                "variable": col,
                "test": "Chi-square",
                "statistic": round(chi2, 2),
                "p_value": p,
                "effect_size": round(cramers_v(table), 3),
                "effect_measure": "Cramer's V",
            }
        )
    for col in ["compa_ratio", "avg_weekly_hours", "overtime_hours_month"]:
        a = df.loc[df["attrited"] == 1, col].dropna()
        b = df.loc[df["attrited"] == 0, col].dropna()
        u, p = stats.mannwhitneyu(a, b, alternative="two-sided")
        # common-language effect size: P(leaver > stayer)
        cles = u / (len(a) * len(b))
        rows.append(
            {
                "variable": col,
                "test": "Mann-Whitney U",
                "statistic": round(u, 0),
                "p_value": p,
                "effect_size": round(cles, 3),
                "effect_measure": "P(leaver > stayer)",
            }
        )
    out = pd.DataFrame(rows)
    out["significant_at_5pct"] = out["p_value"] < 0.05
    return out


# ------------------------------------------------------------------------- regression
def _design(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    X = pd.get_dummies(df[["department"]], drop_first=False).astype(float)
    X = X.drop(columns=["department_Engineering"])  # reference group
    X["compa_ratio_per_0.1"] = (df["compa_ratio"] - 1.0) / 0.1
    X["weekly_hours_per_5h"] = (df["avg_weekly_hours"] - 43) / 5
    X["perf_low_1_2"] = (df["performance_group"] == "Low (1-2)").astype(float)
    X["perf_top_5"] = (df["performance_group"] == "Top (5)").astype(float)
    X["perf_unrated"] = (df["performance_group"] == "Unrated").astype(float)
    X["job_level"] = df["job_level"].astype(float)
    for band in TENURE_BANDS[:-1]:  # reference = 5+ years
        X[f"tenure_{band}"] = (df["tenure_band"].astype(str) == band).astype(float)
    X.insert(0, "intercept", 1.0)
    return X, list(X.columns)


def poisson_rate_model(people: pd.DataFrame, max_iter: int = 50) -> pd.DataFrame:
    """Poisson regression with a log(person-years) offset, fitted by IRLS.

    exp(coef) is the attrition RATE RATIO: 1.30 means 30% higher exit rate, holding the
    other variables fixed. The exposure offset handles people observed for different times.
    """
    df = person_periods(people)  # tenure band is time-varying
    X, cols = _design(df)
    Xm = X.to_numpy()
    y = df["seg_exit"].to_numpy(dtype=float)
    offset = np.log(df["seg_years"].to_numpy())

    beta = np.zeros(Xm.shape[1])
    beta[0] = np.log(y.sum() / np.exp(offset).sum())
    for _ in range(max_iter):
        eta = Xm @ beta + offset
        mu = np.exp(eta)
        z = eta - offset + (y - mu) / mu
        W = mu
        XtW = Xm.T * W
        new = np.linalg.solve(XtW @ Xm, XtW @ z)
        if np.max(np.abs(new - beta)) < 1e-9:
            beta = new
            break
        beta = new
    mu = np.exp(Xm @ beta + offset)
    cov = np.linalg.inv((Xm.T * mu) @ Xm)
    se = np.sqrt(np.diag(cov))
    zscore = beta / se
    p = 2 * (1 - stats.norm.cdf(np.abs(zscore)))
    out = pd.DataFrame(
        {
            "term": cols,
            "rate_ratio": np.exp(beta),
            "ci_low": np.exp(beta - 1.96 * se),
            "ci_high": np.exp(beta + 1.96 * se),
            "p_value": p,
        }
    )
    out = out[out["term"] != "intercept"].reset_index(drop=True)
    out["significant_at_5pct"] = out["p_value"] < 0.05
    return out.round({"rate_ratio": 3, "ci_low": 3, "ci_high": 3})
