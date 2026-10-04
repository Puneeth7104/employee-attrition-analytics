"""Generate a SYNTHETIC employee dataset for the attrition analysis.

No real people are represented. The data contains no names, ages, gender,
addresses or contact details (data minimisation): only an anonymous
employee_id plus the work-related fields needed for the analysis.

Run:  python src/generate_data.py
Out:  data/employees.csv
"""
from pathlib import Path

import numpy as np
import pandas as pd

SEED = 42
N_EMPLOYEES = 2600
WINDOW_START = pd.Timestamp("2023-01-01")
SNAPSHOT = pd.Timestamp("2026-09-30")  # last day of observation

DEPARTMENTS = {
    # name: (share of workforce, attrition multiplier)
    "Engineering": (0.28, 1.00),
    "Sales": (0.20, 1.55),
    "Customer Support": (0.16, 1.40),
    "Operations": (0.12, 1.15),
    "Marketing": (0.08, 1.05),
    "Finance": (0.08, 0.70),
    "HR": (0.05, 0.80),
    "Product": (0.03, 0.95),
}

ROLES = {
    "Engineering": ["Software Engineer", "QA Engineer", "DevOps Engineer", "Data Engineer"],
    "Sales": ["Sales Executive", "Account Manager", "Business Development Rep"],
    "Customer Support": ["Support Agent", "Support Specialist", "Team Lead - Support"],
    "Operations": ["Operations Analyst", "Logistics Coordinator", "Process Specialist"],
    "Marketing": ["Marketing Executive", "Content Specialist", "Digital Marketer"],
    "Finance": ["Accountant", "Financial Analyst", "Payroll Specialist"],
    "HR": ["HR Generalist", "Recruiter", "HR Business Partner"],
    "Product": ["Product Analyst", "Product Manager", "UX Designer"],
}

# level -> midpoint annual salary (INR lakhs per annum)
LEVEL_MID = {1: 4.0, 2: 7.0, 3: 12.0, 4: 20.0, 5: 32.0}
LEVEL_SHARE = [0.30, 0.32, 0.22, 0.12, 0.04]


def tenure_multiplier(months: np.ndarray) -> np.ndarray:
    """Attrition risk by tenure: spikes in the first year, settles later."""
    m = np.ones_like(months, dtype=float)
    m = np.where(months < 3, 1.2, m)
    m = np.where((months >= 3) & (months < 12), 2.0, m)
    m = np.where((months >= 12) & (months < 24), 1.25, m)
    m = np.where((months >= 24) & (months < 60), 0.95, m)
    m = np.where(months >= 60, 0.75, m)
    return m


def main() -> None:
    rng = np.random.default_rng(SEED)
    n = N_EMPLOYEES

    dept_names = list(DEPARTMENTS)
    dept_p = np.array([DEPARTMENTS[d][0] for d in dept_names])
    dept = rng.choice(dept_names, size=n, p=dept_p / dept_p.sum())
    role = np.array([rng.choice(ROLES[d]) for d in dept])
    level = rng.choice([1, 2, 3, 4, 5], size=n, p=LEVEL_SHARE)

    # Hire dates: a mix of long-tenured staff and a steady stream of new hires.
    # Employees are only included if they were employed at some point in the window.
    hire = pd.to_datetime(
        rng.integers(
            pd.Timestamp("2016-01-01").value // 10**9,
            pd.Timestamp("2026-08-31").value // 10**9,
            size=n,
        ),
        unit="s",
    ).normalize()
    # Weight toward recent hires so the early-tenure analysis has enough data.
    recent = rng.random(n) < 0.35
    hire_recent = pd.to_datetime(
        rng.integers(
            WINDOW_START.value // 10**9,
            pd.Timestamp("2026-08-31").value // 10**9,
            size=n,
        ),
        unit="s",
    ).normalize()
    hire = pd.Series(np.where(recent, hire_recent, hire))

    # Compensation: compa-ratio = salary / level midpoint (market position).
    compa = np.clip(rng.normal(1.0, 0.12, size=n), 0.65, 1.45)
    mid = np.array([LEVEL_MID[l] for l in level])
    salary = np.round(mid * compa, 2)

    performance = rng.choice([1, 2, 3, 4, 5], size=n, p=[0.05, 0.15, 0.50, 0.22, 0.08])

    # Workload indicators.
    dept_hours_shift = np.where(np.isin(dept, ["Sales", "Customer Support", "Engineering"]), 2.0, 0.0)
    weekly_hours = np.clip(rng.normal(43 + dept_hours_shift, 4.5, size=n), 32, 65).round(1)
    overtime_hours_month = np.clip((weekly_hours - 40) * 4 + rng.normal(0, 3, size=n), 0, None).round(1)
    months_since_promo_offset = rng.integers(0, 72, size=n)

    # ---- simulate monthly exit events inside the observation window -------------
    base_hazard = 0.0085
    dept_mult = np.array([DEPARTMENTS[d][1] for d in dept])
    comp_mult = np.exp(-3.2 * (compa - 1.0))  # underpaid -> higher risk
    ot_mult = np.exp(0.035 * np.clip(weekly_hours - 44, 0, None))  # overwork -> higher risk
    perf_mult = np.select([performance <= 2, performance == 5], [1.35, 1.25], default=1.0)
    level_mult = np.select([level == 1, level >= 4], [1.15, 0.85], default=1.0)
    static = base_hazard * dept_mult * comp_mult * ot_mult * perf_mult * level_mult

    months = pd.date_range(WINDOW_START, SNAPSHOT, freq="MS")
    exit_date = pd.Series(pd.NaT, index=range(n), dtype="datetime64[ns]")
    alive = hire <= SNAPSHOT
    active_in_window = np.zeros(n, dtype=bool)

    for month_start in months:
        month_end = month_start + pd.offsets.MonthEnd(0)
        employed = alive & (hire <= month_end) & exit_date.isna()
        if not employed.any():
            continue
        active_in_window |= employed.to_numpy()
        tenure_m = ((month_start - hire).dt.days.to_numpy() / 30.4).clip(min=0)
        promo_gap = months_since_promo_offset + (month_start - WINDOW_START).days / 30.4
        promo_mult = np.where(promo_gap > 36, 1.25, 1.0)
        p = np.clip(static * tenure_multiplier(tenure_m) * promo_mult, 0, 0.5)
        leaves = (rng.random(n) < p) & employed.to_numpy()
        for idx in np.where(leaves)[0]:
            day = int(rng.integers(0, month_end.day))
            d = month_start + pd.Timedelta(days=day)
            exit_date.iloc[idx] = max(d, hire.iloc[idx] + pd.Timedelta(days=7))

    exit_date = exit_date.where(exit_date <= SNAPSHOT)
    attrited = exit_date.notna().astype(int)
    end = exit_date.fillna(SNAPSHOT)
    tenure_months = ((end - hire).dt.days / 30.4375).round(1)

    df = pd.DataFrame(
        {
            "employee_id": [f"EMP{str(i + 1).zfill(5)}" for i in range(n)],
            "department": dept,
            "role": role,
            "job_level": level,
            "hire_date": hire.dt.date,
            "exit_date": exit_date.dt.date,
            "tenure_months": tenure_months,
            "annual_salary_lpa": salary,
            "compa_ratio": compa.round(3),
            "performance_rating": performance,
            "avg_weekly_hours": weekly_hours,
            "overtime_hours_month": overtime_hours_month,
            "attrited": attrited,
        }
    )

    df = df[active_in_window].reset_index(drop=True)

    # ---- inject realistic messiness so the data-quality step has something to find
    n_rows = len(df)
    miss_perf = rng.choice(n_rows, size=int(0.02 * n_rows), replace=False)
    df.loc[miss_perf, "performance_rating"] = np.nan
    miss_hours = rng.choice(n_rows, size=int(0.015 * n_rows), replace=False)
    df.loc[miss_hours, "avg_weekly_hours"] = np.nan
    dupes = df.sample(5, random_state=SEED)
    df = pd.concat([df, dupes], ignore_index=True)

    out = Path(__file__).resolve().parents[1] / "data" / "employees.csv"
    df.to_csv(out, index=False)
    print(f"Wrote {out}  rows={len(df)}  attrited={int(df['attrited'].sum())}")


if __name__ == "__main__":
    main()
