"""Run the full attrition analysis and write results to outputs/ and data/.

Run:  python src/run_analysis.py
"""
import json
from pathlib import Path

import pandas as pd

import attrition_lib as al

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
OUT.mkdir(exist_ok=True)


def main() -> None:
    raw = pd.read_csv(ROOT / "data" / "employees.csv")
    df, quality = al.clean(raw)
    df = al.add_derived(df)

    (OUT / "data_quality.json").write_text(json.dumps(quality, indent=2))

    for dim in ["department", "role", "comp_band", "hours_band", "performance_group", "job_level"]:
        al.rate_table(df, dim).to_csv(OUT / f"attrition_by_{dim}.csv", index=False)

    al.tenure_rate_table(df).to_csv(OUT / "attrition_by_tenure_band.csv", index=False)

    monthly = al.monthly_attrition(df)
    monthly.to_csv(OUT / "monthly_attrition.csv", index=False)

    early_all = al.early_tenure(df)
    early_dept = al.early_tenure(df, "department")
    early_all.to_csv(OUT / "early_tenure_overall.csv", index=False)
    early_dept.to_csv(OUT / "early_tenure_by_department.csv", index=False)

    tests = al.association_tests(df)
    tests.to_csv(OUT / "association_tests.csv", index=False)
    model = al.poisson_rate_model(df)
    model.to_csv(OUT / "rate_ratio_model.csv", index=False)

    # headline numbers
    last12 = monthly.tail(12)
    overall = {
        "employees": int(len(df)),
        "exits": int(df["attrited"].sum()),
        "active_at_snapshot": int((df["attrited"] == 0).sum()),
        "person_years": round(float(df["exposure_years"].sum()), 1),
        "annual_attrition_pct_overall": round(df["attrited"].sum() / df["exposure_years"].sum() * 100, 1),
        "annual_attrition_pct_last_12m": float(last12["annual_attrition_pct_t12m"].iloc[-1]),
        "avg_monthly_attrition_pct_last_12m": round(float(last12["monthly_attrition_pct"].mean()), 2),
        "early_exit_90d_pct": float(early_all["exit_90d_pct"].iloc[0]),
        "early_exit_12m_pct": float(early_all["exit_12m_pct"].iloc[0]),
        "window_start": str(al.WINDOW_START.date()),
        "snapshot": str(al.SNAPSHOT.date()),
    }
    (OUT / "headline_metrics.json").write_text(json.dumps(overall, indent=2))

    print(json.dumps(overall, indent=2))
    print("\nBy department:\n", pd.read_csv(OUT / "attrition_by_department.csv").to_string(index=False))
    print("\nBy tenure band:\n", pd.read_csv(OUT / "attrition_by_tenure_band.csv").to_string(index=False))
    print("\nBy comp band:\n", pd.read_csv(OUT / "attrition_by_comp_band.csv").to_string(index=False))
    print("\nBy hours band:\n", pd.read_csv(OUT / "attrition_by_hours_band.csv").to_string(index=False))
    print("\nTests:\n", tests.to_string(index=False))
    print("\nRate-ratio model:\n", model.to_string(index=False))
    print("\nEarly tenure by dept:\n", early_dept.to_string(index=False))
    print("\nQuality:\n", json.dumps(quality, indent=2))


if __name__ == "__main__":
    main()
