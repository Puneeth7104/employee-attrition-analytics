"""Precompute everything the static GitHub Pages dashboard needs -> docs/data.js

Run:  python src/build_static.py
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

import attrition_lib as al

ROOT = Path(__file__).resolve().parents[1]


def records(frame: pd.DataFrame) -> list[dict]:
    frame = frame.copy()
    for c in frame.columns:
        if str(frame[c].dtype) == "category":
            frame[c] = frame[c].astype(str)
    frame = frame.replace({np.nan: None})
    return json.loads(frame.to_json(orient="records", date_format="iso"))


def slice_payload(df: pd.DataFrame) -> dict:
    monthly = al.monthly_attrition(df)
    monthly["month"] = monthly["month"].dt.strftime("%Y-%m")
    early = al.early_tenure(df)
    t12 = monthly["annual_attrition_pct_t12m"].dropna()
    return {
        "kpis": {
            "employees": int(len(df)),
            "exits": int(df["attrited"].sum()),
            "annual_rate": round(float(df["attrited"].sum() / df["exposure_years"].sum() * 100), 1),
            "t12m_rate": float(t12.iloc[-1]) if len(t12) else None,
            "early_12m": float(early["exit_12m_pct"].iloc[0]),
            "early_90d": float(early["exit_90d_pct"].iloc[0]),
        },
        "monthly": records(monthly[["month", "headcount_end", "hires", "exits", "monthly_attrition_pct", "annual_attrition_pct_t12m"]]),
        "tenure": records(al.tenure_rate_table(df)),
        "comp": records(al.rate_table(df, "comp_band")),
        "hours": records(al.rate_table(df, "hours_band")),
        "performance": records(al.rate_table(df, "performance_group")),
        "level": records(al.rate_table(df, "job_level")),
    }


def main() -> None:
    raw = pd.read_csv(ROOT / "data" / "employees.csv")
    clean, quality = al.clean(raw)
    df = al.add_derived(clean)

    payload = {
        "meta": {
            "window_start": str(al.WINDOW_START.date()),
            "snapshot": str(al.SNAPSHOT.date()),
            "synthetic": True,
        },
        "slices": {"All": slice_payload(df)},
        "departments": records(al.rate_table(df, "department").sort_values("annual_attrition_pct", ascending=False)),
        "roles": records(al.rate_table(df, "role").sort_values("annual_attrition_pct", ascending=False)),
        "early_by_dept": records(al.early_tenure(df, "department").sort_values("exit_12m_pct", ascending=False)),
        "model": records(al.poisson_rate_model(df)),
        "tests": records(al.association_tests(df)),
        "quality": quality,
    }
    for dept in sorted(df["department"].unique()):
        payload["slices"][dept] = slice_payload(df[df["department"] == dept])

    # data.js (not .json) so the page also works when opened directly from disk
    out = ROOT / "docs" / "data.js"
    out.write_text("window.DASHBOARD_DATA = " + json.dumps(payload, separators=(",", ":")) + ";\n")
    print(f"Wrote {out}  ({out.stat().st_size/1024:.0f} KB)  slices={list(payload['slices'])}")


if __name__ == "__main__":
    main()
