"""Monitoring summary for the term-deposit ranker (BAN6800 final project).

Run from the repository root:  python -m src.monitoring
Writes reports/monitoring_summary.json (group-level aggregates only, no customer rows).
"""
import json
from datetime import datetime, timezone
from math import sqrt

import pandas as pd

from app.main import CUTOFFS, FEATURES, MODEL
from m4.common import load, split, split_stratified

BASE_RATE_ALERT_PTS = 5.0
RECALL_GAP_ALERT_PTS = 10.0
MIN_SUBSCRIBERS = 30
MIX_COLUMNS = ["contact", "poutcome", "month"]
OUT_PATH = "reports/monitoring_summary.json"


def wilson(k, n, z=1.96):
    """95% Wilson interval for a proportion k/n."""
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / d
    half = z * sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / d
    return (centre - half, centre + half)


def drift_summary(df):
    earlier, later = split(df)
    r1 = earlier["y"].mean() * 100
    r2 = later["y"].mean() * 100
    out = {
        "rows_earlier": len(earlier), "rows_later": len(later),
        "base_rate_earlier_pct": round(r1, 2), "base_rate_later_pct": round(r2, 2),
        "shift_pts": round(r2 - r1, 2),
        "alert": bool(abs(r2 - r1) > BASE_RATE_ALERT_PTS),
        "mix": [],
    }
    for col in MIX_COLUMNS:
        a = earlier[col].value_counts(normalize=True) * 100
        b = later[col].value_counts(normalize=True) * 100
        for cat in sorted(set(a.index) | set(b.index)):
            e, l = float(a.get(cat, 0.0)), float(b.get(cat, 0.0))
            out["mix"].append({"feature": col, "category": str(cat),
                               "earlier_pct": round(e, 2), "later_pct": round(l, 2),
                               "change_pts": round(l - e, 2)})
    return out


def fairness_summary(df):
    _, test = split_stratified(df)
    test = test.copy()
    test["days_since_previous_contact"] = test["days_since_previous_contact"].astype(float)
    score = MODEL.predict_proba(test[FEATURES])[:, 1]
    cutoff = test["age_band"].map(CUTOFFS).astype(float).to_numpy()
    selected = score >= cutoff
    y = test["y"].to_numpy()
    overall_recall = selected[y == 1].mean() * 100
    rows = []
    for band in CUTOFFS:
        m = (test["age_band"] == band).to_numpy()
        subs = int(((y == 1) & m).sum())
        hit = int((selected & (y == 1) & m).sum())
        recall = hit / subs * 100 if subs else 0.0
        lo, hi = wilson(hit, subs)
        gap = recall - overall_recall
        rows.append({
            "age_band": band, "records": int(m.sum()), "subscribers": subs,
            "recall_pct": round(recall, 1),
            "ci_low_pct": round(lo * 100, 1), "ci_high_pct": round(hi * 100, 1),
            "selection_rate_pct": round(selected[m].mean() * 100, 1),
            "gap_pts": round(gap, 1),
            "enough_subscribers": subs >= MIN_SUBSCRIBERS,
            "alert": bool(subs >= MIN_SUBSCRIBERS and abs(gap) > RECALL_GAP_ALERT_PTS),
        })
    return {"test_records": len(test), "test_subscribers": int(y.sum()),
            "overall_recall_pct": round(overall_recall, 1),
            "overall_selection_rate_pct": round(selected.mean() * 100, 1),
            "age_bands": rows}


def main():
    df = load()
    summary = {
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "model": "term_deposit_ranker v1 (no economic indicators) with age-band cutoffs",
        "rules": {"base_rate_shift_alert_pts": BASE_RATE_ALERT_PTS,
                  "recall_gap_alert_pts": RECALL_GAP_ALERT_PTS,
                  "min_subscribers_per_band": MIN_SUBSCRIBERS},
        "drift": drift_summary(df),
        "fairness": fairness_summary(df),
    }
    with open(OUT_PATH, "w") as f:
        json.dump(summary, f, indent=2)
    d, fa = summary["drift"], summary["fairness"]
    print("DRIFT: base rate", d["base_rate_earlier_pct"], "% ->", d["base_rate_later_pct"],
          "% | shift", d["shift_pts"], "pts | alert:", d["alert"])
    print("FAIRNESS: overall recall", fa["overall_recall_pct"], "% on",
          fa["test_records"], "records,", fa["test_subscribers"], "subscribers")
    print(pd.DataFrame(fa["age_bands"]).to_string(index=False))
    print("Saved", OUT_PATH)


if __name__ == "__main__":
    main()
