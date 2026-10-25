import numpy as np, pandas as pd, mlflow
from scipy.stats import binomtest
from fairlearn.metrics import (demographic_parity_ratio, demographic_parity_difference,
                               equalized_odds_difference)

mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("bank-marketing-m4")

d = pd.read_csv("m4/outputs/scores_stratified_no_macro_gboost.csv", index_col=0)
n_sel = int(round(len(d) * 0.30))
sel = np.zeros(len(d), int)
sel[np.argsort(-d["score"].values, kind="stable")[:n_sel]] = 1
d["sel"] = sel
overall_recall = d.loc[d.y == 1, "sel"].mean()
MIN_SUBS, GAP, DIR_SCREEN = 30, 0.10, 0.80
print(f"Overall recall at the 30% cut: {overall_recall:.3f} | rows {len(d)} | subscribers {int(d.y.sum())}\n")

rows = []
for g in ["age_band", "marital", "education"]:
    for level, x in d.groupby(g):
        pos, neg = x[x.y == 1], x[x.y == 0]
        k, m = int(pos.sel.sum()), len(pos)
        ci = binomtest(k, m).proportion_ci(method="wilson") if m > 0 else None
        rows.append({"attribute": g, "group": level, "rows": len(x), "subscribers": m,
                     "selection_rate": x.sel.mean(),
                     "recall": k / m if m else np.nan,
                     "recall_ci_low": ci.low if ci else np.nan,
                     "recall_ci_high": ci.high if ci else np.nan,
                     "false_positive_rate": neg.sel.mean() if len(neg) else np.nan,
                     "recall_gap_vs_overall": (k / m - overall_recall) if m else np.nan,
                     "judged": m >= MIN_SUBS})
t = pd.DataFrame(rows)
t["gap_flag"] = t["judged"] & (t["recall_gap_vs_overall"].abs() > GAP)
t.round(3).to_csv("reports/fairness_metrics.csv", index=False)
pd.set_option("display.width", 250); pd.set_option("display.max_columns", None)
print(t.round(3).to_string(index=False))

print("\nSummary on judged groups only:")
summary = {}
with mlflow.start_run(run_name="50_fairness_deployable"):
    for g in ["age_band", "marital", "education"]:
        ok = t[(t.attribute == g) & t.judged]["group"]
        dj = d[d[g].isin(ok)]
        dpr = demographic_parity_ratio(dj.y, dj.sel, sensitive_features=dj[g])
        dpd = demographic_parity_difference(dj.y, dj.sel, sensitive_features=dj[g])
        eod = equalized_odds_difference(dj.y, dj.sel, sensitive_features=dj[g])
        nflag = int(t[(t.attribute == g)]["gap_flag"].sum())
        print(f"{g}: disparate impact ratio = {dpr:.3f} (screen {DIR_SCREEN}) | "
              f"demographic parity diff = {dpd:.3f} | equalized odds diff = {eod:.3f} | "
              f"groups beyond 10-pt recall gap = {nflag}")
        mlflow.log_metrics({f"{g}_dir": dpr, f"{g}_dp_diff": dpd,
                            f"{g}_eo_diff": eod, f"{g}_gap_flags": nflag})
    mlflow.log_artifact("reports/fairness_metrics.csv")
