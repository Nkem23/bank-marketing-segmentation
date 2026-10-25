import json, numpy as np, pandas as pd, joblib, mlflow
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from fairlearn.metrics import demographic_parity_ratio
from common import load, split_stratified, CAT, NUM

MACRO = ["emp.var.rate", "cons.price.idx", "cons.conf.idx", "euribor3m", "nr.employed"]
num = [c for c in NUM if c not in MACRO]
FEATS = CAT + num
BUDGET, GAP, MIN_SUBS = 0.30, 0.10, 30

mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("bank-marketing-m4")

df = load()
train, test = split_stratified(df)

numt = Pipeline([("imp", SimpleImputer(strategy="median", add_indicator=True)), ("sc", StandardScaler())])
prep = ColumnTransformer([("num", numt, num), ("cat", OneHotEncoder(handle_unknown="ignore"), CAT)])
pipe_cv = Pipeline([("prep", prep), ("clf", GradientBoostingClassifier(
    n_estimators=200, max_depth=3, learning_rate=0.1, subsample=0.8, random_state=42))])
oof = cross_val_predict(pipe_cv, train[FEATS], train["y"],
                        cv=StratifiedKFold(5, shuffle=True, random_state=42),
                        method="predict_proba", n_jobs=-1)[:, 1]
tr = train.assign(score=oof)
bands = sorted(tr["age_band"].unique())


def thresholds(t):
    return {b: float(np.quantile(tr.loc[(tr.age_band == b) & (tr.y == 1), "score"], 1 - t)) for b in bands}


def share(t):
    th = thresholds(t)
    return (tr["score"].values >= tr["age_band"].map(th).values).mean()


lo, hi = 0.0, 1.0
for _ in range(40):
    mid = (lo + hi) / 2
    if share(mid) > BUDGET:
        hi = mid
    else:
        lo = mid
t_star = (lo + hi) / 2
th = thresholds(t_star)
print(f"Common target recall on training data: {t_star:.3f}")
print("Score cutoffs by age band:", {k: round(v, 3) for k, v in th.items()})

pipe = joblib.load("models/stratified_no_macro_gboost.joblib")
test = test.copy()
test["score"] = pipe.predict_proba(test[FEATS])[:, 1]
n_sel = int(round(len(test) * BUDGET))
b = np.zeros(len(test), int)
b[np.argsort(-test["score"].values, kind="stable")[:n_sel]] = 1
test["sel_before"] = b
test["sel_after"] = (test["score"].values >= test["age_band"].map(th).values).astype(int)


def table(col):
    rows = []
    overall = test.loc[test.y == 1, col].mean()
    for g in ["age_band", "marital", "education"]:
        for level, x in test.groupby(g):
            pos = x[x.y == 1]
            r = pos[col].mean() if len(pos) else np.nan
            rows.append({"attribute": g, "group": level, "subscribers": len(pos),
                         "selection_rate": x[col].mean(), "recall": r,
                         "gap": r - overall if len(pos) else np.nan, "judged": len(pos) >= MIN_SUBS})
    return pd.DataFrame(rows), overall


tb, ob = table("sel_before")
ta, oa = table("sel_after")
m = tb.merge(ta, on=["attribute", "group", "subscribers", "judged"], suffixes=("_before", "_after"))
m["flag_before"] = m.judged & (m.gap_before.abs() > GAP)
m["flag_after"] = m.judged & (m.gap_after.abs() > GAP)
m.round(3).to_csv("reports/fairness_mitigation.csv", index=False)
pd.set_option("display.width", 250); pd.set_option("display.max_columns", None)
print()
print(m[["attribute", "group", "subscribers", "selection_rate_before", "selection_rate_after",
         "recall_before", "recall_after", "gap_before", "gap_after", "judged", "flag_after"]].round(3).to_string(index=False))

rng = np.random.default_rng(42)
y = test["y"].values
sa = test["sel_after"].values
bs = [sa[idx][y[idx] == 1].mean() for idx in (rng.integers(0, len(y), len(y)) for _ in range(500))]
blo, bhi = np.percentile(bs, [2.5, 97.5])
print(f"\nOverall capture: before = {ob:.3f} | after = {oa:.3f} (95% CI {blo:.3f}-{bhi:.3f}) | "
      f"share of customers selected after = {test.sel_after.mean():.3f}")
res = {"capture_before": ob, "capture_after": oa, "capture_after_ci_low": blo,
       "capture_after_ci_high": bhi, "selected_share_after": test.sel_after.mean(), "t_star": t_star}
for g in ["age_band", "marital", "education"]:
    ok = m[(m.attribute == g) & m.judged]["group"]
    dj = test[test[g].isin(ok)]
    d0 = demographic_parity_ratio(dj.y, dj.sel_before, sensitive_features=dj[g])
    d1 = demographic_parity_ratio(dj.y, dj.sel_after, sensitive_features=dj[g])
    f0 = int(m[m.attribute == g].flag_before.sum()); f1 = int(m[m.attribute == g].flag_after.sum())
    print(f"{g}: disparate impact ratio {d0:.3f} -> {d1:.3f} | groups beyond 10-pt gap {f0} -> {f1}")
    res.update({f"{g}_dir_before": d0, f"{g}_dir_after": d1, f"{g}_flags_before": f0, f"{g}_flags_after": f1})

json.dump({"age_band_cutoffs": th, "target_recall_on_training_oof": t_star, "budget": BUDGET},
          open("models/age_band_cutoffs.json", "w"), indent=2)
with mlflow.start_run(run_name="60_mitigation_age_band_cutoffs"):
    mlflow.log_metrics(res)
    mlflow.log_artifact("reports/fairness_mitigation.csv")
    mlflow.log_artifact("models/age_band_cutoffs.json")
