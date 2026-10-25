import numpy as np, pandas as pd, joblib, mlflow
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from common import load, split_stratified, capture_at, CAT, NUM

MACRO = ["emp.var.rate", "cons.price.idx", "cons.conf.idx", "euribor3m", "nr.employed"]
NUMF = [c for c in NUM if c not in MACRO]
FEATS = CAT + NUMF
mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("bank-marketing-m4")

df = load()
train, test = split_stratified(df)
pipe = joblib.load("models/stratified_no_macro_gboost.joblib")
y = test["y"].values
rng = np.random.default_rng(42)


def ev(X):
    s = pipe.predict_proba(X[FEATS])[:, 1]
    return roc_auc_score(y, s), capture_at(y, s)


def band_gaps(s):
    n = int(round(len(y) * 0.30))
    sel = np.zeros(len(y), int)
    sel[np.argsort(-s, kind="stable")[:n]] = 1
    t = test.assign(sel=sel)
    overall = t.loc[t.y == 1, "sel"].mean()
    r = t[t.y == 1].groupby("age_band")["sel"].agg(["mean", "size"])
    flags = int(((r["size"] >= 30) & ((r["mean"] - overall).abs() > 0.10)).sum())
    return r["mean"].round(3).to_dict(), flags


rows = []
a0, c0 = ev(test)
rows.append({"test": "baseline (deployable model)", "roc_auc": a0, "capture30": c0})

for mult in [0.1, 0.5]:
    res = []
    for _ in range(20):
        X = test.copy()
        X["age"] = (X["age"] + rng.normal(0, mult * train["age"].std(), len(X))).round().clip(17, 100)
        res.append(ev(X))
    res = np.array(res)
    rows.append({"test": f"age noise, {mult} SD (mean of 20 draws)",
                 "roc_auc": res[:, 0].mean(), "capture30": res[:, 1].mean()})

X = test.copy()
for c in ["job", "marital", "education", "default", "housing", "loan"]:
    mode = train.loc[train[c] != "unknown", c].mode()[0]
    X[c] = X[c].replace("unknown", mode)
a, c = ev(X)
rows.append({"test": "'unknown' replaced by most common category", "roc_auc": a, "capture30": c})

F2 = [f for f in FEATS if f not in ("age", "marital", "education")]
cat2 = [f for f in CAT if f in F2]
num2 = [f for f in NUMF if f in F2]
numt = Pipeline([("imp", SimpleImputer(strategy="median", add_indicator=True)), ("sc", StandardScaler())])
prep2 = ColumnTransformer([("num", numt, num2), ("cat", OneHotEncoder(handle_unknown="ignore"), cat2)])
p2 = Pipeline([("prep", prep2), ("clf", GradientBoostingClassifier(
    n_estimators=200, max_depth=3, learning_rate=0.1, subsample=0.8, random_state=42))])
p2.fit(train[F2], train["y"])
s2 = p2.predict_proba(test[F2])[:, 1]
rows.append({"test": "refit without age, marital, education", "roc_auc": roc_auc_score(y, s2),
             "capture30": capture_at(y, s2)})

out = pd.DataFrame(rows)
out["capture_change_pts"] = (out["capture30"] - c0) * 100
pd.set_option("display.width", 200)
print(out.round(4).to_string(index=False))
out.round(4).to_csv("reports/sensitivity.csv", index=False)

s0 = pipe.predict_proba(test[FEATS])[:, 1]
r0, f0 = band_gaps(s0)
r2, f2 = band_gaps(s2)
print("\nAge-band recall at the 30% cut, deployable model:", r0, "| bands beyond 10-pt gap:", f0)
print("Age-band recall at the 30% cut, refit without protected attributes:", r2, "| bands beyond 10-pt gap:", f2)

with mlflow.start_run(run_name="80_sensitivity"):
    for _, r in out.iterrows():
        k = r["test"][:25].replace(" ", "_").replace(",", "").replace("'", "")
        mlflow.log_metrics({f"{k}_capture30": r["capture30"], f"{k}_auc": r["roc_auc"]})
    mlflow.log_metrics({"refit_no_protected_gap_flags": f2, "deployable_gap_flags": f0})
    mlflow.log_artifact("reports/sensitivity.csv")
