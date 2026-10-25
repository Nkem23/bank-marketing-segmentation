import os, joblib, mlflow, mlflow.sklearn, numpy as np, pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from common import load, split, split_stratified, CAT, NUM

MACRO = ["emp.var.rate", "cons.price.idx", "cons.conf.idx", "euribor3m", "nr.employed"]
VARIANTS = {"no_macro": MACRO, "no_macro_no_month": MACRO + ["month"]}

mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("bank-marketing-m4")


def prep_for(cat, num):
    numt = Pipeline([("imp", SimpleImputer(strategy="median", add_indicator=True)),
                     ("sc", StandardScaler())])
    return ColumnTransformer([("num", numt, num),
                              ("cat", OneHotEncoder(handle_unknown="ignore"), cat)])


def new_model(name):
    if name == "logreg":
        return LogisticRegression(max_iter=2000, class_weight="balanced", C=0.1)
    return GradientBoostingClassifier(n_estimators=200, max_depth=3, learning_rate=0.1,
                                      subsample=0.8, random_state=42)


def cap(y, s, frac=0.30):
    n = int(round(len(y) * frac))
    t = np.argsort(-s, kind="stable")[:n]
    return y[t].sum() / y.sum()


df = load()
rng = np.random.default_rng(42)
rows = []
for split_name, (train, test) in [("stratified", split_stratified(df)), ("time_ordered", split(df))]:
    y = test["y"].values
    for vname, drop in VARIANTS.items():
        cat = [c for c in CAT if c not in drop]
        num = [c for c in NUM if c not in drop]
        feats = cat + num
        for mname in ["logreg", "gboost"]:
            pipe = Pipeline([("prep", prep_for(cat, num)), ("clf", new_model(mname))])
            pipe.fit(train[feats], train["y"])
            s = pipe.predict_proba(test[feats])[:, 1]
            auc, c = roc_auc_score(y, s), cap(y, s)
            lo = hi = np.nan
            if split_name == "stratified":
                b = []
                for _ in range(500):
                    i = rng.integers(0, len(y), len(y))
                    b.append(cap(y[i], s[i]))
                lo, hi = np.percentile(b, [2.5, 97.5])
            with mlflow.start_run(run_name=f"21_{vname}_{mname}_{split_name}"):
                mlflow.set_tags({"split": split_name, "model": mname, "variant": vname})
                mlflow.log_param("dropped_features", ",".join(drop))
                mlflow.log_param("settings", "reused from tuned full model, no re-tuning")
                mlflow.log_metrics({"roc_auc": auc, "capture_at_30": c})
                if split_name == "stratified":
                    mlflow.log_metrics({"capture_at_30_ci_low": lo, "capture_at_30_ci_high": hi})
                    mlflow.sklearn.log_model(pipe, name="model", serialization_format="cloudpickle")
            joblib.dump(pipe, f"models/{split_name}_{vname}_{mname}.joblib")
            out = test[["y", "campaign", "age_band", "marital", "education"]].copy()
            out["score"] = s
            out.to_csv(f"m4/outputs/scores_{split_name}_{vname}_{mname}.csv")
            rows.append({"split": split_name, "variant": vname, "model": mname,
                         "roc_auc": round(auc, 4), "capture30": round(c, 4),
                         "ci_low": round(lo, 4), "ci_high": round(hi, 4)})
print(pd.DataFrame(rows).to_string(index=False))
