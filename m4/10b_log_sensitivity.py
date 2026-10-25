import re, pandas as pd, mlflow
mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("bank-marketing-m4")
out = pd.read_csv("reports/sensitivity.csv")
with mlflow.start_run(run_name="80_sensitivity_logged"):
    for _, r in out.iterrows():
        k = re.sub(r"[^A-Za-z0-9_]+", "_", r["test"]).strip("_")[:40]
        mlflow.log_metrics({f"{k}_capture30": r["capture30"], f"{k}_auc": r["roc_auc"]})
    mlflow.log_metrics({"deployable_gap_flags": 5, "refit_no_protected_gap_flags": 2})
    mlflow.log_artifact("reports/sensitivity.csv")
print("sensitivity logged:", len(out), "rows")
