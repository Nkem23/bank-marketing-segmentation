import mlflow, numpy as np
from sklearn.metrics import roc_auc_score
from common import load, split, split_stratified, capture_at

mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("bank-marketing-m4")
df = load()
rng = np.random.default_rng(42)
rank = {"success": 2, "failure": 1, "nonexistent": 0}


def run(name, train, test):
    y = test["y"].values
    rule = test["poutcome"].map(rank).values + rng.random(len(test)) * 1e-6
    rand_caps = [capture_at(y, rng.random(len(y))) for _ in range(200)]
    res = {
        "test_rows": len(test),
        "test_subscribers": int(y.sum()),
        "test_prevalence": float(y.mean()),
        "train_prevalence": float(train["y"].mean()),
        "majority_accuracy": float(1 - y.mean()),
        "random_capture_at_30": float(np.mean(rand_caps)),
        "rule_capture_at_30": float(capture_at(y, rule)),
        "rule_roc_auc": float(roc_auc_score(y, rule)),
    }
    with mlflow.start_run(run_name=name):
        mlflow.log_metrics(res)
    print(name)
    for k, v in res.items():
        print("  ", k, round(v, 4))


run("10_baselines_stratified", *split_stratified(df))
run("10_baselines_time_ordered", *split(df))
