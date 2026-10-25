import os, joblib, mlflow, mlflow.sklearn, numpy as np
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold, TimeSeriesSplit
from sklearn.metrics import roc_auc_score, average_precision_score
from common import load, split, split_stratified, make_preprocessor, FEATURES, GROUPS

MODELS = {
    "logreg": (LogisticRegression(max_iter=2000, class_weight="balanced"),
               {"clf__C": [0.01, 0.1, 1, 10]}),
    "gboost": (GradientBoostingClassifier(subsample=0.8, random_state=42),
               {"clf__n_estimators": [100, 200], "clf__max_depth": [2, 3],
                "clf__learning_rate": [0.05, 0.1]}),
}


def metrics_at(y, s, frac=0.30):
    n = int(round(len(y) * frac))
    top = np.argsort(-s, kind="stable")[:n]
    tp = y[top].sum()
    prec, rec = tp / n, tp / y.sum()
    return {"roc_auc": roc_auc_score(y, s), "pr_auc": average_precision_score(y, s),
            "capture_at_30": rec, "lift_at_30": rec / frac, "precision_at_30": prec,
            "f1_at_30": 2 * prec * rec / (prec + rec)}


def main():
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("bank-marketing-m4")
    df = load()
    os.makedirs("models", exist_ok=True)
    os.makedirs("m4/outputs", exist_ok=True)
    plans = [
        ("stratified", split_stratified(df), StratifiedKFold(5, shuffle=True, random_state=42)),
        ("time_ordered", split(df), TimeSeriesSplit(4)),
    ]
    for split_name, (train, test), cv in plans:
        scores = test[["y", "campaign"] + GROUPS].copy()
        for name, (est, grid) in MODELS.items():
            pipe = Pipeline([("prep", make_preprocessor()), ("clf", est)])
            gs = GridSearchCV(pipe, grid, scoring="roc_auc", cv=cv, n_jobs=-1)
            gs.fit(train[FEATURES], train["y"])
            best = gs.best_estimator_
            s = best.predict_proba(test[FEATURES])[:, 1]
            m = metrics_at(test["y"].values, s)
            with mlflow.start_run(run_name=f"20_{name}_{split_name}"):
                mlflow.set_tags({"split": split_name, "model": name})
                mlflow.log_params(gs.best_params_)
                mlflow.log_param("cv", type(cv).__name__)
                mlflow.log_metric("cv_roc_auc", gs.best_score_)
                mlflow.log_metrics(m)
                mlflow.sklearn.log_model(best, name="model", serialization_format="cloudpickle")
            joblib.dump(best, f"models/{split_name}_{name}.joblib")
            scores[name] = s
            print(split_name, name, "| best:", gs.best_params_,
                  "| cv_auc:", round(gs.best_score_, 4),
                  "| test:", {k: round(v, 4) for k, v in m.items()})
        scores.to_csv(f"m4/outputs/scores_{split_name}.csv")


if __name__ == "__main__":
    main()
