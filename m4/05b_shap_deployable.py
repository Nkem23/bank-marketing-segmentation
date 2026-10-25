import os, joblib, numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt, shap, mlflow
from common import load, split_stratified, CAT, NUM

MACRO = ["emp.var.rate", "cons.price.idx", "cons.conf.idx", "euribor3m", "nr.employed"]
FEATS = CAT + [c for c in NUM if c not in MACRO]

os.makedirs("reports/figures", exist_ok=True)
mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("bank-marketing-m4")

df = load()
train, test = split_stratified(df)
pipe = joblib.load("models/stratified_no_macro_gboost.joblib")
prep, clf = pipe.named_steps["prep"], pipe.named_steps["clf"]
X = prep.transform(test[FEATS])
X = X.toarray() if hasattr(X, "toarray") else np.asarray(X)
names = [n.split("__", 1)[-1] for n in prep.get_feature_names_out()]

explainer = shap.TreeExplainer(clf)
sv = explainer.shap_values(X)
sv = sv[1] if isinstance(sv, list) else sv
base = float(np.ravel(explainer.expected_value)[0])

imp = pd.Series(np.abs(sv).mean(axis=0), index=names).sort_values(ascending=False)
imp.to_csv("m4/outputs/shap_importance_deployable.csv")
print("Top 12 features by mean |SHAP| (log-odds):")
print(imp.head(12).round(4).to_string())

plt.figure()
shap.summary_plot(sv, X, feature_names=names, max_display=12, show=False)
plt.savefig("reports/figures/fig6_shap_summary_deployable.png", dpi=200, bbox_inches="tight")
plt.close()

score = pipe.predict_proba(test[FEATS])[:, 1]
order = np.argsort(-score, kind="stable")
cut = int(round(len(score) * 0.30))
picks = {"top_ranked": order[0], "borderline": order[cut - 1], "bottom_ranked": order[-1]}
for label, i in picks.items():
    ex = shap.Explanation(values=sv[i], base_values=base, data=X[i], feature_names=names)
    shap.plots.waterfall(ex, max_display=8, show=False)
    plt.savefig(f"reports/figures/fig7_shap_local_{label}_deployable.png", dpi=200, bbox_inches="tight")
    plt.close()
    row = test.iloc[i]
    print(f"\n{label}: predicted probability = {score[i]:.3f}")
    print(row[["age", "job", "marital", "education", "contact", "month", "poutcome", "previous"]].to_dict())
    top = pd.Series(sv[i], index=names).sort_values(key=np.abs, ascending=False).head(4)
    print("main drivers (log-odds):", top.round(3).to_dict())

with mlflow.start_run(run_name="41_shap_deployable"):
    mlflow.log_artifact("m4/outputs/shap_importance_deployable.csv")
    mlflow.log_artifacts("reports/figures")
print("\nFigures saved in reports/figures/")
