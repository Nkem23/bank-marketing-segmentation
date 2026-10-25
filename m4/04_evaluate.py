import os, numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
from sklearn.metrics import roc_curve, roc_auc_score, ConfusionMatrixDisplay, accuracy_score

os.makedirs("reports/figures", exist_ok=True)
mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("bank-marketing-m4")

s = pd.read_csv("m4/outputs/scores_stratified.csv", index_col=0)
y = s["y"].values
camp = s["campaign"].values
models = {"Logistic Regression": s["logreg"].values, "Gradient Boosting": s["gboost"].values}


def top_idx(sc, frac=0.30):
    return np.argsort(-sc, kind="stable")[: int(round(len(sc) * frac))]


def cap(yy, sc):
    return yy[top_idx(sc)].sum() / yy.sum()


rng = np.random.default_rng(42)
boot = {k: [] for k in models}
diff = []
for _ in range(1000):
    i = rng.integers(0, len(y), len(y))
    c = {k: cap(y[i], v[i]) for k, v in models.items()}
    for k in models:
        boot[k].append(c[k])
    diff.append(c["Gradient Boosting"] - c["Logistic Regression"])

results = {}
print("Test rows:", len(y), "| subscribers:", int(y.sum()))
random_cps = camp.sum() / y.sum()
for k, v in models.items():
    lo, hi = np.percentile(boot[k], [2.5, 97.5])
    t = top_idx(v)
    cps = camp[t].sum() / y[t].sum()
    pred = np.zeros(len(y), int); pred[t] = 1
    acc = accuracy_score(y, pred)
    results[k] = (cap(y, v), lo, hi, cps, acc)
    print(f"{k}: capture@30 = {cap(y, v):.3f} (95% CI {lo:.3f}-{hi:.3f}) | "
          f"contacts/subscriber in selected = {cps:.2f} vs {random_cps:.2f} overall | "
          f"accuracy at the 30% cut = {acc:.3f} (majority class = {1 - y.mean():.3f})")
dlo, dhi = np.percentile(diff, [2.5, 97.5])
print(f"GB minus LR capture difference: {np.mean(diff):.4f} (95% CI {dlo:.4f} to {dhi:.4f})")

# Figure 1: ROC
plt.figure(figsize=(5.5, 5))
for k, v in models.items():
    fpr, tpr, _ = roc_curve(y, v)
    plt.plot(fpr, tpr, label=f"{k} (AUC {roc_auc_score(y, v):.3f})")
plt.plot([0, 1], [0, 1], "--", color="grey", label="Random")
plt.xlabel("False positive rate"); plt.ylabel("True positive rate")
plt.title("ROC curves, stratified hold-out set"); plt.legend(); plt.tight_layout()
plt.savefig("reports/figures/fig1_roc.png", dpi=200); plt.close()

# Figure 2: cumulative gain
plt.figure(figsize=(5.5, 5))
for k, v in models.items():
    o = np.argsort(-v, kind="stable")
    plt.plot(np.arange(1, len(y) + 1) / len(y), np.cumsum(y[o]) / y.sum(), label=k)
plt.plot([0, 1], [0, 1], "--", color="grey", label="Random selection")
plt.axvline(0.30, color="k", lw=0.8); plt.axhline(0.70, color="red", lw=0.8, ls=":", label="70% target")
plt.xlabel("Share of customers contacted (ranked by score)")
plt.ylabel("Share of subscribers captured")
plt.title("Cumulative gain, stratified hold-out set"); plt.legend(); plt.tight_layout()
plt.savefig("reports/figures/fig2_gain.png", dpi=200); plt.close()

# Figure 3: confusion matrix at the 30% cut, Gradient Boosting
t = top_idx(models["Gradient Boosting"])
pred = np.zeros(len(y), int); pred[t] = 1
ConfusionMatrixDisplay.from_predictions(y, pred, display_labels=["No", "Yes"], colorbar=False)
plt.title("Gradient Boosting, top 30% selected as 'contact'")
plt.tight_layout(); plt.savefig("reports/figures/fig3_confusion.png", dpi=200); plt.close()

with mlflow.start_run(run_name="30_evaluation_stratified"):
    for k, (c, lo, hi, cps, acc) in results.items():
        tag = k.split()[0].lower()
        mlflow.log_metrics({f"{tag}_capture30": c, f"{tag}_capture30_ci_low": lo,
                            f"{tag}_capture30_ci_high": hi, f"{tag}_contacts_per_sub": cps})
    mlflow.log_metrics({"gb_minus_lr_capture_ci_low": dlo, "gb_minus_lr_capture_ci_high": dhi})
    mlflow.log_artifacts("reports/figures")
print("Figures saved in reports/figures/")
