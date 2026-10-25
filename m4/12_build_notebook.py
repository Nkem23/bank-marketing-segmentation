import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []
def md(t): cells.append(nbf.v4.new_markdown_cell(t))
def code(t): cells.append(nbf.v4.new_code_cell(t))

md("""# SHAP Analysis: Term-Deposit Outreach Ranker (BAN6800, Module 4)
This notebook explains the proposed model (`models/stratified_no_macro_gboost.joblib`, Gradient Boosting without the five economic indicators) on the stratified hold-out test set. SHAP values are in log-odds: positive values push a customer's score up, negative values push it down. It also contrasts the proposed model with the full-feature model, which includes the economic indicators.

**What SHAP can and cannot say.** It describes what the model relies on. It does not show that changing a feature would change a customer's behaviour, and correlated features (for example channel and month, which partly reflect the campaign period) share credit unpredictably.""")

code("""import os, sys, warnings
warnings.filterwarnings("ignore")
if os.path.basename(os.getcwd()) == "notebooks":
    os.chdir("..")
sys.path.insert(0, "m4")
import numpy as np, pandas as pd, joblib, shap
import matplotlib.pyplot as plt
from common import load, split_stratified, CAT, NUM, FEATURES

MACRO = ["emp.var.rate", "cons.price.idx", "cons.conf.idx", "euribor3m", "nr.employed"]
FEATS = CAT + [c for c in NUM if c not in MACRO]
df = load()
train, test = split_stratified(df)
print("test rows:", len(test), "| subscribers:", int(test["y"].sum()))""")

code("""def explain(path, feats):
    pipe = joblib.load(path)
    prep, clf = pipe.named_steps["prep"], pipe.named_steps["clf"]
    X = prep.transform(test[feats])
    X = X.toarray() if hasattr(X, "toarray") else np.asarray(X)
    names = [n.split("__", 1)[-1] for n in prep.get_feature_names_out()]
    ex = shap.TreeExplainer(clf)
    sv = ex.shap_values(X)
    sv = sv[1] if isinstance(sv, list) else sv
    base = float(np.ravel(ex.expected_value)[0])
    return pipe, X, names, sv, base

pipe, X, names, sv, base = explain("models/stratified_no_macro_gboost.joblib", FEATS)
imp = pd.Series(np.abs(sv).mean(axis=0), index=names).sort_values(ascending=False)
print("Proposed model: top 12 features by mean |SHAP| (log-odds)")
imp.head(12).round(4)""")

code("""shap.summary_plot(sv, X, feature_names=names, max_display=12)""")

md("""### Your interpretation: global importance (proposed model)
*(Replace this cell with 3 to 4 sentences in your own words. Cover: which features matter most; which of them the bank controls (channel, timing) and which it does not (age, history); and one caution about reading the plot as cause and effect.)*""")

code("""full_pipe, Xf, names_f, sv_f, base_f = explain("models/stratified_gboost.joblib", FEATURES)
imp_f = pd.Series(np.abs(sv_f).mean(axis=0), index=names_f).sort_values(ascending=False)
print("Full model (with economic indicators): top 8 features by mean |SHAP|")
display(imp_f.head(8).round(4))
shap.summary_plot(sv_f, Xf, feature_names=names_f, max_display=10)""")

md("""### Your interpretation: why the economic indicators were removed
*(Replace this cell with 3 to 4 sentences in your own words. Cover: what the full model relied on; why those features cannot separate one customer from another on the same day; and what removing them cost: 73.5% capture became 67.2%.)*""")

code("""PLAIN = {
    "contact_cellular": "contacted on a mobile phone",
    "contact_telephone": "contacted on a landline",
    "missingindicator_days_since_previous_contact": "whether a previous-contact date is on record",
    "age": "age",
    "default_no": "no credit default on record",
    "poutcome_success": "previous campaign was a success",
    "poutcome_failure": "previous campaign failed",
}
def plain(n):
    return PLAIN.get(n, n.replace("_", " = ", 1) if "_" in n else n)

score = pipe.predict_proba(test[FEATS])[:, 1]
order = np.argsort(-score, kind="stable")
cut = int(round(len(score) * 0.30))
picks = {"top-ranked customer": order[0], "borderline customer (rank at the 30% cut)": order[cut - 1], "bottom-ranked customer": order[-1]}
for label, i in picks.items():
    row = test.iloc[i]
    print("=" * 70)
    print(label, "| predicted probability:", round(float(score[i]), 3))
    print(row[["age", "job", "marital", "education", "contact", "month", "poutcome", "previous"]].to_dict())
    top = pd.Series(sv[i], index=names).sort_values(key=np.abs, ascending=False).head(4)
    for n, v in top.items():
        print(f"   {plain(n)}: {v:+.2f} log-odds ({'raises' if v > 0 else 'lowers'} the score)")
    shap.plots.waterfall(shap.Explanation(values=sv[i], base_values=base, feature_names=names), max_display=8)""")

md("""### Your interpretation: local explanations
*(Replace this cell with 3 to 4 sentences in your own words, as you would explain it to a relationship manager. For each of the three customers say in plain language what pushed the score up or down. Say what a manager could do with this and what they should not read into it.)*""")

code("""age_idx = names.index("age")
t = pd.DataFrame({"age_band": test["age_band"].values, "age_shap": sv[:, age_idx]})
print("Average SHAP contribution of the age feature, by age band (log-odds)")
t.groupby("age_band")["age_shap"].agg(["mean", "size"]).round(3)""")

md("""### Your interpretation: age and fairness
*(Replace this cell with 3 to 4 sentences in your own words. Cover: how the age contribution differs by band; how this links to the fairness table (young and 65+ customers selected far more often than 35-54s); and why removing the age column alone did not remove the gaps.)*""")

md("""### Limitations of this analysis
SHAP explains the model's behaviour on this data, not why customers subscribe. Channel, month and previous-campaign status partly reflect when and how the bank ran its campaigns, so their contributions mix customer behaviour with campaign design. Contributions are computed on the stratified test set only. The waterfall plots show log-odds contributions, with no raw feature values.""")

nb["cells"] = cells
nb["metadata"]["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
nbf.write(nb, "notebooks/05_shap_analysis.ipynb")
print("notebook written:", len(cells), "cells")
