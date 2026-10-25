import itertools, json, numpy as np, pandas as pd, joblib, dice_ml, mlflow
from common import load, split_stratified, CAT, NUM
from src.bias import AGE_BINS, AGE_LABELS

MACRO = ["emp.var.rate", "cons.price.idx", "cons.conf.idx", "euribor3m", "nr.employed"]
NUMF = [c for c in NUM if c not in MACRO]
FEATS = CAT + NUMF
LEVERS = ["contact", "month", "day_of_week"]
CUT = json.load(open("models/age_band_cutoffs.json"))["age_band_cutoffs"]
pipe = joblib.load("models/stratified_no_macro_gboost.joblib")

mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("bank-marketing-m4")


def cutoff_for(age):
    lab = pd.cut(pd.Series(np.asarray(age)), bins=AGE_BINS, labels=AGE_LABELS, right=False).astype(str)
    return lab.map(CUT).values


class Selector:
    """Model plus age-band cutoffs: class 1 = customer makes the outreach list."""
    def __init__(self, p):
        self.p = p

    def score(self, X):
        X = X[FEATS].copy()
        X["days_since_previous_contact"] = X["days_since_previous_contact"].replace(999, np.nan)
        return self.p.predict_proba(X)[:, 1]

    def predict_proba(self, X):
        q = (self.score(X) >= cutoff_for(X["age"].values)).astype(float)
        return np.column_stack([1 - q, q])

    def predict(self, X):
        return self.predict_proba(X)[:, 1].round().astype(int)


def prep(d):
    d = d[FEATS].copy().reset_index(drop=True)
    d["days_since_previous_contact"] = d["days_since_previous_contact"].fillna(999)
    return d


df = load()
train, test = split_stratified(df)
sel = Selector(pipe)
tr, te = prep(train), prep(test)
tr["selected"] = sel.predict(tr[FEATS])

# Part 1: DiCE on three non-selected customers
d = dice_ml.Data(dataframe=tr, continuous_features=["age", "previous", "days_since_previous_contact"],
                 outcome_name="selected")
m = dice_ml.Model(model=sel, backend="sklearn")
exp = dice_ml.Dice(d, m, method="random")

te["score"] = sel.score(te)
te["cutoff"] = cutoff_for(te["age"].values)
te["selected"] = (te["score"] >= te["cutoff"]).astype(int)
ns = te[te["selected"] == 0].copy()
ns["margin"] = ns["score"] - ns["cutoff"]
ns = ns.sort_values("margin", ascending=False)
picks = {"just_below_cutoff": ns.iloc[10], "mid_below": ns.iloc[500], "far_below": ns.iloc[2500]}
for label, row in picks.items():
    print(f"\n=== {label}: score {row['score']:.3f} vs cutoff {row['cutoff']:.3f} ===")
    print(row[["age", "job", "marital", "education", "contact", "month", "day_of_week", "poutcome", "previous"]].to_dict())
    q = pd.DataFrame([row[FEATS].to_dict()])
    for c in tr[FEATS].columns:
        q[c] = q[c].astype(tr[c].dtype)
    try:
        res = exp.generate_counterfactuals(q, total_CFs=3, desired_class="opposite",
                                           features_to_vary=LEVERS)
        cf = res.cf_examples_list[0].final_cfs_df
        if cf is None or len(cf) == 0:
            print("No counterfactual found by changing only channel, month and weekday.")
        else:
            print("Counterfactuals (changed levers only):")
            print(cf[LEVERS].to_string(index=False))
    except Exception as e:
        print("DiCE found no counterfactual:", type(e).__name__, str(e)[:120])

# Part 2: how many non-selected customers can reach the list by changing only the levers?
space = pd.DataFrame(list(itertools.product(*[sorted(tr[c].unique()) for c in LEVERS])), columns=LEVERS)
base = ns[FEATS].drop(columns=LEVERS).copy()
base["rid"] = np.arange(len(base))
base["orig_month"] = ns["month"].values
base["orig_contact"] = ns["contact"].values
big = base.merge(space, how="cross")
big["ok"] = (sel.score(big) >= cutoff_for(big["age"].values))
reach_any = big.groupby("rid")["ok"].any().mean()
reach_nomonth = big[big["month"] == big["orig_month"]].groupby("rid")["ok"].any().mean()
reach_channel = big[(big["month"] == big["orig_month"]) & (big["day_of_week"].notna())].copy()
print(f"\nNon-selected customers in test set: {len(ns)}")
print(f"Could reach the list by changing channel/month/weekday (any): {reach_any:.1%}")
print(f"Could reach the list by changing channel/weekday only (same month): {reach_nomonth:.1%}")
with mlflow.start_run(run_name="70_counterfactuals"):
    mlflow.log_metrics({"reach_any_lever": reach_any, "reach_channel_weekday_only": reach_nomonth})
