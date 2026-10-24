import pandas as pd
from fairlearn.metrics import MetricFrame, count, selection_rate

AGE_BINS = [17, 25, 35, 45, 55, 65, 200]
AGE_LABELS = ["17-24", "25-34", "35-44", "45-54", "55-64", "65+"]
SENSITIVE_COLUMNS = ["age_band", "marital", "education"]
MIN_GROUP_SIZE = 100


def add_age_band(df):
    """Return a copy of df with an age_band column."""
    df = df.copy()
    df["age_band"] = pd.cut(
        df["age"], bins=AGE_BINS, labels=AGE_LABELS, right=False
    ).astype(str)
    return df


def representation_report(df, column, min_group_size=MIN_GROUP_SIZE):
    """Group sizes and subscription rates for one column, using Fairlearn."""
    subscribed = (df["y"] == "yes").astype(int)
    frame = MetricFrame(
        metrics={"rows": count, "subscription_rate": selection_rate},
        y_true=subscribed,
        y_pred=subscribed,
        sensitive_features=df[column],
    )
    report = frame.by_group.copy()
    report["rows"] = report["rows"].astype(int)
    report["share_of_rows"] = report["rows"] / len(df)
    subscribers = subscribed.groupby(df[column]).sum()
    report["share_of_subscribers"] = subscribers / subscribed.sum()
    report["rate_vs_overall"] = report["subscription_rate"] - subscribed.mean()
    report["small_group"] = report["rows"] < min_group_size
    report = report.reset_index().rename(columns={column: "group"})
    report.insert(0, "attribute", column)
    return report


def build_bias_report(df, min_group_size=MIN_GROUP_SIZE):
    """Representation report for age band, marital status and education."""
    df = add_age_band(df)
    reports = [representation_report(df, c, min_group_size) for c in SENSITIVE_COLUMNS]
    return pd.concat(reports, ignore_index=True)