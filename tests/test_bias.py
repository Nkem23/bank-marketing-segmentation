import pandas as pd

from src.bias import build_bias_report


def make_df():
    return pd.DataFrame(
        {
            "age": [20, 22, 30, 40, 70, 71],
            "marital": ["single", "single", "married", "married", "married", "divorced"],
            "education": ["basic.4y"] * 3 + ["high.school"] * 3,
            "y": ["yes", "no", "no", "no", "yes", "no"],
        }
    )


def test_bias_report_shares_and_rates():
    report = build_bias_report(make_df(), min_group_size=2)

    marital = report[report["attribute"] == "marital"].set_index("group")
    assert marital["rows"].to_dict() == {"divorced": 1, "married": 3, "single": 2}
    assert abs(marital["share_of_rows"].sum() - 1.0) < 1e-9
    assert abs(marital["share_of_subscribers"].sum() - 1.0) < 1e-9
    assert marital.loc["single", "subscription_rate"] == 0.5


def test_bias_report_flags_small_groups_and_bands_age():
    report = build_bias_report(make_df(), min_group_size=2)

    marital = report[report["attribute"] == "marital"].set_index("group")
    assert bool(marital.loc["divorced", "small_group"]) is True
    assert bool(marital.loc["married", "small_group"]) is False

    ages = report[report["attribute"] == "age_band"].set_index("group")
    assert ages.loc["17-24", "rows"] == 2
    assert ages.loc["65+", "rows"] == 2