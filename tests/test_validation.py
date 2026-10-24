import pandas as pd

from src.preprocessing import add_pdays_features
from src.validation import CATEGORY_VALUES, ORIGINAL_COLUMNS, validate_data


def make_valid_df():
    rows = []
    for i in range(10):
        row = {col: values[0] for col, values in CATEGORY_VALUES.items()}
        row["y"] = "no"
        row["age"] = 30 + i
        row["duration"] = 100 + i
        row["campaign"] = 1
        row["pdays"] = 999
        row["previous"] = 0
        row["emp.var.rate"] = 1.1
        row["cons.price.idx"] = 93.9
        row["cons.conf.idx"] = -36.4
        row["euribor3m"] = 4.8
        row["nr.employed"] = 5191.0
        rows.append(row)
    df = pd.DataFrame(rows)[ORIGINAL_COLUMNS]
    df.loc[0, "y"] = "yes"
    return add_pdays_features(df)


def test_validate_data_passes_on_clean_data(tmp_path):
    result = validate_data(make_valid_df(), project_dir=str(tmp_path))

    assert result["gx_success"] is True


def test_validate_data_fails_on_bad_values(tmp_path):
    df = make_valid_df()
    df.loc[0, "y"] = "maybe"
    df.loc[1, "age"] = 5

    result = validate_data(df, project_dir=str(tmp_path))

    assert result["gx_success"] is False