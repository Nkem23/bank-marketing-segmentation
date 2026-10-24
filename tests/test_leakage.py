import pandas as pd

from src.preprocessing import select_model_inputs


def test_select_model_inputs_removes_leakage_columns_only():
    df = pd.DataFrame(
        {"age": [30, 40], "duration": [100, 200], "campaign": [1, 2], "y": ["no", "yes"]}
    )

    result = select_model_inputs(df)

    assert list(result.columns) == ["age", "y"]
    assert list(df.columns) == ["age", "duration", "campaign", "y"]