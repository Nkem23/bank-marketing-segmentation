import pytest
import pandas as pd

from src.preprocessing import EXPECTED_COLUMNS, load_data


def create_test_csv(path, columns=None):
    if columns is None:
        columns = EXPECTED_COLUMNS

    df = pd.DataFrame(
        [
            [0] * len(columns),
            [1] * len(columns),
        ],
        columns=columns,
    )

    df.to_csv(path, sep=";", index=False)


def test_load_data_success(tmp_path):
    csv_path = tmp_path / "test_data.csv"
    create_test_csv(csv_path)

    df = load_data(csv_path, expected_rows=2)

    assert df.shape == (2, len(EXPECTED_COLUMNS))
    assert list(df.columns) == EXPECTED_COLUMNS



def test_load_data_wrong_row_count(tmp_path):
    csv_path = tmp_path / "test_data.csv"
    create_test_csv(csv_path)

    with pytest.raises(ValueError, match="Expected 3 rows"):
        load_data(csv_path, expected_rows=3)



def test_load_data_wrong_schema(tmp_path):
    csv_path = tmp_path / "test_data.csv"
    bad_columns = ["customer_age" if col == "age" else col for col in EXPECTED_COLUMNS]
    create_test_csv(csv_path, columns=bad_columns)

    with pytest.raises(ValueError, match="Schema mismatch"):
        load_data(csv_path, expected_rows=2)



def test_load_data_reorders_columns(tmp_path):
    csv_path = tmp_path / "test_data.csv"
    shuffled_columns = list(reversed(EXPECTED_COLUMNS))
    create_test_csv(csv_path, columns=shuffled_columns)

    df = load_data(csv_path, expected_rows=2)

    assert list(df.columns) == EXPECTED_COLUMNS
