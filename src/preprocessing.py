import pandas as pd

EXPECTED_ROWS = 41188

EXPECTED_COLUMNS = [
    "age",
    "job",
    "marital",
    "education",
    "default",
    "housing",
    "loan",
    "contact",
    "month",
    "day_of_week",
    "duration",
    "campaign",
    "pdays",
    "previous",
    "poutcome",
    "emp.var.rate",
    "cons.price.idx",
    "cons.conf.idx",
    "euribor3m",
    "nr.employed",
    "y",
]


def load_data(file_path, expected_rows=EXPECTED_ROWS):
    """Load the bank marketing CSV and check it against the expected schema.

    Raises ValueError if the row count differs from expected_rows, or if
    columns are missing or unexpected. Column order is not enforced; the
    returned DataFrame always has columns in EXPECTED_COLUMNS order.
    """
    df = pd.read_csv(file_path, sep=";")

    if len(df) != expected_rows:
        raise ValueError(
            f"Expected {expected_rows} rows, but found {len(df)}."
        )

    missing = set(EXPECTED_COLUMNS) - set(df.columns)
    unexpected = set(df.columns) - set(EXPECTED_COLUMNS)
    if missing or unexpected:
        raise ValueError(
            f"Schema mismatch. Missing columns: {sorted(missing)}. "
            f"Unexpected columns: {sorted(unexpected)}."
        )

    return df[EXPECTED_COLUMNS]


def add_pdays_features(df):
    """Split pdays into a contacted-before flag and a days-since-contact column.

    previously_contacted comes from the previous column, not pdays, because
    4,110 rows have pdays == 999 but a recorded earlier contact.
    days_since_previous_contact is NaN where pdays == 999.
    """
    df = df.copy()
    df["previously_contacted"] = (df["previous"] > 0).astype(int)
    df["days_since_previous_contact"] = df["pdays"].where(df["pdays"] != 999)
    return df


def drop_duplicate_rows(df):
    """Drop exact duplicate rows, keeping the first occurrence."""
    return df.drop_duplicates().reset_index(drop=True)