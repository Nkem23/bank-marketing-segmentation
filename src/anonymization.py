import hashlib
import re

import pandas as pd

from src.bias import add_age_band

PII_NAME_TOKENS = {
    "name", "firstname", "lastname", "surname", "email", "phone", "mobile",
    "address", "street", "postcode", "zip", "ssn", "passport", "iban",
    "account", "card", "dob", "birthdate", "nin", "bvn", "id",
}
EMAIL_PATTERN = r"[^@\s]+@[^@\s]+\.[A-Za-z]{2,}"
PHONE_PATTERN = r"\+?\d(?:[\s\-().]?\d){8,14}"
QUASI_IDENTIFIERS = ["age_band", "job", "marital", "education"]


def find_pii_columns(df):
    """Return {column: reason} for columns that look like direct identifiers.

    A column is flagged if a word in its name is on the PII list, or if its
    text values contain email or phone number patterns.
    """
    flagged = {}
    for column in df.columns:
        tokens = set(re.split(r"[^a-z0-9]+", str(column).lower()))
        if tokens & PII_NAME_TOKENS:
            flagged[column] = "column name"
            continue
        if pd.api.types.is_string_dtype(df[column]):
            values = df[column].astype(str)
            if values.str.contains(EMAIL_PATTERN, regex=True).any():
                flagged[column] = "email pattern in values"
            elif values.str.contains(PHONE_PATTERN, regex=True).any():
                flagged[column] = "phone pattern in values"
    return flagged


def drop_pii_columns(df):
    """Drop flagged columns. Returns (clean_df, {column: reason})."""
    flagged = find_pii_columns(df)
    return df.drop(columns=list(flagged)), flagged


def pseudonymise(series, salt):
    """Replace each value with a salted SHA-256 hash (first 16 characters)."""
    return series.astype(str).map(
        lambda value: hashlib.sha256((salt + value).encode("utf-8")).hexdigest()[:16]
    )


def k_anonymity_report(df, quasi_identifiers=QUASI_IDENTIFIERS, k=5):
    """Measure re-identification risk from combinations of quasi-identifiers.

    Customers who share the same combination of values form a group. A group
    smaller than k is easier to single out.
    """
    if "age_band" in quasi_identifiers and "age_band" not in df.columns:
        df = add_age_band(df)
    sizes = df.groupby(quasi_identifiers, observed=True).size()
    below = sizes[sizes < k]
    return {
        "k": k,
        "groups": int(len(sizes)),
        "smallest_group": int(sizes.min()),
        "groups_below_k": int(len(below)),
        "rows_below_k": int(below.sum()),
        "share_rows_below_k": float(below.sum() / len(df)),
    }