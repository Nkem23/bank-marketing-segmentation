import pandas as pd
from faker import Faker

from src.anonymization import (
    drop_pii_columns,
    find_pii_columns,
    k_anonymity_report,
    pseudonymise,
)
from src.validation import ORIGINAL_COLUMNS


def make_synthetic_pii_fixture():
    """Fake customers built with Faker. Test fixture only, not real data."""
    fake = Faker("en_US")
    Faker.seed(42)
    return pd.DataFrame(
        {
            "col_a": [fake.email() for _ in range(5)],
            "col_b": [fake.phone_number() for _ in range(5)],
            "customer_name": [fake.name() for _ in range(5)],
            "age": [30, 31, 32, 33, 34],
            "job": ["admin.", "services", "admin.", "retired", "student"],
        }
    )


def test_find_pii_columns_detects_names_and_value_patterns():
    flagged = find_pii_columns(make_synthetic_pii_fixture())

    assert flagged["col_a"] == "email pattern in values"
    assert flagged["col_b"] == "phone pattern in values"
    assert flagged["customer_name"] == "column name"
    assert "age" not in flagged
    assert "job" not in flagged


def test_drop_pii_columns_removes_only_flagged_columns():
    df = make_synthetic_pii_fixture()

    clean, flagged = drop_pii_columns(df)

    assert list(clean.columns) == ["age", "job"]
    assert len(flagged) == 3
    assert "col_a" in df.columns


def test_original_dataset_columns_are_not_flagged():
    df = pd.DataFrame(columns=ORIGINAL_COLUMNS)

    assert find_pii_columns(df) == {}


def test_pseudonymise_is_stable_salted_and_hides_values():
    ids = pd.Series(["C001", "C002", "C001"])

    first = pseudonymise(ids, salt="s1")
    second = pseudonymise(ids, salt="s1")
    other_salt = pseudonymise(ids, salt="s2")

    assert first.tolist() == second.tolist()
    assert first[0] == first[2]
    assert first[0] != first[1]
    assert first[0] != other_salt[0]
    assert "C001" not in first.tolist()


def test_k_anonymity_report_counts_small_groups():
    df = pd.DataFrame(
        {
            "age": [30, 30, 30, 30, 30, 70],
            "job": ["admin."] * 5 + ["retired"],
            "marital": ["married"] * 6,
            "education": ["high.school"] * 6,
        }
    )

    report = k_anonymity_report(df, k=5)

    assert report["groups"] == 2
    assert report["smallest_group"] == 1
    assert report["groups_below_k"] == 1
    assert report["rows_below_k"] == 1