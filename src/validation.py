import great_expectations as gx
import pandas as pd
from great_expectations import expectations as gxe

CATEGORY_VALUES = {
    "job": ["admin.", "blue-collar", "entrepreneur", "housemaid", "management",
            "retired", "self-employed", "services", "student", "technician",
            "unemployed", "unknown"],
    "marital": ["divorced", "married", "single", "unknown"],
    "education": ["basic.4y", "basic.6y", "basic.9y", "high.school", "illiterate",
                  "professional.course", "university.degree", "unknown"],
    "default": ["no", "yes", "unknown"],
    "housing": ["no", "yes", "unknown"],
    "loan": ["no", "yes", "unknown"],
    "contact": ["cellular", "telephone"],
    "month": ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep",
              "oct", "nov", "dec"],
    "day_of_week": ["mon", "tue", "wed", "thu", "fri"],
    "poutcome": ["failure", "nonexistent", "success"],
    "y": ["no", "yes"],
}

ORIGINAL_COLUMNS = [
    "age", "job", "marital", "education", "default", "housing", "loan",
    "contact", "month", "day_of_week", "duration", "campaign", "pdays",
    "previous", "poutcome", "emp.var.rate", "cons.price.idx", "cons.conf.idx",
    "euribor3m", "nr.employed", "y",
]


def build_expectations():
    """Return the list of Great Expectations checks for the cleaned data."""
    checks = [
        gxe.ExpectTableColumnsToMatchSet(
            column_set=ORIGINAL_COLUMNS + ["previously_contacted",
                                           "days_since_previous_contact"],
            exact_match=True,
        ),
        gxe.ExpectCompoundColumnsToBeUnique(column_list=ORIGINAL_COLUMNS),
        gxe.ExpectColumnValuesToBeBetween(column="age", min_value=17, max_value=100),
        gxe.ExpectColumnValuesToBeInSet(column="previously_contacted", value_set=[0, 1]),
        gxe.ExpectColumnValuesToBeBetween(
            column="days_since_previous_contact", min_value=0, max_value=27
        ),
    ]
    for column in ORIGINAL_COLUMNS:
        checks.append(gxe.ExpectColumnValuesToNotBeNull(column=column))
    for column, allowed in CATEGORY_VALUES.items():
        checks.append(gxe.ExpectColumnValuesToBeInSet(column=column, value_set=allowed))
    return checks


def validate_data(df, project_dir="gx_project"):
    """Run the validation suite on df and build the Data Docs report.

    Returns a dict with the overall result, the 'unknown' count per column,
    the pdays contradiction count and the target rate.
    """
    context = gx.get_context(mode="file", project_root_dir=project_dir)

    data_source = context.data_sources.add_or_update_pandas(name="bank_marketing")
    asset = data_source.add_dataframe_asset(name="cleaned_data")
    batch_definition = asset.add_batch_definition_whole_dataframe("whole_dataframe")

    suite = context.suites.add_or_update(gx.ExpectationSuite(name="bank_marketing_suite"))
    suite.expectations = []
    for check in build_expectations():
        suite.add_expectation(check)
    suite.save()

    validation_definition = context.validation_definitions.add_or_update(
        gx.ValidationDefinition(name="bank_marketing_validation",
                                data=batch_definition, suite=suite)
    )
    checkpoint = context.checkpoints.add_or_update(
        gx.Checkpoint(
            name="bank_marketing_checkpoint",
            validation_definitions=[validation_definition],
            actions=[gx.checkpoint.UpdateDataDocsAction(name="update_data_docs")],
        )
    )
    result = checkpoint.run(batch_parameters={"dataframe": df})

    unknown_counts = (df[list(CATEGORY_VALUES)] == "unknown").sum()
    contradictions = int(((df["pdays"] == 999) & (df["previous"] > 0)).sum())
    target_rate = float((df["y"] == "yes").mean())
    target_rate_ok = 0.10 <= target_rate <= 0.125

    return {
        "success": bool(result.success) and target_rate_ok,
        "gx_success": bool(result.success),
        "target_rate": target_rate,
        "target_rate_ok": target_rate_ok,
        "unknown_counts": unknown_counts[unknown_counts > 0].to_dict(),
        "pdays_contradictions": contradictions,
        "result": result,
    }