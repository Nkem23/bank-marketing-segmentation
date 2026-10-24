from pathlib import Path

from prefect import flow, task

from src.audit import get_audit_logger, log_event
from src.preprocessing import add_pdays_features, drop_duplicate_rows, load_data
from src.validation import validate_data

RAW_PATH = "data/raw/bank-additional-full.csv"
OUTPUT_PATH = "data/processed/bank_marketing_clean.csv"


@task
def load_step(raw_path, logger, expected_rows):
    df = load_data(raw_path, expected_rows=expected_rows)
    log_event(logger, "load", "read_csv_and_check_schema",
              rows_out=len(df), file_path=raw_path)
    return df


@task
def pdays_step(df, logger):
    result = add_pdays_features(df)
    log_event(logger, "transform", "add_pdays_features",
              rows_in=len(df), rows_out=len(result),
              detail="added previously_contacted and days_since_previous_contact")
    return result


@task
def dedupe_step(df, logger):
    result = drop_duplicate_rows(df)
    log_event(logger, "clean", "drop_duplicate_rows",
              rows_in=len(df), rows_out=len(result),
              detail=f"removed {len(df) - len(result)} duplicate rows")
    return result


@task
def validate_step(df, logger, project_dir):
    outcome = validate_data(df, project_dir=project_dir)
    log_event(logger, "validate", "great_expectations_suite",
              rows_in=len(df), rows_out=len(df),
              detail=(f"success={outcome['success']}; "
                      f"target_rate={outcome['target_rate']:.4f}; "
                      f"pdays_contradictions={outcome['pdays_contradictions']}"))
    if not outcome["success"]:
        raise ValueError("Validation failed. See the Great Expectations report.")
    return outcome


@task
def save_step(df, output_path, logger):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    log_event(logger, "output", "write_clean_csv",
              rows_in=len(df), rows_out=len(df), file_path=output_path)
    return str(output_path)


@flow(name="bank-marketing-pipeline")
def run_pipeline(raw_path=RAW_PATH, output_path=OUTPUT_PATH,
                 log_path="logs/audit.log", gx_dir="gx_project",
                 expected_rows=41188):
    logger = get_audit_logger(log_path)
    df = load_step(raw_path, logger, expected_rows)
    df = pdays_step(df, logger)
    df = dedupe_step(df, logger)
    validate_step(df, logger, gx_dir)
    return save_step(df, output_path, logger)


if __name__ == "__main__":
    run_pipeline()