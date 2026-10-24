import json

from src.audit import file_sha256, get_audit_logger, log_event


def test_log_event_writes_one_json_line(tmp_path):
    data_file = tmp_path / "data.csv"
    data_file.write_text("a;b\n1;2\n")
    log_path = tmp_path / "audit.log"
    logger = get_audit_logger(log_path)

    log_event(logger, "load", "read_csv", rows_in=None, rows_out=1, file_path=data_file)

    lines = log_path.read_text().strip().splitlines()
    assert len(lines) == 1
    entry = json.loads(lines[0])
    assert entry["step"] == "load"
    assert entry["rows_out"] == 1
    assert entry["file_sha256"] == file_sha256(data_file)


def test_log_event_appends_and_never_stores_data_values(tmp_path):
    data_file = tmp_path / "data.csv"
    data_file.write_text("secret_value\n")
    log_path = tmp_path / "audit.log"
    logger = get_audit_logger(log_path)

    log_event(logger, "load", "read_csv", file_path=data_file)
    log_event(logger, "clean", "drop_duplicates", rows_in=10, rows_out=9)

    text = log_path.read_text()
    assert len(text.strip().splitlines()) == 2
    assert "secret_value" not in text