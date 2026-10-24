import getpass
import hashlib
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

DEFAULT_LOG_PATH = "logs/audit.log"


def file_sha256(file_path):
    """Return the SHA-256 checksum of a file, read in chunks."""
    digest = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def get_audit_logger(log_path=DEFAULT_LOG_PATH):
    """Return a logger that appends one line per event to log_path."""
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(f"audit.{log_path.resolve()}")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    if not logger.handlers:
        handler = logging.FileHandler(log_path, encoding="utf-8")
        handler.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(handler)
    return logger


def log_event(logger, step, action, rows_in=None, rows_out=None,
              file_path=None, detail=None):
    """Write one audit entry as a JSON line.

    Records who did what, when, to which file and how many rows. It never
    records data values, only counts, the file path and its checksum.
    """
    entry = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "user": getpass.getuser(),
        "step": step,
        "action": action,
        "rows_in": rows_in,
        "rows_out": rows_out,
        "file_path": str(file_path) if file_path is not None else None,
        "file_sha256": file_sha256(file_path) if file_path is not None else None,
        "detail": detail,
    }
    logger.info(json.dumps(entry))
    return entry