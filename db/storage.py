"""
storage.py
Lightweight SQLite persistence for past scan results.

Table: scan_results
    run_id              TEXT  — UUID for this scan run
    run_timestamp       TEXT  — ISO-8601 timestamp of the run
    dest_host           TEXT
    uri                 TEXT
    total_requests      INTEGER
    unique_clients      INTEGER
    est_sleep_sec       REAL
    timing_cv           REAL
    payload_cv          REAL
    domain_entropy      REAL
    composite_score     REAL
    window_coverage_pct REAL
    resp_consistency_pct REAL
"""

from __future__ import annotations

import logging
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

_DEFAULT_DB_PATH = Path(__file__).parent / "scan_results.db"

_CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS scan_results (
    run_id               TEXT,
    run_timestamp        TEXT,
    dest_host            TEXT,
    uri                  TEXT,
    total_requests       INTEGER,
    unique_clients       INTEGER,
    est_sleep_sec        REAL,
    timing_cv            REAL,
    payload_cv           REAL,
    domain_entropy       REAL,
    composite_score      REAL,
    window_coverage_pct  REAL,
    resp_consistency_pct REAL
);
"""

_RESULT_COLUMNS = [
    "dest_host", "uri", "total_requests", "unique_clients",
    "est_sleep_sec", "timing_cv", "payload_cv", "domain_entropy",
    "composite_score", "window_coverage_pct", "resp_consistency_pct",
]


def _get_connection(db_path: Path) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    conn.execute(_CREATE_TABLE)
    conn.commit()
    return conn


def save_results(results_df: pd.DataFrame, db_path: Path = _DEFAULT_DB_PATH) -> str:
    """
    Persist *results_df* to SQLite with a new run_id.

    Parameters
    ----------
    results_df:
        Output of engine.detect_beacons().
    db_path:
        Path to the SQLite database file.  Created if absent.

    Returns
    -------
    str
        The run_id UUID string for this batch.
    """
    if results_df is None or results_df.empty:
        logger.info("No results to save.")
        return ""

    run_id = str(uuid.uuid4())
    run_ts = datetime.now(timezone.utc).isoformat()

    rows = []
    for _, row in results_df.iterrows():
        rows.append(
            (
                run_id,
                run_ts,
                str(row.get("dest_host", "")),
                str(row.get("uri", "")),
                int(row.get("total_requests", 0)),
                int(row.get("unique_clients", 0)),
                float(row.get("est_sleep_sec", 0.0)),
                float(row.get("timing_cv", 0.0)),
                float(row.get("payload_cv", 0.0)),
                float(row.get("domain_entropy", 0.0)),
                float(row.get("composite_score", 0.0)),
                float(row.get("window_coverage_pct", 0.0)),
                float(row.get("resp_consistency_pct", 0.0)),
            )
        )

    try:
        conn = _get_connection(db_path)
        conn.executemany(
            """
            INSERT INTO scan_results VALUES
            (?,?,?,?,?,?,?,?,?,?,?,?,?)
            """,
            rows,
        )
        conn.commit()
        conn.close()
        logger.info("Saved %d rows under run_id %s", len(rows), run_id)
    except Exception as exc:  # noqa: BLE001
        logger.error("Failed to save results: %s", exc)

    return run_id


def load_history(db_path: Path = _DEFAULT_DB_PATH, limit: int = 500) -> pd.DataFrame:
    """
    Load the most recent *limit* rows from the scan history.

    Returns an empty DataFrame if the database does not exist yet.
    """
    if not db_path.exists():
        return pd.DataFrame(columns=["run_id", "run_timestamp"] + _RESULT_COLUMNS)

    try:
        conn = _get_connection(db_path)
        df = pd.read_sql_query(
            f"SELECT * FROM scan_results ORDER BY run_timestamp DESC LIMIT {limit}",
            conn,
        )
        conn.close()
        return df
    except Exception as exc:  # noqa: BLE001
        logger.error("Failed to load history: %s", exc)
        return pd.DataFrame()
