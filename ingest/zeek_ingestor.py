"""
zeek_ingestor.py
Loads Zeek conn.log and http.log via zat, joins them on uid,
and returns a canonical-schema DataFrame.
"""

from __future__ import annotations

import os
import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

CANONICAL_COLUMNS = [
    "timestamp",
    "client_ip",
    "dest_host",
    "uri",
    "protocol",
    "method",
    "bytes_sent",
    "bytes_received",
]

# Mapping from Zeek conn.log service field values to our protocol labels
_SERVICE_MAP = {
    "http": "http",
    "ssl": "https",
    "dns": "dns",
}


def _load_log(log_path: Path) -> pd.DataFrame | None:
    """Load a single Zeek TSV log file using zat, return None if unavailable."""
    if not log_path.exists():
        logger.warning("Zeek log not found: %s", log_path)
        return None
    try:
        # zat 0.4+ uses zat.log_to_dataframe (not zat.log_to_df)
        from zat.log_to_dataframe import LogToDataFrame  # type: ignore

        reader = LogToDataFrame()
        # ts_index=True (default) sets 'ts' as the index; reset so it becomes
        # a regular column that our rename logic can find.
        # aggressive_category=False keeps string columns as object dtype so
        # joins/merges work cleanly without Categorical category mismatch errors.
        df = reader.create_dataframe(str(log_path), ts_index=True, aggressive_category=False)
        df = df.reset_index()  # moves 'ts' from index back to a named column
        return df
    except Exception as exc:  # noqa: BLE001
        logger.error("Failed to load %s: %s", log_path, exc)
        return None


def _load_tsv_log(log_path: Path) -> pd.DataFrame | None:
    """
    Fallback: parse a Zeek TSV log manually without zat.
    Handles the standard Zeek header format (#fields, #types, etc.)
    """
    if not log_path.exists():
        return None
    try:
        headers = []
        data_lines = []
        separator = "\t"
        with open(log_path, "r") as fh:
            for line in fh:
                line = line.rstrip("\n")
                if line.startswith("#separator"):
                    sep_hex = line.split()[1]
                    separator = bytes.fromhex(sep_hex.replace("\\x", "")).decode()
                elif line.startswith("#fields"):
                    headers = line.split(separator)[1:]
                elif line.startswith("#"):
                    continue
                else:
                    data_lines.append(line.split(separator))
        if not headers or not data_lines:
            return None
        df = pd.DataFrame(data_lines, columns=headers)
        # Replace Zeek's '-' missing sentinel with NaN
        df.replace("-", pd.NA, inplace=True)
        return df
    except Exception as exc:  # noqa: BLE001
        logger.error("Manual TSV parse failed for %s: %s", log_path, exc)
        return None


def load_zeek_logs(log_dir: str | Path) -> pd.DataFrame:
    """
    Load conn.log and http.log from *log_dir*, join on uid, and return
    a DataFrame in the canonical schema.

    Parameters
    ----------
    log_dir:
        Directory containing Zeek log files (conn.log, optionally http.log).

    Returns
    -------
    pd.DataFrame
        Canonical-schema DataFrame, one row per connection.  If http.log is
        absent, dest_host falls back to id.resp_h and uri is empty string.
    """
    log_dir = Path(log_dir)

    # ------------------------------------------------------------------ #
    # 1.  Load conn.log — required                                         #
    # ------------------------------------------------------------------ #
    conn_path = log_dir / "conn.log"
    conn_df = _load_log(conn_path)
    if conn_df is None:
        conn_df = _load_tsv_log(conn_path)
    if conn_df is None or conn_df.empty:
        raise FileNotFoundError(f"conn.log not found or empty in {log_dir}")

    # zat preserves dotted column names (e.g. 'id.orig_h') after reset_index();
    # the manual TSV fallback also preserves them. No normalization needed —
    # both code paths use the dotted form, which our rename_conn dict handles.

    # Rename zat's dotted column representations (zat converts dots to _)
    rename_conn = {
        "id.orig_h": "client_ip",
        "id_orig_h": "client_ip",
        "id.resp_h": "resp_h",
        "id_resp_h": "resp_h",
        "ts": "timestamp",
        "orig_bytes": "bytes_sent",
        "resp_bytes": "bytes_received",
        "service": "service",
        "uid": "uid",
    }
    conn_df.rename(columns={k: v for k, v in rename_conn.items() if k in conn_df.columns}, inplace=True)

    # Ensure required columns exist
    for col in ("client_ip", "resp_h", "timestamp", "uid"):
        if col not in conn_df.columns:
            raise ValueError(f"conn.log missing expected column after rename: {col}")

    # Coerce numeric columns
    for col in ("bytes_sent", "bytes_received"):
        if col in conn_df.columns:
            conn_df[col] = pd.to_numeric(conn_df[col], errors="coerce").fillna(0).astype(int)
        else:
            conn_df[col] = 0

    # Parse timestamp — handle both paths:
    #   zat path  : 'ts' is already a pandas Timestamp/datetime64 (tz-naive UTC)
    #   TSV path  : 'ts' is a string representation of a Unix epoch float
    import pandas.api.types as pat
    if pat.is_datetime64_any_dtype(conn_df["timestamp"]):
        # Already datetime — ensure UTC timezone is set
        if conn_df["timestamp"].dt.tz is None:
            conn_df["timestamp"] = conn_df["timestamp"].dt.tz_localize("UTC")
        else:
            conn_df["timestamp"] = conn_df["timestamp"].dt.tz_convert("UTC")
    else:
        # String epoch from manual TSV parse — coerce to float then parse
        conn_df["timestamp"] = pd.to_datetime(
            pd.to_numeric(conn_df["timestamp"], errors="coerce"),
            unit="s",
            utc=True,
            errors="coerce",
        )

    # Protocol from service field
    if "service" in conn_df.columns:
        conn_df["protocol"] = conn_df["service"].map(_SERVICE_MAP).fillna("other")
    else:
        conn_df["protocol"] = "other"

    # ------------------------------------------------------------------ #
    # 2.  Load http.log — optional                                         #
    # ------------------------------------------------------------------ #
    http_path = log_dir / "http.log"
    http_df = _load_log(http_path)
    if http_df is None:
        http_df = _load_tsv_log(http_path)

    if http_df is not None and not http_df.empty:
        rename_http = {
            "id.orig_h": "client_ip_h",
            "id_orig_h": "client_ip_h",
            "host": "dest_host",
            "uri": "uri",
            "method": "method",
            "uid": "uid",
        }
        http_df.rename(columns={k: v for k, v in rename_http.items() if k in http_df.columns}, inplace=True)

        keep_http = [c for c in ("uid", "dest_host", "uri", "method") if c in http_df.columns]
        http_subset = http_df[keep_http].drop_duplicates(subset=["uid"])

        merged = conn_df.merge(http_subset, on="uid", how="left")
    else:
        logger.info("http.log not available; dest_host will fall back to resp_h.")
        merged = conn_df.copy()
        for col in ("dest_host", "uri", "method"):
            merged[col] = pd.NA

    # ------------------------------------------------------------------ #
    # 3.  Build canonical schema                                           #
    # ------------------------------------------------------------------ #
    # dest_host: prefer http.log host, fall back to resp IP
    if "dest_host" not in merged.columns:
        merged["dest_host"] = pd.NA
    merged["dest_host"] = merged["dest_host"].where(
        merged["dest_host"].notna() & (merged["dest_host"].astype(str) != "nan"),
        other=merged["resp_h"],
    )

    merged["uri"] = merged.get("uri", pd.Series("", index=merged.index)).fillna("").astype(str)
    merged["method"] = merged.get("method", pd.Series("", index=merged.index)).fillna("").astype(str)

    # For HTTP rows upgrade protocol label
    if "dest_host" in merged.columns:
        http_mask = merged["uri"].str.len() > 0
        merged.loc[http_mask & (merged["protocol"] == "other"), "protocol"] = "http"

    canonical = merged[CANONICAL_COLUMNS].copy()
    canonical["client_ip"] = canonical["client_ip"].astype(str)
    canonical["dest_host"] = canonical["dest_host"].astype(str)
    canonical["timestamp"] = pd.to_datetime(canonical["timestamp"], utc=True)
    canonical["bytes_sent"] = canonical["bytes_sent"].astype(int)
    canonical["bytes_received"] = canonical["bytes_received"].astype(int)

    canonical.dropna(subset=["timestamp", "client_ip"], inplace=True)
    canonical.sort_values("timestamp", inplace=True)
    canonical.reset_index(drop=True, inplace=True)

    return canonical


if __name__ == "__main__":
    import sys

    log_dir = sys.argv[1] if len(sys.argv) > 1 else "."
    df = load_zeek_logs(log_dir)
    print("Shape:", df.shape)
    print("Dtypes:\n", df.dtypes)
    print("\n", df.head(20).to_string())
