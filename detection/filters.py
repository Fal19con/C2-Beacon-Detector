"""
filters.py
Applies the four hard filters to a canonical-schema DataFrame.

Each filter is applied to groups of (dest_host, uri).  Groups that fail any
filter are dropped.  Only groups that survive ALL four filters proceed to the
composite scorer.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Default thresholds (all exposed as keyword arguments so the dashboard
# sliders can override them without touching code)
# ---------------------------------------------------------------------------
DEFAULT_MIN_WINDOW_COVERAGE_PCT: float = 0.30   # 30%
DEFAULT_MIN_TOTAL_REQUESTS: int = 20
DEFAULT_TOLERANCE_PCT: float = 0.50             # ±50% around median resp size
DEFAULT_MIN_RESP_IN_BOUNDS_PCT: float = 0.70    # 70% must be within bounds
DEFAULT_JITTER_TOLERANCE_PCT: float = 0.40      # ±40% around est sleep (covers 30-50% C2 jitter)
DEFAULT_MIN_DELTA_IN_BOUNDS_PCT: float = 0.55   # 55% of clean deltas in bounds
DEFAULT_MIN_CLEAN_DELTAS: int = 5               # require ≥5 after stripping
DEFAULT_OUTLIER_PERCENTILE: float = 90.0        # strip deltas above this pct


def _compute_active_days(timestamps: pd.Series) -> int:
    """Return the number of distinct calendar dates (UTC) in *timestamps*."""
    return timestamps.dt.date.nunique()


def _shannon_entropy(value: str) -> float:
    """Compute the Shannon entropy of a string (bits)."""
    if not value:
        return 0.0
    import math
    freq = {}
    for ch in value:
        freq[ch] = freq.get(ch, 0) + 1
    n = len(value)
    return -sum((c / n) * math.log2(c / n) for c in freq.values())


def apply_filters(
    df: pd.DataFrame,
    *,
    min_window_coverage_pct: float = DEFAULT_MIN_WINDOW_COVERAGE_PCT,
    min_total_requests: int = DEFAULT_MIN_TOTAL_REQUESTS,
    tolerance_pct: float = DEFAULT_TOLERANCE_PCT,
    min_resp_in_bounds_pct: float = DEFAULT_MIN_RESP_IN_BOUNDS_PCT,
    jitter_tolerance_pct: float = DEFAULT_JITTER_TOLERANCE_PCT,
    min_delta_in_bounds_pct: float = DEFAULT_MIN_DELTA_IN_BOUNDS_PCT,
    min_clean_deltas: int = DEFAULT_MIN_CLEAN_DELTAS,
    outlier_percentile: float = DEFAULT_OUTLIER_PERCENTILE,
) -> pd.DataFrame:
    """
    Apply the four hard filters to *df* grouped by dest_host.

    Returns a DataFrame with one row per surviving group, containing the
    pre-computed fields needed by scoring.py:

        dest_host, uri, total_requests, active_days, total_days,
        window_coverage_pct, resp_median, resp_in_bounds_pct,
        est_sleep_sec, clean_deltas (as list), delta_in_bounds_pct,
        bytes_received_list (as list), client_ips (as list)
    """
    if df.empty:
        return pd.DataFrame()

    # Total calendar days spanned by the dataset (for coverage denominator)
    total_days = max(1, (df["timestamp"].max() - df["timestamp"].min()).days + 1)

    survivors: list[dict[str, Any]] = []

    # Group by destination host: beaconing is a host-level C2 channel.
    # Grouping by (dest_host, uri) falsely fragments traffic when malleable C2 profiles
    # rotate URI paths or query strings across requests.
    grouped = df.groupby("dest_host", sort=False)

    for dest_host, group in grouped:
        group = group.sort_values("timestamp")
        total_requests = len(group)

        # Dominant URI for reporting (prefer non-empty URI)
        non_empty_uris = group["uri"][group["uri"].astype(str).str.strip() != ""]
        if not non_empty_uris.empty:
            uri = str(non_empty_uris.value_counts().index[0])
        else:
            uri = ""

        # ---------------------------------------------------------------- #
        # Filter 1: Persistence / window coverage                           #
        # ---------------------------------------------------------------- #
        active_days = _compute_active_days(group["timestamp"])
        coverage = active_days / total_days
        if coverage < min_window_coverage_pct:
            logger.debug(
                "FILTER1 drop: %s  coverage=%.2f < %.2f",
                dest_host, coverage, min_window_coverage_pct,
            )
            continue

        # ---------------------------------------------------------------- #
        # Filter 2: Volume threshold                                        #
        # ---------------------------------------------------------------- #
        if total_requests < min_total_requests:
            logger.debug(
                "FILTER2 drop: %s  requests=%d < %d",
                dest_host, total_requests, min_total_requests,
            )
            continue

        # ---------------------------------------------------------------- #
        # Filter 4: Timing delta uniformity with outlier stripping         #
        # (Evaluated first to establish timing regularity for Filter 3)     #
        # ---------------------------------------------------------------- #
        # Support multi-client beacons: if multiple infected clients beacon #
        # to the same dest_host (lateral movement), interleaved timestamps  #
        # can mask periodicity. We evaluate the timing uniformity of both  #
        # the aggregate stream and each client stream, taking the best.    #
        streams: list[tuple[str, np.ndarray]] = [("ALL", group["timestamp"].sort_values().values)]
        if group["client_ip"].nunique() > 1:
            for c_ip in group["client_ip"].unique():
                c_df = group[group["client_ip"] == c_ip]
                if len(c_df) >= min_clean_deltas:
                    streams.append((str(c_ip), c_df["timestamp"].sort_values().values))

        best_cv = 999.0
        best_clean_deltas = np.array([])
        best_sleep = 0.0
        best_delta_in_bounds_pct = 0.0

        for _stream_label, ts_values in streams:
            ts_series = pd.Series(ts_values)
            deltas_all = ts_series.diff().dropna().dt.total_seconds().values
            # Filter sub-250ms bursts (TCP connection reuse / pipelined sub-resources)
            # while preserving fast beacons with 0.5s - 1.0s sleep
            deltas = deltas_all[deltas_all >= 0.25]
            if len(deltas) < min_clean_deltas:
                continue

            # Strip overnight-gap / lid-closed outliers above outlier_percentile
            p_outlier = float(np.percentile(deltas, outlier_percentile))
            clean_d = deltas[deltas <= p_outlier]
            if len(clean_d) < min_clean_deltas:
                continue

            est_sleep_cand = float(np.median(clean_d))
            if est_sleep_cand <= 0:
                continue

            cv_cand = float(np.std(clean_d) / np.mean(clean_d))
            lower_d = est_sleep_cand * (1.0 - jitter_tolerance_pct)
            upper_d = est_sleep_cand * (1.0 + jitter_tolerance_pct)
            in_b_d = np.sum((clean_d >= lower_d) & (clean_d <= upper_d)) / len(clean_d)

            if cv_cand < best_cv:
                best_cv = cv_cand
                best_clean_deltas = clean_d
                best_sleep = est_sleep_cand
                best_delta_in_bounds_pct = in_b_d

        if len(best_clean_deltas) < min_clean_deltas:
            logger.debug("FILTER4 drop: %s  not enough clean deltas", dest_host)
            continue

        # Allow candidates if in-bounds percentage meets threshold, or if overall timing CV <= 0.42
        # (mathematically covering real-world C2 frameworks with up to 60-70% jitter)
        if best_delta_in_bounds_pct < min_delta_in_bounds_pct and best_cv > 0.42:
            logger.debug(
                "FILTER4 drop: %s  delta_in_bounds=%.2f < %.2f and timing_cv=%.4f > 0.42",
                dest_host, best_delta_in_bounds_pct, min_delta_in_bounds_pct, best_cv,
            )
            continue

        # ---------------------------------------------------------------- #
        # Filter 3: Payload size consistency                               #
        # Evaluates bytes_received and bytes_sent, including 0-byte        #
        # check-ins (HTTP 204 No Content), per-method slices, and active   #
        # tasking sessions where timing periodicity confirms beaconing.    #
        # ---------------------------------------------------------------- #
        best_payload_pct = 0.0
        best_payload_median = 0.0
        best_payload_series = group["bytes_received"].values.astype(float)

        for col in ("bytes_received", "bytes_sent"):
            if col not in group.columns:
                continue
            col_vals = group[col].values.astype(float)
            col_med = float(np.median(col_vals))

            if col_med == 0:
                # 0-byte payloads (e.g. HTTP 204 No Content check-ins, TCP SYN pings)
                in_b = np.sum(col_vals == 0)
                pct = in_b / total_requests
                if pct > best_payload_pct:
                    best_payload_pct = pct
                    best_payload_median = 0.0
                    best_payload_series = col_vals
            else:
                lower = col_med * (1.0 - tolerance_pct)
                upper = col_med * (1.0 + tolerance_pct)
                in_b = np.sum((col_vals >= lower) & (col_vals <= upper))
                pct = in_b / total_requests
                if pct > best_payload_pct:
                    best_payload_pct = pct
                    best_payload_median = col_med
                    best_payload_series = col_vals

            # Evaluate per-method consistency if multiple methods exist
            if "method" in group.columns and group["method"].nunique() > 1:
                for method_val in group["method"].unique():
                    sub_method = group[group["method"] == method_val]
                    if len(sub_method) >= 5:
                        m_vals = sub_method[col].values.astype(float)
                        m_med = float(np.median(m_vals))
                        if m_med == 0:
                            in_b = np.sum(m_vals == 0)
                            pct = in_b / len(m_vals)
                        else:
                            lower = m_med * (1.0 - tolerance_pct)
                            upper = m_med * (1.0 + tolerance_pct)
                            in_b = np.sum((m_vals >= lower) & (m_vals <= upper))
                            pct = in_b / len(m_vals)
                        if pct > best_payload_pct:
                            best_payload_pct = pct
                            best_payload_median = m_med
                            best_payload_series = m_vals

        # Pass if payload consistency is high OR if timing periodicity is strong (CV <= 0.35)
        # to ensure active C2 tasking sessions (variable command payloads) are not dropped.
        if best_payload_pct < min_resp_in_bounds_pct and best_cv > 0.35:
            logger.debug(
                "FILTER3 drop: %s  best_payload_in_bounds=%.2f < %.2f and timing_cv=%.4f > 0.35",
                dest_host, best_payload_pct, min_resp_in_bounds_pct, best_cv,
            )
            continue

        # ---- Survived all four filters --------------------------------- #
        survivors.append(
            {
                "dest_host": dest_host,
                "uri": uri,
                "total_requests": total_requests,
                "active_days": active_days,
                "total_days": total_days,
                "window_coverage_pct": coverage,
                "resp_median": best_payload_median,
                "resp_in_bounds_pct": best_payload_pct,
                "est_sleep_sec": best_sleep,
                "clean_deltas": best_clean_deltas.tolist(),
                "delta_in_bounds_pct": best_delta_in_bounds_pct,
                "bytes_received_list": best_payload_series.tolist(),
                "client_ips": group["client_ip"].tolist(),
            }
        )

    if not survivors:
        return pd.DataFrame()

    return pd.DataFrame(survivors)
