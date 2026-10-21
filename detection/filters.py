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
DEFAULT_MIN_RESP_IN_BOUNDS_PCT: float = 0.80    # 80% must be within bounds
DEFAULT_JITTER_TOLERANCE_PCT: float = 0.30      # ±30% around est sleep
DEFAULT_MIN_DELTA_IN_BOUNDS_PCT: float = 0.70   # 70% of clean deltas in bounds
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
    Apply the four hard filters to *df* grouped by (dest_host, uri).

    Returns a DataFrame with one row per surviving group, containing the
    pre-computed fields needed by scoring.py:

        dest_host, uri, total_requests, active_days, total_days,
        window_coverage_pct, resp_median, resp_in_bounds_pct,
        est_sleep_sec, clean_deltas (as list), delta_in_bounds_pct,
        bytes_received (as list)
    """
    if df.empty:
        return pd.DataFrame()

    # Total calendar days spanned by the dataset (for coverage denominator)
    total_days = max(1, (df["timestamp"].max() - df["timestamp"].min()).days + 1)

    survivors: list[dict[str, Any]] = []

    grouped = df.groupby(["dest_host", "uri"], sort=False)

    for (dest_host, uri), group in grouped:
        group = group.sort_values("timestamp")
        total_requests = len(group)

        # ---------------------------------------------------------------- #
        # Filter 1: Persistence / window coverage                           #
        # ---------------------------------------------------------------- #
        active_days = _compute_active_days(group["timestamp"])
        coverage = active_days / total_days
        if coverage < min_window_coverage_pct:
            logger.debug(
                "FILTER1 drop: %s%s  coverage=%.2f < %.2f",
                dest_host, uri, coverage, min_window_coverage_pct,
            )
            continue

        # ---------------------------------------------------------------- #
        # Filter 2: Volume threshold                                        #
        # ---------------------------------------------------------------- #
        if total_requests < min_total_requests:
            logger.debug(
                "FILTER2 drop: %s%s  requests=%d < %d",
                dest_host, uri, total_requests, min_total_requests,
            )
            continue

        # ---------------------------------------------------------------- #
        # Filter 3: Response size consistency (median-based)               #
        # ---------------------------------------------------------------- #
        resp = group["bytes_received"].values.astype(float)
        resp_median = float(np.median(resp))
        if resp_median == 0:
            logger.debug("FILTER3 drop: %s%s  median bytes_received=0", dest_host, uri)
            continue

        lower = resp_median * (1.0 - tolerance_pct)
        upper = resp_median * (1.0 + tolerance_pct)
        in_bounds = np.sum((resp >= lower) & (resp <= upper))
        resp_in_bounds_pct = in_bounds / total_requests
        if resp_in_bounds_pct < min_resp_in_bounds_pct:
            logger.debug(
                "FILTER3 drop: %s%s  resp_in_bounds=%.2f < %.2f",
                dest_host, uri, resp_in_bounds_pct, min_resp_in_bounds_pct,
            )
            continue

        # ---------------------------------------------------------------- #
        # Filter 4: Timing delta uniformity with outlier stripping          #
        # ---------------------------------------------------------------- #
        ts_sorted = group["timestamp"].sort_values()
        deltas_all = ts_sorted.diff().dropna().dt.total_seconds().values

        # Drop sub-1-second deltas (burst / sub-resource noise)
        deltas = deltas_all[deltas_all >= 1.0]

        if len(deltas) < min_clean_deltas:
            logger.debug(
                "FILTER4 drop: %s%s  not enough deltas before strip (%d)",
                dest_host, uri, len(deltas),
            )
            continue

        # Strip overnight-gap / lid-closed outliers above 90th percentile
        p90 = float(np.percentile(deltas, outlier_percentile))
        clean_deltas = deltas[deltas <= p90]

        if len(clean_deltas) < min_clean_deltas:
            logger.debug(
                "FILTER4 drop: %s%s  not enough clean deltas (%d) after p%d strip",
                dest_host, uri, len(clean_deltas), int(outlier_percentile),
            )
            continue

        est_sleep = float(np.median(clean_deltas))
        if est_sleep == 0:
            continue

        lower_d = est_sleep * (1.0 - jitter_tolerance_pct)
        upper_d = est_sleep * (1.0 + jitter_tolerance_pct)
        in_bounds_d = np.sum((clean_deltas >= lower_d) & (clean_deltas <= upper_d))
        delta_in_bounds_pct = in_bounds_d / len(clean_deltas)

        if delta_in_bounds_pct < min_delta_in_bounds_pct:
            logger.debug(
                "FILTER4 drop: %s%s  delta_in_bounds=%.2f < %.2f",
                dest_host, uri, delta_in_bounds_pct, min_delta_in_bounds_pct,
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
                "resp_median": resp_median,
                "resp_in_bounds_pct": resp_in_bounds_pct,
                "est_sleep_sec": est_sleep,
                "clean_deltas": clean_deltas.tolist(),
                "delta_in_bounds_pct": delta_in_bounds_pct,
                "bytes_received_list": resp.tolist(),
                "client_ips": group["client_ip"].tolist(),
            }
        )

    if not survivors:
        return pd.DataFrame()

    return pd.DataFrame(survivors)
