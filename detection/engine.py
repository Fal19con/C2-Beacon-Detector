"""
engine.py
Orchestrates the detection pipeline: hard filters → composite scoring.

Public API:
    detect_beacons(df, **threshold_kwargs) -> pd.DataFrame
"""

from __future__ import annotations

import logging

import pandas as pd

from detection.filters import (
    apply_filters,
    DEFAULT_MIN_WINDOW_COVERAGE_PCT,
    DEFAULT_MIN_TOTAL_REQUESTS,
    DEFAULT_TOLERANCE_PCT,
    DEFAULT_MIN_RESP_IN_BOUNDS_PCT,
    DEFAULT_JITTER_TOLERANCE_PCT,
    DEFAULT_MIN_DELTA_IN_BOUNDS_PCT,
    DEFAULT_MIN_CLEAN_DELTAS,
    DEFAULT_OUTLIER_PERCENTILE,
)
from detection.scoring import score_survivors

logger = logging.getLogger(__name__)


def detect_beacons(
    df: pd.DataFrame,
    *,
    # Filter 1: persistence
    min_window_coverage_pct: float = DEFAULT_MIN_WINDOW_COVERAGE_PCT,
    # Filter 2: volume
    min_total_requests: int = DEFAULT_MIN_TOTAL_REQUESTS,
    # Filter 3: response size
    tolerance_pct: float = DEFAULT_TOLERANCE_PCT,
    min_resp_in_bounds_pct: float = DEFAULT_MIN_RESP_IN_BOUNDS_PCT,
    # Filter 4: timing
    jitter_tolerance_pct: float = DEFAULT_JITTER_TOLERANCE_PCT,
    min_delta_in_bounds_pct: float = DEFAULT_MIN_DELTA_IN_BOUNDS_PCT,
    min_clean_deltas: int = DEFAULT_MIN_CLEAN_DELTAS,
    outlier_percentile: float = DEFAULT_OUTLIER_PERCENTILE,
) -> pd.DataFrame:
    """
    Run the full beacon-detection pipeline on *df*.

    Parameters
    ----------
    df:
        Canonical-schema DataFrame (output of any ingestor).
    **threshold_kwargs:
        All filter thresholds are keyword-only arguments with sensible defaults.
        The Streamlit dashboard overrides these via sidebar sliders.

    Returns
    -------
    pd.DataFrame
        Ranked results sorted by composite_score descending.  Columns:
            dest_host, uri, total_requests, unique_clients, est_sleep_sec,
            timing_cv, payload_cv, domain_entropy, composite_score,
            window_coverage_pct, resp_consistency_pct
        Returns an empty DataFrame if no candidates survive the filters.
    """
    if df is None or df.empty:
        logger.warning("detect_beacons called with empty DataFrame")
        return pd.DataFrame()

    logger.info("Starting detection on %d rows …", len(df))

    # Step 1 — hard filters
    filter_survivors = apply_filters(
        df,
        min_window_coverage_pct=min_window_coverage_pct,
        min_total_requests=min_total_requests,
        tolerance_pct=tolerance_pct,
        min_resp_in_bounds_pct=min_resp_in_bounds_pct,
        jitter_tolerance_pct=jitter_tolerance_pct,
        min_delta_in_bounds_pct=min_delta_in_bounds_pct,
        min_clean_deltas=min_clean_deltas,
        outlier_percentile=outlier_percentile,
    )

    logger.info("%d groups survived hard filters", len(filter_survivors))

    if filter_survivors.empty:
        return pd.DataFrame()

    # Step 2 — composite scoring
    results = score_survivors(filter_survivors)

    logger.info("%d candidates in final ranked output", len(results))
    return results
