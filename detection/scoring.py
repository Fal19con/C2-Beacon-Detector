"""
scoring.py
Computes the composite beacon score for groups that survived the hard filters.

Inputs come from the filter_survivors DataFrame produced by filters.apply_filters().
"""

from __future__ import annotations

import math
import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Scoring weights (must sum to 1.0)
WEIGHT_TIMING = 0.70
WEIGHT_PAYLOAD = 0.20
WEIGHT_ENTROPY = 0.10

# Shannon entropy normalisation ceiling: a fully random 4-char domain has
# ~2 bits; we normalise by 4.5 so a "maximally suspicious" DGA-ish domain
# saturates the entropy component at 1.0
ENTROPY_NORM = 4.5


def _shannon_entropy(value: str) -> float:
    """Compute Shannon entropy (bits) of a string."""
    if not value:
        return 0.0
    freq: dict[str, int] = {}
    for ch in value:
        freq[ch] = freq.get(ch, 0) + 1
    n = len(value)
    return -sum((c / n) * math.log2(c / n) for c in freq.values())


def score_survivors(filter_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute composite beacon scores for the filter-survivor rows.

    Parameters
    ----------
    filter_df:
        Output of filters.apply_filters() — one row per (dest_host, uri) group
        that passed all four hard filters.

    Returns
    -------
    pd.DataFrame
        Ranked results with columns:
            dest_host, uri, total_requests, unique_clients, est_sleep_sec,
            timing_cv, payload_cv, domain_entropy, composite_score,
            window_coverage_pct, resp_consistency_pct
        Sorted by composite_score descending.
    """
    if filter_df.empty:
        return pd.DataFrame(
            columns=[
                "dest_host", "uri", "total_requests", "unique_clients",
                "est_sleep_sec", "timing_cv", "payload_cv", "domain_entropy",
                "composite_score", "window_coverage_pct", "resp_consistency_pct",
            ]
        )

    records: list[dict] = []

    for _, row in filter_df.iterrows():
        dest_host: str = str(row["dest_host"])
        uri: str = str(row["uri"])

        clean_deltas = np.array(row["clean_deltas"], dtype=float)
        bytes_received = np.array(row["bytes_received_list"], dtype=float)
        client_ips: list[str] = row["client_ips"]

        # ---------------------------------------------------------------- #
        # Timing CV                                                         #
        # ---------------------------------------------------------------- #
        if len(clean_deltas) < 2 or clean_deltas.mean() == 0:
            timing_cv = 1.0  # worst case
        else:
            timing_cv = float(clean_deltas.std() / clean_deltas.mean())

        timing_score = max(0.0, 1.0 - timing_cv)

        # ---------------------------------------------------------------- #
        # Payload CV                                                        #
        # ---------------------------------------------------------------- #
        if len(bytes_received) < 2 or bytes_received.mean() == 0:
            payload_cv = 1.0
        else:
            payload_cv = float(bytes_received.std() / bytes_received.mean())

        payload_score = max(0.0, 1.0 - payload_cv)

        # ---------------------------------------------------------------- #
        # Domain Shannon entropy (DGA signal)                               #
        # Only score the hostname label (left of first dot), not the TLD,  #
        # because TLDs are legitimately short.                              #
        # ---------------------------------------------------------------- #
        label = dest_host.split(".")[0] if "." in dest_host else dest_host
        domain_entropy = _shannon_entropy(label)
        entropy_score = min(domain_entropy / ENTROPY_NORM, 1.0)

        # ---------------------------------------------------------------- #
        # Composite score                                                   #
        # ---------------------------------------------------------------- #
        composite_score = (
            timing_score * WEIGHT_TIMING
            + payload_score * WEIGHT_PAYLOAD
            + entropy_score * WEIGHT_ENTROPY
        )

        # ---------------------------------------------------------------- #
        # Contextual fields (not weighted into the score)                   #
        # ---------------------------------------------------------------- #
        unique_clients = len(set(client_ips))

        records.append(
            {
                "dest_host": dest_host,
                "uri": uri,
                "total_requests": int(row["total_requests"]),
                "unique_clients": unique_clients,
                "est_sleep_sec": float(row["est_sleep_sec"]),
                "timing_cv": round(timing_cv, 4),
                "payload_cv": round(payload_cv, 4),
                "domain_entropy": round(domain_entropy, 4),
                "composite_score": round(composite_score, 4),
                "window_coverage_pct": round(float(row["window_coverage_pct"]) * 100, 1),
                "resp_consistency_pct": round(float(row["resp_in_bounds_pct"]) * 100, 1),
                # Pass through raw delta list for drill-down histogram
                "clean_deltas": row["clean_deltas"],
            }
        )

    result = pd.DataFrame(records)
    result.sort_values("composite_score", ascending=False, inplace=True)
    result.reset_index(drop=True, inplace=True)
    return result
