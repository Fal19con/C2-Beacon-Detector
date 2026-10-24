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
        if len(bytes_received) < 2:
            payload_cv = 1.0
        elif float(bytes_received.std()) == 0.0:
            # Perfectly uniform payload size (e.g. all 0 bytes or all 128 bytes)
            payload_cv = 0.0
        elif bytes_received.mean() == 0:
            payload_cv = 1.0
        else:
            payload_cv = float(bytes_received.std() / bytes_received.mean())

        payload_score = max(0.0, 1.0 - payload_cv)

        # ---------------------------------------------------------------- #
        # Domain Shannon entropy (DGA signal & IP detection)                #
        # If dest_host is an IP address (or IP:port), assign a baseline     #
        # suspicious indicator (2.25 bits / 4.5 = 0.50). For domain names,  #
        # evaluate max entropy across labels excluding TLD to catch DGAs.   #
        # ---------------------------------------------------------------- #
        import re
        host_clean = dest_host.split(":")[0]
        if re.match(r"^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}$", host_clean):
            domain_entropy = 2.25
        else:
            parts = host_clean.split(".")
            if len(parts) <= 1:
                domain_entropy = _shannon_entropy(host_clean)
            else:
                domain_entropy = max(_shannon_entropy(p) for p in parts[:-1])

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
