"""
synthetic_generator.py
Generates fake benign + C2 beacon traffic in the canonical schema DataFrame
for testing the detection engine without real Zeek logs.
"""

from __future__ import annotations

import math
import random
import string

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Canonical column names and dtypes — must match the schema exactly
# ---------------------------------------------------------------------------
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


def _random_subdomain(rng: np.random.Generator, length: int = 12) -> str:
    """Return a high-entropy, DGA-looking subdomain label."""
    chars = string.ascii_lowercase + string.digits
    return "".join(rng.choice(list(chars), size=length))


def generate_sample_logs(
    num_benign: int = 15_000,
    num_days: int = 5,
    beacon_interval_sec: float = 60.0,
    beacon_jitter_pct: float = 0.15,
    seed: int | None = 42,
) -> pd.DataFrame:
    """
    Generate a canonical-schema DataFrame containing realistic benign traffic
    and one injected C2 beacon for algorithm development and unit testing.

    Parameters
    ----------
    num_benign:
        Approximate number of benign connection rows to generate.
    num_days:
        Width of the synthetic time window in days (used for all traffic).
    beacon_interval_sec:
        Target sleep interval for the injected beacon (seconds).
    beacon_jitter_pct:
        Maximum relative jitter applied to each beacon wake-up (±fraction).
    seed:
        NumPy random seed for reproducibility.  Pass None for non-deterministic.

    Returns
    -------
    pd.DataFrame
        Rows in the canonical schema, sorted by timestamp ascending.
    """
    rng = np.random.default_rng(seed)
    random.seed(int(seed) if seed is not None else None)

    window_start = pd.Timestamp("2024-01-01 00:00:00", tz="UTC")
    window_end = window_start + pd.Timedelta(days=num_days)
    total_seconds = num_days * 86_400

    # ------------------------------------------------------------------
    # Benign traffic
    # ------------------------------------------------------------------
    benign_domains = [
        ("www.google.com", "https"),
        ("www.youtube.com", "https"),
        ("api.github.com", "https"),
        ("update.microsoft.com", "https"),
        ("cdn.jsdelivr.net", "https"),
        ("ocsp.digicert.com", "http"),
        ("connectivity-check.ubuntu.com", "http"),
        ("safebrowsing.googleapis.com", "https"),
        ("clients1.google.com", "https"),
        ("push.services.mozilla.com", "https"),
    ]

    benign_uris = [
        "/",
        "/favicon.ico",
        "/api/v1/status",
        "/update/check",
        "/telemetry",
        "/heartbeat",
        "/cdn-cgi/trace",
        "/robots.txt",
        "/health",
        "/static/main.js",
    ]

    # Five simulated internal hosts
    client_ips = [f"10.0.0.{i}" for i in range(1, 6)]

    rows: list[dict] = []

    for _ in range(num_benign):
        domain, proto = benign_domains[int(rng.integers(0, len(benign_domains)))]
        ts_offset = float(rng.uniform(0, total_seconds))
        ts = window_start + pd.Timedelta(seconds=ts_offset)
        method = "GET" if rng.random() < 0.85 else "POST"
        uri = benign_uris[int(rng.integers(0, len(benign_uris)))]
        # Log-normal response sizes with variety
        resp_bytes = max(100, int(rng.lognormal(mean=9.5, sigma=1.5)))
        sent_bytes = max(40, int(rng.lognormal(mean=6.0, sigma=1.0)))
        cip = client_ips[int(rng.integers(0, len(client_ips)))]

        rows.append(
            {
                "timestamp": ts,
                "client_ip": cip,
                "dest_host": domain,
                "uri": uri if proto == "http" else "",
                "protocol": proto,
                "method": method,
                "bytes_sent": sent_bytes,
                "bytes_received": resp_bytes,
            }
        )

    # ------------------------------------------------------------------
    # Injected beacon
    # ------------------------------------------------------------------
    # Use a high-entropy DGA-looking subdomain so Shannon entropy scoring fires.
    beacon_label = _random_subdomain(rng, 14)
    beacon_host = f"{beacon_label}.c2infra.net"
    beacon_client = "10.0.0.99"
    beacon_uri = "/j/session"

    # Walk through the window second-by-second, firing at each wake-up
    t = window_start + pd.Timedelta(seconds=float(rng.uniform(0, beacon_interval_sec)))

    while t < window_end:
        # Simulate overnight gap: if time is between 22:00 and 06:00 UTC
        # skip forward to 06:00 of the next day (forcing a large outlier delta)
        local_hour = t.hour
        if 22 <= local_hour or local_hour < 6:
            # Jump to 06:00 next day
            next_wakeup = (t + pd.Timedelta(days=1)).normalize() + pd.Timedelta(hours=6)
            t = next_wakeup
            if t >= window_end:
                break

        # ±jitter
        jitter = float(rng.uniform(-beacon_jitter_pct, beacon_jitter_pct))
        actual_interval = beacon_interval_sec * (1.0 + jitter)

        # ~90% small check-in, ~10% large tasking response
        if rng.random() < 0.90:
            resp = max(100, int(rng.normal(512, 30)))
            sent = max(40, int(rng.normal(128, 10)))
        else:
            resp = max(4096, int(rng.normal(16_384, 2048)))
            sent = max(256, int(rng.normal(1024, 128)))

        rows.append(
            {
                "timestamp": t,
                "client_ip": beacon_client,
                "dest_host": beacon_host,
                "uri": beacon_uri,
                "protocol": "http",
                "method": "POST",
                "bytes_sent": sent,
                "bytes_received": resp,
            }
        )

        t = t + pd.Timedelta(seconds=actual_interval)

    # ------------------------------------------------------------------
    # Build DataFrame, enforce canonical schema dtypes
    # ------------------------------------------------------------------
    df = pd.DataFrame(rows, columns=CANONICAL_COLUMNS)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df["bytes_sent"] = df["bytes_sent"].astype(int)
    df["bytes_received"] = df["bytes_received"].astype(int)
    df["client_ip"] = df["client_ip"].astype(str)
    df["dest_host"] = df["dest_host"].astype(str)
    df["uri"] = df["uri"].astype(str)
    df["protocol"] = df["protocol"].astype(str)
    df["method"] = df["method"].astype(str)

    df.sort_values("timestamp", inplace=True)
    df.reset_index(drop=True, inplace=True)

    return df, beacon_host  # Return beacon_host for test introspection


def get_beacon_info(seed: int = 42) -> dict:
    """
    Return the beacon parameters for a given seed without full data generation.
    Useful in tests that need to know the expected beacon host/uri.
    """
    rng = np.random.default_rng(seed)
    beacon_label = _random_subdomain(rng, 14)
    return {
        "dest_host": f"{beacon_label}.c2infra.net",
        "uri": "/j/session",
        "interval_sec": 60.0,
    }


if __name__ == "__main__":
    df, beacon_host = generate_sample_logs()
    print("Shape:", df.shape)
    print("Dtypes:\n", df.dtypes)
    print("\nHead:\n", df.head(10).to_string())
    print("\nBeacon host:", beacon_host)
    print("\nBeacon rows:", (df["dest_host"] == beacon_host).sum())
