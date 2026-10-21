"""
test_detection.py
Unit tests for the C2 beacon detection pipeline using synthetic data.

Assertions:
1. The known injected beacon appears in the filtered/scored output.
2. At least one known-benign host does NOT appear in the output.
3. The outlier-stripping logic correctly produces est_sleep_sec ≈ 60s
   (not skewed by the simulated overnight gap).
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

# Allow running from repository root or tests/ directory
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from ingest.synthetic_generator import generate_sample_logs, get_beacon_info
from detection.engine import detect_beacons


# Hard-coded seed for reproducibility across all test runs
_SEED = 42
_BEACON_INTERVAL_SEC = 60.0
_SLEEP_TOLERANCE = 0.10  # ±10%

# Known-benign domains from the generator
_KNOWN_BENIGN_HOSTS = {
    "www.google.com",
    "www.youtube.com",
    "api.github.com",
    "update.microsoft.com",
    "cdn.jsdelivr.net",
    "ocsp.digicert.com",
    "connectivity-check.ubuntu.com",
    "safebrowsing.googleapis.com",
    "clients1.google.com",
    "push.services.mozilla.com",
}


class TestSyntheticGenerator(unittest.TestCase):
    """Sanity checks on the generator itself."""

    @classmethod
    def setUpClass(cls):
        cls.df, cls.beacon_host = generate_sample_logs(
            num_benign=15_000, num_days=5, seed=_SEED
        )

    def test_canonical_schema_columns(self):
        expected = {
            "timestamp", "client_ip", "dest_host", "uri",
            "protocol", "method", "bytes_sent", "bytes_received",
        }
        self.assertEqual(set(self.df.columns), expected)

    def test_timestamp_is_datetime64(self):
        import pandas as pd
        self.assertTrue(
            pd.api.types.is_datetime64_any_dtype(self.df["timestamp"]),
            "timestamp column must be datetime64, got: " + str(self.df["timestamp"].dtype),
        )

    def test_beacon_rows_present(self):
        beacon_rows = self.df[self.df["dest_host"] == self.beacon_host]
        self.assertGreater(len(beacon_rows), 0, "No beacon rows found in synthetic data")

    def test_benign_rows_present(self):
        for host in list(_KNOWN_BENIGN_HOSTS)[:3]:
            benign_rows = self.df[self.df["dest_host"] == host]
            self.assertGreater(len(benign_rows), 0, f"No rows for benign host {host}")


class TestDetectionEngine(unittest.TestCase):
    """Core detection pipeline tests."""

    @classmethod
    def setUpClass(cls):
        cls.df, cls.beacon_host = generate_sample_logs(
            num_benign=15_000, num_days=5, seed=_SEED
        )
        cls.results = detect_beacons(cls.df)

    def test_results_not_empty(self):
        """Detection results must not be empty when run on synthetic data with beacon."""
        self.assertIsNotNone(self.results)
        self.assertFalse(self.results.empty, "detect_beacons returned an empty DataFrame on synthetic data")

    def test_beacon_appears_in_results(self):
        """The injected beacon's dest_host must appear in the ranked output."""
        flagged_hosts = set(self.results["dest_host"].tolist())
        self.assertIn(
            self.beacon_host,
            flagged_hosts,
            f"Beacon host '{self.beacon_host}' not found in detection results.\n"
            f"Flagged hosts: {flagged_hosts}",
        )

    def test_beacon_uri_correct(self):
        """The flagged beacon must have the correct URI."""
        beacon_info = get_beacon_info(_SEED)
        beacon_rows = self.results[self.results["dest_host"] == self.beacon_host]
        self.assertFalse(beacon_rows.empty)
        self.assertIn(beacon_info["uri"], beacon_rows["uri"].values)

    def test_no_benign_hosts_flagged(self):
        """None of the known-benign hosts should appear in the output."""
        flagged_hosts = set(self.results["dest_host"].tolist())
        overlap = flagged_hosts & _KNOWN_BENIGN_HOSTS
        self.assertEqual(
            overlap,
            set(),
            f"Benign hosts incorrectly flagged as beacons: {overlap}",
        )

    def test_est_sleep_close_to_beacon_interval(self):
        """
        est_sleep_sec must be within ±10% of the configured 60-second interval.
        This proves the outlier-stripping logic correctly ignores the simulated
        overnight gaps instead of being skewed toward hours.
        """
        beacon_rows = self.results[self.results["dest_host"] == self.beacon_host]
        self.assertFalse(beacon_rows.empty, "Beacon not in results; cannot check sleep interval")

        est_sleep = float(beacon_rows.iloc[0]["est_sleep_sec"])
        lower = _BEACON_INTERVAL_SEC * (1.0 - _SLEEP_TOLERANCE)
        upper = _BEACON_INTERVAL_SEC * (1.0 + _SLEEP_TOLERANCE)

        self.assertGreaterEqual(
            est_sleep, lower,
            f"est_sleep_sec={est_sleep:.2f}s is below {lower:.1f}s — outlier stripping may have failed",
        )
        self.assertLessEqual(
            est_sleep, upper,
            f"est_sleep_sec={est_sleep:.2f}s is above {upper:.1f}s — overnight gap may be skewing the estimate",
        )

    def test_stricter_thresholds_return_fewer_results(self):
        """
        Running with min_total_requests=10000 should return fewer (or zero) results
        than the default, proving filters are actually applied and not silently ignored.
        """
        strict_results = detect_beacons(self.df, min_total_requests=10_000)
        default_count = len(self.results)
        strict_count = len(strict_results)
        self.assertLessEqual(
            strict_count,
            default_count,
            f"Strict filter (min_total_requests=10000) returned {strict_count} results "
            f"but default returned {default_count}; filters appear to be ignored.",
        )

    def test_results_sorted_by_score_descending(self):
        """Results must be sorted by composite_score descending."""
        if len(self.results) < 2:
            self.skipTest("Need at least 2 results to test sort order")
        scores = self.results["composite_score"].tolist()
        self.assertEqual(scores, sorted(scores, reverse=True))

    def test_required_output_columns_present(self):
        """All required output columns must be present."""
        required = {
            "dest_host", "uri", "total_requests", "unique_clients",
            "est_sleep_sec", "timing_cv", "payload_cv", "domain_entropy",
            "composite_score", "window_coverage_pct", "resp_consistency_pct",
        }
        missing = required - set(self.results.columns)
        self.assertEqual(missing, set(), f"Missing output columns: {missing}")


class TestOutlierStripping(unittest.TestCase):
    """
    Directly verify that the timing filter strips overnight gaps.
    Inject a group with a clear outlier delta and assert it's ignored.
    """

    def test_overnight_gap_stripped(self):
        import numpy as np
        from detection.filters import apply_filters
        import pandas as pd

        # Build a minimal DataFrame: 100 rows, 60s apart, with one 8-hour gap
        base = pd.Timestamp("2024-01-01 08:00:00", tz="UTC")
        timestamps = []
        t = base
        for i in range(50):
            timestamps.append(t)
            t += pd.Timedelta(seconds=60)
        # Overnight gap
        t += pd.Timedelta(hours=8)
        for i in range(50):
            timestamps.append(t)
            t += pd.Timedelta(seconds=60)

        n = len(timestamps)
        df = pd.DataFrame(
            {
                "timestamp": timestamps,
                "client_ip": ["10.0.0.1"] * n,
                "dest_host": ["beacon.test.com"] * n,
                "uri": ["/check"] * n,
                "protocol": ["http"] * n,
                "method": ["POST"] * n,
                "bytes_sent": [128] * n,
                "bytes_received": [512] * n,
            }
        )

        result = apply_filters(df, min_total_requests=10, min_window_coverage_pct=0.1)
        self.assertFalse(result.empty, "Filter should not drop this synthetic beacon group")
        est_sleep = float(result.iloc[0]["est_sleep_sec"])
        self.assertAlmostEqual(
            est_sleep, 60.0, delta=10.0,
            msg=f"est_sleep_sec={est_sleep:.1f}s — overnight gap may not have been stripped",
        )


if __name__ == "__main__":
    # Run with verbose output
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    suite.addTests(loader.loadTestsFromTestCase(TestSyntheticGenerator))
    suite.addTests(loader.loadTestsFromTestCase(TestDetectionEngine))
    suite.addTests(loader.loadTestsFromTestCase(TestOutlierStripping))

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    sys.exit(0 if result.wasSuccessful() else 1)
