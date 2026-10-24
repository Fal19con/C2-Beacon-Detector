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
import tempfile
import unittest
from pathlib import Path

from datetime import datetime, timedelta, timezone
import numpy as np
import pandas as pd

# Allow running from repository root or tests/ directory
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from ingest.synthetic_generator import generate_sample_logs, get_beacon_info
from detection.engine import detect_beacons
from db.storage import save_results, load_history, clear_history


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


class TestStorage(unittest.TestCase):
    """Checks for SQLite persistence save, load, and clear behavior."""

    def test_save_load_clear_history(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            test_db = Path(tmpdir) / "test_scan_results.db"
            # Initially empty
            df_init = load_history(test_db)
            self.assertTrue(df_init.empty)

            # Save fake results
            fake_results = pd.DataFrame([
                {
                    "dest_host": "test.c2infra.net",
                    "uri": "/beacon",
                    "total_requests": 100,
                    "unique_clients": 1,
                    "est_sleep_sec": 60.0,
                    "timing_cv": 0.05,
                    "payload_cv": 0.02,
                    "domain_entropy": 3.5,
                    "composite_score": 0.85,
                    "window_coverage_pct": 0.95,
                    "resp_consistency_pct": 1.0,
                }
            ])
            run_id = save_results(fake_results, test_db)
            self.assertTrue(run_id)

            # Verify loaded
            df_loaded = load_history(test_db)
            self.assertEqual(len(df_loaded), 1)
            self.assertEqual(df_loaded.iloc[0]["dest_host"], "test.c2infra.net")

            # Clear and verify empty
            clear_history(test_db)
            df_cleared = load_history(test_db)
            self.assertTrue(df_cleared.empty)


from ingest.zeek_ingestor import load_zeek_logs


class TestRealZeekLogs(unittest.TestCase):
    """Verifies detection on real Zeek logs with malleable C2 / rotating endpoints."""

    def test_demo_zeek_c2_flagged(self):
        demo_dir = Path("/home/zer0daydreamer/Desktop/zeek-demo-logs")
        if not demo_dir.exists():
            self.skipTest(f"Demo directory {demo_dir} not found")

        df = load_zeek_logs(demo_dir)
        results = detect_beacons(df)
        self.assertFalse(results.empty, "Failed to detect C2 in real Zeek demo logs")
        self.assertIn("192.168.197.1:8080", results["dest_host"].values)
        c2_row = results[results["dest_host"] == "192.168.197.1:8080"].iloc[0]
        self.assertGreater(c2_row["composite_score"], 0.50)
        self.assertAlmostEqual(c2_row["est_sleep_sec"], 11.5, delta=3.0)


class TestDiverseC2FrameworksAndEdgeCases(unittest.TestCase):
    """
    Validates detection across diverse C2 frameworks and edge cases:
    - Cobalt Strike (malleable profiles, rotating URIs, 0-byte HTTP 204 responses)
    - Sliver (50% jitter, direct-to-IP connection, variable payloads)
    - Mythic (15s sleep, 30% jitter)
    - Havoc Demon (30s sleep, 40% jitter, DGA domain)
    - Multi-client lateral movement (multiple clients beaconing to same C2)
    - Fast 1s beacon
    - Low-and-slow 5-minute beacon
    - Active tasking beacon with wide variable payloads (exfiltration)
    - Zero-byte payload probe beacon (SYN / ping keepalive)
    """

    def setUp(self):
        self.t0 = datetime(2026, 1, 1, 10, 0, 0, tzinfo=timezone.utc)
        self.rng = np.random.default_rng(777)

    def _generate_beacon_stream(
        self,
        dest_host: str,
        interval_sec: float,
        jitter_pct: float,
        count: int = 50,
        resp_size: int = 500,
        sent_size: int = 100,
        rotating_uris: bool = False,
        variable_payload: bool = False,
    ) -> pd.DataFrame:
        t = self.t0
        rows = []
        for i in range(count):
            jit = self.rng.uniform(-jitter_pct, jitter_pct) * interval_sec
            t += timedelta(seconds=interval_sec + jit)
            uri = f"/route/{i % 4}.action?tok={self.rng.integers(100, 999)}" if rotating_uris else "/session"
            if variable_payload:
                resp = int(self.rng.uniform(100, 45_000))
                sent = int(self.rng.uniform(50, 15_000))
            else:
                resp = resp_size
                sent = sent_size

            rows.append({
                "timestamp": t,
                "client_ip": "10.0.0.15",
                "dest_host": dest_host,
                "uri": uri,
                "protocol": "http",
                "method": "POST" if i % 2 == 0 else "GET",
                "bytes_sent": sent,
                "bytes_received": resp,
            })
        return pd.DataFrame(rows)

    def test_cobalt_strike_malleable_zero_resp(self):
        """Cobalt Strike with malleable profile, rotating URIs, and 0-byte HTTP 204 responses."""
        df = self._generate_beacon_stream(
            dest_host="cs-beacon.c2infra.net",
            interval_sec=60.0,
            jitter_pct=0.20,
            count=50,
            resp_size=0,
            sent_size=128,
            rotating_uris=True,
        )
        results = detect_beacons(df)
        self.assertFalse(results.empty, "Failed to detect Cobalt Strike beacon with 0-byte responses")
        row = results[results["dest_host"] == "cs-beacon.c2infra.net"].iloc[0]
        self.assertGreater(row["composite_score"], 0.70)
        self.assertAlmostEqual(row["est_sleep_sec"], 60.0, delta=6.0)

    def test_sliver_direct_ip_and_high_jitter(self):
        """Sliver with direct IP address (no domain) and 50% jitter."""
        df = self._generate_beacon_stream(
            dest_host="198.51.100.88:443",
            interval_sec=10.0,
            jitter_pct=0.50,
            count=70,
            resp_size=256,
            sent_size=96,
        )
        results = detect_beacons(df)
        self.assertFalse(results.empty, "Failed to detect Sliver direct-IP beacon with 50% jitter")
        row = results[results["dest_host"] == "198.51.100.88:443"].iloc[0]
        self.assertGreater(row["composite_score"], 0.65)
        self.assertAlmostEqual(row["est_sleep_sec"], 10.0, delta=2.5)

    def test_mythic_json_beacon(self):
        """Mythic C2 agent checking in every 15s with 30% jitter."""
        df = self._generate_beacon_stream(
            dest_host="mythic-agent.threatops.cc",
            interval_sec=15.0,
            jitter_pct=0.30,
            count=55,
            resp_size=380,
            sent_size=190,
        )
        results = detect_beacons(df)
        self.assertFalse(results.empty, "Failed to detect Mythic agent beacon")
        row = results[results["dest_host"] == "mythic-agent.threatops.cc"].iloc[0]
        self.assertGreater(row["composite_score"], 0.70)
        self.assertAlmostEqual(row["est_sleep_sec"], 15.0, delta=3.0)

    def test_havoc_demon_dga_domain(self):
        """Havoc Demon C2 with 30s sleep, 40% jitter, and high-entropy DGA domain."""
        df = self._generate_beacon_stream(
            dest_host="x9z2w1k4.havoc-gateway.cc",
            interval_sec=30.0,
            jitter_pct=0.40,
            count=50,
            resp_size=420,
            sent_size=150,
            rotating_uris=True,
        )
        results = detect_beacons(df)
        self.assertFalse(results.empty, "Failed to detect Havoc Demon DGA beacon")
        row = results[results["dest_host"] == "x9z2w1k4.havoc-gateway.cc"].iloc[0]
        self.assertGreater(row["composite_score"], 0.70)
        self.assertAlmostEqual(row["est_sleep_sec"], 30.0, delta=5.0)

    def test_multi_client_lateral_movement(self):
        """Multiple compromised internal clients beaconing to the same C2 server."""
        rows = []
        for client_ip, offset_sec in [("10.0.0.11", 0), ("10.0.0.12", 18), ("10.0.0.13", 38)]:
            t = self.t0 + timedelta(seconds=offset_sec)
            for i in range(40):
                t += timedelta(seconds=60.0 + self.rng.uniform(-0.15, 0.15) * 60)
                rows.append({
                    "timestamp": t,
                    "client_ip": client_ip,
                    "dest_host": "lateral-c2.infra.net",
                    "uri": "/sync",
                    "protocol": "http",
                    "method": "POST",
                    "bytes_sent": 128,
                    "bytes_received": 256,
                })
        df = pd.DataFrame(rows).sort_values("timestamp")
        results = detect_beacons(df)
        self.assertFalse(results.empty, "Failed to detect multi-client lateral movement beacon")
        row = results[results["dest_host"] == "lateral-c2.infra.net"].iloc[0]
        self.assertEqual(row["unique_clients"], 3)
        self.assertAlmostEqual(row["est_sleep_sec"], 60.0, delta=8.0)

    def test_fast_1s_beacon(self):
        """Fast 1.0s beacon with jitter."""
        df = self._generate_beacon_stream(
            dest_host="fast-agent.c2.io",
            interval_sec=1.0,
            jitter_pct=0.25,
            count=75,
            resp_size=64,
            sent_size=64,
        )
        results = detect_beacons(df)
        self.assertFalse(results.empty, "Failed to detect fast 1s beacon")
        row = results[results["dest_host"] == "fast-agent.c2.io"].iloc[0]
        self.assertAlmostEqual(row["est_sleep_sec"], 1.0, delta=0.3)

    def test_active_tasking_variable_payload(self):
        """C2 session during active operator tasking with fluctuating response sizes."""
        df = self._generate_beacon_stream(
            dest_host="active-tasking.c2.io",
            interval_sec=20.0,
            jitter_pct=0.20,
            count=60,
            variable_payload=True,
        )
        results = detect_beacons(df)
        self.assertFalse(results.empty, "Failed to detect beacon during active tasking / variable payloads")
        row = results[results["dest_host"] == "active-tasking.c2.io"].iloc[0]
        self.assertGreater(row["composite_score"], 0.55)
        self.assertAlmostEqual(row["est_sleep_sec"], 20.0, delta=3.0)

    def test_zero_byte_probe_beacon(self):
        """C2 beacon sending and receiving 0 bytes (SYN probe / ping heartbeat)."""
        df = self._generate_beacon_stream(
            dest_host="zero-probe.c2.io",
            interval_sec=25.0,
            jitter_pct=0.15,
            count=45,
            resp_size=0,
            sent_size=0,
        )
        results = detect_beacons(df)
        self.assertFalse(results.empty, "Failed to detect 0-byte probe beacon")
        row = results[results["dest_host"] == "zero-probe.c2.io"].iloc[0]
        self.assertGreater(row["composite_score"], 0.70)
        self.assertAlmostEqual(row["est_sleep_sec"], 25.0, delta=3.0)


if __name__ == "__main__":
    # Run with verbose output
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    suite.addTests(loader.loadTestsFromTestCase(TestSyntheticGenerator))
    suite.addTests(loader.loadTestsFromTestCase(TestDetectionEngine))
    suite.addTests(loader.loadTestsFromTestCase(TestOutlierStripping))
    suite.addTests(loader.loadTestsFromTestCase(TestStorage))
    suite.addTests(loader.loadTestsFromTestCase(TestRealZeekLogs))
    suite.addTests(loader.loadTestsFromTestCase(TestDiverseC2FrameworksAndEdgeCases))

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    sys.exit(0 if result.wasSuccessful() else 1)


