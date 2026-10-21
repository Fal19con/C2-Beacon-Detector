# C2 Beacon Detector

A network traffic analysis tool that ingests connection logs and flags hosts exhibiting periodic "beaconing" behavior characteristic of malware C2 check-ins.

---

## Quick Start

### 1. Install dependencies

```bash
pip install pandas numpy streamlit plotly requests zat
```

### 2. Launch the dashboard

```bash
cd c2-beacon-detector
streamlit run dashboard/app.py
```

Open `http://localhost:8501` in your browser. Select **Synthetic sample data** to run immediately — no Zeek logs needed.

### 3. Run against real Zeek logs

Point the dashboard to a directory containing `conn.log` (and optionally `http.log`). The ingestor handles both formats automatically.

---

## Repository Structure

```
c2-beacon-detector/
  ingest/
    zeek_ingestor.py       # Loads conn.log + http.log via zat, joins on uid
    synthetic_generator.py  # Generates fake benign + beacon traffic
  detection/
    filters.py             # 4 hard filters (persistence, volume, resp-size, timing)
    scoring.py             # CV + Shannon entropy composite scoring
    engine.py              # Orchestrates filters → scoring
  enrichment/
    virustotal.py           # Optional domain reputation lookup
  dashboard/
    app.py                  # Streamlit UI
  db/
    storage.py              # SQLite persistence of past runs
  tests/
    test_detection.py       # Unit tests using synthetic data
```

---

## Detection Pipeline

### Canonical Schema

Every ingestor outputs exactly these columns:

| Column           | Type      | Notes                                              |
|------------------|-----------|----------------------------------------------------|
| `timestamp`      | datetime  | Request time (UTC, timezone-aware)                 |
| `client_ip`      | string    | Internal source host                               |
| `dest_host`      | string    | Domain if known, else destination IP               |
| `uri`            | string    | HTTP path, empty string if not HTTP                |
| `protocol`       | string    | `http`, `https`, `dns`, or `other`                 |
| `method`         | string    | GET/POST, empty if not HTTP                        |
| `bytes_sent`     | int       | Request size (orig_bytes)                          |
| `bytes_received` | int       | Response size (resp_bytes) — key field             |

### Stage 1: Hard Filters (detection/filters.py)

Groups of `(dest_host, uri)` are evaluated in order:

1. **Persistence / window coverage** — host must be active on ≥30% of dataset days
2. **Volume threshold** — group must have ≥20 total requests
3. **Response size consistency** — ≥80% of responses must be within ±50% of the group's median response size
4. **Timing delta uniformity with outlier stripping** — consecutive inter-request deltas are stripped of sub-1s noise and p90 outliers (overnight gaps, lid-closed laptops), then ≥70% of cleaned deltas must fall within ±30% of the estimated sleep interval

Only groups surviving all four filters proceed to scoring.

### Stage 2: Composite Scoring (detection/scoring.py)

For each surviving group:

```
timing_score    = max(0, 1 - timing_cv)      # CV = std/mean of cleaned deltas
payload_score   = max(0, 1 - payload_cv)     # CV of bytes_received
entropy_score   = min(domain_entropy / 4.5, 1.0)  # Shannon entropy of hostname label

composite_score = (timing_score × 0.70)
               + (payload_score × 0.20)
               + (entropy_score × 0.10)
```

Higher scores indicate more beacon-like behavior. Timing regularity is weighted most heavily; domain randomness (DGA signal) provides a tertiary boost.

---

## Configuration

All thresholds are configurable via Streamlit sidebar sliders or directly as keyword arguments to `detect_beacons()`:

| Parameter                  | Default | Description                                    |
|----------------------------|---------|------------------------------------------------|
| `min_window_coverage_pct`  | 0.30    | Min fraction of dataset days with activity     |
| `min_total_requests`       | 20      | Min total connections in the group             |
| `tolerance_pct`            | 0.50    | Response size tolerance band (±50%)            |
| `min_resp_in_bounds_pct`   | 0.80    | Min fraction of responses in size band         |
| `jitter_tolerance_pct`     | 0.30    | Timing jitter tolerance (±30%)                 |
| `min_delta_in_bounds_pct`  | 0.70    | Min fraction of clean deltas in timing band    |
| `min_clean_deltas`         | 5       | Min deltas after outlier stripping             |
| `outlier_percentile`       | 90.0    | Percentile above which deltas are stripped     |

---

## Enrichment

If a VirusTotal API v3 key is provided in the sidebar, flagged candidates are looked up against the VT domain reputation API. Results show malicious engine count out of total engines. Missing keys or failed requests degrade gracefully to "N/A" without crashing the UI.

---

## Testing

```bash
cd c2-beacon-detector
python -m pytest tests/test_detection.py -v
```

The test suite asserts:
- The injected beacon appears in the detection output
- `est_sleep_sec` is within ±10% of the configured 60-second interval (proving outlier stripping works)
- No known-benign hosts appear in the output
- Stricter thresholds return fewer results (proving filters are actually applied)

---

## Future Work (Out of Scope)

The following are intentional non-goals in this version:

- **ML models** — Isolation Forest, autoencoders, clustering
- **Alternative frontends** — Flask/React
- **Heavy databases** — ClickHouse, PostgreSQL
- **Additional enrichment** — WHOIS, GeoIP, SSL certificate analysis
- **CDN-abuse / trusted-SaaS C2 detection** — Teams/Slack/OneDrive abuse
- **Domain-fronting detection**
- **Docker packaging**

---

## Architecture Decisions

**Median over mean for response size filtering** — resistant to the occasional large "tasking" response that real C2 beacons produce when delivering payloads.

**p90 outlier stripping for timing deltas** — correctly handles overnight gaps, lid-closed laptops, and "business hours only" beacons that would otherwise inflate the estimated sleep interval from 60s to hours.

**Entropy on hostname label only** — scoring entropy on the full FQDN would penalise legitimate multi-label CDN hostnames; scoring only the leftmost label isolates the DGA signal.

**`unique_clients` as context, not score** — a beacon hitting only 1-2 internal hosts is more suspicious than one hit by hundreds (which is more likely a CDN), but adding it to the weighted score would require tuning that varies too much across environments.
