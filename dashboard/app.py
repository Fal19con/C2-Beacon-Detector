"""
app.py — C2 Beacon Detector Streamlit Dashboard
"""

from __future__ import annotations

import sys
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ── Path setup (allow running from any working directory) ──────────────────
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from ingest.synthetic_generator import generate_sample_logs
from ingest.zeek_ingestor import load_zeek_logs
from detection.engine import detect_beacons
from enrichment.virustotal import enrich_host
from db.storage import save_results, load_history

logging.basicConfig(level=logging.WARNING)

# ── Page config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="C2 Beacon Detector",
    page_icon="🕵️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────
st.markdown(
    """
<style>
/* Import Google Font */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

/* Global */
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

/* Dark background accent */
.stApp { background: linear-gradient(135deg, #0a0e1a 0%, #0d1526 50%, #0a0e1a 100%); }

/* Sidebar */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0f1a2e 0%, #0a1020 100%);
    border-right: 1px solid #1e3a5f;
}

/* Metric cards */
[data-testid="metric-container"] {
    background: linear-gradient(135deg, #0f2040 0%, #0a1830 100%);
    border: 1px solid #1e4080;
    border-radius: 12px;
    padding: 16px;
    box-shadow: 0 4px 20px rgba(0,120,255,0.15);
}

/* Threat level badge */
.threat-critical { color: #ff4444; font-weight: 700; }
.threat-high     { color: #ff8c00; font-weight: 700; }
.threat-medium   { color: #ffd700; font-weight: 700; }
.threat-low      { color: #44ff88; font-weight: 700; }

/* Section headers */
h2, h3 { color: #4db8ff !important; }

/* Styled info box */
.info-box {
    background: rgba(0, 100, 200, 0.15);
    border-left: 4px solid #0080ff;
    border-radius: 4px;
    padding: 12px 16px;
    margin: 8px 0;
}
</style>
""",
    unsafe_allow_html=True,
)

# ── Header ─────────────────────────────────────────────────────────────────
col_logo, col_title = st.columns([1, 10])
with col_logo:
    st.markdown("## 🕵️")
with col_title:
    st.markdown(
        "<h1 style='margin:0; color:#4db8ff; font-weight:700;'>C2 Beacon Detector</h1>"
        "<p style='margin:0; color:#7fa8d0; font-size:0.9rem;'>"
        "Network traffic analysis · Periodic beaconing detection · Threat intelligence</p>",
        unsafe_allow_html=True,
    )

st.markdown("---")

# ══════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("### 📡 Data Source")

    data_source = st.radio(
        "Select data source",
        ["🧪 Synthetic sample data", "📁 Zeek log directory", "📄 Upload CSV"],
        key="data_source_radio",
    )

    raw_df: pd.DataFrame | None = None
    source_label = ""

    # ── Synthetic ──────────────────────────────────────────────────────
    if data_source == "🧪 Synthetic sample data":
        synth_seed = st.number_input("Random seed", value=42, min_value=0, max_value=99999, step=1)
        num_benign = st.slider("Benign connections", 5000, 30000, 15000, step=1000)
        num_days = st.slider("Simulation days", 3, 14, 5)
        if st.button("🔄 Generate / Refresh", use_container_width=True):
            st.cache_data.clear()

        @st.cache_data(show_spinner="Generating synthetic traffic …")
        def _load_synthetic(seed, nb, nd):
            df, beacon_host = generate_sample_logs(
                num_benign=nb, num_days=nd, seed=seed
            )
            return df, beacon_host

        raw_df, _beacon_host = _load_synthetic(int(synth_seed), num_benign, num_days)
        source_label = f"Synthetic ({len(raw_df):,} rows)"
        st.success(f"✅ {source_label}")
        st.caption(f"Injected beacon: `{_beacon_host}`")

    # ── Zeek directory ─────────────────────────────────────────────────
    elif data_source == "📁 Zeek log directory":
        log_dir = st.text_input("Zeek log directory path", placeholder="/var/log/zeek/current")
        if log_dir and st.button("📂 Load Zeek Logs", use_container_width=True):
            with st.spinner("Loading Zeek logs …"):
                try:
                    raw_df = load_zeek_logs(log_dir)
                    source_label = f"Zeek ({len(raw_df):,} rows)"
                    st.success(f"✅ {source_label}")
                except Exception as exc:
                    st.error(f"❌ {exc}")

    # ── CSV upload ─────────────────────────────────────────────────────
    elif data_source == "📄 Upload CSV":
        uploaded = st.file_uploader("Upload CSV (canonical schema)", type=["csv"])
        if uploaded:
            try:
                raw_df = pd.read_csv(uploaded, parse_dates=["timestamp"])
                raw_df["timestamp"] = pd.to_datetime(raw_df["timestamp"], utc=True)
                source_label = f"CSV ({len(raw_df):,} rows)"
                st.success(f"✅ {source_label}")
            except Exception as exc:
                st.error(f"❌ Parse error: {exc}")

    st.markdown("---")

    # ── Detection thresholds ───────────────────────────────────────────
    st.markdown("### ⚙️ Detection Thresholds")

    with st.expander("Filter 1 — Persistence", expanded=False):
        min_window_coverage_pct = st.slider(
            "Min window coverage (%)", 10, 100, 30, 5,
            help="% of days in dataset where the host must be active"
        ) / 100.0

    with st.expander("Filter 2 — Volume", expanded=False):
        min_total_requests = st.slider(
            "Min total requests", 5, 500, 20, 5,
            help="Minimum total connections to the (host, uri) group"
        )

    with st.expander("Filter 3 — Response Size Consistency", expanded=False):
        tolerance_pct = st.slider(
            "Response size tolerance (±%)", 10, 200, 50, 10,
            help="Allowed deviation from median response size"
        ) / 100.0
        min_resp_in_bounds_pct = st.slider(
            "Min % in bounds", 50, 100, 80, 5,
            help="% of responses that must be within the tolerance band"
        ) / 100.0

    with st.expander("Filter 4 — Timing Uniformity", expanded=False):
        jitter_tolerance_pct = st.slider(
            "Timing jitter tolerance (±%)", 5, 100, 30, 5,
            help="Allowed deviation from estimated sleep interval"
        ) / 100.0
        min_delta_in_bounds_pct = st.slider(
            "Min % deltas in bounds", 50, 100, 70, 5,
            help="% of cleaned time deltas that must be within the jitter band"
        ) / 100.0
        min_clean_deltas = st.slider(
            "Min clean deltas required", 3, 20, 5, 1,
            help="Minimum deltas remaining after outlier stripping"
        )
        outlier_percentile = st.slider(
            "Outlier strip percentile", 70, 99, 90, 1,
            help="Strip time deltas above this percentile (handles overnight gaps)"
        )

    st.markdown("---")

    # ── VirusTotal ─────────────────────────────────────────────────────
    st.markdown("### 🔍 VirusTotal (optional)")
    vt_api_key = st.text_input(
        "VT API Key", type="password",
        placeholder="Paste your v3 API key …",
        help="If provided, flagged hosts will be enriched with VT reputation data"
    )

# ══════════════════════════════════════════════════════════════════════════
# MAIN AREA — Run detection if we have data
# ══════════════════════════════════════════════════════════════════════════
if raw_df is None or raw_df.empty:
    st.markdown(
        """
        <div class="info-box">
        👈 Select a data source in the sidebar to get started.<br>
        The synthetic sample option works immediately — no files needed.
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.stop()

# Run the detection engine
with st.spinner("🔎 Running beacon detection …"):
    results_df = detect_beacons(
        raw_df,
        min_window_coverage_pct=min_window_coverage_pct,
        min_total_requests=min_total_requests,
        tolerance_pct=tolerance_pct,
        min_resp_in_bounds_pct=min_resp_in_bounds_pct,
        jitter_tolerance_pct=jitter_tolerance_pct,
        min_delta_in_bounds_pct=min_delta_in_bounds_pct,
        min_clean_deltas=min_clean_deltas,
        outlier_percentile=float(outlier_percentile),
    )

# Persist results to SQLite (non-blocking; failures are logged not raised)
try:
    save_results(results_df)
except Exception:
    pass

# ── Summary metrics ────────────────────────────────────────────────────────
total_groups = raw_df.groupby(["dest_host", "uri"]).ngroups
flagged = len(results_df)
top_score = float(results_df["composite_score"].max()) if flagged else 0.0

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Total (host, uri) Groups", f"{total_groups:,}")
with col2:
    st.metric("🚨 Flagged Candidates", flagged, delta=None)
with col3:
    st.metric("🔝 Highest Score", f"{top_score:.3f}")
with col4:
    st.metric("📊 Ingested Rows", f"{len(raw_df):,}")

st.markdown("---")

# ── Results table ──────────────────────────────────────────────────────────
st.markdown("### 🚨 Beacon Candidates")

if results_df.empty:
    st.info("✅ No beacon candidates detected with current thresholds. Try relaxing the filters.")
else:
    # VirusTotal enrichment (only if key provided)
    display_df = results_df.drop(columns=["clean_deltas"], errors="ignore").copy()

    if vt_api_key:
        with st.spinner("🔍 Enriching with VirusTotal …"):
            vt_results = []
            for host in display_df["dest_host"]:
                vt_data = enrich_host(host, vt_api_key)
                vt_results.append(vt_data)
            display_df["vt_malicious"] = [r["vt_malicious_count"] for r in vt_results]
            display_df["vt_engines"] = [r["vt_total_engines"] for r in vt_results]
            display_df["vt_error"] = [r["vt_error"] for r in vt_results]

    # Threat level helper
    def _threat_label(score: float) -> str:
        if score >= 0.85:
            return "🔴 CRITICAL"
        elif score >= 0.70:
            return "🟠 HIGH"
        elif score >= 0.50:
            return "🟡 MEDIUM"
        else:
            return "🟢 LOW"

    display_df.insert(0, "Threat", display_df["composite_score"].apply(_threat_label))

    # Rename for readability
    col_rename = {
        "dest_host": "Destination Host",
        "uri": "URI",
        "total_requests": "Total Reqs",
        "unique_clients": "Unique Clients",
        "est_sleep_sec": "Est. Sleep (s)",
        "timing_cv": "Timing CV",
        "payload_cv": "Payload CV",
        "domain_entropy": "Domain Entropy",
        "composite_score": "Composite Score",
        "window_coverage_pct": "Window Coverage %",
        "resp_consistency_pct": "Resp. Consistency %",
    }
    display_renamed = display_df.rename(columns=col_rename)

    # width='stretch' replaces use_container_width=True (Streamlit ≥1.66)
    st.dataframe(
        display_renamed,
        width="stretch",
        hide_index=True,
        column_config={
            "Composite Score": st.column_config.ProgressColumn(
                "Composite Score", min_value=0, max_value=1, format="%.3f"
            ),
            "Est. Sleep (s)": st.column_config.NumberColumn(format="%.1f s"),
        },
    )

    # ── CSV download ───────────────────────────────────────────────────
    csv_bytes = display_renamed.to_csv(index=False).encode()
    st.download_button(
        "⬇️ Download Results CSV",
        csv_bytes,
        "beacon_results.csv",
        "text/csv",
    )

    st.markdown("---")

    # ── Scatter plot ───────────────────────────────────────────────────
    st.markdown("### 📈 Sleep Interval vs. Composite Score")

    plot_df = results_df.drop(columns=["clean_deltas"], errors="ignore").copy()
    plot_df["est_sleep_sec"] = plot_df["est_sleep_sec"].clip(lower=0.1)

    fig = px.scatter(
        plot_df,
        x="est_sleep_sec",
        y="composite_score",
        size="total_requests",
        color="composite_score",
        hover_name="dest_host",
        hover_data={"uri": True, "unique_clients": True, "timing_cv": ":.4f", "payload_cv": ":.4f"},
        color_continuous_scale="Plasma",
        log_x=True,
        title="Beacon candidates: sleep interval vs. detection score",
        labels={
            "est_sleep_sec": "Estimated Sleep Interval (seconds, log scale)",
            "composite_score": "Composite Score",
            "total_requests": "Total Requests",
        },
        template="plotly_dark",
        size_max=40,
    )
    fig.update_layout(
        paper_bgcolor="rgba(10,15,30,0.8)",
        plot_bgcolor="rgba(10,15,30,0.8)",
        font=dict(family="Inter", color="#c0d8f0"),
        coloraxis_colorbar=dict(title="Score"),
        height=450,
    )
    fig.update_xaxes(gridcolor="#1a3050", zeroline=False)
    fig.update_yaxes(gridcolor="#1a3050", zeroline=False, range=[0, 1.05])
    # width='stretch' replaces use_container_width=True (Streamlit ≥1.66)
    st.plotly_chart(fig, width="stretch")

    st.markdown("---")

    # ── Per-candidate drill-downs ──────────────────────────────────────
    st.markdown("### 🔬 Candidate Drill-Down")

    for i, row in results_df.iterrows():
        host = row["dest_host"]
        uri_val = row["uri"]
        score = row["composite_score"]
        deltas = row.get("clean_deltas", [])

        label = f"{_threat_label(score)}  **{host}**{uri_val}  (score: {score:.3f})"
        with st.expander(label):
            dcol1, dcol2, dcol3 = st.columns(3)
            with dcol1:
                st.metric("Est. Sleep", f"{row['est_sleep_sec']:.1f}s")
                st.metric("Total Requests", int(row["total_requests"]))
            with dcol2:
                st.metric("Timing CV", f"{row['timing_cv']:.4f}")
                st.metric("Payload CV", f"{row['payload_cv']:.4f}")
            with dcol3:
                st.metric("Domain Entropy", f"{row['domain_entropy']:.4f}")
                st.metric("Unique Clients", int(row["unique_clients"]))

            if deltas:
                fig_hist = go.Figure()
                fig_hist.add_trace(
                    go.Histogram(
                        x=deltas,
                        nbinsx=40,
                        marker_color="#0080ff",
                        opacity=0.85,
                        name="Inter-request delta (s)",
                    )
                )
                fig_hist.update_layout(
                    title=f"Timing delta distribution — {host}",
                    xaxis_title="Delta (seconds)",
                    yaxis_title="Count",
                    template="plotly_dark",
                    paper_bgcolor="rgba(10,15,30,0.8)",
                    plot_bgcolor="rgba(10,15,30,0.8)",
                    font=dict(family="Inter", color="#c0d8f0"),
                    height=300,
                    showlegend=False,
                )
                fig_hist.update_xaxes(gridcolor="#1a3050")
                fig_hist.update_yaxes(gridcolor="#1a3050")
                # width='stretch' replaces use_container_width=True (Streamlit ≥1.66)
                st.plotly_chart(fig_hist, width="stretch")

                arr = np.array(deltas)
                st.caption(
                    f"Cleaned delta stats — "
                    f"median: **{np.median(arr):.1f}s** · "
                    f"std: **{np.std(arr):.1f}s** · "
                    f"CV: **{np.std(arr)/np.mean(arr):.4f}** · "
                    f"n={len(arr)}"
                )

# ── Raw log preview ────────────────────────────────────────────────────────
with st.expander("🗂️ Raw Log Preview (first 10 rows)", expanded=False):
    st.caption("Canonical-schema ingested data — sanity-check your ingestor output here.")
    st.dataframe(raw_df.head(10), width="stretch", hide_index=True)
    st.caption(f"dtypes: {dict(raw_df.dtypes)}")

# ── Scan history ───────────────────────────────────────────────────────────
with st.expander("🗄️ Past Scan History (SQLite)", expanded=False):
    history_df = load_history()
    if history_df.empty:
        st.info("No past scans stored yet.")
    else:
        st.dataframe(history_df, width="stretch", hide_index=True)
        st.caption(f"{len(history_df)} rows across {history_df['run_id'].nunique()} runs")
