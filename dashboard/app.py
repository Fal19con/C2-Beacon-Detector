"""
app.py — C2 Beacon Detector Streamlit Dashboard (Hack The Box Aesthetic)
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

from ingest.synthetic_generator import generate_sample_logs, CANONICAL_COLUMNS
from ingest.zeek_ingestor import load_zeek_logs
from detection.engine import detect_beacons
from enrichment.virustotal import enrich_host
from db.storage import save_results, load_history

logging.basicConfig(level=logging.WARNING)

# ── Page config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="C2 Beacon Detector",
    page_icon=":material/radar:",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS: Hack The Box Theme ──────────────────────────────────────────
st.markdown(
    """
<style>
/* Import Google Font */
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600&family=Inter:wght@300;400;500;600;700&display=swap');

/* Global Font & Theme */
html, body, [class*="css"] {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    color: #E6EDF3;
}

code, pre, .stCodeBlock {
    font-family: 'JetBrains+Mono', monospace !important;
}

/* Background: Near-black navy (#0D1117 to #141D2B) */
.stApp {
    background: linear-gradient(145deg, #0D1117 0%, #101725 45%, #141D2B 100%);
    color: #E6EDF3;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background-color: #101826 !important;
    border-right: 1px solid #1E293B;
}

/* Metric cards: Surface background one shade lighter (#162234) */
[data-testid="stMetric"], [data-testid="metric-container"] {
    background: #162234 !important;
    border: 1px solid #223249 !important;
    border-radius: 8px !important;
    padding: 16px !important;
    box-shadow: 0 4px 14px rgba(0, 0, 0, 0.35) !important;
}

[data-testid="stMetricValue"] {
    color: #9FEF00 !important;
    font-weight: 700 !important;
}

/* Primary buttons: Neon green accent (#9FEF00) */
button[kind="primary"], .stButton > button {
    background-color: #162234 !important;
    color: #9FEF00 !important;
    border: 1px solid #9FEF00 !important;
    border-radius: 6px !important;
    font-weight: 600 !important;
    letter-spacing: 0.3px;
    transition: all 0.2s ease-in-out;
}

button[kind="primary"]:hover, .stButton > button:hover {
    background-color: #9FEF00 !important;
    color: #0D1117 !important;
    box-shadow: 0 0 12px rgba(159, 239, 0, 0.45) !important;
}

/* Download button */
.stDownloadButton > button {
    background-color: #162234 !important;
    color: #9FEF00 !important;
    border: 1px solid #9FEF00 !important;
    border-radius: 6px !important;
    font-weight: 600 !important;
    transition: all 0.2s ease-in-out;
}

.stDownloadButton > button:hover {
    background-color: #9FEF00 !important;
    color: #0D1117 !important;
    box-shadow: 0 0 12px rgba(159, 239, 0, 0.45) !important;
}

/* Expanders */
div[data-testid="stExpander"] {
    background-color: #141D2B !important;
    border: 1px solid #223249 !important;
    border-radius: 8px !important;
    margin-bottom: 12px;
}

/* Headings */
h1, h2, h3, h4 {
    color: #E6EDF3 !important;
    font-weight: 700 !important;
}

h1 .accent, h2 .accent, h3 .accent {
    color: #9FEF00 !important;
}

/* Status severity pill/chip badges */
.badge {
    display: inline-flex;
    align-items: center;
    gap: 4px;
    padding: 3px 10px;
    border-radius: 9999px;
    font-size: 0.75rem;
    font-weight: 700;
    letter-spacing: 0.5px;
    text-transform: uppercase;
}

.badge-critical {
    background: rgba(255, 68, 68, 0.15);
    color: #FF4444;
    border: 1px solid rgba(255, 68, 68, 0.4);
}

.badge-high {
    background: rgba(255, 140, 0, 0.15);
    color: #FF8C00;
    border: 1px solid rgba(255, 140, 0, 0.4);
}

.badge-medium {
    background: rgba(255, 215, 0, 0.15);
    color: #FFD700;
    border: 1px solid rgba(255, 215, 0, 0.4);
}

.badge-low {
    background: rgba(159, 239, 0, 0.15);
    color: #9FEF00;
    border: 1px solid rgba(159, 239, 0, 0.4);
}

/* Info Box */
.info-box {
    background: #162234;
    border-left: 4px solid #9FEF00;
    border-radius: 6px;
    padding: 14px 18px;
    margin: 12px 0;
    color: #C9D1D9;
}

/* Active highlights */
.highlight-score {
    color: #9FEF00;
    font-weight: bold;
}
</style>
""",
    unsafe_allow_html=True,
)

# ── Header ─────────────────────────────────────────────────────────────────
col_logo, col_title = st.columns([1, 11])
with col_logo:
    st.markdown("<h1 style='margin:0; color:#9FEF00;'>:material/radar:</h1>", unsafe_allow_html=True)
with col_title:
    st.markdown(
        "<h1 style='margin:0; font-weight:700;'><span style='color:#9FEF00;'>C2</span> Beacon Detector</h1>"
        "<p style='margin:0; color:#8B949E; font-size:0.92rem;'>"
        "Statistical timing & payload consistency · Periodic beaconing detection · HTB SecOps edition</p>",
        unsafe_allow_html=True,
    )

st.markdown("---")

# ══════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("### :material/sensors: Data Source")

    data_source = st.radio(
        "Select data source",
        [
            ":material/science: Synthetic sample data",
            ":material/folder: Zeek log directory",
            ":material/upload_file: Upload CSV",
        ],
        key="data_source_radio",
    )

    # Initialize zeek_df in session state if missing, or clear when switching data source
    if "zeek_df" not in st.session_state or data_source != ":material/folder: Zeek log directory":
        st.session_state.zeek_df = None

    raw_df: pd.DataFrame | None = None
    source_label = ""

    # ── Synthetic ──────────────────────────────────────────────────────
    if data_source == ":material/science: Synthetic sample data":
        synth_seed = st.number_input("Random seed", value=42, min_value=0, max_value=99999, step=1)
        num_benign = st.slider("Benign connections", 5000, 30000, 15000, step=1000)
        num_days = st.slider("Simulation days", 3, 14, 5)
        if st.button("Generate / Refresh", icon=":material/refresh:", width="stretch"):
            st.cache_data.clear()

        @st.cache_data(show_spinner="Generating synthetic traffic …")
        def _load_synthetic(seed: int, nb: int, nd: int) -> tuple[pd.DataFrame, str]:
            df, beacon_host = generate_sample_logs(
                num_benign=nb, num_days=nd, seed=seed
            )
            return df, beacon_host

        raw_df, _beacon_host = _load_synthetic(int(synth_seed), num_benign, num_days)
        source_label = f"Synthetic ({len(raw_df):,} rows)"
        st.success(source_label, icon=":material/check_circle:")
        st.caption(f"Injected beacon: `{_beacon_host}`")

    # ── Zeek directory ─────────────────────────────────────────────────
    elif data_source == ":material/folder: Zeek log directory":
        log_dir = st.text_input("Zeek log directory path", placeholder="/var/log/zeek/current")
        button_clicked = log_dir and st.button("Load Zeek Logs", icon=":material/folder_open:", width="stretch")
        if button_clicked:
            with st.spinner("Loading Zeek logs …"):
                try:
                    st.session_state.zeek_df = load_zeek_logs(log_dir)
                    st.success(f"Zeek ({len(st.session_state.zeek_df):,} rows)", icon=":material/check_circle:")
                except Exception as exc:
                    st.session_state.zeek_df = None
                    st.error(str(exc), icon=":material/error:")

        raw_df = st.session_state.zeek_df
        if raw_df is not None:
            source_label = f"Zeek ({len(raw_df):,} rows)"
            if not button_clicked:
                st.caption(f"Loaded: **{source_label}**")

    # ── CSV upload ─────────────────────────────────────────────────────
    elif data_source == ":material/upload_file: Upload CSV":
        uploaded = st.file_uploader("Upload CSV (canonical schema)", type=["csv"])
        if uploaded:
            uploaded_df: pd.DataFrame | None = None
            try:
                uploaded_df = pd.read_csv(uploaded)
            except Exception as exc:
                st.error(f"CSV read error: {exc}", icon=":material/error:")

            if uploaded_df is not None:
                # 1. Validate required columns
                missing_cols = [col for col in CANONICAL_COLUMNS if col not in uploaded_df.columns]
                if missing_cols:
                    st.error(
                        f"CSV schema error: missing required column(s): {', '.join(missing_cols)}. "
                        f"Expected columns: {', '.join(CANONICAL_COLUMNS)}",
                        icon=":material/error:",
                    )
                else:
                    # 2. Validate and coerce data types
                    coerced_df = uploaded_df[CANONICAL_COLUMNS].copy()
                    valid = True

                    # Coerce timestamp
                    try:
                        coerced_df["timestamp"] = pd.to_datetime(
                            coerced_df["timestamp"], utc=True, format="mixed", errors="raise"
                        )
                    except Exception as exc:
                        st.error(f"CSV validation error: failed to parse 'timestamp' as datetime ({exc})", icon=":material/error:")
                        valid = False

                    # Coerce bytes_sent and bytes_received to integer
                    if valid:
                        for num_col in ("bytes_sent", "bytes_received"):
                            try:
                                coerced_df[num_col] = pd.to_numeric(coerced_df[num_col], errors="raise").astype(int)
                            except Exception as exc:
                                st.error(
                                    f"CSV validation error: column '{num_col}' must be numeric integer ({exc})",
                                    icon=":material/error:",
                                )
                                valid = False
                                break

                    # Coerce string columns
                    if valid:
                        for str_col in ("client_ip", "dest_host", "uri", "protocol", "method"):
                            coerced_df[str_col] = coerced_df[str_col].fillna("").astype(str)

                        coerced_df.dropna(subset=["timestamp", "client_ip"], inplace=True)
                        coerced_df.sort_values("timestamp", inplace=True)
                        coerced_df.reset_index(drop=True, inplace=True)

                        raw_df = coerced_df
                        source_label = f"CSV ({len(raw_df):,} rows)"
                        st.success(source_label, icon=":material/check_circle:")

    st.markdown("---")

    # ── Detection thresholds ───────────────────────────────────────────
    st.markdown("### :material/tune: Detection Thresholds")

    with st.expander("Filter 1 — Persistence", expanded=False, icon=":material/timelapse:"):
        min_window_coverage_pct = st.slider(
            "Min window coverage (%)", 10, 100, 30, 5,
            help="% of days in dataset where the host must be active"
        ) / 100.0

    with st.expander("Filter 2 — Volume", expanded=False, icon=":material/bar_chart:"):
        min_total_requests = st.slider(
            "Min total requests", 5, 500, 20, 5,
            help="Minimum total connections to the (host, uri) group"
        )

    with st.expander("Filter 3 — Response Size Consistency", expanded=False, icon=":material/data_usage:"):
        tolerance_pct = st.slider(
            "Response size tolerance (±%)", 10, 200, 50, 10,
            help="Allowed deviation from median response size"
        ) / 100.0
        min_resp_in_bounds_pct = st.slider(
            "Min % in bounds", 50, 100, 80, 5,
            help="% of responses that must be within the tolerance band"
        ) / 100.0

    with st.expander("Filter 4 — Timing Uniformity", expanded=False, icon=":material/schedule:"):
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
    st.markdown("### :material/security: VirusTotal (optional)")
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
        :material/arrow_back: Select a data source in the sidebar to get started.<br>
        The synthetic sample option works immediately — no files needed.
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.stop()

# Run the detection engine
with st.spinner("Running beacon detection …"):
    try:
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
    except Exception as exc:
        st.error(f"Detection engine failed: {exc}", icon=":material/error:")
        st.stop()

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
    st.metric("Total Groups", f"{total_groups:,}", icon=":material/hub:")
with col2:
    st.metric("Flagged Candidates", flagged, delta=None, icon=":material/warning:")
with col3:
    st.metric("Highest Score", f"{top_score:.3f}", icon=":material/speed:")
with col4:
    st.metric("Ingested Rows", f"{len(raw_df):,}", icon=":material/dataset:")

st.markdown("---")

# ── Results table ──────────────────────────────────────────────────────────
st.markdown("### :material/warning: Beacon Candidates")

if results_df.empty:
    st.info("No beacon candidates detected with current thresholds. Try relaxing the filters.", icon=":material/check_circle:")
else:
    # VirusTotal enrichment (only if key provided)
    display_df = results_df.drop(columns=["clean_deltas"], errors="ignore").copy()

    if vt_api_key:
        with st.spinner("Enriching with VirusTotal …"):
            vt_results = []
            for host in display_df["dest_host"]:
                vt_data = enrich_host(host, vt_api_key)
                vt_results.append(vt_data)
            display_df["vt_malicious"] = [r["vt_malicious_count"] for r in vt_results]
            display_df["vt_engines"] = [r["vt_total_engines"] for r in vt_results]
            display_df["vt_error"] = [r["vt_error"] for r in vt_results]

    # Threat level label helper with Material Symbols
    def _threat_badge_text(score: float) -> str:
        if score >= 0.85:
            return ":material/dangerous: CRITICAL"
        elif score >= 0.70:
            return ":material/warning: HIGH"
        elif score >= 0.50:
            return ":material/info: MEDIUM"
        else:
            return ":material/check_circle: LOW"

    display_df.insert(0, "Threat", display_df["composite_score"].apply(_threat_badge_text))

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
        "Download Results CSV",
        csv_bytes,
        "beacon_results.csv",
        "text/csv",
        icon=":material/download:",
    )

    st.markdown("---")

    # ── Scatter plot ───────────────────────────────────────────────────
    st.markdown("### :material/show_chart: Sleep Interval vs. Composite Score")

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
        color_continuous_scale=[
            (0.0, "#162234"),
            (0.5, "#00E5FF"),
            (1.0, "#9FEF00"),
        ],
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
        paper_bgcolor="#101725",
        plot_bgcolor="#101725",
        font=dict(family="Inter", color="#E6EDF3"),
        coloraxis_colorbar=dict(title="Score"),
        height=450,
    )
    fig.update_xaxes(gridcolor="#1E293B", zeroline=False)
    fig.update_yaxes(gridcolor="#1E293B", zeroline=False, range=[0, 1.05])
    st.plotly_chart(fig, width="stretch")

    st.markdown("---")

    # ── Per-candidate drill-downs ──────────────────────────────────────
    st.markdown("### :material/biotech: Candidate Drill-Down")

    for i, row in results_df.iterrows():
        host = row["dest_host"]
        uri_val = row["uri"]
        score = row["composite_score"]
        deltas = row.get("clean_deltas", [])

        if score >= 0.85:
            badge_html = "<span class='badge badge-critical'>CRITICAL</span>"
            threat_icon = ":material/dangerous:"
        elif score >= 0.70:
            badge_html = "<span class='badge badge-high'>HIGH</span>"
            threat_icon = ":material/warning:"
        elif score >= 0.50:
            badge_html = "<span class='badge badge-medium'>MEDIUM</span>"
            threat_icon = ":material/info:"
        else:
            badge_html = "<span class='badge badge-low'>LOW</span>"
            threat_icon = ":material/check_circle:"

        label = f"{threat_icon} {host}{uri_val} · score: {score:.3f}"
        with st.expander(label, icon=threat_icon):
            st.markdown(f"Status: {badge_html} &nbsp;&nbsp; Destination: `<strong>{host}{uri_val}</strong>`", unsafe_allow_html=True)
            dcol1, dcol2, dcol3 = st.columns(3)
            with dcol1:
                st.metric("Est. Sleep", f"{row['est_sleep_sec']:.1f}s", icon=":material/schedule:")
                st.metric("Total Requests", int(row["total_requests"]), icon=":material/swap_vert:")
            with dcol2:
                st.metric("Timing CV", f"{row['timing_cv']:.4f}", icon=":material/timeline:")
                st.metric("Payload CV", f"{row['payload_cv']:.4f}", icon=":material/analytics:")
            with dcol3:
                st.metric("Domain Entropy", f"{row['domain_entropy']:.4f}", icon=":material/lock:")
                st.metric("Unique Clients", int(row["unique_clients"]), icon=":material/devices:")

            if deltas:
                fig_hist = go.Figure()
                fig_hist.add_trace(
                    go.Histogram(
                        x=deltas,
                        nbinsx=40,
                        marker_color="#9FEF00",
                        opacity=0.85,
                        name="Inter-request delta (s)",
                    )
                )
                fig_hist.update_layout(
                    title=f"Timing delta distribution — {host}",
                    xaxis_title="Delta (seconds)",
                    yaxis_title="Count",
                    template="plotly_dark",
                    paper_bgcolor="#101725",
                    plot_bgcolor="#101725",
                    font=dict(family="Inter", color="#E6EDF3"),
                    height=300,
                    showlegend=False,
                )
                fig_hist.update_xaxes(gridcolor="#1E293B")
                fig_hist.update_yaxes(gridcolor="#1E293B")
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
with st.expander("Raw Log Preview (first 10 rows)", expanded=False, icon=":material/table_rows:"):
    st.caption("Canonical-schema ingested data — sanity-check your ingestor output here.")
    st.dataframe(raw_df.head(10), width="stretch", hide_index=True)
    st.caption(f"dtypes: {dict(raw_df.dtypes)}")

# ── Scan history ───────────────────────────────────────────────────────────
with st.expander("Past Scan History (SQLite)", expanded=False, icon=":material/history:"):
    history_df = load_history()
    if history_df.empty:
        st.info("No past scans stored yet.", icon=":material/info:")
    else:
        st.dataframe(history_df, width="stretch", hide_index=True)
        st.caption(f"{len(history_df)} rows across {history_df['run_id'].nunique()} runs")
