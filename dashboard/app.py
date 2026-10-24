"""
app.py — C2 Beacon Detector Streamlit Dashboard (Hack The Box Aesthetic)
"""

from __future__ import annotations

import sys
import logging
import importlib
from datetime import datetime
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

# Hot-reload detection and ingestor modules on every rerun to prevent stale sys.modules caching
for _mod in ("detection.filters", "detection.scoring", "detection.engine", "ingest.zeek_ingestor"):
    if _mod in sys.modules:
        importlib.reload(sys.modules[_mod])

from ingest.synthetic_generator import generate_sample_logs, CANONICAL_COLUMNS
from ingest.zeek_ingestor import load_zeek_logs
from detection.engine import detect_beacons
from enrichment.virustotal import enrich_host
from db.storage import save_results, load_history, clear_history

logging.basicConfig(level=logging.WARNING)

# ── Page config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="C2 Beacon Detector",
    page_icon=":material/radar:",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS: Cyber SecOps Theme ──────────────────────────────────────────
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
st.markdown(
    """
    <div style="display: flex; align-items: center; gap: 16px; margin: 0 0 20px 0; padding-bottom: 18px; border-bottom: 1px solid #1E293B;">
        <div style="
            width: 52px;
            height: 52px;
            min-width: 52px;
            border-radius: 12px;
            background: radial-gradient(circle, rgba(159, 239, 0, 0.18) 0%, rgba(159, 239, 0, 0.05) 70%, transparent 100%);
            border: 1px solid rgba(159, 239, 0, 0.35);
            box-shadow: 0 0 18px rgba(159, 239, 0, 0.15);
            display: flex;
            align-items: center;
            justify-content: center;
        ">
            <svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="#9FEF00" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round">
                <circle cx="12" cy="12" r="10" stroke-opacity="0.4"/>
                <circle cx="12" cy="12" r="6" stroke-opacity="0.7"/>
                <circle cx="12" cy="12" r="2" fill="#9FEF00"/>
                <line x1="12" y1="12" x2="19" y2="5"/>
            </svg>
        </div>
        <div>
            <h1 style="margin: 0; font-size: 2.1rem; font-weight: 700; line-height: 1.2; letter-spacing: -0.5px; color: #E6EDF3;">
                <span style="color: #9FEF00;">C2</span> Beacon Detector
            </h1>
            <p style="margin: 4px 0 0 0; color: #8B949E; font-size: 0.92rem; font-weight: 400; letter-spacing: 0.2px;">
                Statistical timing &amp; payload consistency · Periodic beaconing detection
            </p>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

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
        default_dir = "/home/zer0daydreamer/Desktop/zeek-demo-logs" if Path("/home/zer0daydreamer/Desktop/zeek-demo-logs").exists() else ""
        log_dir = st.text_input(
            "Zeek log directory path",
            value=default_dir,
            placeholder="/var/log/zeek/current",
            help="Directory containing conn.log and http.log",
        )

        live_polling = st.toggle(
            "Real-time Live Fetch",
            value=True,
            help="Continuously poll and stream newly captured Zeek logs automatically",
        )

        col_ref1, col_ref2 = st.columns([1, 1])
        with col_ref1:
            poll_interval = st.selectbox(
                "Polling interval",
                [1, 2, 3, 5, 10],
                index=1,
                format_func=lambda s: f"{s}s",
                disabled=not live_polling,
            )
        with col_ref2:
            manual_fetch = st.button("Fetch Now", icon=":material/refresh:", width="stretch")

        is_valid_dir = bool(log_dir and Path(log_dir).is_dir() and (Path(log_dir) / "conn.log").exists())

        if not is_valid_dir:
            if log_dir:
                st.error("Directory not found or conn.log missing.", icon=":material/error:")
            raw_df = None
        else:
            try:
                raw_df = load_zeek_logs(log_dir)
                st.session_state.zeek_df = raw_df
                source_label = f"Zeek ({len(raw_df):,} rows)"
                if live_polling:
                    st.success(f"Streaming: {source_label}", icon=":material/sensors:")
                    st.caption(f"Last updated: **{datetime.now().strftime('%H:%M:%S')}** (polling every {poll_interval}s)")
                else:
                    st.success(f"Loaded: {source_label}", icon=":material/check_circle:")
            except Exception as exc:
                raw_df = None
                st.error(f"Failed to load Zeek logs: {exc}", icon=":material/error:")

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
            help="Minimum total connections to the destination host"
        )

    with st.expander("Filter 3 — Response Size Consistency", expanded=False, icon=":material/data_usage:"):
        tolerance_pct = st.slider(
            "Response size tolerance (±%)", 10, 200, 50, 10,
            help="Allowed deviation from median payload size"
        ) / 100.0
        min_resp_in_bounds_pct = st.slider(
            "Min % in bounds", 50, 100, 70, 5,
            help="% of payload sizes that must be within the tolerance band"
        ) / 100.0

    with st.expander("Filter 4 — Timing Uniformity", expanded=False, icon=":material/schedule:"):
        jitter_tolerance_pct = st.slider(
            "Timing jitter tolerance (±%)", 5, 100, 40, 5,
            help="Allowed deviation from estimated sleep interval (handles 30-50% jitter)"
        ) / 100.0
        min_delta_in_bounds_pct = st.slider(
            "Min % deltas in bounds", 50, 100, 55, 5,
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
# MAIN AREA — Detection & Visualizations
# ══════════════════════════════════════════════════════════════════════════
threshold_kwargs = {
    "min_window_coverage_pct": min_window_coverage_pct,
    "min_total_requests": min_total_requests,
    "tolerance_pct": tolerance_pct,
    "min_resp_in_bounds_pct": min_resp_in_bounds_pct,
    "jitter_tolerance_pct": jitter_tolerance_pct,
    "min_delta_in_bounds_pct": min_delta_in_bounds_pct,
    "min_clean_deltas": min_clean_deltas,
    "outlier_percentile": float(outlier_percentile),
}


def render_main_dashboard(data_df: pd.DataFrame) -> None:
    if data_df is None or data_df.empty:
        st.info(
            "Select a valid data source in the sidebar to get started. The synthetic sample option works immediately — no files needed.",
            icon=":material/arrow_back:",
        )
        return

    # Run the detection engine
    try:
        results_df = detect_beacons(data_df, **threshold_kwargs)
    except Exception as exc:
        st.error(f"Detection engine failed: {exc}", icon=":material/error:")
        return

    # Persist results to SQLite (non-blocking)
    try:
        last_hash = st.session_state.get("_last_saved_results_hash")
        cur_hash = hash(tuple(results_df["composite_score"].round(3).tolist())) if not results_df.empty else 0
        if cur_hash != 0 and cur_hash != last_hash:
            save_results(results_df)
            st.session_state["_last_saved_results_hash"] = cur_hash
    except Exception:
        pass

    # ── Summary metrics ────────────────────────────────────────────────────
    total_groups = data_df["dest_host"].nunique()
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
        st.metric("Ingested Rows", f"{len(data_df):,}", icon=":material/dataset:")

    st.markdown("---")

    # ── Results table ──────────────────────────────────────────────────────
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

        # Threat level label helper
        def _threat_badge_text(score: float) -> str:
            if score >= 0.85:
                return "🚨 CRITICAL"
            elif score >= 0.70:
                return "⚠️ HIGH"
            elif score >= 0.50:
                return "⚡ MEDIUM"
            else:
                return "🟢 LOW"

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

        # ── Scatter plot: Threat Landscape & C2 Benchmark ──────────────────
        st.markdown("### :material/show_chart: Sleep Interval vs. Composite Score")

        plot_df = results_df.drop(columns=["clean_deltas"], errors="ignore").copy()
        plot_df["est_sleep_sec"] = plot_df["est_sleep_sec"].clip(lower=0.1)

        fig = go.Figure()

        # Threat zone background bands
        fig.add_hrect(
            y0=0.85, y1=1.05,
            fillcolor="rgba(255, 68, 68, 0.07)", line_width=0,
            annotation_text="🚨 CRITICAL THREAT", annotation_position="top left",
            annotation=dict(font_color="#FF5252", font_size=10, font_family="Inter"),
        )
        fig.add_hrect(
            y0=0.70, y1=0.85,
            fillcolor="rgba(255, 170, 0, 0.07)", line_width=0,
            annotation_text="⚠️ HIGH THREAT", annotation_position="top left",
            annotation=dict(font_color="#FFAA00", font_size=10, font_family="Inter"),
        )
        fig.add_hrect(
            y0=0.50, y1=0.70,
            fillcolor="rgba(0, 229, 255, 0.05)", line_width=0,
            annotation_text="⚡ MEDIUM THREAT", annotation_position="top left",
            annotation=dict(font_color="#00E5FF", font_size=10, font_family="Inter"),
        )

        # Baseline detection threshold line (0.50)
        fig.add_hline(
            y=0.50,
            line=dict(color="#00E5FF", width=1.5, dash="dash"),
            annotation_text="⚠️ Detection Threshold (0.50)",
            annotation_position="bottom left",
            annotation=dict(font_color="#8B949E", font_size=10, font_family="Inter"),
        )

        # C2 framework default sleep benchmarks
        benchmarks = [
            (10, "Mythic / Sliver (10s)"),
            (30, "Sliver Default (30s)"),
            (60, "Cobalt Strike Default (60s)"),
        ]
        for sleep_val, label in benchmarks:
            fig.add_vline(
                x=sleep_val,
                line=dict(color="#21262D", width=1, dash="dot"),
                annotation_text=label,
                annotation_position="top right",
                annotation=dict(font_color="#6E7681", font_size=9, font_family="Inter"),
            )

        # Candidate scatter bubble trace
        sizes = np.clip(np.sqrt(plot_df["total_requests"]) * 2.2, 22, 48) if not plot_df.empty else []
        fig.add_trace(
            go.Scatter(
                x=plot_df["est_sleep_sec"],
                y=plot_df["composite_score"],
                mode="markers",
                marker=dict(
                    size=sizes,
                    color=plot_df["composite_score"],
                    colorscale=[
                        [0.0, "#162234"],
                        [0.5, "#00E5FF"],
                        [1.0, "#9FEF00"],
                    ],
                    showscale=True,
                    colorbar=dict(
                        title=dict(text="Score", font=dict(color="#E6EDF3", family="Inter")),
                        tickfont=dict(color="#8B949E", family="Inter"),
                        len=0.75,
                    ),
                    line=dict(color="#9FEF00", width=2),
                ),
                text=[
                    f"<b>{r.dest_host}</b><br>"
                    f"URI: {r.uri or '/'}<br>"
                    f"Composite Score: {r.composite_score:.3f}<br>"
                    f"Est. Sleep: {r.est_sleep_sec:.1f}s<br>"
                    f"Total Requests: {r.total_requests:,}<br>"
                    f"Timing CV: {r.timing_cv:.4f}<br>"
                    f"Payload CV: {r.payload_cv:.4f}"
                    for _, r in plot_df.iterrows()
                ],
                hoverinfo="text",
                name="Flagged Candidates",
            )
        )

        # Callout annotations for candidates
        for _, r in plot_df.iterrows():
            fig.add_annotation(
                x=r.est_sleep_sec,
                y=r.composite_score,
                text=(
                    f"🎯 <b>{r.dest_host}</b><br>"
                    f"<span style='color:#9FEF00;'>Score: {r.composite_score:.3f}</span> · "
                    f"Sleep: ~{r.est_sleep_sec:.1f}s · "
                    f"Req: {int(r.total_requests):,}"
                ),
                showarrow=True,
                arrowhead=2,
                arrowsize=1,
                arrowwidth=1.5,
                arrowcolor="#9FEF00",
                ax=60,
                ay=-50,
                bgcolor="#162234",
                bordercolor="#00E5FF",
                borderwidth=1,
                borderpad=7,
                font=dict(family="Inter", size=11, color="#E6EDF3"),
            )

        # Log scale x-axis bounded logically around sleep intervals
        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="#101725",
            plot_bgcolor="#101725",
            font=dict(family="Inter", color="#E6EDF3"),
            xaxis=dict(
                type="log",
                title="Estimated Sleep Interval (seconds, log scale)",
                gridcolor="#1E293B",
                range=[0, 2.5],
            ),
            yaxis=dict(
                title="Composite Detection Score",
                range=[0, 1.05],
                gridcolor="#1E293B",
                zeroline=False,
            ),
            height=460,
            margin=dict(l=40, r=40, t=40, b=40),
            showlegend=False,
        )
        st.plotly_chart(fig, width="stretch")

        st.caption(
            f"ℹ️ **Analysis Insight**: Displays destination hosts that passed all 4 beacon filters. "
            f"In this live capture, **{len(results_df)}** active C2 beacon channel was identified out of "
            f"**{total_groups}** total evaluated network destinations (benign background traffic did not meet "
            f"volume/periodicity thresholds)."
        )

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
            with st.expander(label, icon=threat_icon, expanded=True, key=f"drilldown_{host}"):
                st.markdown(
                    f"Status: {badge_html} &nbsp;&nbsp; Destination: "
                    f"<span style='color:#9FEF00; font-family:monospace; font-weight:600; font-size:1.05rem;'>{host}{uri_val}</span>",
                    unsafe_allow_html=True,
                )
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
                    arr = np.array(deltas)
                    med_delta = float(np.median(arr))
                    std_delta = float(np.std(arr))
                    cv_delta = float(std_delta / np.mean(arr)) if np.mean(arr) > 0 else 0.0

                    # 50% Jitter boundary window around median sleep
                    lower_jitter = med_delta * 0.50
                    upper_jitter = med_delta * 1.50
                    in_jitter_cnt = int(np.sum((arr >= lower_jitter) & (arr <= upper_jitter)))
                    in_jitter_pct = (in_jitter_cnt / len(arr)) * 100 if len(arr) > 0 else 0.0

                    tab_dist, tab_pulse = st.tabs([
                        ":material/bar_chart: Interval Distribution & Jitter",
                        ":material/timeline: Heartbeat Pulse Timeline",
                    ])

                    with tab_dist:
                        kcol1, kcol2, kcol3, kcol4 = st.columns(4)
                        with kcol1:
                            st.caption(f"🎯 Target Sleep: **{med_delta:.1f}s**")
                        with kcol2:
                            st.caption(f"📏 Jitter Window (±50%): **{lower_jitter:.1f}s – {upper_jitter:.1f}s**")
                        with kcol3:
                            st.caption(f"📈 In-Window Bound: **{in_jitter_pct:.1f}%** ({in_jitter_cnt}/{len(arr)})")
                        with kcol4:
                            st.caption(f"📉 Timing CV: **{cv_delta:.4f}** (Uniform)")

                        fig_hist = go.Figure()

                        # Shaded Jitter Envelope
                        fig_hist.add_vrect(
                            x0=lower_jitter,
                            x1=upper_jitter,
                            fillcolor="rgba(0, 229, 255, 0.08)",
                            line_width=0,
                            annotation_text="Jitter Envelope (±50%)",
                            annotation_position="top left",
                            annotation=dict(font_color="#00E5FF", font_size=10, font_family="Inter"),
                        )

                        # Histogram with defined bar borders
                        fig_hist.add_trace(
                            go.Histogram(
                                x=deltas,
                                nbinsx=32,
                                marker=dict(
                                    color="#9FEF00",
                                    line=dict(color="#0D1117", width=1.5),
                                ),
                                opacity=0.90,
                                name="Interval (s)",
                            )
                        )

                        # Target sleep vertical reference line
                        fig_hist.add_vline(
                            x=med_delta,
                            line=dict(color="#00E5FF", width=2.5, dash="dash"),
                            annotation_text=f"🎯 Sleep: {med_delta:.1f}s",
                            annotation_position="top right",
                            annotation=dict(font_color="#00E5FF", font_size=11, font_family="Inter"),
                        )

                        fig_hist.update_layout(
                            title=f"Timing delta distribution & Jitter Envelope — {host}",
                            xaxis_title="Inter-Request Delta (seconds)",
                            yaxis_title="Packet Count",
                            template="plotly_dark",
                            paper_bgcolor="#101725",
                            plot_bgcolor="#101725",
                            font=dict(family="Inter", color="#E6EDF3"),
                            height=320,
                            showlegend=False,
                            margin=dict(l=30, r=30, t=40, b=30),
                        )
                        fig_hist.update_xaxes(gridcolor="#1E293B")
                        fig_hist.update_yaxes(gridcolor="#1E293B")
                        st.plotly_chart(fig_hist, width="stretch")

                    with tab_pulse:
                        # Sequence heartbeat plot for the last 80 check-ins
                        recent_deltas = deltas[-80:]
                        seq_indices = list(range(1, len(recent_deltas) + 1))

                        fig_pulse = go.Figure()

                        # Jitter envelope horizontal band
                        fig_pulse.add_hrect(
                            y0=lower_jitter,
                            y1=upper_jitter,
                            fillcolor="rgba(0, 229, 255, 0.08)",
                            line_width=0,
                            annotation_text="Configured Jitter Range [5.8s – 17.2s]",
                            annotation_position="bottom right",
                            annotation=dict(font_color="#00E5FF", font_size=10, font_family="Inter"),
                        )

                        # Median sleep center line
                        fig_pulse.add_hline(
                            y=med_delta,
                            line=dict(color="#00E5FF", width=1.5, dash="dash"),
                            annotation_text=f"Sleep Interval ({med_delta:.1f}s)",
                            annotation_position="top left",
                            annotation=dict(font_color="#00E5FF", font_size=10, font_family="Inter"),
                        )

                        # Connected heartbeat trace
                        fig_pulse.add_trace(
                            go.Scatter(
                                x=seq_indices,
                                y=recent_deltas,
                                mode="lines+markers",
                                line=dict(color="rgba(159, 239, 0, 0.5)", width=1.5),
                                marker=dict(
                                    color="#9FEF00",
                                    size=6,
                                    line=dict(color="#00E5FF", width=1),
                                ),
                                name="Inter-checkin Delta",
                            )
                        )

                        fig_pulse.update_layout(
                            title=f"Heartbeat Pulse (Chronological Check-ins) — {host}",
                            xaxis_title="Recent Check-in Sequence (#1 → #80)",
                            yaxis_title="Interval to Previous Request (s)",
                            template="plotly_dark",
                            paper_bgcolor="#101725",
                            plot_bgcolor="#101725",
                            font=dict(family="Inter", color="#E6EDF3"),
                            height=320,
                            showlegend=False,
                            margin=dict(l=30, r=30, t=40, b=30),
                        )
                        fig_pulse.update_xaxes(gridcolor="#1E293B")
                        fig_pulse.update_yaxes(gridcolor="#1E293B")
                        st.plotly_chart(fig_pulse, width="stretch")

                        st.caption(
                            "⚡ **Heartbeat Proof**: In a true automated C2 beacon, consecutive arrival intervals "
                            "oscillate strictly within the jitter boundary. Human browsing, by comparison, produces "
                            "irregular clustering with erratic multi-minute gaps."
                        )

    # ── Raw log preview ────────────────────────────────────────────────────
    with st.expander("Raw Log Preview (first 10 rows)", expanded=False, icon=":material/table_rows:"):
        st.caption("Canonical-schema ingested data — sanity-check your ingestor output here.")
        st.dataframe(data_df.head(10), width="stretch", hide_index=True)
        st.caption(f"dtypes: {dict(data_df.dtypes)}")

    # ── Scan history ───────────────────────────────────────────────────────
    with st.expander("Past Scan History (SQLite)", expanded=False, icon=":material/history:"):
        history_df = load_history()
        if history_df.empty:
            st.info("No past scans stored yet.", icon=":material/info:")
        else:
            st.dataframe(history_df, width="stretch", hide_index=True)
            col_hist_info, col_hist_btn = st.columns([4, 1])
            with col_hist_info:
                st.caption(f"{len(history_df)} rows across {history_df['run_id'].nunique()} runs")
            with col_hist_btn:
                if st.button("Clear History", icon=":material/delete:", width="stretch"):
                    clear_history()
                    st.rerun()


# ── Execution: Real-time Live Polling vs. Static View ───────────────────────
if raw_df is None or raw_df.empty:
    st.info(
        "Select a valid data source in the sidebar to get started. The synthetic sample option works immediately — no files needed.",
        icon=":material/arrow_back:",
    )
else:
    if data_source == ":material/folder: Zeek log directory" and is_valid_dir and live_polling:
        st.markdown(
            f"<div style='display:flex; align-items:center; gap:8px; margin: 0 0 16px 0; font-size:0.85rem; color:#8B949E;'>"
            f"<span style='display:inline-block; width:9px; height:9px; border-radius:50%; background:#9FEF00; box-shadow:0 0 10px #9FEF00;'></span>"
            f"<strong style='color:#9FEF00;'>LIVE REAL-TIME STREAMING</strong> · "
            f"Auto-refreshed at <strong>{datetime.now().strftime('%H:%M:%S')}</strong> · "
            f"Ingested <strong>{len(raw_df):,}</strong> rows"
            f"</div>",
            unsafe_allow_html=True,
        )
    render_main_dashboard(raw_df)

if data_source == ":material/folder: Zeek log directory" and is_valid_dir and live_polling:
    import time
    time.sleep(poll_interval)
    st.rerun()


