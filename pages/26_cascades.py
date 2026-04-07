"""Page 26: Information Cascade Detection — Forwarding chains and amplifiers."""

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import polars as pl

from src.page_logger import log_page_entry, log_page_error
from src.state import (
    render_date_filter, load_filtered_edge_fact, load_nonhuman_emails,
)
from src.analytics.cascade import (
    detect_cascades, compute_cascade_metrics, identify_amplifiers,
)
from src.export import download_csv_button


@st.cache_data(show_spinner="Detecting information cascades...", ttl=3600)
def _cached_cascades(start_date, end_date, max_delay, min_chain):
    ef = load_filtered_edge_fact(start_date, end_date)
    cascades = detect_cascades(ef, max_delay_minutes=max_delay, min_chain_length=min_chain)
    metrics = compute_cascade_metrics(cascades)
    amplifiers = identify_amplifiers(cascades)
    return cascades, metrics, amplifiers


st.set_page_config(page_title="Information Cascades", layout="wide")
_page_log = log_page_entry("26_cascades")
st.title("Information Cascade Detection")
st.caption(
    "Detecting forwarding chains (A emails B, then B emails C within a time window) "
    "to identify information flow patterns and amplifier nodes."
)

start_date, end_date = render_date_filter()

# --- Sidebar controls ---
with st.sidebar:
    st.subheader("Cascade Settings")
    max_delay = st.slider(
        "Max forwarding delay (minutes)", 5, 240, 60, step=5,
        help="Maximum time between receiving and forwarding a message",
    )
    min_chain = st.slider(
        "Minimum chain length", 2, 10, 3,
        help="Minimum number of hops to count as a cascade",
    )

cascades, metrics, amplifiers = _cached_cascades(start_date, end_date, max_delay, min_chain)

# --- Nonhuman filter ---
nonhuman_emails = load_nonhuman_emails(start_date, end_date)
filter_nonhuman = st.session_state.get("exclude_nonhuman", True)
if filter_nonhuman and nonhuman_emails:
    nh_list = list(nonhuman_emails)
    amplifiers = amplifiers.filter(~pl.col("email").is_in(nh_list))

if len(cascades) == 0:
    st.info(
        "No information cascades detected with current settings. "
        "Try increasing the forwarding delay or reducing the minimum chain length."
    )
    st.stop()

# --- Section 1: KPI Row ---
st.divider()
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Total Cascades", f"{len(metrics):,}")
with col2:
    avg_depth = metrics["depth"].mean() if len(metrics) > 0 else 0
    st.metric("Avg Chain Depth", f"{avg_depth:.1f}")
with col3:
    max_depth = metrics["depth"].max() if len(metrics) > 0 else 0
    st.metric("Max Chain Depth", f"{max_depth}")
with col4:
    avg_vel = metrics["velocity_seconds"].mean() if len(metrics) > 0 else 0
    st.metric("Avg Forwarding Speed", f"{avg_vel:.0f}s")

# --- Section 2: Largest Cascades ---
st.divider()
st.subheader("Largest Cascades")
st.caption("Cascades ranked by depth (longest forwarding chains).")

if len(metrics) > 0:
    top_cascades = metrics.head(20)
    fig_depth = px.bar(
        top_cascades.to_pandas(),
        x="cascade_id", y="depth",
        color="breadth",
        color_continuous_scale="Viridis",
        title="Top Cascades by Depth",
        labels={"depth": "Chain Depth", "breadth": "Unique Recipients", "cascade_id": "Cascade"},
    )
    fig_depth.update_layout(height=400)
    st.plotly_chart(fig_depth, use_container_width=True)

    st.dataframe(top_cascades.to_pandas(), use_container_width=True)
    download_csv_button(metrics, "cascade_metrics.csv")

# --- Section 3: Amplifier Rankings ---
st.divider()
st.subheader("Amplifier Rankings")
st.caption("People who receive information and forward it to many others — key nodes in information flow.")

if len(amplifiers) > 0:
    top_amp = amplifiers.head(25)
    fig_amp = px.bar(
        top_amp.to_pandas(),
        x="email", y="amplifier_score",
        color="avg_fanout",
        color_continuous_scale="OrRd",
        title="Top Information Amplifiers",
        labels={"amplifier_score": "Amplifier Score", "avg_fanout": "Avg Fan-out"},
    )
    fig_amp.update_layout(height=400, xaxis_tickangle=-45)
    st.plotly_chart(fig_amp, use_container_width=True)

    st.dataframe(top_amp.to_pandas(), use_container_width=True)
    download_csv_button(amplifiers, "amplifiers.csv")
else:
    st.info("No amplifier nodes identified.")

# --- Section 4: Cascade Timeline ---
st.divider()
st.subheader("Cascade Timeline")
st.caption("When cascades occur over the date range.")

if len(metrics) > 0 and "duration_minutes" in metrics.columns:
    # Get start times from raw cascade data
    cascade_starts = (
        cascades.group_by("cascade_id")
        .agg(pl.col("timestamp").min().alias("start_time"))
    )
    timeline = metrics.join(cascade_starts, on="cascade_id", how="left")

    if "start_time" in timeline.columns:
        fig_timeline = px.scatter(
            timeline.to_pandas(),
            x="start_time", y="depth",
            size="breadth", color="seed_sender",
            title="Cascade Activity Over Time",
            labels={"start_time": "Time", "depth": "Chain Depth", "breadth": "Recipients"},
        )
        fig_timeline.update_layout(height=400, showlegend=False)
        st.plotly_chart(fig_timeline, use_container_width=True)

# --- Section 5: Cascade Detail Explorer ---
st.divider()
st.subheader("Cascade Detail")
st.caption("Select a cascade to see the full forwarding chain.")

if len(metrics) > 0:
    cascade_ids = metrics["cascade_id"].to_list()[:50]
    depth_lookup = dict(zip(metrics["cascade_id"].to_list(), metrics["depth"].to_list()))
    selected_id = st.selectbox(
        "Select Cascade",
        cascade_ids,
        format_func=lambda x: f"Cascade {x} (depth {depth_lookup.get(x, '?')})",
    )

    if selected_id is not None:
        chain = cascades.filter(pl.col("cascade_id") == selected_id).sort("step")
        st.dataframe(chain.to_pandas(), use_container_width=True)

        # Simple flow visualization
        if len(chain) > 0:
            nodes = []
            for row in chain.iter_rows(named=True):
                if row["from_email"] not in nodes:
                    nodes.append(row["from_email"])
                if row["to_email"] not in nodes:
                    nodes.append(row["to_email"])

            node_idx = {n: i for i, n in enumerate(nodes)}
            sources = [node_idx[r["from_email"]] for r in chain.iter_rows(named=True)]
            targets = [node_idx[r["to_email"]] for r in chain.iter_rows(named=True)]
            values = [1] * len(sources)

            fig_flow = go.Figure(go.Sankey(
                node=dict(label=nodes, pad=15, thickness=20),
                link=dict(source=sources, target=targets, value=values),
            ))
            fig_flow.update_layout(title=f"Cascade {selected_id} Flow", height=300)
            st.plotly_chart(fig_flow, use_container_width=True)
