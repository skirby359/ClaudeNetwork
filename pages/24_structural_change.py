"""Page 24: Structural Change Detection — Community evolution analysis."""

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import polars as pl

from src.page_logger import log_page_entry, log_page_error
from src.state import (
    render_date_filter, load_filtered_edge_fact, load_nonhuman_emails,
)
from src.analytics.temporal_network import (
    build_monthly_snapshots, compute_community_stability,
)
from src.analytics.structural_change import (
    classify_community_shifts, track_node_switches,
    compute_switch_rates, nmi_drop_alerts, build_community_flow,
)
from src.export import download_csv_button


@st.cache_data(show_spinner="Analyzing structural changes...", ttl=3600)
def _cached_structural(start_date, end_date):
    ef = load_filtered_edge_fact(start_date, end_date)
    snapshots = build_monthly_snapshots(ef)
    stability = compute_community_stability(snapshots)
    shifts = classify_community_shifts(snapshots)
    node_sw = track_node_switches(snapshots)
    switch_rates = compute_switch_rates(node_sw)
    alerts = nmi_drop_alerts(stability)
    flow = build_community_flow(snapshots)
    return stability, shifts, node_sw, switch_rates, alerts, flow


st.set_page_config(page_title="Structural Change Detection", layout="wide")
_page_log = log_page_entry("24_structural_change")
st.title("Structural Change Detection")
st.caption("Detecting when and how organizational structure shifts over time.")

start_date, end_date = render_date_filter()

stability, shifts, node_switches, switch_rates, alerts, flow = _cached_structural(
    start_date, end_date,
)

if len(stability) == 0:
    st.warning("Need at least 2 months of data for structural change analysis.")
    st.stop()

# --- Nonhuman filter for switch rates ---
nonhuman_emails = load_nonhuman_emails(start_date, end_date)
filter_nonhuman = st.session_state.get("exclude_nonhuman", True)
if filter_nonhuman and nonhuman_emails:
    switch_rates = switch_rates.filter(~pl.col("email").is_in(list(nonhuman_emails)))

# --- Section 1: NMI Timeline with Alert Zones ---
st.divider()
st.subheader("Community Stability Over Time")
st.caption("Normalized Mutual Information between consecutive months. Lower = more change.")

fig_nmi = go.Figure()
fig_nmi.add_trace(go.Scatter(
    x=stability["month_pair"].to_list(),
    y=stability["nmi"].to_list(),
    mode="lines+markers",
    name="NMI",
    line=dict(color="#4e79a7", width=2),
    marker=dict(size=8),
))
# Alert zones
fig_nmi.add_hrect(y0=0, y1=0.3, fillcolor="red", opacity=0.1, line_width=0,
                  annotation_text="Critical", annotation_position="top left")
fig_nmi.add_hrect(y0=0.3, y1=0.5, fillcolor="orange", opacity=0.1, line_width=0,
                  annotation_text="Warning", annotation_position="top left")
fig_nmi.add_hrect(y0=0.7, y1=1.0, fillcolor="green", opacity=0.05, line_width=0,
                  annotation_text="Stable", annotation_position="top left")
fig_nmi.update_layout(
    height=400, yaxis_title="NMI Score", xaxis_title="Month Pair",
    yaxis_range=[0, 1.05],
)
st.plotly_chart(fig_nmi, use_container_width=True)

if len(alerts) > 0:
    st.warning(f"**{len(alerts)} stability alert(s)** detected.")
    st.dataframe(alerts.to_pandas(), use_container_width=True)

# --- Section 2: Shift Classification ---
st.divider()
st.subheader("Change Classification")
st.caption("What kind of structural change occurred each month.")

if len(shifts) > 0:
    # Color-coded table
    shift_colors = {"stable": "green", "split": "orange", "merge": "blue", "reorg": "red"}
    fig_shifts = px.bar(
        shifts.to_pandas(),
        x="month_pair", y="nmi", color="shift_type",
        color_discrete_map={
            "stable": "#59a14f", "split": "#f28e2b",
            "merge": "#4e79a7", "reorg": "#e15759",
            "insufficient_data": "#bab0ac",
        },
        title="Shift Type by Month",
        labels={"nmi": "NMI Score", "shift_type": "Change Type"},
    )
    fig_shifts.update_layout(height=350)
    st.plotly_chart(fig_shifts, use_container_width=True)

    st.dataframe(
        shifts.to_pandas(),
        use_container_width=True,
        height=min(400, len(shifts) * 35 + 40),
    )

# --- Section 3: Frequent Switchers ---
st.divider()
st.subheader("Frequent Community Switchers")
st.caption("People who change communities most often — may indicate role changes or flexible connectors.")

if len(switch_rates) > 0:
    top_switchers = switch_rates.filter(pl.col("n_switches") > 0).head(30)

    if len(top_switchers) > 0:
        fig_sw = px.bar(
            top_switchers.to_pandas(),
            x="email", y="n_switches",
            color="switch_rate",
            color_continuous_scale="OrRd",
            title="Top Community Switchers",
            labels={"n_switches": "Times Switched", "switch_rate": "Switch Rate"},
        )
        fig_sw.update_layout(height=400, xaxis_tickangle=-45)
        st.plotly_chart(fig_sw, use_container_width=True)

        st.dataframe(top_switchers.to_pandas(), use_container_width=True)
        download_csv_button(top_switchers, "frequent_switchers.csv")
    else:
        st.success("No community switches detected in the selected period.")

# --- Section 4: Community Flow (Sankey) ---
st.divider()
st.subheader("Community Membership Flow")
st.caption("How people move between communities across months.")

if len(flow) > 0:
    # Build Sankey diagram
    all_labels = sorted(set(flow["source"].to_list() + flow["target"].to_list()))
    label_idx = {label: i for i, label in enumerate(all_labels)}

    fig_sankey = go.Figure(go.Sankey(
        node=dict(
            label=all_labels,
            pad=15,
            thickness=20,
        ),
        link=dict(
            source=[label_idx[s] for s in flow["source"].to_list()],
            target=[label_idx[t] for t in flow["target"].to_list()],
            value=flow["value"].to_list(),
        ),
    ))
    fig_sankey.update_layout(
        title="Community Flow Between Months",
        height=max(500, len(all_labels) * 15),
    )
    st.plotly_chart(fig_sankey, use_container_width=True)
else:
    st.info("Not enough months for flow analysis.")
