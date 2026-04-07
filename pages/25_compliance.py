"""Page 25: Compliance Pattern Detection — Blackouts, spikes, key dates, after-hours."""

import datetime as dt

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import polars as pl

from src.page_logger import log_page_entry, log_page_error
from src.state import (
    render_date_filter, load_filtered_message_fact, load_filtered_edge_fact,
    load_person_dim, load_nonhuman_emails,
)
from src.analytics.compliance import (
    detect_blackout_windows, detect_external_spikes,
    key_date_gap_analysis, detect_after_hours_clusters,
)
from src.export import download_csv_button


@st.cache_data(show_spinner="Detecting communication blackouts...", ttl=3600)
def _cached_blackouts(start_date, end_date, gap_hours):
    mf = load_filtered_message_fact(start_date, end_date)
    return detect_blackout_windows(mf, min_gap_hours=gap_hours)


@st.cache_data(show_spinner="Detecting external spikes...", ttl=3600)
def _cached_external_spikes(start_date, end_date, z_thresh):
    ef = load_filtered_edge_fact(start_date, end_date)
    pd_dim = load_person_dim()
    return detect_external_spikes(ef, pd_dim, z_threshold=z_thresh)


@st.cache_data(show_spinner="Detecting after-hours clusters...", ttl=3600)
def _cached_after_hours(start_date, end_date):
    ef = load_filtered_edge_fact(start_date, end_date)
    return detect_after_hours_clusters(ef)


st.set_page_config(page_title="Compliance Patterns", layout="wide")
_page_log = log_page_entry("25_compliance")
st.title("Compliance Pattern Detection")
st.caption("Communication anomalies relevant to compliance review: blackouts, external contact spikes, and after-hours clusters.")

start_date, end_date = render_date_filter()

# --- Sidebar controls ---
with st.sidebar:
    st.subheader("Compliance Thresholds")
    gap_hours = st.slider("Blackout gap (hours)", 24, 168, 48, step=12,
                          help="Minimum gap to flag as a blackout window")
    z_threshold = st.slider("External spike z-score", 1.5, 4.0, 2.5, step=0.5,
                            help="Standard deviations above baseline to flag as spike")

# --- Nonhuman filter ---
nonhuman_emails = load_nonhuman_emails(start_date, end_date)
filter_nonhuman = st.session_state.get("exclude_nonhuman", True)
nh_list = list(nonhuman_emails) if filter_nonhuman and nonhuman_emails else []

# --- Section 1: Communication Blackout Windows ---
st.divider()
st.subheader("Communication Blackout Windows")
st.caption(
    f"Active senders (5+ msgs/week avg) who go silent for {gap_hours}+ hours. "
    "May indicate leave, reassignment, or deliberate communication avoidance."
)

blackouts = _cached_blackouts(start_date, end_date, gap_hours)
if nh_list:
    blackouts = blackouts.filter(~pl.col("from_email").is_in(nh_list))

if len(blackouts) > 0:
    st.warning(f"**{len(blackouts)} blackout window(s)** detected.")

    # Top blackouts chart
    top_blackouts = blackouts.head(30)
    fig_bo = px.bar(
        top_blackouts.to_pandas(),
        x="from_email", y="gap_hours",
        color="avg_weekly_volume",
        color_continuous_scale="OrRd",
        title="Longest Communication Gaps",
        labels={"gap_hours": "Gap (hours)", "avg_weekly_volume": "Avg Weekly Volume"},
    )
    fig_bo.update_layout(height=400, xaxis_tickangle=-45)
    st.plotly_chart(fig_bo, use_container_width=True)

    st.dataframe(
        blackouts.head(50).to_pandas(),
        use_container_width=True,
        height=min(400, len(blackouts) * 35 + 40),
    )
    download_csv_button(blackouts, "blackout_windows.csv")
else:
    st.success("No significant communication blackouts detected.")

# --- Section 2: External Contact Spikes ---
st.divider()
st.subheader("External Contact Spikes")
st.caption(
    f"Weeks where a person's external contact count exceeds {z_threshold} standard deviations "
    "above their baseline. May indicate new external relationships or data exfiltration."
)

spikes = _cached_external_spikes(start_date, end_date, z_threshold)
if nh_list:
    spikes = spikes.filter(~pl.col("from_email").is_in(nh_list))

spike_only = spikes.filter(pl.col("is_spike"))

if len(spike_only) > 0:
    st.warning(f"**{len(spike_only)} external spike event(s)** detected.")

    # Timeline of spikes
    fig_spikes = px.scatter(
        spike_only.to_pandas(),
        x="week_start", y="external_contacts",
        color="from_email", size="zscore",
        title="External Contact Spikes Over Time",
        labels={"external_contacts": "Unique External Contacts", "zscore": "Z-Score"},
    )
    fig_spikes.update_layout(height=400)
    st.plotly_chart(fig_spikes, use_container_width=True)

    st.dataframe(spike_only.head(50).to_pandas(), use_container_width=True)
    download_csv_button(spike_only, "external_spikes.csv")
else:
    st.success("No external contact spikes detected.")

# --- Section 3: Key Date Analysis ---
st.divider()
st.subheader("Key Date Analysis")
st.caption("Compare communication volume before and after significant dates.")

# User input for key dates
with st.expander("Add Key Dates", expanded=False):
    col_date, col_label = st.columns([1, 2])
    with col_date:
        input_date = st.date_input(
            "Date", value=start_date, min_value=start_date, max_value=end_date,
            key="compliance_key_date",
        )
    with col_label:
        input_label = st.text_input("Label", placeholder="e.g., Contract signing, Layoff announcement",
                                    key="compliance_key_label")

    if st.button("Add Date"):
        if "_compliance_key_dates" not in st.session_state:
            st.session_state._compliance_key_dates = []
        if input_label:
            st.session_state._compliance_key_dates.append({
                "date": input_date, "label": input_label,
            })
            st.rerun()

key_dates = st.session_state.get("_compliance_key_dates", [])
if key_dates:
    st.write(f"**{len(key_dates)} key date(s)** configured.")
    mf = load_filtered_message_fact(start_date, end_date)
    kd_analysis = key_date_gap_analysis(mf, key_dates)

    if len(kd_analysis) > 0:
        fig_kd = go.Figure()
        for row in kd_analysis.iter_rows(named=True):
            fig_kd.add_trace(go.Bar(
                name=row["label"],
                x=["Before", "After"],
                y=[row["pre_window_volume"], row["post_window_volume"]],
            ))
        fig_kd.update_layout(
            barmode="group", title="Volume Before vs After Key Dates",
            height=350, yaxis_title="Message Count",
        )
        st.plotly_chart(fig_kd, use_container_width=True)
        st.dataframe(kd_analysis.to_pandas(), use_container_width=True)
else:
    st.info("Add key dates above to analyze communication patterns around significant events.")

# --- Section 4: After-Hours Clusters ---
st.divider()
st.subheader("After-Hours Clusters")
st.caption("Groups who consistently communicate outside business hours (before 7 AM or after 6 PM).")

ah_clusters = _cached_after_hours(start_date, end_date)
if nh_list and len(ah_clusters) > 0:
    # Filter clusters that are predominantly nonhuman
    ah_clusters = ah_clusters.filter(pl.col("n_members") > 0)

if len(ah_clusters) > 0:
    st.info(f"**{len(ah_clusters)} after-hours cluster(s)** detected.")

    col_a, col_b = st.columns([2, 1])
    with col_a:
        fig_ah = px.bar(
            ah_clusters.to_pandas(),
            x="cluster_id", y="total_after_hours_msgs",
            color="n_members",
            color_continuous_scale="Purples",
            title="After-Hours Clusters by Message Volume",
            labels={
                "total_after_hours_msgs": "After-Hours Messages",
                "n_members": "Members",
            },
        )
        fig_ah.update_layout(height=350)
        st.plotly_chart(fig_ah, use_container_width=True)

    with col_b:
        for row in ah_clusters.head(5).iter_rows(named=True):
            with st.expander(f"Cluster {row['cluster_id']} ({row['n_members']} people)"):
                st.write(f"**Messages:** {row['total_after_hours_msgs']:,}")
                st.write(f"**Peak hour:** {row['peak_hour']}:00")
                members = row["members"]
                if isinstance(members, list):
                    for m in members[:20]:
                        st.write(f"- {m}")
                    if len(members) > 20:
                        st.write(f"... and {len(members) - 20} more")

    download_csv_button(
        ah_clusters.drop("members"), "after_hours_clusters.csv",
    )
else:
    st.success("No significant after-hours communication clusters detected.")
