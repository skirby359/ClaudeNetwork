"""Page 1: Executive Summary — Key findings at a glance."""

import streamlit as st
import plotly.express as px

from src.state import (
    load_person_dim, load_message_fact,
    render_date_filter, render_comparison_filter,
    load_filtered_message_fact, load_filtered_edge_fact,
    load_filtered_weekly_agg, load_filtered_broadcast,
)
from src.analytics.volume import compute_sender_concentration
from src.analytics.comparison import compute_period_summary, compute_delta
from src.analytics.narrative import generate_executive_narrative
from src.export import download_csv_button
from src.drilldown import handle_plotly_person_click, handle_plotly_week_click


# ---------------------------------------------------------------------------
# Cached analytics
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner=False, ttl=3600)
def _cached_period_summary(start_date, end_date):
    mf = load_filtered_message_fact(start_date, end_date)
    ef = load_filtered_edge_fact(start_date, end_date)
    return compute_period_summary(mf, ef)


@st.cache_data(show_spinner=False, ttl=3600)
def _cached_sender_concentration(start_date, end_date):
    ef = load_filtered_edge_fact(start_date, end_date)
    return compute_sender_concentration(ef)


@st.cache_data(show_spinner=False, ttl=3600)
def _cached_narrative(start_date, end_date):
    mf = load_filtered_message_fact(start_date, end_date)
    wa = load_filtered_weekly_agg(start_date, end_date)
    ef = load_filtered_edge_fact(start_date, end_date)
    pd_dim = load_person_dim()
    return generate_executive_narrative(mf, wa, ef, pd_dim)


# ---------------------------------------------------------------------------
# Page layout
# ---------------------------------------------------------------------------

st.set_page_config(page_title="Executive Summary", layout="wide")
st.title("Executive Summary")

start_date, end_date = render_date_filter()

# Full dataset bounds for comparison slider
full_mf = load_message_fact()
data_min = full_mf["timestamp"].min().date()
data_max = full_mf["timestamp"].max().date()

# Comparison mode
comp_enabled, comp_start, comp_end = render_comparison_filter(data_min, data_max)

message_fact = load_filtered_message_fact(start_date, end_date)
edge_fact = load_filtered_edge_fact(start_date, end_date)
person_dim = load_person_dim()
weekly_agg = load_filtered_weekly_agg(start_date, end_date)
broadcast = load_filtered_broadcast(start_date, end_date)

# Compute current period summary (cached)
current_summary = _cached_period_summary(start_date, end_date)

# If comparison mode, compute previous period (cached)
delta_info = None
if comp_enabled and comp_start and comp_end:
    prev_summary = _cached_period_summary(comp_start, comp_end)
    delta_info = compute_delta(current_summary, prev_summary)

# Top-level KPIs
c1, c2, c3, c4, c5 = st.columns(5)

with c1:
    if delta_info:
        st.metric("Total Messages", f"{current_summary['total_messages']:,}",
                  delta=f"{delta_info['total_messages']['delta']:+,}")
    else:
        st.metric("Total Messages", f"{len(message_fact):,}")
with c2:
    if delta_info:
        st.metric("Unique Senders", f"{current_summary['unique_senders']:,}",
                  delta=f"{delta_info['unique_senders']['delta']:+,}")
    else:
        st.metric("Unique People", f"{len(person_dim):,}")
with c3:
    internal = person_dim.filter(person_dim["is_internal"])
    st.metric("Internal", f"{len(internal):,}")
with c4:
    total_bytes = message_fact["size_bytes"].sum()
    if delta_info:
        delta_gb = delta_info["total_bytes"]["delta"] / (1024**3)
        st.metric("Total Data", f"{total_bytes / (1024**3):.1f} GB",
                  delta=f"{delta_gb:+.1f} GB")
    else:
        st.metric("Total Data", f"{total_bytes / (1024**3):.1f} GB")
with c5:
    avg_recip = message_fact["n_recipients"].mean()
    if delta_info:
        st.metric("Avg Recipients/Msg", f"{avg_recip:.1f}",
                  delta=f"{delta_info['avg_recipients']['delta']:+.1f}")
    else:
        st.metric("Avg Recipients/Msg", f"{avg_recip:.1f}")

st.divider()

# Volume trend
col_left, col_right = st.columns(2)

with col_left:
    st.subheader("Weekly Message Volume")
    wa = weekly_agg.to_pandas()
    fig = px.line(wa, x="week_start", y="msg_count", title="Messages per Week")
    fig.update_layout(height=350)
    ev_weekly = st.plotly_chart(fig, width="stretch", on_select="rerun", key="p01_weekly")
    handle_plotly_week_click(ev_weekly, "p01_weekly", start_date, end_date)

with col_right:
    st.subheader("Sender Concentration")
    conc = _cached_sender_concentration(start_date, end_date)
    st.write(f"**Gini coefficient:** {conc['gini']:.3f}")
    st.write(f"**Top 5 senders:** {conc['top_5_share']:.1%} of all messages")
    st.write(f"**Top 10 senders:** {conc['top_10_share']:.1%} of all messages")
    st.write(f"**Top 20 senders:** {conc['top_20_share']:.1%} of all messages")

    top_senders = conc["top_senders"].head(10).to_pandas()
    fig2 = px.bar(top_senders, x="from_email", y="count", title="Top 10 Senders")
    fig2.update_layout(height=350, xaxis_tickangle=-45)
    ev_senders = st.plotly_chart(fig2, width="stretch", on_select="rerun", key="p01_senders")
    handle_plotly_person_click(ev_senders, "p01_senders", start_date, end_date)

st.divider()

# Key insights
st.subheader("Key Findings")
col_a, col_b = st.columns(2)

with col_a:
    ah_rate = message_fact["is_after_hours"].mean()
    we_rate = message_fact["is_weekend"].mean()
    if delta_info:
        st.metric("After-Hours Rate", f"{ah_rate:.1%}",
                  delta=f"{delta_info['after_hours_rate']['delta']:+.1%}")
        st.metric("Weekend Rate", f"{we_rate:.1%}",
                  delta=f"{delta_info['weekend_rate']['delta']:+.1%}")
    else:
        st.info(f"**After-hours rate:** {ah_rate:.1%} of messages sent outside business hours")
        st.info(f"**Weekend rate:** {we_rate:.1%} of messages sent on weekends")

with col_b:
    n_blast = len(message_fact.filter(message_fact["n_recipients"] > 10))
    st.info(f"**Large blasts (>10 recipients):** {n_blast:,} messages ({n_blast/max(len(message_fact),1):.1%})")
    top_blaster = broadcast.head(1)
    if len(top_blaster) > 0:
        st.info(f"**Top broadcaster:** {top_blaster['from_email'][0]} ({top_blaster['total_msgs'][0]:,} msgs)")

st.divider()

# Executive narrative (cached)
st.subheader("Executive Narrative")
narrative = _cached_narrative(start_date, end_date)
st.markdown(narrative)

# Export
st.divider()
download_csv_button(weekly_agg, "weekly_aggregation.csv", "Download Weekly Data")
