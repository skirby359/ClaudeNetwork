"""Page 20: Period Comparison — Side-by-side analysis of two time periods."""

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import polars as pl

from src.page_logger import log_page_entry, log_page_error
from src.state import (
    render_date_filter, render_comparison_filter,
    load_message_fact,
    load_filtered_message_fact, load_filtered_edge_fact,
    load_filtered_weekly_agg,
)
from src.analytics.comparison import compute_period_summary, compute_delta
from src.analytics.volume import compute_sender_concentration
from src.export import download_csv_button


# ---------------------------------------------------------------------------
# Cached analytics
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner=False, ttl=3600)
def _cached_summary(start_date, end_date):
    mf = load_filtered_message_fact(start_date, end_date)
    ef = load_filtered_edge_fact(start_date, end_date)
    return compute_period_summary(mf, ef)


@st.cache_data(show_spinner=False, ttl=3600)
def _cached_weekly(start_date, end_date):
    return load_filtered_weekly_agg(start_date, end_date)


@st.cache_data(show_spinner=False, ttl=3600)
def _cached_top_senders(start_date, end_date, n=15):
    ef = load_filtered_edge_fact(start_date, end_date)
    return (
        ef.group_by("from_email")
        .agg(pl.len().alias("count"))
        .sort("count", descending=True)
        .head(n)
    )


# ---------------------------------------------------------------------------
# Page layout
# ---------------------------------------------------------------------------

st.set_page_config(page_title="Period Comparison", layout="wide")
_page_log = log_page_entry("20_comparison")
st.title("Period Comparison")
st.caption("Compare key metrics between two time periods side by side.")

start_date, end_date = render_date_filter()

full_mf = load_message_fact()
data_min = full_mf["timestamp"].min().date()
data_max = full_mf["timestamp"].max().date()

comp_enabled, comp_start, comp_end = render_comparison_filter(data_min, data_max)

message_fact = load_filtered_message_fact(start_date, end_date)

if len(message_fact) == 0:
    st.warning("No messages in selected date range.")
    st.stop()

current_summary = _cached_summary(start_date, end_date)

if not comp_enabled or comp_start is None or comp_end is None:
    st.info("Enable **Compare to previous period** in the sidebar to see a side-by-side comparison.")

    # Show current period summary only
    st.subheader("Current Period Summary")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Messages", f"{current_summary['total_messages']:,}")
    with c2:
        st.metric("Unique Senders", f"{current_summary['unique_senders']:,}")
    with c3:
        st.metric("After-Hours Rate", f"{current_summary['after_hours_rate']:.1%}")
    with c4:
        st.metric("Avg Recipients", f"{current_summary['avg_recipients']:.1f}")
    st.stop()

# Comparison enabled
prev_summary = _cached_summary(comp_start, comp_end)
delta_info = compute_delta(current_summary, prev_summary)

# Delta KPIs
st.subheader("Key Metric Changes")
c1, c2, c3, c4, c5 = st.columns(5)
with c1:
    st.metric("Messages", f"{current_summary['total_messages']:,}",
              delta=f"{delta_info['total_messages']['delta']:+,}")
with c2:
    st.metric("Unique Senders", f"{current_summary['unique_senders']:,}",
              delta=f"{delta_info['unique_senders']['delta']:+,}")
with c3:
    st.metric("Unique Recipients", f"{current_summary['unique_recipients']:,}",
              delta=f"{delta_info['unique_recipients']['delta']:+,}")
with c4:
    st.metric("After-Hours Rate", f"{current_summary['after_hours_rate']:.1%}",
              delta=f"{delta_info['after_hours_rate']['delta']:+.1%}")
with c5:
    gb_current = current_summary["total_bytes"] / (1024**3)
    gb_delta = delta_info["total_bytes"]["delta"] / (1024**3)
    st.metric("Total Data", f"{gb_current:.1f} GB", delta=f"{gb_delta:+.1f} GB")

st.divider()

# Side-by-side volume trends
st.subheader("Volume Trends Comparison")
col_a, col_b = st.columns(2)

with col_a:
    st.markdown(f"**Current Period** ({start_date} to {end_date})")
    wa_current = _cached_weekly(start_date, end_date)
    if len(wa_current) > 0:
        fig = px.bar(wa_current.to_pandas(), x="week_start", y="msg_count",
                     title="Weekly Volume (Current)")
        fig.update_layout(height=300)
        st.plotly_chart(fig, width="stretch")

with col_b:
    st.markdown(f"**Previous Period** ({comp_start} to {comp_end})")
    wa_prev = _cached_weekly(comp_start, comp_end)
    if len(wa_prev) > 0:
        fig2 = px.bar(wa_prev.to_pandas(), x="week_start", y="msg_count",
                      title="Weekly Volume (Previous)")
        fig2.update_layout(height=300)
        st.plotly_chart(fig2, width="stretch")

st.divider()

# Top senders comparison
st.subheader("Top Senders Comparison")
col_c, col_d = st.columns(2)

with col_c:
    current_senders = _cached_top_senders(start_date, end_date)
    fig3 = px.bar(current_senders.to_pandas(), x="from_email", y="count",
                  title="Top 15 Senders (Current)")
    fig3.update_layout(height=350, xaxis_tickangle=-45)
    st.plotly_chart(fig3, width="stretch")

with col_d:
    prev_senders = _cached_top_senders(comp_start, comp_end)
    fig4 = px.bar(prev_senders.to_pandas(), x="from_email", y="count",
                  title="Top 15 Senders (Previous)")
    fig4.update_layout(height=350, xaxis_tickangle=-45)
    st.plotly_chart(fig4, width="stretch")

st.divider()

# Full delta table
st.subheader("All Metric Deltas")
rows = []
for key, info in delta_info.items():
    label = key.replace("_", " ").title()
    if isinstance(info["current"], float):
        rows.append({
            "Metric": label,
            "Current": f"{info['current']:.3f}",
            "Previous": f"{info['previous']:.3f}",
            "Delta": f"{info['delta']:+.3f}",
            "Change %": f"{info['pct']:+.1f}%",
        })
    else:
        rows.append({
            "Metric": label,
            "Current": f"{info['current']:,}",
            "Previous": f"{info['previous']:,}",
            "Delta": f"{info['delta']:+,}",
            "Change %": f"{info['pct']:+.1f}%",
        })
delta_df = pl.DataFrame(rows)
st.dataframe(delta_df.to_pandas(), width="stretch")
download_csv_button(delta_df, "period_comparison.csv")
