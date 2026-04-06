"""Page 19: Narrative Insights — Auto-generated executive narrative with comparison."""

import streamlit as st
import polars as pl

from src.state import (
    render_date_filter, render_comparison_filter,
    load_message_fact, load_person_dim,
    load_filtered_message_fact, load_filtered_edge_fact,
    load_filtered_weekly_agg,
)
from src.analytics.narrative import generate_executive_narrative
from src.analytics.comparison import compute_period_summary, compute_delta


# ---------------------------------------------------------------------------
# Cached analytics
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner=False, ttl=3600)
def _cached_narrative(start_date, end_date):
    mf = load_filtered_message_fact(start_date, end_date)
    wa = load_filtered_weekly_agg(start_date, end_date)
    ef = load_filtered_edge_fact(start_date, end_date)
    pd_dim = load_person_dim()
    return generate_executive_narrative(mf, wa, ef, pd_dim)


@st.cache_data(show_spinner=False, ttl=3600)
def _cached_period_summary(start_date, end_date):
    mf = load_filtered_message_fact(start_date, end_date)
    ef = load_filtered_edge_fact(start_date, end_date)
    return compute_period_summary(mf, ef)


# ---------------------------------------------------------------------------
# Page layout
# ---------------------------------------------------------------------------

st.set_page_config(page_title="Narrative Insights", layout="wide")
st.title("Narrative Insights")
st.caption("Auto-generated analysis of key communication patterns.")

start_date, end_date = render_date_filter()

# Full dataset bounds for comparison slider
full_mf = load_message_fact()
data_min = full_mf["timestamp"].min().date()
data_max = full_mf["timestamp"].max().date()

# Comparison mode
comp_enabled, comp_start, comp_end = render_comparison_filter(data_min, data_max)

message_fact = load_filtered_message_fact(start_date, end_date)

if len(message_fact) == 0:
    st.warning("No messages in selected date range.")
    st.stop()

# Current period narrative
st.subheader("Executive Narrative")
narrative = _cached_narrative(start_date, end_date)
st.markdown(narrative)

st.divider()

# Period summary KPIs
st.subheader("Period KPIs")
current_summary = _cached_period_summary(start_date, end_date)

delta_info = None
if comp_enabled and comp_start and comp_end:
    prev_summary = _cached_period_summary(comp_start, comp_end)
    delta_info = compute_delta(current_summary, prev_summary)

c1, c2, c3, c4 = st.columns(4)
with c1:
    if delta_info:
        st.metric("Messages", f"{current_summary['total_messages']:,}",
                  delta=f"{delta_info['total_messages']['delta']:+,}")
    else:
        st.metric("Messages", f"{current_summary['total_messages']:,}")
with c2:
    if delta_info:
        st.metric("Unique Senders", f"{current_summary['unique_senders']:,}",
                  delta=f"{delta_info['unique_senders']['delta']:+,}")
    else:
        st.metric("Unique Senders", f"{current_summary['unique_senders']:,}")
with c3:
    if delta_info:
        st.metric("After-Hours Rate", f"{current_summary['after_hours_rate']:.1%}",
                  delta=f"{delta_info['after_hours_rate']['delta']:+.1%}")
    else:
        st.metric("After-Hours Rate", f"{current_summary['after_hours_rate']:.1%}")
with c4:
    if delta_info:
        st.metric("Avg Recipients", f"{current_summary['avg_recipients']:.1f}",
                  delta=f"{delta_info['avg_recipients']['delta']:+.1f}")
    else:
        st.metric("Avg Recipients", f"{current_summary['avg_recipients']:.1f}")

# Comparison narrative
if comp_enabled and comp_start and comp_end:
    st.divider()
    st.subheader("Comparison Period Narrative")
    comp_narrative = _cached_narrative(comp_start, comp_end)
    st.markdown(comp_narrative)

    st.divider()
    st.subheader("Period-over-Period Changes")
    if delta_info:
        changes = []
        for key, info in delta_info.items():
            label = key.replace("_", " ").title()
            if isinstance(info["current"], float):
                changes.append({
                    "Metric": label,
                    "Current": f"{info['current']:.2f}",
                    "Previous": f"{info['previous']:.2f}",
                    "Change": f"{info['delta']:+.2f}",
                    "Change %": f"{info['pct']:+.1f}%",
                })
            else:
                changes.append({
                    "Metric": label,
                    "Current": f"{info['current']:,}",
                    "Previous": f"{info['previous']:,}",
                    "Change": f"{info['delta']:+,}",
                    "Change %": f"{info['pct']:+.1f}%",
                })
        st.dataframe(pl.DataFrame(changes).to_pandas(), width="stretch")
