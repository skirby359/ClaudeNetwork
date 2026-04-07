"""Page 3: Time Norms — When do people communicate?"""

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import polars as pl

from src.page_logger import log_page_entry, log_page_error
from src.state import (
    render_date_filter, load_filtered_message_fact,
    load_nonhuman_emails,
)
from src.analytics.timing_analytics import (
    compute_hour_day_heatmap,
    compute_after_hours_by_week,
    compute_burstiness,
)
from src.drilldown import handle_plotly_person_click, handle_plotly_week_click


# ---------------------------------------------------------------------------
# Cached analytics
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner=False, ttl=3600)
def _cached_heatmap(start_date, end_date, exclude_nonhuman):
    mf = load_filtered_message_fact(start_date, end_date)
    if exclude_nonhuman:
        nonhuman = load_nonhuman_emails(start_date, end_date)
        mf = mf.filter(~pl.col("from_email").is_in(list(nonhuman)))
    return compute_hour_day_heatmap(mf)


@st.cache_data(show_spinner=False, ttl=3600)
def _cached_after_hours(start_date, end_date, exclude_nonhuman):
    mf = load_filtered_message_fact(start_date, end_date)
    if exclude_nonhuman:
        nonhuman = load_nonhuman_emails(start_date, end_date)
        mf = mf.filter(~pl.col("from_email").is_in(list(nonhuman)))
    return compute_after_hours_by_week(mf)


@st.cache_data(show_spinner=False, ttl=3600)
def _cached_burstiness(start_date, end_date, exclude_nonhuman, top_n):
    mf = load_filtered_message_fact(start_date, end_date)
    if exclude_nonhuman:
        nonhuman = load_nonhuman_emails(start_date, end_date)
        mf = mf.filter(~pl.col("from_email").is_in(list(nonhuman)))
    return compute_burstiness(mf, top_n=top_n)


@st.cache_data(show_spinner=False, ttl=3600)
def _cached_hourly_volume(start_date, end_date, exclude_nonhuman):
    mf = load_filtered_message_fact(start_date, end_date)
    if exclude_nonhuman:
        nonhuman = load_nonhuman_emails(start_date, end_date)
        mf = mf.filter(~pl.col("from_email").is_in(list(nonhuman)))
    return (
        mf.group_by("hour")
        .agg(pl.len().alias("count"))
        .sort("hour")
    )


# ---------------------------------------------------------------------------
# Page layout
# ---------------------------------------------------------------------------

st.set_page_config(page_title="Time Norms", layout="wide")
_page_log = log_page_entry("03_time_norms")
st.title("Time Norms")

start_date, end_date = render_date_filter()

message_fact = load_filtered_message_fact(start_date, end_date)
if len(message_fact) == 0:
    st.warning("No data in selected date range.")
    st.stop()

# --- Nonhuman filter ---
nonhuman_emails = load_nonhuman_emails(start_date, end_date)

filter_on = st.session_state.get("exclude_nonhuman", True)

if filter_on:
    st.caption(f"Filtered out {len(nonhuman_emails)} nonhuman addresses.")

# Hour x Day heatmap
st.subheader("Message Activity Heatmap")
heatmap_data = _cached_heatmap(start_date, end_date, filter_on)

day_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
matrix = np.zeros((24, 7))
for row in heatmap_data.iter_rows(named=True):
    matrix[row["hour"]][row["day_of_week"]] = row["msg_count"]

fig = go.Figure(data=go.Heatmap(
    z=matrix,
    x=day_names,
    y=list(range(24)),
    colorscale="YlOrRd",
    hovertemplate="Day: %{x}<br>Hour: %{y}:00<br>Messages: %{z}<extra></extra>",
))
fig.update_layout(
    height=500,
    title="Messages by Hour and Day of Week",
    yaxis_title="Hour of Day",
    xaxis_title="Day of Week",
    yaxis=dict(dtick=1),
)
st.plotly_chart(fig, width="stretch")

# After-hours trend
st.divider()
st.subheader("After-Hours Messaging Over Time")
ah_weekly = _cached_after_hours(start_date, end_date, filter_on).to_pandas()
fig2 = go.Figure()
fig2.add_trace(go.Scatter(x=ah_weekly["week_start"], y=ah_weekly["after_hours_rate"],
                          name="After-Hours Rate", line=dict(width=2)))
fig2.add_trace(go.Scatter(x=ah_weekly["week_start"], y=ah_weekly["weekend_rate"],
                          name="Weekend Rate", line=dict(width=2, dash="dash")))
fig2.update_layout(height=350, title="After-Hours & Weekend Rates by Week",
                   yaxis_title="Rate", yaxis_tickformat=".0%")
ev_afterhours = st.plotly_chart(fig2, width="stretch", on_select="rerun", key="p03_afterhours")
handle_plotly_week_click(ev_afterhours, "p03_afterhours", start_date, end_date)

# Hourly distribution
st.divider()
st.subheader("Hourly Volume Distribution")
hourly = _cached_hourly_volume(start_date, end_date, filter_on).to_pandas()
fig3 = px.bar(hourly, x="hour", y="count", title="Message Volume by Hour of Day")
fig3.update_layout(height=350, xaxis=dict(dtick=1))
st.plotly_chart(fig3, width="stretch")

# Burstiness
st.divider()
st.subheader("Sender Burstiness")
st.markdown("""
**Burstiness** measures how irregular a sender's activity pattern is.
- B > 0: Bursty (sends in clusters)
- B ~ 0: Random (Poisson-like)
- B < 0: Periodic (evenly spaced)
""")
burst = _cached_burstiness(start_date, end_date, filter_on, 30).to_pandas()
fig4 = px.bar(burst, x="from_email", y="burstiness", color="burstiness",
              color_continuous_scale="RdBu_r", title="Burstiness of Top 30 Senders")
fig4.update_layout(height=400, xaxis_tickangle=-45)
ev_burst = st.plotly_chart(fig4, width="stretch", on_select="rerun", key="p03_burst")
handle_plotly_person_click(ev_burst, "p03_burst", start_date, end_date)
