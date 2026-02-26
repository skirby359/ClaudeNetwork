"""Page 3: Time Norms — When do people communicate?"""

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import polars as pl

from src.state import (
    render_date_filter, load_filtered_message_fact,
)
from src.analytics.timing_analytics import (
    compute_hour_day_heatmap,
    compute_after_hours_by_week,
    compute_burstiness,
)

st.set_page_config(page_title="Time Norms", layout="wide")
st.title("Time Norms")

start_date, end_date = render_date_filter()

message_fact = load_filtered_message_fact(start_date, end_date)

# Hour x Day heatmap
st.subheader("Message Activity Heatmap")
heatmap_data = compute_hour_day_heatmap(message_fact)

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
ah_weekly = compute_after_hours_by_week(message_fact).to_pandas()
fig2 = go.Figure()
fig2.add_trace(go.Scatter(x=ah_weekly["week_start"], y=ah_weekly["after_hours_rate"],
                          name="After-Hours Rate", line=dict(width=2)))
fig2.add_trace(go.Scatter(x=ah_weekly["week_start"], y=ah_weekly["weekend_rate"],
                          name="Weekend Rate", line=dict(width=2, dash="dash")))
fig2.update_layout(height=350, title="After-Hours & Weekend Rates by Week",
                   yaxis_title="Rate", yaxis_tickformat=".0%")
st.plotly_chart(fig2, width="stretch")

# Hourly distribution
st.divider()
st.subheader("Hourly Volume Distribution")
hourly = (
    message_fact.group_by("hour")
    .agg(pl.len().alias("count"))
    .sort("hour")
    .to_pandas()
)
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
burst = compute_burstiness(message_fact, top_n=30).to_pandas()
fig4 = px.bar(burst, x="from_email", y="burstiness", color="burstiness",
              color_continuous_scale="RdBu_r", title="Burstiness of Top 30 Senders")
fig4.update_layout(height=400, xaxis_tickangle=-45)
st.plotly_chart(fig4, width="stretch")
