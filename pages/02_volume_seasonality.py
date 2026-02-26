"""Page 2: Volume & Seasonality — Message flow trends over time."""

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from src.state import (
    render_date_filter,
    load_filtered_edge_fact, load_filtered_weekly_agg,
)
from src.analytics.volume import compute_volume_trends, compute_top_n, compute_sender_concentration

st.set_page_config(page_title="Volume & Seasonality", layout="wide")
st.title("Volume & Seasonality")

start_date, end_date = render_date_filter()

edge_fact = load_filtered_edge_fact(start_date, end_date)
weekly_agg = load_filtered_weekly_agg(start_date, end_date)

# Volume trends with rolling average
st.subheader("Weekly Message Volume with Trend")
trends = compute_volume_trends(weekly_agg)
td = trends.to_pandas()

fig = go.Figure()
fig.add_trace(go.Bar(x=td["week_start"], y=td["msg_count"], name="Weekly Count", opacity=0.5))
fig.add_trace(go.Scatter(x=td["week_start"], y=td["msg_count_4wk_avg"], name="4-Week Avg",
                         line=dict(width=3, color="red")))
fig.update_layout(height=400, title="Messages per Week")
st.plotly_chart(fig, width="stretch")

# Impressions trend
st.subheader("Recipient Impressions Over Time")
fig2 = go.Figure()
fig2.add_trace(go.Bar(x=td["week_start"], y=td["recipient_impressions"], name="Impressions", opacity=0.5))
fig2.add_trace(go.Scatter(x=td["week_start"], y=td["impressions_4wk_avg"], name="4-Week Avg",
                          line=dict(width=3, color="orange")))
fig2.update_layout(height=400, title="Total Recipient Impressions per Week")
st.plotly_chart(fig2, width="stretch")

# Top senders and receivers
st.divider()
st.subheader("Top Senders & Receivers")
top_n = st.slider("Number of top entities", 5, 50, 20)
top = compute_top_n(edge_fact, n=top_n)

col1, col2 = st.columns(2)
with col1:
    st.markdown("**Top Senders**")
    ts = top["top_senders"].to_pandas()
    fig3 = px.bar(ts, x="from_email", y="sent_count", title=f"Top {top_n} Senders")
    fig3.update_layout(height=400, xaxis_tickangle=-45)
    st.plotly_chart(fig3, width="stretch")

with col2:
    st.markdown("**Top Receivers**")
    tr = top["top_receivers"].to_pandas()
    fig4 = px.bar(tr, x="to_email", y="received_count", title=f"Top {top_n} Receivers")
    fig4.update_layout(height=400, xaxis_tickangle=-45)
    st.plotly_chart(fig4, width="stretch")

# Sender concentration
st.divider()
st.subheader("Sender Concentration")
conc = compute_sender_concentration(edge_fact)
st.write(f"**Gini coefficient:** {conc['gini']:.3f} (0=perfectly equal, 1=one sender dominates)")
st.write(f"**Total unique senders:** {conc['total_senders']:,}")
