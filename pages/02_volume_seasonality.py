"""Page 2: Volume & Seasonality — Message flow trends over time."""

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from src.page_logger import log_page_entry, log_page_error
from src.state import (
    render_date_filter,
    load_filtered_edge_fact, load_filtered_weekly_agg,
)
from src.analytics.volume import compute_volume_trends, compute_top_n, compute_sender_concentration
from src.export import download_csv_button
from src.anonymize import anon_df
from src.drilldown import handle_plotly_person_click, handle_plotly_week_click


# ---------------------------------------------------------------------------
# Cached analytics
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner=False, ttl=3600)
def _cached_volume_trends(start_date, end_date):
    wa = load_filtered_weekly_agg(start_date, end_date)
    return compute_volume_trends(wa)


@st.cache_data(show_spinner=False, ttl=3600)
def _cached_top_n(start_date, end_date, n):
    ef = load_filtered_edge_fact(start_date, end_date)
    return compute_top_n(ef, n=n)


@st.cache_data(show_spinner=False, ttl=3600)
def _cached_sender_concentration(start_date, end_date):
    ef = load_filtered_edge_fact(start_date, end_date)
    return compute_sender_concentration(ef)


# ---------------------------------------------------------------------------
# Page layout
# ---------------------------------------------------------------------------

st.set_page_config(page_title="Volume & Seasonality", layout="wide")
_page_log = log_page_entry("02_volume_seasonality")
st.title("Volume & Seasonality")

start_date, end_date = render_date_filter()

edge_fact = load_filtered_edge_fact(start_date, end_date)

if len(edge_fact) == 0:
    st.warning("No data in selected date range.")
    st.stop()

# Volume trends with rolling average (cached)
st.subheader("Weekly Message Volume with Trend")
trends = _cached_volume_trends(start_date, end_date)
td = trends.to_pandas()

fig = go.Figure()
fig.add_trace(go.Bar(x=td["week_start"], y=td["msg_count"], name="Weekly Count", opacity=0.5))
fig.add_trace(go.Scatter(x=td["week_start"], y=td["msg_count_4wk_avg"], name="4-Week Avg",
                         line=dict(width=3, color="red")))
fig.update_layout(height=400, title="Messages per Week")
ev_weekly = st.plotly_chart(fig, width="stretch", on_select="rerun", key="p02_weekly")
handle_plotly_week_click(ev_weekly, "p02_weekly", start_date, end_date)

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
top = _cached_top_n(start_date, end_date, top_n)

col1, col2 = st.columns(2)
with col1:
    st.markdown("**Top Senders**")
    ts = anon_df(top["top_senders"]).to_pandas()
    fig3 = px.bar(ts, x="from_email", y="sent_count", title=f"Top {top_n} Senders")
    fig3.update_layout(height=400, xaxis_tickangle=-45)
    ev_senders = st.plotly_chart(fig3, width="stretch", on_select="rerun", key="p02_senders")
    handle_plotly_person_click(ev_senders, "p02_senders", start_date, end_date)

with col2:
    st.markdown("**Top Receivers**")
    tr = anon_df(top["top_receivers"]).to_pandas()
    fig4 = px.bar(tr, x="to_email", y="received_count", title=f"Top {top_n} Receivers")
    fig4.update_layout(height=400, xaxis_tickangle=-45)
    ev_receivers = st.plotly_chart(fig4, width="stretch", on_select="rerun", key="p02_receivers")
    handle_plotly_person_click(ev_receivers, "p02_receivers", start_date, end_date)

# Sender concentration
st.divider()
st.subheader("Sender Concentration")
conc = _cached_sender_concentration(start_date, end_date)
st.write(f"**Gini coefficient:** {conc['gini']:.3f} (0=perfectly equal, 1=one sender dominates)")
st.write(f"**Total unique senders:** {conc['total_senders']:,}")

st.divider()
download_csv_button(trends, "weekly_volume_trends.csv", "Download Weekly Trends")
