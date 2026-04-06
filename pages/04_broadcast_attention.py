"""Page 4: Broadcast & Attention — Mass-send patterns and inbox load."""

import streamlit as st
import plotly.express as px

from src.state import (
    render_date_filter, load_filtered_message_fact, load_filtered_broadcast,
)
from src.analytics.broadcast_analytics import (
    compute_blast_impact,
    compute_high_blast_senders,
    compute_recipient_distribution,
)
from src.export import download_csv_button
from src.drilldown import handle_plotly_person_click, handle_dataframe_person_click


# ---------------------------------------------------------------------------
# Cached analytics
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner=False, ttl=3600)
def _cached_blast_impact(start_date, end_date):
    mf = load_filtered_message_fact(start_date, end_date)
    return compute_blast_impact(mf)


@st.cache_data(show_spinner=False, ttl=3600)
def _cached_recipient_distribution(start_date, end_date):
    mf = load_filtered_message_fact(start_date, end_date)
    return compute_recipient_distribution(mf)


@st.cache_data(show_spinner=False, ttl=3600)
def _cached_high_blast_senders(start_date, end_date, threshold):
    mf = load_filtered_message_fact(start_date, end_date)
    return compute_high_blast_senders(mf, threshold=threshold)


# ---------------------------------------------------------------------------
# Page layout
# ---------------------------------------------------------------------------

st.set_page_config(page_title="Broadcast & Attention", layout="wide")
st.title("Broadcast & Attention")

start_date, end_date = render_date_filter()

message_fact = load_filtered_message_fact(start_date, end_date)
broadcast = load_filtered_broadcast(start_date, end_date)

if len(message_fact) == 0:
    st.warning("No data in selected date range.")
    st.stop()

# Blast tier breakdown (cached)
st.subheader("Message Tiers by Recipient Count")
tiers = _cached_blast_impact(start_date, end_date).to_pandas()
col1, col2 = st.columns(2)

with col1:
    fig = px.pie(tiers, values="msg_count", names="blast_tier", title="Messages by Tier")
    fig.update_layout(height=350)
    st.plotly_chart(fig, width="stretch")

with col2:
    fig2 = px.pie(tiers, values="total_impressions", names="blast_tier",
                  title="Recipient Impressions by Tier")
    fig2.update_layout(height=350)
    st.plotly_chart(fig2, width="stretch")

st.dataframe(tiers, width="stretch")

# Recipient distribution
st.divider()
st.subheader("Recipient Count Distribution")
dist = _cached_recipient_distribution(start_date, end_date).to_pandas()
fig3 = px.bar(dist.head(30), x="n_recipients", y="msg_count",
              title="Messages by Number of Recipients (top 30 buckets)")
fig3.update_layout(height=350)
st.plotly_chart(fig3, width="stretch")

# CDF
fig4 = px.line(dist, x="n_recipients", y="cumulative_pct",
               title="Cumulative % of Messages by Recipient Count")
fig4.update_layout(height=350, yaxis_title="Cumulative %")
st.plotly_chart(fig4, width="stretch")

# High-blast senders
st.divider()
st.subheader("High-Blast Senders")
threshold = st.slider("Blast threshold (min recipients)", 5, 50, 10)
blasters = _cached_high_blast_senders(start_date, end_date, threshold).to_pandas()
st.write(f"**{len(blasters)} senders** have sent messages to >{threshold} recipients")
blasters_top = blasters.head(30)
ev_blasters = st.dataframe(blasters_top, width="stretch", on_select="rerun", selection_mode="single-row", key="p04_blasters_df")
handle_dataframe_person_click(ev_blasters, blasters_top, "p04_blasters_df", "from_email", start_date, end_date)

# Per-sender broadcast profile
st.divider()
st.subheader("Broadcast Profile: Top Senders")
top_bc = broadcast.head(20).to_pandas()
fig5 = px.bar(top_bc, x="from_email", y="avg_recipients",
              title="Average Recipients per Message (Top 20 by Volume)")
fig5.update_layout(height=400, xaxis_tickangle=-45)
ev_broadcast = st.plotly_chart(fig5, width="stretch", on_select="rerun", key="p04_broadcast")
handle_plotly_person_click(ev_broadcast, "p04_broadcast", start_date, end_date)

st.divider()
download_csv_button(broadcast, "broadcast_profiles.csv", "Download Broadcast Data")
