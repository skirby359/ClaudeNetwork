"""Page 1: Executive Summary — Key findings at a glance."""

import streamlit as st
import plotly.express as px

from src.state import (
    load_person_dim,
    render_date_filter,
    load_filtered_message_fact, load_filtered_edge_fact,
    load_filtered_weekly_agg, load_filtered_broadcast,
)
from src.analytics.volume import compute_sender_concentration

st.set_page_config(page_title="Executive Summary", layout="wide")
st.title("Executive Summary")

start_date, end_date = render_date_filter()

message_fact = load_filtered_message_fact(start_date, end_date)
edge_fact = load_filtered_edge_fact(start_date, end_date)
person_dim = load_person_dim()
weekly_agg = load_filtered_weekly_agg(start_date, end_date)
broadcast = load_filtered_broadcast(start_date, end_date)

# Top-level KPIs
c1, c2, c3, c4, c5 = st.columns(5)

with c1:
    st.metric("Total Messages", f"{len(message_fact):,}")
with c2:
    st.metric("Unique People", f"{len(person_dim):,}")
with c3:
    internal = person_dim.filter(person_dim["is_internal"])
    st.metric("Internal", f"{len(internal):,}")
with c4:
    total_bytes = message_fact["size_bytes"].sum()
    st.metric("Total Data", f"{total_bytes / (1024**3):.1f} GB")
with c5:
    avg_recip = message_fact["n_recipients"].mean()
    st.metric("Avg Recipients/Msg", f"{avg_recip:.1f}")

st.divider()

# Volume trend
col_left, col_right = st.columns(2)

with col_left:
    st.subheader("Weekly Message Volume")
    wa = weekly_agg.to_pandas()
    fig = px.line(wa, x="week_start", y="msg_count", title="Messages per Week")
    fig.update_layout(height=350)
    st.plotly_chart(fig, width="stretch")

with col_right:
    st.subheader("Sender Concentration")
    conc = compute_sender_concentration(edge_fact)
    st.write(f"**Gini coefficient:** {conc['gini']:.3f}")
    st.write(f"**Top 5 senders:** {conc['top_5_share']:.1%} of all messages")
    st.write(f"**Top 10 senders:** {conc['top_10_share']:.1%} of all messages")
    st.write(f"**Top 20 senders:** {conc['top_20_share']:.1%} of all messages")

    top_senders = conc["top_senders"].head(10).to_pandas()
    fig2 = px.bar(top_senders, x="from_email", y="count", title="Top 10 Senders")
    fig2.update_layout(height=350, xaxis_tickangle=-45)
    st.plotly_chart(fig2, width="stretch")

st.divider()

# Key insights
st.subheader("Key Findings")
col_a, col_b = st.columns(2)

with col_a:
    ah_rate = message_fact["is_after_hours"].mean()
    we_rate = message_fact["is_weekend"].mean()
    st.info(f"**After-hours rate:** {ah_rate:.1%} of messages sent outside business hours")
    st.info(f"**Weekend rate:** {we_rate:.1%} of messages sent on weekends")

with col_b:
    n_blast = len(message_fact.filter(message_fact["n_recipients"] > 10))
    st.info(f"**Large blasts (>10 recipients):** {n_blast:,} messages ({n_blast/len(message_fact):.1%})")
    top_blaster = broadcast.head(1)
    if len(top_blaster) > 0:
        st.info(f"**Top broadcaster:** {top_blaster['from_email'][0]} ({top_blaster['total_msgs'][0]:,} msgs)")
