"""Page 12: Search — Look up any email address and see their comprehensive profile."""

import streamlit as st
import plotly.express as px
import polars as pl

from src.page_logger import log_page_entry, log_page_error
from src.state import (
    load_person_dim,
    render_date_filter,
    load_filtered_edge_fact, load_filtered_message_fact,
    load_filtered_graph_metrics,
)
from src.analytics.timing_analytics import compute_burstiness
from src.analytics.anomaly import detect_sender_anomalies
from src.export import download_csv_button
from src.drilldown import (
    handle_plotly_person_click, handle_dataframe_person_click,
)


# ---------------------------------------------------------------------------
# Cached analytics
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner=False, ttl=3600)
def _cached_person_burstiness(start_date, end_date, email):
    mf = load_filtered_message_fact(start_date, end_date)
    return compute_burstiness(mf.filter(pl.col("from_email") == email), top_n=1)


@st.cache_data(show_spinner=False, ttl=3600)
def _cached_sender_anomalies(start_date, end_date):
    ef = load_filtered_edge_fact(start_date, end_date)
    pd_dim = load_person_dim()
    return detect_sender_anomalies(ef, pd_dim)


# ---------------------------------------------------------------------------
# Page layout
# ---------------------------------------------------------------------------

st.set_page_config(page_title="Search", layout="wide")
_page_log = log_page_entry("12_search")
st.title("Email Address Search")

start_date, end_date = render_date_filter()

edge_fact = load_filtered_edge_fact(start_date, end_date)
message_fact = load_filtered_message_fact(start_date, end_date)
person_dim = load_person_dim()

# Pre-populate from query_params (e.g. linked from drill-down dialog)
default_query = st.query_params.get("email", "")

# Search input
query = st.text_input("Enter an email address (or partial match)", value=default_query, placeholder="e.g. bhopp@spokanecounty.org")

if not query.strip():
    st.info("Type an email address above to see their communication profile.")
    st.stop()

query_lower = query.strip().lower()

# Find matching addresses
matches = person_dim.filter(pl.col("email").str.contains(query_lower))

if len(matches) == 0:
    st.warning(f"No email addresses matching **{query}** found.")
    st.stop()

# If multiple matches, let user pick
if len(matches) > 1:
    match_list = matches.sort("total_sent", descending=True)["email"].to_list()
    if len(match_list) > 50:
        st.write(f"**{len(match_list)} matches** — showing top 50 by send volume. Try a more specific query.")
        match_list = match_list[:50]
    selected = st.selectbox("Multiple matches found — select one:", match_list)
else:
    selected = matches["email"][0]

# Get person info
person = person_dim.filter(pl.col("email") == selected)
person_row = person.row(0, named=True)

# Profile header
st.divider()
st.subheader(f"Profile: {selected}")

c1, c2, c3, c4, c5 = st.columns(5)
with c1:
    st.metric("Display Name", person_row.get("display_name") or "N/A")
with c2:
    st.metric("Messages Sent", f"{person_row['total_sent']:,}")
with c3:
    st.metric("Messages Received", f"{person_row['total_received']:,}")
with c4:
    st.metric("Domain", person_row["domain"])
with c5:
    label = "Internal" if person_row["is_internal"] else "External"
    st.metric("Type", label)

# --- Volume Timeline (sent + received per week) ---
st.divider()
st.subheader("Volume Timeline")

sent_weekly = (
    message_fact.filter(pl.col("from_email") == selected)
    .group_by("week_id")
    .agg([pl.len().alias("sent"), pl.col("timestamp").min().alias("week_start")])
)
recv_weekly = (
    edge_fact.filter(pl.col("to_email") == selected)
    .group_by("week_id")
    .agg([pl.len().alias("received"), pl.col("timestamp").min().alias("week_start")])
)

if len(sent_weekly) > 0 or len(recv_weekly) > 0:
    timeline = sent_weekly.join(recv_weekly.drop("week_start"), on="week_id", how="full", coalesce=True)
    timeline = timeline.with_columns([
        pl.col("sent").fill_null(0),
        pl.col("received").fill_null(0),
    ]).sort("week_start")
    timeline_pd = timeline.to_pandas()
    fig_timeline = px.line(timeline_pd, x="week_start", y=["sent", "received"],
                           title="Weekly Send/Receive Volume")
    fig_timeline.update_layout(height=300)
    st.plotly_chart(fig_timeline, width="stretch")

# --- Top 10 Contacts (send + receive) ---
st.divider()
st.subheader("Top 10 Contacts")

sent_to = (
    edge_fact.filter(pl.col("from_email") == selected)
    .group_by("to_email")
    .agg(pl.len().alias("sent_to"))
)
received_from = (
    edge_fact.filter(pl.col("to_email") == selected)
    .group_by("from_email")
    .agg(pl.len().alias("received_from"))
    .rename({"from_email": "to_email"})
)

contacts = sent_to.join(received_from, on="to_email", how="full", coalesce=True)
contacts = contacts.with_columns([
    pl.col("sent_to").fill_null(0),
    pl.col("received_from").fill_null(0),
    (pl.col("sent_to").fill_null(0) + pl.col("received_from").fill_null(0)).alias("total"),
]).sort("total", descending=True).rename({"to_email": "contact"})

if len(contacts) > 0:
    top_contacts = contacts.head(10).to_pandas()
    fig_contacts = px.bar(top_contacts, x="contact", y=["sent_to", "received_from"],
                          title="Top 10 Contacts (Messages Exchanged)", barmode="stack")
    fig_contacts.update_layout(height=350, xaxis_tickangle=-45)
    ev_contacts = st.plotly_chart(fig_contacts, width="stretch", on_select="rerun", key="p12_contacts")
    handle_plotly_person_click(ev_contacts, "p12_contacts", start_date, end_date, field="x")
    download_csv_button(contacts.head(50), f"contacts_{selected.split('@')[0]}.csv")

# --- Community & Co-Members ---
st.divider()
st.subheader("Community Membership")
try:
    graph_metrics = load_filtered_graph_metrics(start_date, end_date)
    person_gm = graph_metrics.filter(pl.col("email") == selected)
    if len(person_gm) > 0:
        comm_id = person_gm["community_id"][0]
        st.write(f"**Community ID:** {comm_id}")
        co_members = graph_metrics.filter(
            (pl.col("community_id") == comm_id) & (pl.col("email") != selected)
        ).sort("pagerank", descending=True)
        st.write(f"**Co-members:** {len(co_members)}")
        if len(co_members) > 0:
            co_pd = co_members.head(20).select(["email", "pagerank", "in_degree", "out_degree"]).to_pandas()
            ev_comembers = st.dataframe(co_pd, width="stretch", on_select="rerun", selection_mode="single-row", key="p12_comembers")
            handle_dataframe_person_click(ev_comembers, co_pd, "p12_comembers", "email", start_date, end_date)
    else:
        st.info("Person not found in current network graph.")
except Exception:
    st.info("Graph metrics not available for this date range.")

# --- Behavioral Metrics ---
st.divider()
st.subheader("Behavioral Metrics")

person_msgs = message_fact.filter(pl.col("from_email") == selected)

if len(person_msgs) > 0:
    col_a, col_b, col_c = st.columns(3)
    with col_a:
        ah_rate = float(person_msgs["is_after_hours"].mean())
        st.metric("After-Hours Rate", f"{ah_rate:.1%}")
    with col_b:
        we_rate = float(person_msgs["is_weekend"].mean())
        st.metric("Weekend Rate", f"{we_rate:.1%}")
    with col_c:
        # Burstiness (cached)
        burstiness = _cached_person_burstiness(start_date, end_date, selected)
        if len(burstiness) > 0:
            b_score = burstiness["burstiness"][0]
            st.metric("Burstiness", f"{b_score:.2f}")
        else:
            st.metric("Burstiness", "N/A")

# --- Anomaly Flags (cached) ---
st.divider()
st.subheader("Anomaly Flags")
try:
    anomalies = _cached_sender_anomalies(start_date, end_date)
    person_anomaly = anomalies.filter(pl.col("from_email") == selected)
    if len(person_anomaly) > 0:
        row = person_anomaly.row(0, named=True)
        st.warning("This person has been flagged as anomalous.")
        ac1, ac2, ac3, ac4 = st.columns(4)
        with ac1:
            st.metric("Volume Z-Score", f"{row['total_sent_zscore']:.1f}")
        with ac2:
            st.metric("Recipients Z-Score", f"{row['unique_recipients_zscore']:.1f}")
        with ac3:
            st.metric("After-Hours Z-Score", f"{row['after_hours_rate_zscore']:.1f}")
        with ac4:
            st.metric("Weekend Z-Score", f"{row['weekend_rate_zscore']:.1f}")
    else:
        st.success("No anomaly flags for this person.")
except Exception:
    st.info("Anomaly detection not available.")

# --- Hour of day pattern ---
st.divider()
st.subheader("Activity by Hour")
if len(person_msgs) > 0:
    hourly = (
        person_msgs.group_by("hour")
        .agg(pl.len().alias("count"))
        .sort("hour")
        .to_pandas()
    )
    fig4 = px.bar(hourly, x="hour", y="count",
                  title="Send Pattern by Hour of Day")
    fig4.update_layout(height=300, xaxis=dict(dtick=1))
    st.plotly_chart(fig4, width="stretch")
