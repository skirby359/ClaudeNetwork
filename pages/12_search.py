"""Page 12: Search — Look up any email address and see their contacts."""

import streamlit as st
import plotly.express as px
import polars as pl

from src.state import (
    load_person_dim,
    render_date_filter,
    load_filtered_edge_fact, load_filtered_message_fact,
)

st.set_page_config(page_title="Search", layout="wide")
st.title("Email Address Search")

start_date, end_date = render_date_filter()

edge_fact = load_filtered_edge_fact(start_date, end_date)
message_fact = load_filtered_message_fact(start_date, end_date)
person_dim = load_person_dim()

# Search input
query = st.text_input("Enter an email address (or partial match)", placeholder="e.g. bhopp@spokanecounty.org")

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

# --- Sent TO (who does this person email?) ---
st.divider()
st.subheader("Sent To")
st.markdown(f"People that **{selected}** sends email to.")

sent_to = (
    edge_fact.filter(pl.col("from_email") == selected)
    .group_by("to_email")
    .agg([
        pl.len().alias("msg_count"),
        pl.col("size_bytes").sum().alias("total_bytes"),
    ])
    .sort("msg_count", descending=True)
)

if len(sent_to) > 0:
    col1, col2 = st.columns([2, 1])
    with col1:
        top_sent = sent_to.head(20).to_pandas()
        fig = px.bar(top_sent, x="to_email", y="msg_count",
                     title=f"Top Recipients (of mail from {selected.split('@')[0]})")
        fig.update_layout(height=400, xaxis_tickangle=-45)
        st.plotly_chart(fig, width="stretch")
    with col2:
        st.write(f"**{len(sent_to)} unique recipients**")
        st.dataframe(sent_to.to_pandas(), width="stretch", height=400)
else:
    st.write("No outgoing messages found in selected date range.")

# --- Received FROM (who emails this person?) ---
st.divider()
st.subheader("Received From")
st.markdown(f"People who send email **to {selected}**.")

received_from = (
    edge_fact.filter(pl.col("to_email") == selected)
    .group_by("from_email")
    .agg([
        pl.len().alias("msg_count"),
        pl.col("size_bytes").sum().alias("total_bytes"),
    ])
    .sort("msg_count", descending=True)
)

if len(received_from) > 0:
    col1, col2 = st.columns([2, 1])
    with col1:
        top_recv = received_from.head(20).to_pandas()
        fig2 = px.bar(top_recv, x="from_email", y="msg_count",
                      title=f"Top Senders (to {selected.split('@')[0]})")
        fig2.update_layout(height=400, xaxis_tickangle=-45)
        st.plotly_chart(fig2, width="stretch")
    with col2:
        st.write(f"**{len(received_from)} unique senders**")
        st.dataframe(received_from.to_pandas(), width="stretch", height=400)
else:
    st.write("No incoming messages found in selected date range.")

# --- Activity over time ---
st.divider()
st.subheader("Activity Over Time")

person_msgs = message_fact.filter(pl.col("from_email") == selected)
if len(person_msgs) > 0:
    weekly = (
        person_msgs.group_by("week_id")
        .agg([
            pl.len().alias("sent_count"),
            pl.col("timestamp").min().alias("week_start"),
        ])
        .sort("week_start")
        .to_pandas()
    )
    fig3 = px.bar(weekly, x="week_start", y="sent_count",
                  title=f"Weekly Send Volume for {selected.split('@')[0]}")
    fig3.update_layout(height=300)
    st.plotly_chart(fig3, width="stretch")

# --- Hour of day pattern ---
if len(person_msgs) > 0:
    hourly = (
        person_msgs.group_by("hour")
        .agg(pl.len().alias("count"))
        .sort("hour")
        .to_pandas()
    )
    fig4 = px.bar(hourly, x="hour", y="count",
                  title=f"Send Pattern by Hour of Day")
    fig4.update_layout(height=300, xaxis=dict(dtick=1))
    st.plotly_chart(fig4, width="stretch")
