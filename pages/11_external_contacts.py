"""Page 11: External Contacts — Top external addresses by send and receive volume."""

import streamlit as st
import plotly.express as px
import polars as pl

from src.state import (
    load_person_dim,
    render_date_filter, load_filtered_edge_fact,
)

st.set_page_config(page_title="External Contacts", layout="wide")
st.title("External Contacts")

st.markdown("Identifies the most active **external** email addresses — people outside "
            "the internal domain(s) who send to or receive from the organization.")

start_date, end_date = render_date_filter()

edge_fact = load_filtered_edge_fact(start_date, end_date)
person_dim = load_person_dim()

external = person_dim.filter(~pl.col("is_internal"))
external_emails = external["email"].to_list()

# KPIs
total_people = len(person_dim)
n_external = len(external)
c1, c2, c3 = st.columns(3)
with c1:
    st.metric("External Addresses", f"{n_external:,}")
with c2:
    st.metric("Internal Addresses", f"{total_people - n_external:,}")
with c3:
    st.metric("External %", f"{n_external / max(total_people, 1) * 100:.1f}%")

st.divider()

# --- Top external SENDERS (external → internal) ---
st.subheader("Top External Senders")
st.markdown("External addresses that send the most email **into** the organization.")

external_senders = (
    edge_fact.filter(pl.col("from_email").is_in(external_emails))
    .group_by("from_email")
    .agg([
        pl.len().alias("msgs_sent"),
        pl.col("to_email").n_unique().alias("unique_recipients"),
        pl.col("size_bytes").sum().alias("total_bytes"),
    ])
    .sort("msgs_sent", descending=True)
)

top_n = st.slider("Number to show", 10, 100, 30, key="ext_sender_n")
top_ext_senders = external_senders.head(top_n).to_pandas()

fig = px.bar(top_ext_senders, x="from_email", y="msgs_sent",
             hover_data=["unique_recipients", "total_bytes"],
             title=f"Top {top_n} External Senders")
fig.update_layout(height=450, xaxis_tickangle=-45)
st.plotly_chart(fig, width="stretch")
st.dataframe(top_ext_senders, width="stretch")

# --- Top external RECEIVERS (internal → external) ---
st.divider()
st.subheader("Top External Receivers")
st.markdown("External addresses that receive the most email **from** the organization.")

external_receivers = (
    edge_fact.filter(pl.col("to_email").is_in(external_emails))
    .group_by("to_email")
    .agg([
        pl.len().alias("msgs_received"),
        pl.col("from_email").n_unique().alias("unique_senders"),
        pl.col("size_bytes").sum().alias("total_bytes"),
    ])
    .sort("msgs_received", descending=True)
)

top_ext_receivers = external_receivers.head(top_n).to_pandas()

fig2 = px.bar(top_ext_receivers, x="to_email", y="msgs_received",
              hover_data=["unique_senders", "total_bytes"],
              title=f"Top {top_n} External Receivers")
fig2.update_layout(height=450, xaxis_tickangle=-45)
st.plotly_chart(fig2, width="stretch")
st.dataframe(top_ext_receivers, width="stretch")

# --- Top external domains ---
st.divider()
st.subheader("Top External Domains")

external_with_domain = external.select(["email", "domain"])

# Sent by domain
domain_sent = (
    edge_fact.filter(pl.col("from_email").is_in(external_emails))
    .join(external_with_domain, left_on="from_email", right_on="email", how="left")
    .group_by("domain")
    .agg(pl.len().alias("msgs_sent"))
    .sort("msgs_sent", descending=True)
)

# Received by domain
domain_recv = (
    edge_fact.filter(pl.col("to_email").is_in(external_emails))
    .join(external_with_domain, left_on="to_email", right_on="email", how="left")
    .group_by("domain")
    .agg(pl.len().alias("msgs_received"))
    .sort("msgs_received", descending=True)
)

domain_stats = domain_sent.join(domain_recv, on="domain", how="full", coalesce=True)
domain_stats = domain_stats.with_columns([
    pl.col("msgs_sent").fill_null(0),
    pl.col("msgs_received").fill_null(0),
])
domain_stats = domain_stats.with_columns(
    (pl.col("msgs_sent") + pl.col("msgs_received")).alias("total")
).sort("total", descending=True)

col1, col2 = st.columns(2)
with col1:
    ds = domain_stats.head(20).to_pandas()
    fig3 = px.bar(ds, x="domain", y=["msgs_sent", "msgs_received"],
                  title="Top 20 External Domains", barmode="group")
    fig3.update_layout(height=400, xaxis_tickangle=-45)
    st.plotly_chart(fig3, width="stretch")

with col2:
    st.dataframe(domain_stats.head(30).to_pandas(), width="stretch")
