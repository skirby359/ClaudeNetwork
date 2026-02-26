"""Page 5: Artifact vs Ping — Message size and purpose patterns."""

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import polars as pl

from src.state import (
    render_date_filter, load_filtered_message_fact,
)

st.set_page_config(page_title="Artifact vs Ping", layout="wide")
st.title("Artifact vs Ping")

st.markdown("""
**Artifacts** are large, content-rich messages (reports, documents, formal communication).
**Pings** are small, quick messages (acknowledgments, short replies, coordination).

We classify by message size to distinguish these patterns.
""")

start_date, end_date = render_date_filter()

message_fact = load_filtered_message_fact(start_date, end_date)

# Size distribution
st.subheader("Message Size Distribution")
size_kb = (message_fact["size_bytes"] / 1024).to_numpy()

fig = go.Figure(data=[go.Histogram(x=size_kb, nbinsx=100, name="Message Size")])
fig.update_layout(height=400, title="Distribution of Message Sizes",
                  xaxis_title="Size (KB)", yaxis_title="Count",
                  xaxis=dict(range=[0, 200]))
st.plotly_chart(fig, width="stretch")

# Classify messages
ping_threshold = st.slider("Ping threshold (KB)", 1, 20, 5)
artifact_threshold = st.slider("Artifact threshold (KB)", 20, 200, 50)

classified = message_fact.with_columns([
    pl.when(pl.col("size_bytes") < ping_threshold * 1024).then(pl.lit("ping"))
    .when(pl.col("size_bytes") >= artifact_threshold * 1024).then(pl.lit("artifact"))
    .otherwise(pl.lit("standard"))
    .alias("msg_type"),
])

# Breakdown
st.subheader("Message Type Breakdown")
type_counts = classified.group_by("msg_type").agg(pl.len().alias("count")).to_pandas()

col1, col2 = st.columns(2)
with col1:
    fig2 = px.pie(type_counts, values="count", names="msg_type", title="Messages by Type")
    fig2.update_layout(height=350)
    st.plotly_chart(fig2, width="stretch")

with col2:
    type_by_sender = (
        classified.group_by(["from_email", "msg_type"])
        .agg(pl.len().alias("count"))
        .sort("count", descending=True)
    )
    ping_senders = type_by_sender.filter(pl.col("msg_type") == "ping").head(10).to_pandas()
    fig3 = px.bar(ping_senders, x="from_email", y="count", title="Top Ping Senders")
    fig3.update_layout(height=350, xaxis_tickangle=-45)
    st.plotly_chart(fig3, width="stretch")

# Size vs Recipients scatter
st.divider()
st.subheader("Message Size vs Recipient Count")
sample = message_fact.sample(n=min(5000, len(message_fact)), seed=42)
sd = sample.to_pandas()
fig4 = px.scatter(sd, x="n_recipients", y="size_bytes",
                  opacity=0.3, title="Size vs Recipients (sampled)")
fig4.update_layout(height=400, yaxis_title="Size (bytes)", xaxis_title="Number of Recipients")
st.plotly_chart(fig4, width="stretch")

# Average size by hour
st.divider()
st.subheader("Average Message Size by Hour")
hourly_size = (
    message_fact.group_by("hour")
    .agg(pl.col("size_bytes").mean().alias("avg_size"))
    .sort("hour")
    .to_pandas()
)
fig5 = px.bar(hourly_size, x="hour", y="avg_size", title="Average Message Size by Hour")
fig5.update_layout(height=350, xaxis=dict(dtick=1), yaxis_title="Avg Size (bytes)")
st.plotly_chart(fig5, width="stretch")
