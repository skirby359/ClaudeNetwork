"""Page 9: Coordination & Churn — Community structure and activity patterns."""

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import polars as pl
import numpy as np

from src.state import (
    load_person_dim,
    render_date_filter,
    load_filtered_edge_fact, load_filtered_message_fact,
    load_filtered_graph_metrics,
)

st.set_page_config(page_title="Coordination & Churn", layout="wide")
st.title("Coordination & Churn")

start_date, end_date = render_date_filter()

edge_fact = load_filtered_edge_fact(start_date, end_date)
message_fact = load_filtered_message_fact(start_date, end_date)
person_dim = load_person_dim()
graph_metrics = load_filtered_graph_metrics(start_date, end_date)

# Community sizes
st.subheader("Community Structure")
comm_sizes = (
    graph_metrics.group_by("community_id")
    .agg([
        pl.len().alias("members"),
        pl.col("pagerank").sum().alias("total_pagerank"),
        pl.col("pagerank").mean().alias("avg_pagerank"),
    ])
    .sort("members", descending=True)
)

fig = px.bar(comm_sizes.to_pandas(), x="community_id", y="members",
             title="Community Sizes", color="avg_pagerank",
             color_continuous_scale="Viridis")
fig.update_layout(height=350)
st.plotly_chart(fig, width="stretch")

# Cross-community communication
st.divider()
st.subheader("Cross-Community Communication")
st.markdown("How much do communities talk to each other vs. within themselves?")

edge_enriched = (
    edge_fact.select(["from_email", "to_email"])
    .join(graph_metrics.select(["email", "community_id"]), left_on="from_email", right_on="email", how="left")
    .rename({"community_id": "from_community"})
    .join(graph_metrics.select(["email", "community_id"]), left_on="to_email", right_on="email", how="left")
    .rename({"community_id": "to_community"})
)

cross_comm = (
    edge_enriched.group_by(["from_community", "to_community"])
    .agg(pl.len().alias("msg_count"))
    .sort("msg_count", descending=True)
)

total_msgs = cross_comm["msg_count"].sum()
internal_msgs = cross_comm.filter(pl.col("from_community") == pl.col("to_community"))["msg_count"].sum()
cross_pct = (1 - internal_msgs / total_msgs) * 100 if total_msgs > 0 else 0

st.metric("Cross-Community Message Rate", f"{cross_pct:.1f}%")

top_communities = comm_sizes.head(10)["community_id"].to_list()
cross_filtered = cross_comm.filter(
    pl.col("from_community").is_in(top_communities) & pl.col("to_community").is_in(top_communities)
)

n_comm = len(top_communities)
matrix = np.zeros((n_comm, n_comm))
comm_idx = {c: i for i, c in enumerate(top_communities)}
for row in cross_filtered.iter_rows(named=True):
    fc = row["from_community"]
    tc = row["to_community"]
    if fc in comm_idx and tc in comm_idx:
        matrix[comm_idx[fc]][comm_idx[tc]] = row["msg_count"]

fig2 = go.Figure(data=go.Heatmap(
    z=matrix, x=[str(c) for c in top_communities], y=[str(c) for c in top_communities],
    colorscale="Blues",
    hovertemplate="From Community %{y} -> To Community %{x}: %{z} messages<extra></extra>",
))
fig2.update_layout(height=400, title="Inter-Community Message Flow (Top 10 Communities)",
                   xaxis_title="To Community", yaxis_title="From Community")
st.plotly_chart(fig2, width="stretch")

# Activity churn
st.divider()
st.subheader("Sender Activity Churn")
st.markdown("How many unique senders are active each week?")

weekly_senders = (
    message_fact.group_by("week_id")
    .agg([
        pl.col("from_email").n_unique().alias("active_senders"),
        pl.col("timestamp").min().alias("week_start"),
    ])
    .sort("week_start")
    .to_pandas()
)

fig3 = px.line(weekly_senders, x="week_start", y="active_senders",
               title="Unique Active Senders per Week")
fig3.update_layout(height=350)
st.plotly_chart(fig3, width="stretch")

# Community member list
st.divider()
st.subheader("Community Members")
selected_community = st.selectbox(
    "Select Community",
    sorted(graph_metrics["community_id"].unique().to_list()),
)
members = (
    graph_metrics.filter(pl.col("community_id") == selected_community)
    .join(person_dim.select(["email", "display_name"]), on="email", how="left")
    .sort("pagerank", descending=True)
    .to_pandas()
)
st.write(f"**{len(members)} members** in Community {selected_community}")
st.dataframe(members, width="stretch")
