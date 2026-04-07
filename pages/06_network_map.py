"""Page 6: Network Map — Interactive communication network visualization."""

import streamlit as st
import plotly.graph_objects as go
import polars as pl
import colorsys

from pyvis.network import Network

from src.page_logger import log_page_entry, log_page_error
from src.state import (
    load_person_dim,
    render_date_filter,
    load_filtered_edge_fact, load_filtered_graph_metrics,
)
from src.export import download_csv_button
from src.drilldown import handle_dataframe_person_click

st.set_page_config(page_title="Network Map", layout="wide")
_page_log = log_page_entry("06_network_map")
st.title("Network Map")

start_date, end_date = render_date_filter()

edge_fact = load_filtered_edge_fact(start_date, end_date)

if len(edge_fact) == 0:
    st.warning("No data in selected date range.")
    st.stop()

person_dim = load_person_dim()
graph_metrics = load_filtered_graph_metrics(start_date, end_date)

# Community counts — computed once and reused below for coloring
all_comm_counts = graph_metrics.group_by("community_id").agg(pl.len().alias("n"))
valid_communities = set(
    all_comm_counts.filter(pl.col("n") > 2)["community_id"].to_list()
)

# Community size bounds for the slider
comm_sizes = all_comm_counts.filter(pl.col("n") > 2)["n"]
min_comm_size = int(comm_sizes.min()) if len(comm_sizes) > 0 else 3
max_comm_size = int(comm_sizes.max()) if len(comm_sizes) > 0 else 3

# Summary stats
st.subheader("Network Overview")
c1, c2, c3, c4 = st.columns(4)
with c1:
    st.metric("Nodes (People)", f"{len(graph_metrics):,}")
with c2:
    n_edges = len(edge_fact.select(["from_email", "to_email"]).unique())
    st.metric("Unique Edges", f"{n_edges:,}")
with c3:
    st.metric("Communities (>2)", f"{len(valid_communities)}")
with c4:
    avg_degree = (graph_metrics["in_degree"] + graph_metrics["out_degree"]).mean()
    st.metric("Avg Degree", f"{avg_degree:.1f}")

# Sidebar filters
with st.sidebar:
    st.subheader("Network Filters")

    # Community size range slider
    size_range = st.slider(
        "Community size",
        min_value=min_comm_size,
        max_value=max_comm_size,
        value=(min_comm_size, max_comm_size),
        help="Show only communities within this member-count range",
    )

    # Determine which communities pass the size filter
    selected_communities = set(
        all_comm_counts.filter(
            (pl.col("n") >= size_range[0]) & (pl.col("n") <= size_range[1])
        )["community_id"].to_list()
    ) & valid_communities

    st.caption(f"{len(selected_communities)} communities in range")

    # Search/highlight input
    search_query = st.text_input("Search & highlight node", placeholder="e.g. bhopp")

# Interactive PyVis network (top N nodes for performance)
st.divider()
st.subheader("Interactive Network (Top Communicators)")
top_n = st.slider("Number of people to show", 20, 200, 50)

# Filter graph_metrics by selected communities
filtered_metrics = graph_metrics.filter(
    pl.col("community_id").is_in(list(selected_communities))
) if selected_communities else graph_metrics

top_people = (
    filtered_metrics.with_columns(
        (pl.col("in_degree") + pl.col("out_degree")).alias("total_degree")
    )
    .sort("total_degree", descending=True)
    .head(top_n)
)
top_emails = set(top_people["email"].to_list())

edge_pairs = (
    edge_fact.group_by(["from_email", "to_email"])
    .agg(pl.len().alias("weight"))
    .filter(
        pl.col("from_email").is_in(list(top_emails))
        & pl.col("to_email").is_in(list(top_emails))
    )
)

net = Network(height="600px", width="100%", bgcolor="#222222", font_color="white",
              directed=True)
net.barnes_hut(gravity=-3000, central_gravity=0.3, spring_length=100)

# Color communities with >2 members; lump the rest as "other"
community_ids = [c for c in top_people["community_id"].unique().to_list() if c in selected_communities]
colors = {}
for i, cid in enumerate(community_ids):
    hue = i / max(len(community_ids), 1)
    r, g, b = colorsys.hsv_to_rgb(hue, 0.7, 0.9)
    colors[cid] = f"rgb({int(r*255)},{int(g*255)},{int(b*255)})"

search_lower = search_query.strip().lower() if search_query else ""

for row in top_people.iter_rows(named=True):
    email = row["email"]
    size = max(5, min(50, row["total_degree"] / 10))
    color = colors.get(row["community_id"], "#999999")
    label = email.split("@")[0]

    # Search highlight: matched nodes get gold color + larger size
    if search_lower and search_lower in email.lower():
        color = "#FFD700"  # gold
        size = max(size, 30)

    net.add_node(email, label=label, size=size, color=color,
                 title=f"{email}\nDegree: {row['total_degree']}\nCommunity: {row['community_id']}")

for row in edge_pairs.iter_rows(named=True):
    weight = row["weight"]
    net.add_edge(row["from_email"], row["to_email"],
                 value=min(weight, 20), title=f"{weight} messages")

html = net.generate_html()
st.components.v1.html(html, height=620, scrolling=True)

# Community breakdown — only communities with >2 members
st.divider()
st.subheader("Community Breakdown")
comm_stats = (
    graph_metrics.group_by("community_id")
    .agg([
        pl.len().alias("members"),
        pl.col("pagerank").sum().alias("total_pagerank"),
    ])
    .filter(pl.col("members") > 2)
    .sort("members", descending=True)
)
st.write(f"Showing **{len(comm_stats)}** communities with more than 2 members")
comm_stats_pd = comm_stats.to_pandas()
st.dataframe(comm_stats_pd, width="stretch")
download_csv_button(comm_stats, "community_breakdown.csv")
