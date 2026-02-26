"""Page 7: Bottlenecks & Routing — Who are the critical connectors?"""

import streamlit as st
import plotly.express as px
import polars as pl

from src.state import (
    load_person_dim,
    render_date_filter,
    load_filtered_graph_metrics,
)

st.set_page_config(page_title="Bottlenecks & Routing", layout="wide")
st.title("Bottlenecks & Routing")

start_date, end_date = render_date_filter()

person_dim = load_person_dim()
graph_metrics = load_filtered_graph_metrics(start_date, end_date)

# Join graph metrics with person_dim for names
enriched = graph_metrics.join(
    person_dim.select(["email", "display_name", "is_internal"]),
    on="email",
    how="left",
)

# Hub detection
st.subheader("Hubs — People Who Send to Many")
st.markdown("High **out-degree** nodes: people who distribute information widely.")
hubs = enriched.sort("out_degree", descending=True).head(20).to_pandas()
fig = px.bar(hubs, x="email", y="out_degree", color="community_id",
             title="Top 20 Hubs (by Weighted Out-Degree)")
fig.update_layout(height=400, xaxis_tickangle=-45)
st.plotly_chart(fig, width="stretch")

# Broker detection
st.divider()
st.subheader("Brokers — Information Gatekeepers")
st.markdown("High **betweenness centrality** nodes: people who bridge different groups.")
brokers = enriched.sort("betweenness_centrality", descending=True).head(20).to_pandas()
fig2 = px.bar(brokers, x="email", y="betweenness_centrality", color="community_id",
              title="Top 20 Brokers (by Betweenness Centrality)")
fig2.update_layout(height=400, xaxis_tickangle=-45)
st.plotly_chart(fig2, width="stretch")

# PageRank
st.divider()
st.subheader("PageRank — Overall Importance")
st.markdown("**PageRank** considers not just how many people email you, but how important those people are.")
pr_top = enriched.sort("pagerank", descending=True).head(20).to_pandas()
fig3 = px.bar(pr_top, x="email", y="pagerank", color="community_id",
              title="Top 20 by PageRank")
fig3.update_layout(height=400, xaxis_tickangle=-45)
st.plotly_chart(fig3, width="stretch")

# Silo detection
st.divider()
st.subheader("Potential Silos — Isolated Groups")
st.markdown("People with **high degree** but **low betweenness** mostly communicate within their group.")

silo_candidates = enriched.filter(
    (pl.col("in_degree") + pl.col("out_degree") > enriched["in_degree"].median())
    & (pl.col("betweenness_centrality") < enriched["betweenness_centrality"].quantile(0.25))
)
st.write(f"**{len(silo_candidates)} potential silo members** identified")
st.dataframe(
    silo_candidates.select(["email", "display_name", "community_id", "in_degree",
                            "out_degree", "betweenness_centrality"])
    .sort("in_degree", descending=True)
    .head(30)
    .to_pandas(),
    width="stretch",
)

# Full metrics table
st.divider()
st.subheader("All Graph Metrics")
st.dataframe(
    enriched.select(["email", "display_name", "in_degree", "out_degree",
                     "betweenness_centrality", "pagerank", "community_id"])
    .sort("pagerank", descending=True)
    .to_pandas(),
    width="stretch",
)
