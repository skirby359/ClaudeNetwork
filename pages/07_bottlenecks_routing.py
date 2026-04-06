"""Page 7: Bottlenecks & Routing — Who are the critical connectors?"""

import streamlit as st
import plotly.express as px
import polars as pl

from src.state import (
    load_person_dim,
    render_date_filter,
    load_filtered_edge_fact, load_filtered_graph_metrics,
    load_nonhuman_emails,
)
from src.export import download_csv_button
from src.drilldown import handle_plotly_person_click, handle_dataframe_person_click

st.set_page_config(page_title="Bottlenecks & Routing", layout="wide")
st.title("Bottlenecks & Routing")

start_date, end_date = render_date_filter()

person_dim = load_person_dim()
edge_fact = load_filtered_edge_fact(start_date, end_date)

if len(edge_fact) == 0:
    st.warning("No data in selected date range.")
    st.stop()

graph_metrics = load_filtered_graph_metrics(start_date, end_date)

# --- Nonhuman filter (cached) ---
nonhuman_emails = load_nonhuman_emails(start_date, end_date)

filter_on = st.checkbox("Hide nonhuman addresses (copiers, bots, system accounts)", value=True)

if filter_on:
    graph_metrics = graph_metrics.filter(~pl.col("email").is_in(list(nonhuman_emails)))
    st.caption(f"Filtered out {len(nonhuman_emails)} nonhuman addresses.")

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
ev_hubs = st.plotly_chart(fig, width="stretch", on_select="rerun", key="p07_hubs")
handle_plotly_person_click(ev_hubs, "p07_hubs", start_date, end_date)

# Broker detection
st.divider()
st.subheader("Brokers — Information Gatekeepers")
st.markdown("High **betweenness centrality** nodes: people who bridge different groups.")
brokers = enriched.sort("betweenness_centrality", descending=True).head(20).to_pandas()
fig2 = px.bar(brokers, x="email", y="betweenness_centrality", color="community_id",
              title="Top 20 Brokers (by Betweenness Centrality)")
fig2.update_layout(height=400, xaxis_tickangle=-45)
ev_brokers = st.plotly_chart(fig2, width="stretch", on_select="rerun", key="p07_brokers")
handle_plotly_person_click(ev_brokers, "p07_brokers", start_date, end_date)

# PageRank
st.divider()
st.subheader("PageRank — Overall Importance")
st.markdown("**PageRank** considers not just how many people email you, but how important those people are.")
pr_top = enriched.sort("pagerank", descending=True).head(20).to_pandas()
fig3 = px.bar(pr_top, x="email", y="pagerank", color="community_id",
              title="Top 20 by PageRank")
fig3.update_layout(height=400, xaxis_tickangle=-45)
ev_pagerank = st.plotly_chart(fig3, width="stretch", on_select="rerun", key="p07_pagerank")
handle_plotly_person_click(ev_pagerank, "p07_pagerank", start_date, end_date)

# Silo detection
st.divider()
st.subheader("Potential Silos — Isolated Groups")
st.markdown("People with **high degree** but **low betweenness** mostly communicate within their group.")

silo_candidates = enriched.filter(
    (pl.col("in_degree") + pl.col("out_degree") > enriched["in_degree"].median())
    & (pl.col("betweenness_centrality") < enriched["betweenness_centrality"].quantile(0.25))
)
st.write(f"**{len(silo_candidates)} potential silo members** identified")
silos_pd = (
    silo_candidates.select(["email", "display_name", "community_id", "in_degree",
                            "out_degree", "betweenness_centrality"])
    .sort("in_degree", descending=True)
    .head(30)
    .to_pandas()
)
ev_silos = st.dataframe(silos_pd, width="stretch", on_select="rerun", selection_mode="single-row", key="p07_silos_df")
handle_dataframe_person_click(ev_silos, silos_pd, "p07_silos_df", "email", start_date, end_date)

# Full metrics table
st.divider()
st.subheader("All Graph Metrics")
metrics_pd = (
    enriched.select(["email", "display_name", "in_degree", "out_degree",
                     "betweenness_centrality", "pagerank", "community_id"])
    .sort("pagerank", descending=True)
    .to_pandas()
)
ev_metrics = st.dataframe(metrics_pd, width="stretch", on_select="rerun", selection_mode="single-row", key="p07_metrics_df")
handle_dataframe_person_click(ev_metrics, metrics_pd, "p07_metrics_df", "email", start_date, end_date)
download_csv_button(
    enriched.select(["email", "display_name", "in_degree", "out_degree",
                     "betweenness_centrality", "pagerank", "community_id"])
    .sort("pagerank", descending=True),
    "graph_metrics.csv",
)
