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
from src.anonymize import anon_df
from src.drilldown import handle_plotly_person_click, handle_dataframe_person_click

st.set_page_config(page_title="Bottlenecks & Routing", layout="wide")
st.title("Bottlenecks & Routing")

st.markdown("""
**How to read this page:**
- **Information Spreaders**: People who send to many others. They distribute information widely.
- **Cross-Group Connectors**: People who bridge different groups. If they leave, groups lose contact.
- **Importance Score**: How central someone is — not just by volume, but by who communicates with them.
- **Within-Group Focus**: Active people who mostly communicate within their own group.
""")

start_date, end_date = render_date_filter()

person_dim = load_person_dim()
edge_fact = load_filtered_edge_fact(start_date, end_date)

if len(edge_fact) == 0:
    st.warning("No data in selected date range.")
    st.stop()

graph_metrics = load_filtered_graph_metrics(start_date, end_date)

# Use global nonhuman filter
filter_on = st.session_state.get("exclude_nonhuman", True)
if filter_on:
    nonhuman_emails = load_nonhuman_emails(start_date, end_date)
    graph_metrics = graph_metrics.filter(~pl.col("email").is_in(list(nonhuman_emails)))

# Join graph metrics with person_dim for names
enriched = graph_metrics.join(
    person_dim.select(["email", "display_name", "is_internal"]),
    on="email",
    how="left",
)

# Use community labels if available
comm_col = "community_label" if "community_label" in enriched.columns else "community_id"

# Information Spreaders
st.subheader("Information Spreaders")
st.caption("People who send to the most others. High outgoing connection volume.")
hubs = anon_df(enriched.sort("out_degree", descending=True).head(20))
fig = px.bar(hubs.to_pandas(), x="email", y="out_degree",
             title="Top 20 Information Spreaders",
             labels={"email": "Person", "out_degree": "Outgoing Message Volume"})
fig.update_layout(height=400, xaxis_tickangle=-45)
ev_hubs = st.plotly_chart(fig, width="stretch", on_select="rerun", key="p07_hubs")
handle_plotly_person_click(ev_hubs, "p07_hubs", start_date, end_date)

# Cross-Group Connectors
st.divider()
st.subheader("Cross-Group Connectors")
st.caption(
    "People who bridge different communication groups. "
    "Removing them would isolate parts of the organization."
)
brokers = anon_df(enriched.sort("betweenness_centrality", descending=True).head(20))
fig2 = px.bar(brokers.to_pandas(), x="email", y="betweenness_centrality",
              title="Top 20 Cross-Group Connectors",
              labels={"email": "Person", "betweenness_centrality": "Connector Score"},
              color_discrete_sequence=["#e15759"])
fig2.update_layout(height=400, xaxis_tickangle=-45)
ev_brokers = st.plotly_chart(fig2, width="stretch", on_select="rerun", key="p07_brokers")
handle_plotly_person_click(ev_brokers, "p07_brokers", start_date, end_date)

# Importance Score
st.divider()
st.subheader("Importance Score")
st.caption(
    "Ranks people not just by how many messages they get, but by how important "
    "the people who contact them are. A person emailed by many connectors ranks higher."
)
pr_top = anon_df(enriched.sort("pagerank", descending=True).head(20))
fig3 = px.bar(pr_top.to_pandas(), x="email", y="pagerank",
              title="Top 20 by Importance Score",
              labels={"email": "Person", "pagerank": "Importance Score"})
fig3.update_layout(height=400, xaxis_tickangle=-45)
ev_pagerank = st.plotly_chart(fig3, width="stretch", on_select="rerun", key="p07_pagerank")
handle_plotly_person_click(ev_pagerank, "p07_pagerank", start_date, end_date)

# Within-Group Focus
st.divider()
st.subheader("Within-Group Focus")
st.caption(
    "Active people with low cross-group connectivity. "
    "They communicate frequently but mostly within their own group."
)

silo_candidates = enriched.filter(
    (pl.col("in_degree") + pl.col("out_degree") > enriched["in_degree"].median())
    & (pl.col("betweenness_centrality") < enriched["betweenness_centrality"].quantile(0.25))
)
st.write(f"**{len(silo_candidates)} people** identified with within-group focus")
silos_display = anon_df(
    silo_candidates.select(["email", "display_name", comm_col, "in_degree",
                            "out_degree", "betweenness_centrality"])
    .sort("in_degree", descending=True)
    .head(30)
)
silos_pd = silos_display.to_pandas()
ev_silos = st.dataframe(silos_pd, width="stretch", on_select="rerun", selection_mode="single-row", key="p07_silos_df")
handle_dataframe_person_click(ev_silos, silos_pd, "p07_silos_df", "email", start_date, end_date)

# Full metrics table with renamed columns
st.divider()
st.subheader("All People Metrics")
metrics_display = anon_df(
    enriched.select([
        pl.col("email"),
        pl.col("display_name").alias("Name"),
        pl.col("in_degree").alias("Incoming Volume"),
        pl.col("out_degree").alias("Outgoing Volume"),
        pl.col("betweenness_centrality").alias("Connector Score"),
        pl.col("pagerank").alias("Importance Score"),
        pl.col(comm_col).alias("Group"),
    ])
    .sort("Importance Score", descending=True)
)
metrics_pd = metrics_display.to_pandas()
ev_metrics = st.dataframe(metrics_pd, width="stretch", on_select="rerun", selection_mode="single-row", key="p07_metrics_df")
handle_dataframe_person_click(ev_metrics, metrics_pd, "p07_metrics_df", "email", start_date, end_date)
download_csv_button(metrics_display, "people_metrics.csv")
