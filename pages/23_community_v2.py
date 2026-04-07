"""Page 23: Community Detection v2 — Leiden multi-resolution analysis."""

import streamlit as st
import plotly.express as px
import polars as pl

from src.state import (
    render_date_filter, load_filtered_edge_fact,
    load_nonhuman_emails, load_person_dim,
)
from src.analytics.network import build_graph
from src.analytics.community_leiden import (
    detect_leiden_communities, build_hierarchy_nesting,
    compare_louvain_leiden, HAS_LEIDEN,
)
from src.export import download_csv_button


@st.cache_data(show_spinner="Running Leiden community detection...", ttl=3600)
def _cached_leiden(start_date, end_date):
    ef = load_filtered_edge_fact(start_date, end_date)
    nonhuman = load_nonhuman_emails(start_date, end_date)
    G = build_graph(ef)
    leiden_df = detect_leiden_communities(G, exclude_emails=set(nonhuman))
    comparison = compare_louvain_leiden(G, exclude_emails=set(nonhuman))
    nesting = build_hierarchy_nesting(leiden_df)
    return leiden_df, comparison, nesting


st.set_page_config(page_title="Community Detection v2", layout="wide")
st.title("Community Detection v2 (Leiden)")
st.caption(
    "Multi-resolution community detection with type classification and hierarchy nesting."
    + ("" if HAS_LEIDEN else " **Note:** leidenalg not installed — using Louvain fallback.")
)

start_date, end_date = render_date_filter()

leiden_df, comparison, nesting = _cached_leiden(start_date, end_date)

if len(leiden_df) == 0:
    st.warning("No data in selected date range.")
    st.stop()

# --- Nonhuman filter ---
nonhuman_emails = load_nonhuman_emails(start_date, end_date)
filter_nonhuman = st.session_state.get("exclude_nonhuman", True)
if filter_nonhuman and nonhuman_emails:
    leiden_df = leiden_df.filter(~pl.col("email").is_in(list(nonhuman_emails)))

# --- Section 1: Algorithm Comparison ---
st.divider()
st.subheader("Algorithm Comparison")

cols = st.columns(4)
with cols[0]:
    st.metric("Louvain Communities", comparison["louvain_communities"])
with cols[1]:
    st.metric("Louvain Modularity", comparison["louvain_modularity"])
with cols[2]:
    st.metric("Leiden Communities", comparison.get("leiden_communities", "N/A"))
with cols[3]:
    st.metric("Leiden Modularity", comparison.get("leiden_modularity", "N/A"))

# --- Section 2: Multi-Resolution View ---
st.divider()
st.subheader("Multi-Resolution Communities")
st.caption("Same network analyzed at three granularity levels.")

res_cols = st.columns(3)
for col, (res_name, col_name) in zip(res_cols, [
    ("Coarse", "community_coarse"),
    ("Medium", "community_medium"),
    ("Fine", "community_fine"),
]):
    with col:
        comm_counts = (
            leiden_df.group_by(col_name)
            .agg(pl.len().alias("members"))
            .sort("members", descending=True)
        )
        n_comms = len(comm_counts)
        st.metric(f"{res_name} — Communities", n_comms)

        label_col = f"community_label_{res_name.lower()}"
        label_counts = (
            leiden_df.group_by(label_col)
            .agg(pl.len().alias("members"))
            .sort("members", descending=True)
            .head(15)
        )
        fig = px.bar(
            label_counts.to_pandas(),
            x="members", y=label_col, orientation="h",
            title=f"{res_name} Resolution",
            labels={"members": "Members", label_col: "Community"},
        )
        fig.update_layout(height=400, yaxis=dict(autorange="reversed"))
        st.plotly_chart(fig, use_container_width=True)

# --- Section 3: Community Types ---
st.divider()
st.subheader("Community Type Classification")
st.caption("Communities classified by size and domain composition at medium resolution.")

type_counts = (
    leiden_df.group_by("community_type")
    .agg([
        pl.col("community_medium").n_unique().alias("n_communities"),
        pl.len().alias("total_members"),
    ])
    .sort("total_members", descending=True)
)

col_a, col_b = st.columns([1, 2])
with col_a:
    fig_pie = px.pie(
        type_counts.to_pandas(),
        values="n_communities", names="community_type",
        title="Community Types (by count)",
        color_discrete_sequence=px.colors.qualitative.Set2,
    )
    fig_pie.update_layout(height=350)
    st.plotly_chart(fig_pie, use_container_width=True)

with col_b:
    st.dataframe(type_counts.to_pandas(), use_container_width=True)
    st.markdown("""
    - **pair**: 2-3 members
    - **team**: 4-15 members
    - **department**: >15 members, dominated by one domain
    - **cross-functional**: >15 members, mixed domains
    """)

# --- Section 4: Hierarchy Nesting ---
st.divider()
st.subheader("Community Hierarchy Nesting")
st.caption("How fine-grained communities nest inside coarser ones.")

if len(nesting) > 0:
    # Treemap: coarse -> medium -> fine
    nesting_display = nesting.with_columns([
        pl.lit("All").alias("root"),
        pl.col("community_coarse").cast(pl.Utf8).alias("coarse_str"),
        pl.col("community_medium").cast(pl.Utf8).alias("medium_str"),
        pl.col("community_fine").cast(pl.Utf8).alias("fine_str"),
    ])
    fig_tree = px.treemap(
        nesting_display.to_pandas(),
        path=["root", "coarse_str", "medium_str", "fine_str"],
        values="n_members",
        title="Community Nesting: Coarse > Medium > Fine",
        color="n_members",
        color_continuous_scale="Blues",
    )
    fig_tree.update_layout(height=600)
    st.plotly_chart(fig_tree, use_container_width=True)

# --- Export ---
st.divider()
download_csv_button(leiden_df, "leiden_communities.csv", "Download Community Assignments")
