"""Page 8: Dyads & Asymmetry — Bidirectional relationship analysis."""

import streamlit as st
import plotly.express as px
import polars as pl

from src.state import (
    render_date_filter,
    load_filtered_edge_fact, load_filtered_dyads,
)
from src.analytics.timing_analytics import compute_ping_pong

st.set_page_config(page_title="Dyads & Asymmetry", layout="wide")
st.title("Dyads & Asymmetry")

start_date, end_date = render_date_filter()

edge_fact = load_filtered_edge_fact(start_date, end_date)
dyads = load_filtered_dyads(start_date, end_date)

# Top communication pairs
st.subheader("Strongest Communication Pairs")
st.markdown("Bidirectional pairs ranked by total message exchange.")
top_dyads = dyads.head(30).to_pandas()
fig = px.bar(top_dyads, x=top_dyads.apply(
    lambda r: f"{r['from_email'].split('@')[0]} <-> {r['to_email'].split('@')[0]}", axis=1),
    y="total_pair_msgs", color="asymmetry_ratio",
    color_continuous_scale="RdYlGn_r",
    title="Top 30 Communication Pairs")
fig.update_layout(height=450, xaxis_tickangle=-45, xaxis_title="Pair")
st.plotly_chart(fig, width="stretch")

# Asymmetry analysis
st.divider()
st.subheader("Asymmetry Distribution")
st.markdown("""
**Asymmetry ratio:** 0 = perfectly balanced, 1 = completely one-directional.
""")
asym = dyads.filter(pl.col("total_pair_msgs") >= 5)
fig2 = px.histogram(asym.to_pandas(), x="asymmetry_ratio", nbins=50,
                    title="Distribution of Asymmetry Ratios (pairs with 5+ messages)")
fig2.update_layout(height=350)
st.plotly_chart(fig2, width="stretch")

# Most asymmetric pairs
st.subheader("Most Asymmetric Relationships")
st.markdown("Pairs where communication is heavily one-directional.")
highly_asym = (
    dyads.filter(pl.col("total_pair_msgs") >= 10)
    .sort("asymmetry_ratio", descending=True)
    .head(20)
    .to_pandas()
)
st.dataframe(highly_asym, width="stretch")

# Most balanced pairs
st.divider()
st.subheader("Most Balanced Relationships")
balanced = (
    dyads.filter(pl.col("total_pair_msgs") >= 10)
    .sort("asymmetry_ratio")
    .head(20)
    .to_pandas()
)
st.dataframe(balanced, width="stretch")

# Ping-pong analysis
st.divider()
st.subheader("Ping-Pong Pairs")
st.markdown("Pairs with frequent back-and-forth exchanges — active conversational relationships.")
min_ex = st.slider("Minimum exchanges per direction", 3, 20, 5)
pp = compute_ping_pong(edge_fact, min_exchanges=min_ex).to_pandas()
st.write(f"**{len(pp)} active ping-pong pairs** found")
st.dataframe(pp.head(30), width="stretch")
