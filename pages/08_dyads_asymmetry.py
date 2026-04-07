"""Page 8: Dyads & Asymmetry — Bidirectional relationship analysis."""

import streamlit as st
import plotly.express as px
import polars as pl

from src.page_logger import log_page_entry, log_page_error
from src.state import (
    load_person_dim,
    render_date_filter,
    load_filtered_edge_fact, load_filtered_dyads,
    load_nonhuman_emails,
)
from src.analytics.timing_analytics import compute_ping_pong
from src.export import download_csv_button
from src.drilldown import (
    handle_dyad_chart_click, handle_dataframe_dyad_click,
)


# ---------------------------------------------------------------------------
# Cached analytics
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner=False, ttl=3600)
def _cached_ping_pong(start_date, end_date, exclude_nonhuman, min_exchanges):
    ef = load_filtered_edge_fact(start_date, end_date)
    if exclude_nonhuman:
        nonhuman = load_nonhuman_emails(start_date, end_date)
        ef = ef.filter(
            ~pl.col("from_email").is_in(list(nonhuman))
            & ~pl.col("to_email").is_in(list(nonhuman))
        )
    return compute_ping_pong(ef, min_exchanges=min_exchanges)


# ---------------------------------------------------------------------------
# Page layout
# ---------------------------------------------------------------------------

st.set_page_config(page_title="Dyads & Asymmetry", layout="wide")
_page_log = log_page_entry("08_dyads_asymmetry")
st.title("Dyads & Asymmetry")

start_date, end_date = render_date_filter()

edge_fact = load_filtered_edge_fact(start_date, end_date)

if len(edge_fact) == 0:
    st.warning("No data in selected date range.")
    st.stop()

person_dim = load_person_dim()
dyads = load_filtered_dyads(start_date, end_date)

# --- Nonhuman filter (cached) ---
nonhuman_emails = load_nonhuman_emails(start_date, end_date)

filter_on = st.session_state.get("exclude_nonhuman", True)

if filter_on:
    dyads = dyads.filter(
        ~pl.col("from_email").is_in(list(nonhuman_emails))
        & ~pl.col("to_email").is_in(list(nonhuman_emails))
    )
    st.caption(f"Filtered out {len(nonhuman_emails)} nonhuman addresses.")

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
ev_pairs = st.plotly_chart(fig, width="stretch", on_select="rerun", key="p08_pairs")
handle_dyad_chart_click(ev_pairs, "p08_pairs", top_dyads, start_date, end_date)

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
ev_asym = st.dataframe(highly_asym, width="stretch", on_select="rerun", selection_mode="single-row", key="p08_asym_df")
handle_dataframe_dyad_click(ev_asym, highly_asym, "p08_asym_df", start_date, end_date)

# Most balanced pairs
st.divider()
st.subheader("Most Balanced Relationships")
balanced = (
    dyads.filter(pl.col("total_pair_msgs") >= 10)
    .sort("asymmetry_ratio")
    .head(20)
    .to_pandas()
)
ev_bal = st.dataframe(balanced, width="stretch", on_select="rerun", selection_mode="single-row", key="p08_bal_df")
handle_dataframe_dyad_click(ev_bal, balanced, "p08_bal_df", start_date, end_date)

# Ping-pong analysis (cached)
st.divider()
st.subheader("Ping-Pong Pairs")
st.markdown("Pairs with frequent back-and-forth exchanges — active conversational relationships.")
min_ex = st.slider("Minimum exchanges per direction", 3, 20, 5)
pp = _cached_ping_pong(start_date, end_date, filter_on, min_ex).to_pandas()
st.write(f"**{len(pp)} active ping-pong pairs** found")
pp_top = pp.head(30)
ev_pp = st.dataframe(pp_top, width="stretch", on_select="rerun", selection_mode="single-row", key="p08_pp_df")
handle_dataframe_dyad_click(ev_pp, pp_top, "p08_pp_df", start_date, end_date)
download_csv_button(dyads, "dyad_analysis.csv")
