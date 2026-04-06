"""Page 16: Temporal Network Evolution — How the network changes over time."""

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import polars as pl

from src.state import (
    render_date_filter,
    load_filtered_edge_fact,
)
from src.analytics.temporal_network import (
    build_monthly_snapshots, compute_centrality_trends,
    detect_rising_fading, compute_community_stability,
)
from src.export import download_csv_button
from src.drilldown import handle_dataframe_person_click


# ---------------------------------------------------------------------------
# Cached analytics
# ---------------------------------------------------------------------------

MIN_MONTHS_FOR_TRENDS = 3


@st.cache_data(show_spinner="Building monthly snapshots (this may take a moment)...", ttl=3600)
def _cached_temporal_analysis(start_date, end_date):
    """Cache all temporal analytics together to avoid redundant snapshot builds."""
    ef = load_filtered_edge_fact(start_date, end_date)
    snapshots = build_monthly_snapshots(ef)

    if len(snapshots) < 2:
        return snapshots, None, None, None, None

    trends = compute_centrality_trends(snapshots)
    rising, fading = detect_rising_fading(trends, min_months=MIN_MONTHS_FOR_TRENDS)
    stability = compute_community_stability(snapshots)
    return snapshots, trends, rising, fading, stability


# ---------------------------------------------------------------------------
# Page layout
# ---------------------------------------------------------------------------

st.set_page_config(page_title="Temporal Evolution", layout="wide")
st.title("Temporal Network Evolution")

start_date, end_date = render_date_filter()

edge_fact = load_filtered_edge_fact(start_date, end_date)

if len(edge_fact) == 0:
    st.warning("No edge data in selected date range.")
    st.stop()

# Cached computation
snapshots, centrality_trends, rising, fading, stability = _cached_temporal_analysis(start_date, end_date)

if len(snapshots) < 2:
    st.info("Need at least 2 months of data for temporal analysis. Try selecting a wider date range.")
    st.stop()

n_months = len(snapshots)

# Monthly overview metrics
st.subheader("Monthly Network Growth")
monthly_summary = []
for month, df in sorted(snapshots.items()):
    monthly_summary.append({
        "month": month,
        "node_count": len(df),
        "community_count": df["community_id"].n_unique(),
        "avg_pagerank": float(df["pagerank"].mean()),
    })
summary_df = pl.DataFrame(monthly_summary)

col_left, col_right = st.columns(2)
with col_left:
    fig = px.line(summary_df.to_pandas(), x="month", y="node_count",
                  title="Active People per Month", markers=True)
    fig.update_layout(height=350)
    st.plotly_chart(fig, width="stretch")

with col_right:
    fig2 = px.line(summary_df.to_pandas(), x="month", y="community_count",
                   title="Communities Detected per Month", markers=True)
    fig2.update_layout(height=350)
    st.plotly_chart(fig2, width="stretch")

st.divider()

# Rising stars and fading out
st.subheader("Rising Stars & Fading Out")

st.caption(
    f"Compares average centrality in the first half vs. second half of the "
    f"selected period.  Each person must appear in at least "
    f"**{MIN_MONTHS_FOR_TRENDS} distinct months** to qualify.  "
    f"Your current selection spans **{n_months} months**."
)
if n_months < MIN_MONTHS_FOR_TRENDS:
    st.warning(
        f"Rising/fading detection needs at least {MIN_MONTHS_FOR_TRENDS} months "
        f"of data, but the current range only covers {n_months}.  "
        f"Widen the date range (try the **All** preset) to enable this analysis."
    )

col_a, col_b = st.columns(2)
with col_a:
    st.markdown("**Rising Stars** (increasing centrality)")
    if rising is not None and len(rising) > 0:
        rising_pd = rising.to_pandas()
        ev_rising = st.dataframe(rising_pd, width="stretch", on_select="rerun", selection_mode="single-row", key="p16_rising_df")
        handle_dataframe_person_click(ev_rising, rising_pd, "p16_rising_df", "email", start_date, end_date)
        download_csv_button(rising, "rising_stars.csv")
    else:
        st.info(
            f"Not enough data to detect rising trends.  "
            f"Need people active in {MIN_MONTHS_FOR_TRENDS}+ months."
        )

with col_b:
    st.markdown("**Fading Out** (decreasing centrality)")
    if fading is not None and len(fading) > 0:
        fading_pd = fading.to_pandas()
        ev_fading = st.dataframe(fading_pd, width="stretch", on_select="rerun", selection_mode="single-row", key="p16_fading_df")
        handle_dataframe_person_click(ev_fading, fading_pd, "p16_fading_df", "email", start_date, end_date)
        download_csv_button(fading, "fading_out.csv")
    else:
        st.info(
            f"Not enough data to detect fading trends.  "
            f"Need people active in {MIN_MONTHS_FOR_TRENDS}+ months."
        )

st.divider()

# Community stability
st.subheader("Community Stability Over Time")
if stability is not None and len(stability) > 0:
    fig3 = px.bar(stability.to_pandas(), x="month_pair", y="nmi",
                  title="Community Stability (NMI between consecutive months)",
                  labels={"nmi": "Normalized Mutual Information", "month_pair": "Month Pair"})
    fig3.update_layout(height=350)
    st.plotly_chart(fig3, width="stretch")

st.divider()

# Animated centrality scatter for top-N people
st.subheader("Centrality Animation (Top People)")
top_n = st.slider("Number of people to track", 10, 50, 20)

# Find top people by average pagerank
person_avg = (
    centrality_trends.group_by("email")
    .agg(pl.col("pagerank").mean().alias("avg_pr"))
    .sort("avg_pr", descending=True)
    .head(top_n)
)
top_emails = person_avg["email"].to_list()

anim_data = centrality_trends.filter(pl.col("email").is_in(top_emails)).to_pandas()
if len(anim_data) > 0:
    fig4 = px.scatter(
        anim_data, x="in_degree", y="pagerank",
        animation_frame="month_id", hover_name="email",
        size="out_degree", size_max=30,
        title="Network Position Over Time (use controls to play/pause)",
        labels={"in_degree": "In-Degree", "pagerank": "PageRank"},
    )
    # Slow down the animation: 1500ms per frame, 600ms transition
    fig4.layout.updatemenus[0].buttons[0].args[1]["frame"]["duration"] = 1500
    fig4.layout.updatemenus[0].buttons[0].args[1]["transition"]["duration"] = 600
    fig4.update_layout(height=500)
    st.plotly_chart(fig4, width="stretch")
