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
    load_filtered_graph_metrics, load_nonhuman_emails,
)
from src.drilldown import (
    handle_plotly_community_click, handle_plotly_week_click,
    handle_dataframe_person_click,
)


# ---------------------------------------------------------------------------
# Cached analytics
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner=False, ttl=3600)
def _cached_comm_sizes(start_date, end_date):
    gm = load_filtered_graph_metrics(start_date, end_date)
    return (
        gm.group_by("community_id")
        .agg([
            pl.len().alias("members"),
            pl.col("pagerank").sum().alias("total_pagerank"),
            pl.col("pagerank").mean().alias("avg_pagerank"),
        ])
        .sort("members", descending=True)
    )


@st.cache_data(show_spinner=False, ttl=3600)
def _cached_cross_community(start_date, end_date):
    ef = load_filtered_edge_fact(start_date, end_date)
    gm = load_filtered_graph_metrics(start_date, end_date)
    community_lookup = gm.select(["email", "community_id"])
    edge_enriched = (
        ef.select(["from_email", "to_email"])
        .join(community_lookup, left_on="from_email", right_on="email", how="left")
        .rename({"community_id": "from_community"})
        .join(community_lookup, left_on="to_email", right_on="email", how="left")
        .rename({"community_id": "to_community"})
    )
    return (
        edge_enriched.group_by(["from_community", "to_community"])
        .agg(pl.len().alias("msg_count"))
        .sort("msg_count", descending=True)
    )


@st.cache_data(show_spinner=False, ttl=3600)
def _cached_weekly_senders(start_date, end_date):
    mf = load_filtered_message_fact(start_date, end_date)
    return (
        mf.group_by("week_id")
        .agg([
            pl.col("from_email").n_unique().alias("active_senders"),
            pl.col("timestamp").min().alias("week_start"),
        ])
        .sort("week_start")
    )


# ---------------------------------------------------------------------------
# Page layout
# ---------------------------------------------------------------------------

st.set_page_config(page_title="Coordination & Churn", layout="wide")
st.title("Coordination & Churn")

start_date, end_date = render_date_filter()

edge_fact = load_filtered_edge_fact(start_date, end_date)
message_fact = load_filtered_message_fact(start_date, end_date)

if len(edge_fact) == 0:
    st.warning("No data in selected date range.")
    st.stop()

person_dim = load_person_dim()
graph_metrics = load_filtered_graph_metrics(start_date, end_date)

# --- Nonhuman filter (global toggle) ---
nonhuman_emails = load_nonhuman_emails(start_date, end_date)
filter_nonhuman = st.session_state.get("exclude_nonhuman", True)
if filter_nonhuman and nonhuman_emails:
    nh_list = list(nonhuman_emails)
    graph_metrics = graph_metrics.filter(~pl.col("email").is_in(nh_list))

# Community sizes (cached)
st.subheader("Community Structure")
comm_sizes = _cached_comm_sizes(start_date, end_date)

fig = px.bar(comm_sizes.to_pandas(), x="community_id", y="members",
             title="Community Sizes", color="avg_pagerank",
             color_continuous_scale="Viridis")
fig.update_layout(height=350)
ev_comm = st.plotly_chart(fig, width="stretch", on_select="rerun", key="p09_comm")
handle_plotly_community_click(ev_comm, "p09_comm", start_date, end_date)

# Cross-community communication
st.divider()
st.subheader("Cross-Community Communication")
st.markdown("How much do communities talk to each other vs. within themselves?")

cross_comm = _cached_cross_community(start_date, end_date)

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
fc_list = cross_filtered["from_community"].to_list()
tc_list = cross_filtered["to_community"].to_list()
mc_list = cross_filtered["msg_count"].to_list()
for fc, tc, mc in zip(fc_list, tc_list, mc_list):
    if fc in comm_idx and tc in comm_idx:
        matrix[comm_idx[fc]][comm_idx[tc]] = mc

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

weekly_senders = _cached_weekly_senders(start_date, end_date).to_pandas()

fig3 = px.line(weekly_senders, x="week_start", y="active_senders",
               title="Unique Active Senders per Week")
fig3.update_layout(height=350)
ev_active = st.plotly_chart(fig3, width="stretch", on_select="rerun", key="p09_active")
handle_plotly_week_click(ev_active, "p09_active", start_date, end_date)

# Community member list
st.divider()
st.subheader("Community Members")

all_communities = sorted(graph_metrics["community_id"].unique().to_list())

# Persist selection across reruns via session_state
if "p09_community_value" not in st.session_state:
    st.session_state.p09_community_value = all_communities[0] if all_communities else 0

def _on_community_change():
    st.session_state.p09_community_value = st.session_state._p09_comm_widget

# Ensure stored value is still valid
if st.session_state.p09_community_value not in all_communities:
    st.session_state.p09_community_value = all_communities[0] if all_communities else 0

selected_community = st.selectbox(
    "Select Community",
    all_communities,
    index=all_communities.index(st.session_state.p09_community_value),
    key="_p09_comm_widget",
    on_change=_on_community_change,
)
members = (
    graph_metrics.filter(pl.col("community_id") == selected_community)
    .join(person_dim.select(["email", "display_name"]), on="email", how="left")
    .sort("pagerank", descending=True)
    .to_pandas()
)
st.write(f"**{len(members)} members** in Community {selected_community}")
ev_members = st.dataframe(members, width="stretch", on_select="rerun", selection_mode="single-row", key="p09_members_df")
handle_dataframe_person_click(ev_members, members, "p09_members_df", "email", start_date, end_date)
