"""Page 13: Response Time Analysis — Reply detection and speed metrics."""

import streamlit as st
import plotly.express as px
import polars as pl

from src.state import (
    load_person_dim,
    render_date_filter,
    load_filtered_edge_fact,
)
from src.analytics.response_time import (
    compute_reply_times, compute_person_response_stats,
    compute_department_response_stats,
)
from src.export import download_csv_button
from src.drilldown import handle_plotly_person_click, handle_scatter_person_click


# ---------------------------------------------------------------------------
# Cached analytics
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner="Computing reply times...", ttl=3600)
def _cached_response_analysis(start_date, end_date, scope):
    """Cache reply times + person stats + dept stats together, keyed on (dates, scope)."""
    ef = load_filtered_edge_fact(start_date, end_date)
    pd_dim = load_person_dim()
    internal_emails = set(pd_dim.filter(pl.col("is_internal"))["email"].to_list())
    external_emails = set(pd_dim.filter(~pl.col("is_internal"))["email"].to_list())

    if scope == "Internal only":
        ef_scoped = ef.filter(
            pl.col("from_email").is_in(list(internal_emails))
            & pl.col("to_email").is_in(list(internal_emails))
        )
    elif scope == "External only":
        ef_scoped = ef.filter(
            pl.col("from_email").is_in(list(external_emails))
            | pl.col("to_email").is_in(list(external_emails))
        )
    else:
        ef_scoped = ef

    n_scoped = len(ef_scoped)
    if n_scoped == 0:
        return None, None, None, 0

    reply_times = compute_reply_times(ef_scoped)
    if len(reply_times) == 0:
        return reply_times, None, None, n_scoped

    person_stats = compute_person_response_stats(reply_times)
    dept_stats = compute_department_response_stats(reply_times, pd_dim)
    return reply_times, person_stats, dept_stats, n_scoped


# ---------------------------------------------------------------------------
# Page layout
# ---------------------------------------------------------------------------

st.set_page_config(page_title="Response Time", layout="wide")
st.title("Response Time Analysis")

start_date, end_date = render_date_filter()

edge_fact = load_filtered_edge_fact(start_date, end_date)
person_dim = load_person_dim()

if len(edge_fact) == 0:
    st.warning("No edge data in selected date range.")
    st.stop()

# --- Internal / External scope ---
internal_emails = set(person_dim.filter(pl.col("is_internal"))["email"].to_list())

scope_tab = st.radio(
    "Scope", ["Internal only", "External only", "All addresses"],
    horizontal=True, key="response_scope",
)

# Cached computation
reply_times, person_stats, dept_stats, n_scoped = _cached_response_analysis(
    start_date, end_date, scope_tab
)

if reply_times is None:
    st.info("No edges for the selected scope.")
    st.stop()

if person_stats is None:
    st.info("No reply pairs detected in the selected date range and scope.")
    st.stop()

# KPIs
c1, c2, c3 = st.columns(3)
with c1:
    org_median = reply_times["median_reply_seconds"].median()
    st.metric("Org-Wide Median Reply Time", f"{org_median / 60:.0f} min")
with c2:
    total_replies = int(reply_times["reply_count"].sum())
    reply_pct = total_replies / max(n_scoped, 1)
    st.metric("Messages with Replies", f"{reply_pct:.1%}")
with c3:
    if len(person_stats) > 0:
        fastest = person_stats.row(0, named=True)
        st.metric("Fastest Responder", fastest["email"].split("@")[0])

st.divider()

# Top 20 fastest/slowest responders
col_left, col_right = st.columns(2)

with col_left:
    st.subheader("Fastest Responders (Top 20)")
    fast = person_stats.filter(pl.col("total_replies") >= 5).head(20)
    if len(fast) > 0:
        # Add internal/external label
        fast_enriched = fast.with_columns(
            pl.col("email").is_in(list(internal_emails)).alias("is_internal")
        )
        fast_pd = fast_enriched.with_columns(
            (pl.col("median_response_sec") / 60).alias("median_min")
        ).to_pandas()
        fig = px.bar(fast_pd, x="email", y="median_min",
                     title="Median Response Time (minutes)",
                     color="is_internal",
                     color_discrete_map={True: "#4e79a7", False: "#e15759"})
        fig.update_layout(height=400, xaxis_tickangle=-45)
        ev_fast = st.plotly_chart(fig, width="stretch", on_select="rerun", key="p13_fast")
        handle_plotly_person_click(ev_fast, "p13_fast", start_date, end_date)

with col_right:
    st.subheader("Slowest Responders (Top 20)")
    slow = person_stats.filter(pl.col("total_replies") >= 5).sort("median_response_sec", descending=True).head(20)
    if len(slow) > 0:
        slow_enriched = slow.with_columns(
            pl.col("email").is_in(list(internal_emails)).alias("is_internal")
        )
        slow_pd = slow_enriched.with_columns(
            (pl.col("median_response_sec") / 60).alias("median_min")
        ).to_pandas()
        fig2 = px.bar(slow_pd, x="email", y="median_min",
                      title="Median Response Time (minutes)",
                      color="is_internal",
                      color_discrete_map={True: "#4e79a7", False: "#e15759"})
        fig2.update_layout(height=400, xaxis_tickangle=-45)
        ev_slow = st.plotly_chart(fig2, width="stretch", on_select="rerun", key="p13_slow")
        handle_plotly_person_click(ev_slow, "p13_slow", start_date, end_date)

st.divider()

# Scatter: reply time vs volume
st.subheader("Reply Time vs Volume")
scatter_enriched = person_stats.filter(pl.col("total_replies") >= 3).with_columns([
    (pl.col("median_response_sec") / 3600).alias("median_hours"),
    pl.col("email").is_in(list(internal_emails)).alias("is_internal"),
])
scatter_data = scatter_enriched.to_pandas()
if len(scatter_data) > 0:
    fig3 = px.scatter(scatter_data, x="total_replies", y="median_hours",
                      hover_data=["email"],
                      color="is_internal",
                      color_discrete_map={True: "#4e79a7", False: "#e15759"},
                      title="Response Time vs Reply Volume",
                      labels={"total_replies": "Total Replies", "median_hours": "Median Response (hours)"})
    fig3.update_layout(height=400)
    ev_scatter = st.plotly_chart(fig3, width="stretch", on_select="rerun", key="p13_scatter")
    handle_scatter_person_click(ev_scatter, "p13_scatter", start_date, end_date, customdata_index=0)

# Department breakdown
st.divider()
st.subheader("Department Response Times")
if len(dept_stats) > 0:
    st.dataframe(dept_stats.to_pandas(), width="stretch")
    download_csv_button(dept_stats, "department_response_times.csv")
