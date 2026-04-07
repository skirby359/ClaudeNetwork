"""Page 22: Organizational Health Score — Composite communication health metric."""

import streamlit as st
import plotly.graph_objects as go
import polars as pl

from src.state import (
    render_date_filter,
    load_filtered_message_fact, load_filtered_edge_fact,
    load_filtered_graph_metrics, load_nonhuman_emails,
)
from src.analytics.health_score import compute_health_score, compute_health_trend
from src.analytics.response_time import compute_reply_times


# ---------------------------------------------------------------------------
# Cached analytics
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner="Computing health score...", ttl=3600)
def _cached_health(start_date, end_date):
    mf = load_filtered_message_fact(start_date, end_date)
    ef = load_filtered_edge_fact(start_date, end_date)
    gm = load_filtered_graph_metrics(start_date, end_date)

    # Filter to human-only for meaningful scores
    nonhuman = load_nonhuman_emails(start_date, end_date)
    nh_list = list(nonhuman)
    mf_human = mf.filter(~pl.col("from_email").is_in(nh_list))
    ef_human = ef.filter(
        ~pl.col("from_email").is_in(nh_list)
        & ~pl.col("to_email").is_in(nh_list)
    )

    # Get reply time
    reply_median = None
    try:
        reply_times = compute_reply_times(ef_human)
        if len(reply_times) > 0:
            reply_median = float(reply_times["median_reply_seconds"].median())
    except Exception:
        pass

    return compute_health_score(mf_human, ef_human, gm, reply_median)


# ---------------------------------------------------------------------------
# Page layout
# ---------------------------------------------------------------------------

st.set_page_config(page_title="Health Score", layout="wide")
st.title("Organizational Health Score")
st.caption(
    "A composite 0-100 score measuring communication health from six dimensions. "
    "Based on human communication only (automated senders excluded)."
)

start_date, end_date = render_date_filter()

mf = load_filtered_message_fact(start_date, end_date)
if len(mf) == 0:
    st.warning("No data in selected date range.")
    st.stop()

health = _cached_health(start_date, end_date)
composite = health["composite"]
sub_scores = health["sub_scores"]

# Big number
col_score, col_gauge = st.columns([1, 2])

with col_score:
    if composite >= 70:
        color = "green"
        verdict = "Healthy"
    elif composite >= 50:
        color = "orange"
        verdict = "Moderate"
    else:
        color = "red"
        verdict = "Needs Attention"

    st.markdown(
        f"<div style='text-align:center;padding:20px;'>"
        f"<div style='font-size:72px;font-weight:bold;color:{color};'>{composite:.0f}</div>"
        f"<div style='font-size:24px;color:{color};'>{verdict}</div>"
        f"<div style='font-size:14px;color:#888;'>out of 100</div>"
        f"</div>",
        unsafe_allow_html=True,
    )

with col_gauge:
    # Radar chart
    categories = [s["label"] for s in sub_scores.values()]
    values = [s["value"] for s in sub_scores.values()]
    # Close the polygon
    categories_closed = categories + [categories[0]]
    values_closed = values + [values[0]]

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=values_closed,
        theta=categories_closed,
        fill="toself",
        fillcolor="rgba(78, 121, 167, 0.3)",
        line=dict(color="#4e79a7", width=2),
        name="Your Organization",
    ))
    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 100]),
        ),
        height=400,
        margin=dict(l=60, r=60, t=40, b=40),
        title="Health Dimensions",
    )
    st.plotly_chart(fig, use_container_width=True)

st.divider()

# Sub-score breakdown
st.subheader("Score Breakdown")

for key, score in sub_scores.items():
    col_bar, col_detail = st.columns([3, 2])
    with col_bar:
        val = score["value"]
        bar_color = "#4e79a7" if val >= 60 else "#f28e2b" if val >= 40 else "#e15759"
        st.progress(val / 100, text=f"**{score['label']}**: {val:.0f}/100")
    with col_detail:
        st.caption(score["detail"])

st.divider()

# Interpretation
st.subheader("What This Means")

strengths = [s for s in sub_scores.values() if s["value"] >= 70]
concerns = [s for s in sub_scores.values() if s["value"] < 50]

if strengths:
    st.markdown("**Strengths:**")
    for s in strengths:
        st.write(f"- {s['label']}: {s['detail']}")

if concerns:
    st.markdown("**Areas for Improvement:**")
    for s in concerns:
        st.write(f"- {s['label']}: {s['detail']}")

if not concerns:
    st.success("No major communication health concerns detected.")

# --- Trend over time ---
st.divider()
st.subheader("Health Score Trend")
st.caption("Monthly health score showing how communication patterns evolve over time.")


@st.cache_data(show_spinner="Computing monthly health scores...", ttl=3600)
def _cached_trend(start_date, end_date):
    mf = load_filtered_message_fact(start_date, end_date)
    ef = load_filtered_edge_fact(start_date, end_date)
    gm = load_filtered_graph_metrics(start_date, end_date)
    nonhuman = load_nonhuman_emails(start_date, end_date)
    nh_list = list(nonhuman)
    mf_human = mf.filter(~pl.col("from_email").is_in(nh_list))
    ef_human = ef.filter(
        ~pl.col("from_email").is_in(nh_list)
        & ~pl.col("to_email").is_in(nh_list)
    )
    return compute_health_trend(mf_human, ef_human, gm)


trend = _cached_trend(start_date, end_date)
if len(trend) >= 2:
    fig_trend = go.Figure()
    fig_trend.add_trace(go.Scatter(
        x=trend["month_id"].to_list(),
        y=trend["composite"].to_list(),
        mode="lines+markers",
        name="Composite Score",
        line=dict(color="#4e79a7", width=3),
        marker=dict(size=8),
    ))

    # Add sub-score lines if available
    sub_keys = [c for c in trend.columns if c not in ("month_id", "composite")]
    colors = ["#e15759", "#f28e2b", "#59a14f", "#76b7b2", "#edc948", "#b07aa1"]
    for i, key in enumerate(sub_keys):
        fig_trend.add_trace(go.Scatter(
            x=trend["month_id"].to_list(),
            y=trend[key].to_list(),
            mode="lines",
            name=key.replace("_", " ").title(),
            line=dict(color=colors[i % len(colors)], width=1, dash="dot"),
            opacity=0.7,
        ))

    fig_trend.add_hrect(y0=70, y1=100, fillcolor="green", opacity=0.05, line_width=0)
    fig_trend.add_hrect(y0=0, y1=50, fillcolor="red", opacity=0.05, line_width=0)
    fig_trend.update_layout(
        height=400, yaxis_title="Score", xaxis_title="Month",
        yaxis_range=[0, 105],
    )
    st.plotly_chart(fig_trend, use_container_width=True)
else:
    st.info("Need at least 2 months of data to show a trend.")

st.divider()
st.caption(
    "Methodology: Each dimension is scored 0-100 based on the statistical properties of "
    "human email communication patterns. The composite score is a weighted average. "
    "Higher scores indicate healthier communication patterns. Automated/system emails "
    "are excluded from all calculations."
)
