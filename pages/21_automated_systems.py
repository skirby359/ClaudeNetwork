"""Page 21: Automated Systems — Understanding machine-generated email traffic."""

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import polars as pl

from src.page_logger import log_page_entry, log_page_error
from src.state import (
    render_date_filter,
    load_filtered_message_fact,
    load_nonhuman_emails,
)
from src.analytics.size_forensics import detect_size_templates
from src.analytics.hierarchy import classify_nonhuman_type
from src.export import download_csv_button
from src.anonymize import anon_df


# ---------------------------------------------------------------------------
# Cached analytics
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner=False, ttl=3600)
def _cached_auto_breakdown(start_date, end_date):
    mf = load_filtered_message_fact(start_date, end_date)
    nonhuman = load_nonhuman_emails(start_date, end_date)
    nh_list = list(nonhuman)

    total = len(mf)
    auto_count = len(mf.filter(pl.col("from_email").is_in(nh_list)))
    human_count = total - auto_count

    # Top automated senders with type classification
    auto_senders = (
        mf.filter(pl.col("from_email").is_in(nh_list))
        .group_by("from_email")
        .agg([
            pl.len().alias("msg_count"),
            pl.col("size_bytes").mean().alias("avg_size"),
            pl.col("n_recipients").mean().alias("avg_recipients"),
            pl.col("is_after_hours").mean().alias("after_hours_rate"),
        ])
        .sort("msg_count", descending=True)
    )
    # Add type classification
    types = [classify_nonhuman_type(e) for e in auto_senders["from_email"].to_list()]
    auto_senders = auto_senders.with_columns(pl.Series("system_type", types))

    # Per-system hourly heatmap data (top 15 senders)
    top_systems = auto_senders["from_email"].head(15).to_list()
    system_hourly = (
        mf.filter(pl.col("from_email").is_in(top_systems))
        .group_by(["from_email", "hour"])
        .agg(pl.len().alias("count"))
        .sort(["from_email", "hour"])
    )

    # Hourly distribution: human vs automated
    human_hourly = (
        mf.filter(~pl.col("from_email").is_in(nh_list))
        .group_by("hour").agg(pl.len().alias("human_count")).sort("hour")
    )
    auto_hourly = (
        mf.filter(pl.col("from_email").is_in(nh_list))
        .group_by("hour").agg(pl.len().alias("auto_count")).sort("hour")
    )

    return {
        "total": total,
        "auto_count": auto_count,
        "human_count": human_count,
        "auto_senders": auto_senders,
        "human_hourly": human_hourly,
        "auto_hourly": auto_hourly,
        "n_nonhuman": len(nonhuman),
        "system_hourly": system_hourly,
    }


# ---------------------------------------------------------------------------
# Page layout
# ---------------------------------------------------------------------------

st.set_page_config(page_title="Automated Systems", layout="wide")
_page_log = log_page_entry("21_automated_systems")
st.title("Automated Systems Overview")
st.caption(
    "Understanding the machines: what automated systems dominate your email, "
    "and how filtering them reveals the real human communication patterns."
)

start_date, end_date = render_date_filter()

message_fact = load_filtered_message_fact(start_date, end_date)

if len(message_fact) == 0:
    st.warning("No data in selected date range.")
    st.stop()

data = _cached_auto_breakdown(start_date, end_date)

# KPI row
c1, c2, c3, c4 = st.columns(4)
with c1:
    st.metric("Total Messages", f"{data['total']:,}")
with c2:
    auto_pct = data['auto_count'] / max(data['total'], 1) * 100
    st.metric("Automated", f"{data['auto_count']:,} ({auto_pct:.0f}%)")
with c3:
    human_pct = 100 - auto_pct
    st.metric("Human", f"{data['human_count']:,} ({human_pct:.0f}%)")
with c4:
    st.metric("Automated Addresses", f"{data['n_nonhuman']:,}")

st.divider()

# Donut chart + explanation
col_chart, col_text = st.columns([1, 2])
with col_chart:
    fig_split = px.pie(
        values=[data['human_count'], data['auto_count']],
        names=["Human", "Automated"],
        color_discrete_sequence=["#4e79a7", "#e15759"],
        hole=0.5,
    )
    fig_split.update_layout(height=300, margin=dict(l=10, r=10, t=30, b=10))
    fig_split.update_traces(textinfo="percent+label")
    st.plotly_chart(fig_split, use_container_width=True)

with col_text:
    st.markdown(f"""
    ### What does this mean?

    **{auto_pct:.0f}% of your email traffic is automated** — copiers, scanners, alert systems,
    notification services, and mail infrastructure. This is normal for organizations with
    enterprise systems.

    **When you enable "Exclude automated senders" in the sidebar**, all analysis pages
    focus on the **{human_pct:.0f}% human communication** — giving you accurate insights into
    actual work patterns, collaboration, and organizational structure.

    **{data['n_nonhuman']:,} automated addresses** were detected using two methods:
    - Pattern matching (noreply, copier, scanner, automail, etc.)
    - Behavioral analysis (accounts that only send or only receive with 100+ messages)
    """)

st.divider()

# Top automated senders
st.subheader("Top Automated Senders")
st.caption("The most active machine-generated email accounts.")
auto_senders = data['auto_senders']
if len(auto_senders) > 0:
    top20 = anon_df(auto_senders.head(20)).with_columns(
        (pl.col("avg_size") / 1024).round(1).alias("avg_size_kb"),
        (pl.col("after_hours_rate") * 100).round(1).alias("after_hours_pct"),
        (pl.col("avg_recipients")).round(1),
    )
    fig = px.bar(
        top20.to_pandas(), x="from_email", y="msg_count",
        color="system_type",
        title="Top 20 Automated Senders by Volume",
        labels={"from_email": "System", "msg_count": "Messages", "system_type": "Type"},
    )
    fig.update_layout(height=400, xaxis_tickangle=-45)
    st.plotly_chart(fig, width="stretch")

    st.dataframe(
        top20.select(["from_email", "system_type", "msg_count", "avg_size_kb", "avg_recipients", "after_hours_pct"])
        .rename({
            "from_email": "Address",
            "system_type": "Type",
            "msg_count": "Messages",
            "avg_size_kb": "Avg Size (KB)",
            "avg_recipients": "Avg Recipients",
            "after_hours_pct": "After-Hours %",
        })
        .to_pandas(),
        width="stretch",
    )
    download_csv_button(auto_senders, "automated_senders.csv")
else:
    st.success("No automated senders detected.")

st.divider()

# Human vs Automated hourly patterns
st.subheader("When Do Humans vs Machines Send?")
st.caption("Hourly distribution reveals that automated systems often peak outside business hours.")

human_hourly = data['human_hourly']
auto_hourly = data['auto_hourly']

if len(human_hourly) > 0 and len(auto_hourly) > 0:
    # Merge into one DataFrame for plotting
    hours = pl.DataFrame({"hour": list(range(24))})
    merged = (
        hours
        .join(human_hourly, on="hour", how="left")
        .join(auto_hourly, on="hour", how="left")
        .with_columns([
            pl.col("human_count").fill_null(0),
            pl.col("auto_count").fill_null(0),
        ])
    ).to_pandas()

    fig2 = go.Figure()
    fig2.add_trace(go.Bar(x=merged["hour"], y=merged["human_count"],
                          name="Human", marker_color="#4e79a7", opacity=0.8))
    fig2.add_trace(go.Bar(x=merged["hour"], y=merged["auto_count"],
                          name="Automated", marker_color="#e15759", opacity=0.8))
    fig2.update_layout(
        height=400, barmode="stack",
        title="Messages by Hour of Day",
        xaxis=dict(dtick=1, title="Hour"),
        yaxis_title="Messages",
    )
    st.plotly_chart(fig2, width="stretch")

st.divider()

# System type breakdown
st.subheader("Automated Sender Types")
st.caption("Classification of automated senders by function.")

if len(auto_senders) > 0 and "system_type" in auto_senders.columns:
    type_summary = (
        auto_senders.group_by("system_type")
        .agg([
            pl.len().alias("n_senders"),
            pl.col("msg_count").sum().alias("total_messages"),
        ])
        .sort("total_messages", descending=True)
    )
    col_pie, col_table = st.columns([1, 1])
    with col_pie:
        fig_types = px.pie(
            type_summary.to_pandas(),
            values="total_messages", names="system_type",
            title="Message Volume by System Type",
            color_discrete_sequence=px.colors.qualitative.Set2,
        )
        fig_types.update_layout(height=350)
        st.plotly_chart(fig_types, use_container_width=True)
    with col_table:
        st.dataframe(
            type_summary.rename({
                "system_type": "Type",
                "n_senders": "Senders",
                "total_messages": "Messages",
            }).to_pandas(),
            use_container_width=True,
        )

st.divider()

# Per-system hour-of-day heatmap
st.subheader("Per-System Hourly Activity")
st.caption("When each automated system sends — reveals scheduled jobs, real-time alerts, and always-on services.")

system_hourly = data.get("system_hourly")
if system_hourly is not None and len(system_hourly) > 0:
    pivot = system_hourly.to_pandas().pivot_table(
        index="from_email", columns="hour", values="count", fill_value=0,
    )
    # Ensure all 24 hours present
    for h in range(24):
        if h not in pivot.columns:
            pivot[h] = 0
    pivot = pivot[sorted(pivot.columns)]

    fig_heatmap = px.imshow(
        pivot,
        labels=dict(x="Hour of Day", y="System", color="Messages"),
        title="Automated Sender Activity by Hour",
        aspect="auto",
        color_continuous_scale="YlOrRd",
    )
    fig_heatmap.update_layout(height=max(300, len(pivot) * 28 + 100))
    fig_heatmap.update_xaxes(dtick=1)
    st.plotly_chart(fig_heatmap, use_container_width=True)

st.divider()

# Size templates
st.subheader("Size Templates (Machine Fingerprints)")
st.caption(
    "When many messages share the exact same byte size, they're almost certainly auto-generated. "
    "Each size cluster represents a distinct automated system."
)
templates = detect_size_templates(message_fact)
if len(templates) > 0:
    templates_display = templates.with_columns(
        (pl.col("size_bytes") / 1024).round(2).alias("size_kb")
    ).head(15)
    st.dataframe(
        anon_df(templates_display).select([
            "size_kb", "occurrence_count", "unique_senders", "example_sender",
        ]).rename({
            "size_kb": "Size (KB)",
            "occurrence_count": "Occurrences",
            "unique_senders": "Unique Senders",
            "example_sender": "Example Sender",
        }).to_pandas(),
        width="stretch",
    )
else:
    st.info("No clear size templates detected.")

st.divider()

# Recommendation
st.subheader("Recommendation")
if auto_pct > 50:
    st.warning(
        f"**{auto_pct:.0f}% of your email is automated.** "
        f"Always enable the 'Exclude automated senders' toggle (sidebar) "
        f"for accurate human communication analysis. Without it, volume metrics, "
        f"after-hours rates, and network structure will be dominated by machines."
    )
elif auto_pct > 20:
    st.info(
        f"**{auto_pct:.0f}% of your email is automated.** "
        f"The automated filter is recommended for most analysis pages."
    )
else:
    st.success(
        f"Only **{auto_pct:.0f}% automated** — your email traffic is predominantly human. "
        f"The automated filter will have minimal impact on analysis."
    )
