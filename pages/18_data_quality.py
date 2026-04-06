"""Page 18: Data Quality — Completeness, anomalies, and ingestion stats."""

import streamlit as st
import plotly.express as px
import polars as pl

from src.state import (
    render_date_filter,
    load_message_fact, load_filtered_message_fact,
    load_filtered_edge_fact, load_person_dim,
)
from src.analytics.data_quality import compute_quality_metrics, compute_per_file_stats
from src.ingest.pipeline import get_last_ingestion_stats
from src.export import download_csv_button


# ---------------------------------------------------------------------------
# Cached analytics
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner=False, ttl=3600)
def _cached_quality_metrics(start_date, end_date):
    mf = load_filtered_message_fact(start_date, end_date)
    return compute_quality_metrics(mf)


@st.cache_data(show_spinner=False, ttl=3600)
def _cached_daily_completeness(start_date, end_date):
    """Per-day message counts to spot gaps."""
    mf = load_filtered_message_fact(start_date, end_date)
    return (
        mf.with_columns(pl.col("timestamp").dt.date().alias("date"))
        .group_by("date")
        .agg(pl.len().alias("msg_count"))
        .sort("date")
    )


@st.cache_data(show_spinner=False, ttl=3600)
def _cached_zero_size_senders(start_date, end_date):
    mf = load_filtered_message_fact(start_date, end_date)
    zeros = mf.filter(pl.col("size_bytes") == 0)
    return (
        zeros.group_by("from_email")
        .agg(pl.len().alias("zero_count"))
        .sort("zero_count", descending=True)
    )


# ---------------------------------------------------------------------------
# Page layout
# ---------------------------------------------------------------------------

st.set_page_config(page_title="Data Quality", layout="wide")
st.title("Data Quality Report")

start_date, end_date = render_date_filter()

quality = _cached_quality_metrics(start_date, end_date)

if quality["total_messages"] == 0:
    st.warning("No messages in selected date range.")
    st.stop()

# KPIs
c1, c2, c3, c4 = st.columns(4)
with c1:
    st.metric("Total Messages", f"{quality['total_messages']:,}")
with c2:
    st.metric("Zero-Size", f"{quality['zero_size_count']:,} ({quality['zero_size_pct']:.1%})")
with c3:
    st.metric("Missing Sender Names", f"{quality['missing_name_count']:,} ({quality['missing_name_pct']:.1%})")
with c4:
    st.metric("Duplicate IDs", f"{quality['duplicate_msg_ids']:,}")

st.divider()

# Daily completeness calendar
st.subheader("Daily Message Volume (Completeness Check)")
st.caption("Gaps or sudden drops may indicate missing data or ingestion errors.")
daily = _cached_daily_completeness(start_date, end_date)

if len(daily) > 0:
    fig = px.bar(daily.to_pandas(), x="date", y="msg_count",
                 title="Messages per Day",
                 labels={"date": "Date", "msg_count": "Messages"})
    fig.update_layout(height=350)
    st.plotly_chart(fig, width="stretch")

    # Detect gaps (days with 0 messages within the range)
    all_dates = pl.date_range(daily["date"].min(), daily["date"].max(), eager=True).alias("date")
    all_dates_df = pl.DataFrame({"date": all_dates})
    merged = all_dates_df.join(daily, on="date", how="left")
    gaps = merged.filter(pl.col("msg_count").is_null())
    if len(gaps) > 0:
        st.warning(f"**{len(gaps)} day(s)** with zero messages detected within the date range.")
        st.dataframe(gaps.to_pandas(), width="stretch")
    else:
        st.success("No gaps detected -- every day in the range has at least one message.")

st.divider()

# Zero-size message breakdown
st.subheader("Zero-Size Messages by Sender")
zero_senders = _cached_zero_size_senders(start_date, end_date)
if len(zero_senders) > 0:
    top_zeros = zero_senders.head(20)
    fig2 = px.bar(top_zeros.to_pandas(), x="from_email", y="zero_count",
                  title="Top 20 Senders of Zero-Size Messages")
    fig2.update_layout(height=400, xaxis_tickangle=-45)
    st.plotly_chart(fig2, width="stretch")
    download_csv_button(zero_senders, "zero_size_senders.csv")
else:
    st.success("No zero-size messages in the selected range.")

st.divider()

# Recipient count distribution
st.subheader("Recipient Count Distribution")
mf = load_filtered_message_fact(start_date, end_date)
single_pct = quality["single_recipient_pct"]
st.write(f"**Single-recipient messages:** {quality['single_recipient_count']:,} ({single_pct:.1%})")

recip_dist = (
    mf.group_by("n_recipients")
    .agg(pl.len().alias("count"))
    .sort("n_recipients")
)
if len(recip_dist) > 0:
    fig3 = px.bar(
        recip_dist.filter(pl.col("n_recipients") <= 50).to_pandas(),
        x="n_recipients", y="count",
        title="Message Count by Number of Recipients (capped at 50)",
        labels={"n_recipients": "Recipient Count", "count": "Messages"},
    )
    fig3.update_layout(height=350)
    st.plotly_chart(fig3, width="stretch")

st.divider()

# Per-file ingestion stats
st.subheader("Ingestion Statistics")
ingestion_stats = get_last_ingestion_stats()
if ingestion_stats:
    file_stats = compute_per_file_stats(ingestion_stats)
    st.dataframe(file_stats.to_pandas(), width="stretch")
else:
    st.info("No ingestion stats available. Run the pipeline to generate stats.")

# Full dataset quality (unfiltered)
st.divider()
st.subheader("Full Dataset Quality (All Dates)")
full_mf = load_message_fact()
full_quality = compute_quality_metrics(full_mf)
col_a, col_b = st.columns(2)
with col_a:
    st.write(f"**Total messages (full):** {full_quality['total_messages']:,}")
    st.write(f"**Zero-size (full):** {full_quality['zero_size_count']:,} ({full_quality['zero_size_pct']:.1%})")
    st.write(f"**Missing names (full):** {full_quality['missing_name_count']:,} ({full_quality['missing_name_pct']:.1%})")
with col_b:
    st.write(f"**Single-recipient (full):** {full_quality['single_recipient_count']:,} ({full_quality['single_recipient_pct']:.1%})")
    st.write(f"**Duplicate IDs (full):** {full_quality['duplicate_msg_ids']:,}")
    person_dim = load_person_dim()
    st.write(f"**Unique people:** {len(person_dim):,}")
