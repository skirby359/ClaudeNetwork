"""Page 17: Email Size Forensics — Size patterns, templates, anomalies."""

import streamlit as st
import plotly.express as px
import polars as pl
import numpy as np

from src.state import (
    render_date_filter,
    load_filtered_message_fact,
)
from src.analytics.size_forensics import (
    classify_by_size, detect_size_templates,
    compute_sender_size_profile, detect_size_anomalies,
)
from src.export import download_csv_button
from src.drilldown import handle_plotly_person_click, handle_dataframe_person_click


# ---------------------------------------------------------------------------
# Cached analytics
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner=False, ttl=3600)
def _cached_classify(start_date, end_date):
    mf = load_filtered_message_fact(start_date, end_date)
    return classify_by_size(mf)


@st.cache_data(show_spinner=False, ttl=3600)
def _cached_size_templates(start_date, end_date):
    mf = load_filtered_message_fact(start_date, end_date)
    return detect_size_templates(mf)


@st.cache_data(show_spinner=False, ttl=3600)
def _cached_sender_profiles(start_date, end_date):
    mf = load_filtered_message_fact(start_date, end_date)
    return compute_sender_size_profile(mf)


@st.cache_data(show_spinner=False, ttl=3600)
def _cached_size_anomalies(start_date, end_date):
    mf = load_filtered_message_fact(start_date, end_date)
    return detect_size_anomalies(mf)


# ---------------------------------------------------------------------------
# Page layout
# ---------------------------------------------------------------------------

st.set_page_config(page_title="Size Forensics", layout="wide")
st.title("Email Size Forensics")

start_date, end_date = render_date_filter()

message_fact = load_filtered_message_fact(start_date, end_date)

if len(message_fact) == 0:
    st.warning("No messages in selected date range.")
    st.stop()

# Classify messages (cached)
classified = _cached_classify(start_date, end_date)

# KPIs
c1, c2, c3, c4 = st.columns(4)
with c1:
    avg_size = message_fact["size_bytes"].mean()
    st.metric("Avg Message Size", f"{avg_size / 1024:.1f} KB")
with c2:
    median_size = message_fact["size_bytes"].median()
    st.metric("Median Size", f"{median_size / 1024:.1f} KB")
with c3:
    max_size = message_fact["size_bytes"].max()
    st.metric("Largest Message", f"{max_size / (1024*1024):.1f} MB")
with c4:
    zero_count = len(message_fact.filter(pl.col("size_bytes") == 0))
    st.metric("Zero-Size Messages", f"{zero_count:,}")

st.divider()

# Size distribution histogram (log scale)
col_left, col_right = st.columns(2)

with col_left:
    st.subheader("Size Distribution (Log Scale)")
    sizes = message_fact.filter(pl.col("size_bytes") > 0)["size_bytes"].to_numpy()
    if len(sizes) > 0:
        log_sizes = np.log10(sizes.astype(float))
        fig = px.histogram(x=log_sizes, nbins=50,
                           title="Message Size Distribution",
                           labels={"x": "log10(size in bytes)", "y": "Count"})
        fig.update_layout(height=400)
        st.plotly_chart(fig, width="stretch")

with col_right:
    st.subheader("Size Class Breakdown")
    class_counts = (
        classified.group_by("size_class")
        .agg(pl.len().alias("count"))
        .sort("count", descending=True)
    )
    fig2 = px.pie(class_counts.to_pandas(), names="size_class", values="count",
                  title="Message Size Classes")
    fig2.update_layout(height=400)
    st.plotly_chart(fig2, width="stretch")

st.divider()

# Detected templates (cached)
st.subheader("Detected Size Templates (Auto-Generated?)")
st.markdown("Messages with frequently repeated exact sizes may be auto-generated.")
templates = _cached_size_templates(start_date, end_date)
if len(templates) > 0:
    templates_display = templates.with_columns(
        (pl.col("size_bytes") / 1024).round(1).alias("size_kb")
    ).head(30)
    st.dataframe(templates_display.to_pandas(), width="stretch")
    download_csv_button(templates_display, "size_templates.csv")
else:
    st.info("No frequently repeated exact sizes detected.")

st.divider()

# Per-sender size profile (cached)
st.subheader("Sender Size Profiles")
sender_profiles = _cached_sender_profiles(start_date, end_date)
if len(sender_profiles) > 0:
    top_senders = sender_profiles.filter(pl.col("msg_count") >= 10).head(30)
    top_pd = top_senders.with_columns(
        (pl.col("avg_size") / 1024).alias("avg_size_kb")
    ).to_pandas()
    fig3 = px.bar(top_pd, x="from_email", y="avg_size_kb",
                  color="dominant_class",
                  title="Average Message Size by Sender (Top 30)",
                  labels={"avg_size_kb": "Avg Size (KB)"})
    fig3.update_layout(height=400, xaxis_tickangle=-45)
    ev_profiles = st.plotly_chart(fig3, width="stretch", on_select="rerun", key="p17_profiles")
    handle_plotly_person_click(ev_profiles, "p17_profiles", start_date, end_date)

st.divider()

# Size anomalies (cached)
st.subheader("Size Anomalies")
st.markdown("Messages with unusual size for their sender (z-score > 3).")
anomalies = _cached_size_anomalies(start_date, end_date)
if len(anomalies) > 0:
    st.write(f"**{len(anomalies)} anomalous messages** detected.")
    anom_pd = anomalies.head(50).to_pandas()
    ev_anom = st.dataframe(anom_pd, width="stretch", on_select="rerun", selection_mode="single-row", key="p17_anom_df")
    handle_dataframe_person_click(ev_anom, anom_pd, "p17_anom_df", "from_email", start_date, end_date)
    download_csv_button(anomalies, "size_anomalies.csv")
else:
    st.info("No size anomalies detected.")
