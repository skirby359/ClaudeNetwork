"""Page 10: Risk Register — Anomalies and flags."""

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import polars as pl

from src.state import (
    load_person_dim,
    render_date_filter,
    load_filtered_message_fact, load_filtered_edge_fact,
    load_filtered_weekly_agg,
    load_nonhuman_emails,
)
from src.analytics.anomaly import (
    detect_volume_anomalies,
    detect_sender_anomalies,
)
from src.export import download_csv_button
from src.drilldown import (
    handle_plotly_week_click, handle_dataframe_person_click,
)


# ---------------------------------------------------------------------------
# Cached analytics
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner=False, ttl=3600)
def _cached_volume_anomalies(start_date, end_date, z_threshold):
    wa = load_filtered_weekly_agg(start_date, end_date)
    return detect_volume_anomalies(wa, z_threshold=z_threshold)


@st.cache_data(show_spinner=False, ttl=3600)
def _cached_sender_anomalies(start_date, end_date, exclude_nonhuman, z_threshold):
    ef = load_filtered_edge_fact(start_date, end_date)
    pd_dim = load_person_dim()
    if exclude_nonhuman:
        nonhuman = load_nonhuman_emails(start_date, end_date)
        ef = ef.filter(
            ~pl.col("from_email").is_in(list(nonhuman))
            & ~pl.col("to_email").is_in(list(nonhuman))
        )
    return detect_sender_anomalies(ef, pd_dim, z_threshold=z_threshold)


# ---------------------------------------------------------------------------
# Page layout
# ---------------------------------------------------------------------------

st.set_page_config(page_title="Risk Register", layout="wide")
st.title("Risk Register")

st.markdown("""
This page flags statistical anomalies and potential risks detected in email patterns.
Anomalies are detected using z-score analysis (default threshold: 2.5 standard deviations).
""")

start_date, end_date = render_date_filter()

message_fact = load_filtered_message_fact(start_date, end_date)
edge_fact = load_filtered_edge_fact(start_date, end_date)

if len(message_fact) == 0:
    st.warning("No data in selected date range.")
    st.stop()

person_dim = load_person_dim()
weekly_agg = load_filtered_weekly_agg(start_date, end_date)

# --- Nonhuman filter (cached) ---
nonhuman_emails = load_nonhuman_emails(start_date, end_date)

filter_on = st.checkbox("Hide nonhuman addresses (copiers, bots, system accounts)", value=True)

if filter_on:
    message_fact = message_fact.filter(~pl.col("from_email").is_in(list(nonhuman_emails)))
    edge_fact = edge_fact.filter(
        ~pl.col("from_email").is_in(list(nonhuman_emails))
        & ~pl.col("to_email").is_in(list(nonhuman_emails))
    )
    st.caption(f"Filtered out {len(nonhuman_emails)} nonhuman addresses.")

z_threshold = st.slider("Z-score threshold", 1.5, 4.0, 2.5, 0.1)

# Volume anomalies (cached)
st.subheader("Volume Anomalies")
vol_anom = _cached_volume_anomalies(start_date, end_date, z_threshold)
anomalous_weeks = vol_anom.filter(pl.col("is_volume_anomaly"))

st.write(f"**{len(anomalous_weeks)} anomalous weeks** detected out of {len(vol_anom)}")

vol_pd = vol_anom.to_pandas()
fig = go.Figure()
fig.add_trace(go.Bar(x=vol_pd["week_start"], y=vol_pd["msg_count"],
                     marker_color=["red" if a else "steelblue" for a in vol_pd["is_volume_anomaly"]],
                     name="Weekly Volume"))
fig.update_layout(height=400, title="Weekly Volume (Anomalous Weeks in Red)")
ev_vol = st.plotly_chart(fig, width="stretch", on_select="rerun", key="p10_vol")
handle_plotly_week_click(ev_vol, "p10_vol", start_date, end_date)

if len(anomalous_weeks) > 0:
    st.markdown("**Anomalous weeks:**")
    st.dataframe(anomalous_weeks.select([
        "week_id", "week_start", "msg_count", "volume_zscore"
    ]).to_pandas(), width="stretch")

# Sender anomalies (cached)
st.divider()
st.subheader("Sender Anomalies")
sender_anom = _cached_sender_anomalies(start_date, end_date, filter_on, z_threshold)
st.write(f"**{len(sender_anom)} anomalous senders** detected")

if len(sender_anom) > 0:
    anom_pd = sender_anom.select([
        "from_email", "total_sent", "unique_recipients",
        "after_hours_rate", "weekend_rate",
        "total_sent_zscore", "unique_recipients_zscore",
    ]).to_pandas()
    ev_anom = st.dataframe(anom_pd, width="stretch", on_select="rerun", selection_mode="single-row", key="p10_anom_df")
    handle_dataframe_person_click(ev_anom, anom_pd, "p10_anom_df", "from_email", start_date, end_date)

    st.markdown("**Anomaly breakdown:**")
    col1, col2 = st.columns(2)
    with col1:
        high_vol = len(sender_anom.filter(pl.col("total_sent_zscore").abs() > z_threshold))
        high_recip = len(sender_anom.filter(pl.col("unique_recipients_zscore").abs() > z_threshold))
        st.write(f"- High/low volume: {high_vol} senders")
        st.write(f"- Unusual recipient diversity: {high_recip} senders")
    with col2:
        high_ah = len(sender_anom.filter(pl.col("after_hours_rate_zscore").abs() > z_threshold))
        high_we = len(sender_anom.filter(pl.col("weekend_rate_zscore").abs() > z_threshold))
        st.write(f"- Unusual after-hours rate: {high_ah} senders")
        st.write(f"- Unusual weekend rate: {high_we} senders")

# Self-send detection
st.divider()
st.subheader("Self-Sends")
self_sends = edge_fact.filter(pl.col("from_email") == pl.col("to_email"))
self_send_count = (
    self_sends.group_by("from_email")
    .agg(pl.len().alias("self_send_count"))
    .sort("self_send_count", descending=True)
)
st.write(f"**{len(self_sends)} self-sent messages** ({len(self_sends)/max(len(edge_fact),1)*100:.1f}% of all edges)")
self_pd = self_send_count.head(20).to_pandas()
ev_self = st.dataframe(self_pd, width="stretch", on_select="rerun", selection_mode="single-row", key="p10_self_df")
handle_dataframe_person_click(ev_self, self_pd, "p10_self_df", "from_email", start_date, end_date)

# Summary risk table
st.divider()
st.subheader("Risk Summary")
risks = []
if len(anomalous_weeks) > 0:
    risks.append({"Risk": "Volume anomaly weeks", "Count": len(anomalous_weeks),
                  "Severity": "Medium", "Detail": "Weeks with unusual message volume"})
if len(sender_anom) > 0:
    risks.append({"Risk": "Anomalous senders", "Count": len(sender_anom),
                  "Severity": "Low", "Detail": "Senders with unusual patterns"})
if len(self_sends) > 100:
    risks.append({"Risk": "High self-send volume", "Count": len(self_sends),
                  "Severity": "Info", "Detail": "May indicate archiving or forwarding patterns"})

if risks:
    st.dataframe(pl.DataFrame(risks).to_pandas(), width="stretch")
else:
    st.success("No significant risks detected.")

download_csv_button(sender_anom, "sender_anomalies.csv", "Download Anomalous Senders")
