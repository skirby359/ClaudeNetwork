"""Page 28: Person Comparison — Side-by-side behavioral metrics for two people."""

import streamlit as st
import plotly.graph_objects as go
import polars as pl

from src.page_logger import log_page_entry, log_page_error
from src.state import (
    render_date_filter, load_filtered_edge_fact, load_filtered_message_fact,
    load_filtered_graph_metrics, load_nonhuman_emails, load_person_dim,
)
from src.analytics.response_time import compute_reply_times
from src.export import download_csv_button


@st.cache_data(show_spinner=False, ttl=3600)
def _person_metrics(start_date, end_date, email):
    """Compute comprehensive metrics for one person."""
    ef = load_filtered_edge_fact(start_date, end_date)
    mf = load_filtered_message_fact(start_date, end_date)
    gm = load_filtered_graph_metrics(start_date, end_date)
    pd_dim = load_person_dim()

    # Basic send/receive counts
    sent = ef.filter(pl.col("from_email") == email)
    received = ef.filter(pl.col("to_email") == email)
    sent_msgs = mf.filter(pl.col("from_email") == email)

    n_sent = len(sent)
    n_received = len(received)
    unique_contacts_to = sent["to_email"].n_unique() if n_sent > 0 else 0
    unique_contacts_from = received["from_email"].n_unique() if n_received > 0 else 0

    # After-hours rate
    ah_rate = float(sent_msgs["is_after_hours"].mean()) * 100 if len(sent_msgs) > 0 else 0.0

    # Average message size
    avg_size = float(sent_msgs["size_bytes"].mean()) / 1024 if len(sent_msgs) > 0 else 0.0

    # Average recipients per message
    avg_recipients = float(sent_msgs["n_recipients"].mean()) if len(sent_msgs) > 0 else 0.0

    # Graph metrics — safe extraction helper
    def _get(df, col, default=0):
        if len(df) > 0 and col in df.columns:
            val = df[col].to_list()[0]
            return val if val is not None else default
        return default

    node = gm.filter(pl.col("email") == email)
    betweenness = float(_get(node, "betweenness_centrality", 0.0))
    pagerank = float(_get(node, "pagerank", 0.0))
    community_id = _get(node, "community_id", -1)
    community_label = _get(node, "community_label", f"Group {community_id}")
    in_degree = int(_get(node, "in_degree", 0))
    out_degree = int(_get(node, "out_degree", 0))

    # Reply time
    reply_median = None
    try:
        rt = compute_reply_times(ef)
        person_rt = rt.filter(pl.col("person_b") == email)
        if len(person_rt) > 0:
            reply_median = float(person_rt["median_reply_seconds"].median()) / 60  # minutes
    except Exception:
        pass

    # Hourly distribution
    hourly = (
        sent_msgs.group_by("hour")
        .agg(pl.len().alias("count"))
        .sort("hour")
    ) if len(sent_msgs) > 0 else pl.DataFrame({"hour": [], "count": []})

    # Weekly volume
    weekly = (
        sent_msgs.with_columns(pl.col("timestamp").dt.truncate("1w").alias("week"))
        .group_by("week")
        .agg(pl.len().alias("count"))
        .sort("week")
    ) if len(sent_msgs) > 0 else pl.DataFrame({"week": [], "count": []})

    # Top contacts (sent to)
    top_contacts = (
        sent.group_by("to_email")
        .agg(pl.len().alias("msg_count"))
        .sort("msg_count", descending=True)
        .head(10)
    ) if n_sent > 0 else pl.DataFrame({"to_email": [], "msg_count": []})

    # Display name and department
    name_row = pd_dim.filter(pl.col("email") == email)
    display_name = _get(name_row, "display_name", email) or email
    department = _get(name_row, "department", "Unknown") or "Unknown"

    return {
        "email": email,
        "display_name": display_name,
        "n_sent": n_sent,
        "n_received": n_received,
        "unique_contacts_to": unique_contacts_to,
        "unique_contacts_from": unique_contacts_from,
        "after_hours_pct": round(ah_rate, 1),
        "avg_size_kb": round(avg_size, 1),
        "avg_recipients": round(avg_recipients, 1),
        "betweenness": round(betweenness, 6),
        "pagerank": round(pagerank, 6),
        "community_label": community_label,
        "in_degree": in_degree,
        "out_degree": out_degree,
        "department": department,
        "reply_time_min": round(reply_median, 1) if reply_median else None,
        "hourly": hourly,
        "weekly": weekly,
        "top_contacts": top_contacts,
    }


st.set_page_config(page_title="Person Comparison", layout="wide")
_page_log = log_page_entry("28_person_comparison")
st.title("Person Comparison")
st.caption("Side-by-side behavioral metrics for two people.")

start_date, end_date = render_date_filter()

# Build person list
gm = load_filtered_graph_metrics(start_date, end_date)
person_dim = load_person_dim()
nonhuman_emails = load_nonhuman_emails(start_date, end_date)
filter_nonhuman = st.session_state.get("exclude_nonhuman", True)

available = gm.select("email")
if filter_nonhuman and nonhuman_emails:
    available = available.filter(~pl.col("email").is_in(list(nonhuman_emails)))

# Join display names for the selector
available = available.join(
    person_dim.select(["email", "display_name"]), on="email", how="left",
)
email_list = available["email"].sort().to_list()
name_lookup = dict(zip(available["email"].to_list(), available["display_name"].to_list()))


def _format_option(email):
    name = name_lookup.get(email, "")
    if name and name != email:
        return f"{name} ({email})"
    return email


if len(email_list) < 2:
    st.warning("Need at least 2 people in the selected date range.")
    st.stop()

# Selection
col_sel1, col_sel2 = st.columns(2)
with col_sel1:
    person_a = st.selectbox("Person A", email_list, index=0, format_func=_format_option, key="cmp_a")
with col_sel2:
    default_b = 1 if len(email_list) > 1 else 0
    person_b = st.selectbox("Person B", email_list, index=default_b, format_func=_format_option, key="cmp_b")

if person_a == person_b:
    st.info("Select two different people to compare.")
    st.stop()

# Compute metrics
metrics_a = _person_metrics(start_date, end_date, person_a)
metrics_b = _person_metrics(start_date, end_date, person_b)

# --- KPI Comparison Table ---
st.divider()
st.subheader("Key Metrics")

metric_rows = [
    ("Department", "department", "{}"),
    ("Messages Sent", "n_sent", "{:,}"),
    ("Messages Received", "n_received", "{:,}"),
    ("Unique Contacts (Outbound)", "unique_contacts_to", "{:,}"),
    ("Unique Contacts (Inbound)", "unique_contacts_from", "{:,}"),
    ("After-Hours Rate", "after_hours_pct", "{}%"),
    ("Avg Message Size", "avg_size_kb", "{} KB"),
    ("Avg Recipients/Message", "avg_recipients", "{}"),
    ("Connector Score", "betweenness", "{}"),
    ("Importance Score", "pagerank", "{}"),
    ("In-Degree (Weighted)", "in_degree", "{:,}"),
    ("Out-Degree (Weighted)", "out_degree", "{:,}"),
    ("Community", "community_label", "{}"),
    ("Reply Time (Median)", "reply_time_min", "{} min"),
]

col_label, col_a, col_b = st.columns([2, 1, 1])
with col_label:
    st.markdown("**Metric**")
with col_a:
    name_a = metrics_a["display_name"]
    st.markdown(f"**{name_a}**")
with col_b:
    name_b = metrics_b["display_name"]
    st.markdown(f"**{name_b}**")

for label, key, fmt in metric_rows:
    col_label, col_a, col_b = st.columns([2, 1, 1])
    val_a = metrics_a.get(key)
    val_b = metrics_b.get(key)
    with col_label:
        st.write(label)
    with col_a:
        st.write(fmt.format(val_a) if val_a is not None else "N/A")
    with col_b:
        st.write(fmt.format(val_b) if val_b is not None else "N/A")

# --- Radar Chart ---
st.divider()
st.subheader("Behavioral Profile Comparison")

# Normalize metrics to 0-100 for radar
radar_metrics = ["n_sent", "n_received", "unique_contacts_to", "after_hours_pct", "avg_recipients"]
radar_labels = ["Send Volume", "Receive Volume", "Contact Breadth", "After-Hours %", "Broadcast Rate"]

# Get max for normalization
max_vals = {}
for key in radar_metrics:
    vals = [metrics_a.get(key, 0) or 0, metrics_b.get(key, 0) or 0]
    max_vals[key] = max(vals) if max(vals) > 0 else 1

norm_a = [(metrics_a.get(k, 0) or 0) / max_vals[k] * 100 for k in radar_metrics]
norm_b = [(metrics_b.get(k, 0) or 0) / max_vals[k] * 100 for k in radar_metrics]

fig_radar = go.Figure()
fig_radar.add_trace(go.Scatterpolar(
    r=norm_a + [norm_a[0]],
    theta=radar_labels + [radar_labels[0]],
    fill="toself",
    name=name_a,
    fillcolor="rgba(78, 121, 167, 0.2)",
    line=dict(color="#4e79a7"),
))
fig_radar.add_trace(go.Scatterpolar(
    r=norm_b + [norm_b[0]],
    theta=radar_labels + [radar_labels[0]],
    fill="toself",
    name=name_b,
    fillcolor="rgba(225, 87, 89, 0.2)",
    line=dict(color="#e15759"),
))
fig_radar.update_layout(
    polar=dict(radialaxis=dict(visible=True, range=[0, 105])),
    height=400,
)
st.plotly_chart(fig_radar, use_container_width=True)

# --- Hourly Activity ---
st.divider()
st.subheader("Hourly Activity Pattern")

col_h1, col_h2 = st.columns(2)

for col, metrics, name, color in [
    (col_h1, metrics_a, name_a, "#4e79a7"),
    (col_h2, metrics_b, name_b, "#e15759"),
]:
    with col:
        hourly = metrics["hourly"]
        if len(hourly) > 0:
            # Ensure all 24 hours
            hours = pl.DataFrame({"hour": list(range(24))})
            hourly_full = hours.join(hourly, on="hour", how="left").with_columns(
                pl.col("count").fill_null(0)
            )
            fig_h = go.Figure(go.Bar(
                x=hourly_full["hour"].to_list(),
                y=hourly_full["count"].to_list(),
                marker_color=color,
            ))
            fig_h.update_layout(
                title=name, height=250,
                xaxis=dict(dtick=2, title="Hour"),
                yaxis_title="Messages",
                margin=dict(l=40, r=10, t=40, b=40),
            )
            st.plotly_chart(fig_h, use_container_width=True)
        else:
            st.info(f"No hourly data for {name}")

# --- Weekly Volume Trend ---
st.divider()
st.subheader("Weekly Volume Trend")

fig_weekly = go.Figure()
for metrics, name, color in [
    (metrics_a, name_a, "#4e79a7"),
    (metrics_b, name_b, "#e15759"),
]:
    weekly = metrics["weekly"]
    if len(weekly) > 0:
        fig_weekly.add_trace(go.Scatter(
            x=weekly["week"].to_list(),
            y=weekly["count"].to_list(),
            mode="lines+markers",
            name=name,
            line=dict(color=color),
        ))
fig_weekly.update_layout(height=350, yaxis_title="Messages Sent", xaxis_title="Week")
st.plotly_chart(fig_weekly, use_container_width=True)

# --- Top Contacts ---
st.divider()
st.subheader("Top Contacts")

col_c1, col_c2 = st.columns(2)
for col, metrics, name in [(col_c1, metrics_a, name_a), (col_c2, metrics_b, name_b)]:
    with col:
        st.write(f"**{name}**")
        tc = metrics["top_contacts"]
        if len(tc) > 0:
            st.dataframe(tc.rename({"to_email": "Contact", "msg_count": "Messages"}).to_pandas(),
                         use_container_width=True, height=min(350, len(tc) * 35 + 40))
        else:
            st.info("No outbound messages")
