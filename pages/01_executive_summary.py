"""Page 1: Executive Summary — Key findings at a glance."""

import streamlit as st
import plotly.express as px
import polars as pl

from src.page_logger import log_page_entry, log_page_error
from src.state import (
    load_person_dim, load_message_fact,
    render_date_filter, render_comparison_filter,
    load_filtered_message_fact, load_filtered_edge_fact,
    load_filtered_weekly_agg, load_filtered_broadcast,
    load_filtered_graph_metrics, load_nonhuman_emails,
)
from src.analytics.volume import compute_sender_concentration
from src.analytics.comparison import compute_period_summary, compute_delta
from src.analytics.narrative import generate_executive_narrative
from src.analytics.silos import identify_bridges, simulate_removal
from src.analytics.network import build_graph
from src.analytics.response_time import compute_reply_times, compute_person_response_stats
from src.export import download_csv_button
from src.report import generate_executive_report
from src.anonymize import anon, anon_df
from src.drilldown import handle_plotly_person_click, handle_plotly_week_click


# ---------------------------------------------------------------------------
# Cached analytics
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner=False, ttl=3600)
def _cached_period_summary(start_date, end_date):
    mf = load_filtered_message_fact(start_date, end_date)
    ef = load_filtered_edge_fact(start_date, end_date)
    return compute_period_summary(mf, ef)


@st.cache_data(show_spinner=False, ttl=3600)
def _cached_sender_concentration(start_date, end_date):
    ef = load_filtered_edge_fact(start_date, end_date)
    return compute_sender_concentration(ef)


@st.cache_data(show_spinner=False, ttl=3600)
def _cached_narrative(start_date, end_date):
    mf = load_filtered_message_fact(start_date, end_date)
    wa = load_filtered_weekly_agg(start_date, end_date)
    ef = load_filtered_edge_fact(start_date, end_date)
    pd_dim = load_person_dim()
    return generate_executive_narrative(mf, wa, ef, pd_dim)


@st.cache_data(show_spinner=False, ttl=3600)
def _cached_bridges(start_date, end_date):
    ef = load_filtered_edge_fact(start_date, end_date)
    gm = load_filtered_graph_metrics(start_date, end_date)
    community_lookup = dict(zip(gm["email"].to_list(), gm["community_id"].to_list()))
    G = build_graph(ef)
    return identify_bridges(G, community_lookup)


@st.cache_data(show_spinner=False, ttl=3600)
def _cached_reply_summary(start_date, end_date):
    ef = load_filtered_edge_fact(start_date, end_date)
    nonhuman = load_nonhuman_emails(start_date, end_date)
    # Filter to human-only for meaningful response times
    ef_human = ef.filter(
        ~pl.col("from_email").is_in(list(nonhuman))
        & ~pl.col("to_email").is_in(list(nonhuman))
    )
    reply_times = compute_reply_times(ef_human)
    if len(reply_times) == 0:
        return None, 0, 0
    median_sec = reply_times["median_reply_seconds"].median()
    total_replies = int(reply_times["reply_count"].sum())
    return reply_times, median_sec, total_replies


# ---------------------------------------------------------------------------
# Page layout
# ---------------------------------------------------------------------------

st.set_page_config(page_title="Executive Summary", layout="wide")
_page_log = log_page_entry("01_executive_summary")
st.title("Executive Summary")

start_date, end_date = render_date_filter()

# Full dataset bounds for comparison slider
full_mf = load_message_fact()
data_min = full_mf["timestamp"].min().date()
data_max = full_mf["timestamp"].max().date()

# Comparison mode
comp_enabled, comp_start, comp_end = render_comparison_filter(data_min, data_max)

message_fact = load_filtered_message_fact(start_date, end_date)
edge_fact = load_filtered_edge_fact(start_date, end_date)
person_dim = load_person_dim()
weekly_agg = load_filtered_weekly_agg(start_date, end_date)
nonhuman_emails = load_nonhuman_emails(start_date, end_date)

if len(message_fact) == 0:
    st.warning("No data in selected date range.")
    st.stop()

# =========================================================================
# Section 1: Human vs Machine Communication
# =========================================================================
st.subheader("Communication Composition")

nh_list = list(nonhuman_emails)
human_msgs = len(message_fact.filter(~pl.col("from_email").is_in(nh_list)))
machine_msgs = len(message_fact) - human_msgs
human_pct = human_msgs / max(len(message_fact), 1) * 100

c_h, c_m, c_t, c_p = st.columns([2, 2, 2, 1])
with c_h:
    st.metric("Human Messages", f"{human_msgs:,}")
with c_m:
    st.metric("Automated Messages", f"{machine_msgs:,}")
with c_t:
    st.metric("Total Messages", f"{len(message_fact):,}")
with c_p:
    fig_split = px.pie(
        values=[human_msgs, machine_msgs],
        names=["Human", "Automated"],
        color_discrete_sequence=["#4e79a7", "#e15759"],
        hole=0.6,
    )
    fig_split.update_layout(height=180, margin=dict(l=0, r=0, t=0, b=0), showlegend=False)
    fig_split.update_traces(textinfo="percent")
    st.plotly_chart(fig_split, use_container_width=True)

if machine_msgs > human_msgs:
    st.info(
        f"Only **{human_pct:.0f}%** of messages are human-to-human. "
        f"The remaining **{100-human_pct:.0f}%** comes from automated systems "
        f"(copiers, alerts, notifications). Human-focused analysis filters these out."
    )

st.divider()

# =========================================================================
# Section 2: Key Metrics
# =========================================================================
current_summary = _cached_period_summary(start_date, end_date)

delta_info = None
if comp_enabled and comp_start and comp_end:
    prev_summary = _cached_period_summary(comp_start, comp_end)
    delta_info = compute_delta(current_summary, prev_summary)

c1, c2, c3, c4, c5 = st.columns(5)
with c1:
    st.metric("Unique People", f"{len(person_dim):,}")
with c2:
    internal = person_dim.filter(person_dim["is_internal"])
    st.metric("Internal Staff", f"{len(internal):,}")
with c3:
    external_count = len(person_dim) - len(internal)
    st.metric("External Contacts", f"{external_count:,}")
with c4:
    total_bytes = message_fact["size_bytes"].sum()
    st.metric("Data Volume", f"{total_bytes / (1024**3):.1f} GB")
with c5:
    gm = load_filtered_graph_metrics(start_date, end_date)
    n_communities = gm["community_id"].n_unique() if len(gm) > 0 else 0
    st.metric("Communication Groups", f"{n_communities}")

st.divider()

# =========================================================================
# Section 3: Critical Personnel Risk
# =========================================================================
st.subheader("Critical Personnel Risk")
st.caption("People whose departure would fragment inter-group communication.")

try:
    bridges = _cached_bridges(start_date, end_date)
    if len(bridges) > 0:
        top_bridges = bridges.head(5)
        top_bridges_display = anon_df(top_bridges)

        col_chart, col_detail = st.columns([3, 2])
        with col_chart:
            fig_bridges = px.bar(
                top_bridges_display.to_pandas(),
                x="email", y="communities_bridged",
                title="Top 5 Cross-Group Connectors",
                labels={"email": "Person", "communities_bridged": "Groups Connected"},
                color_discrete_sequence=["#e15759"],
            )
            fig_bridges.update_layout(height=300, xaxis_tickangle=-45)
            ev_bridges = st.plotly_chart(fig_bridges, width="stretch", on_select="rerun", key="p01_bridges")
            handle_plotly_person_click(ev_bridges, "p01_bridges", start_date, end_date)

        with col_detail:
            st.write(f"**{len(bridges)} people** span multiple communication groups.")
            st.write("If any of the top 5 leave, departments may lose contact.")
            for row in top_bridges_display.head(5).iter_rows(named=True):
                st.write(f"- **{anon(row['email'])}** connects {row['communities_bridged']} groups")
    else:
        st.info("Not enough data to identify bridge people in this date range.")
except Exception:
    st.info("Bridge analysis requires a wider date range.")

st.divider()

# =========================================================================
# Section 4: Responsiveness & Work-Life Balance
# =========================================================================
st.subheader("Responsiveness & Work Patterns")

col_resp, col_wlb = st.columns(2)

with col_resp:
    try:
        reply_data, median_sec, total_replies = _cached_reply_summary(start_date, end_date)
        if median_sec and median_sec > 0:
            median_min = median_sec / 60
            r1, r2 = st.columns(2)
            with r1:
                st.metric("Median Reply Time", f"{median_min:.0f} min",
                          help="How fast people typically respond (human messages only)")
            with r2:
                st.metric("Reply Pairs Detected", f"{total_replies:,}",
                          help="Unique sender-recipient pairs with detected replies")
        else:
            st.info("Not enough reply data in this date range.")
    except Exception:
        st.info("Response time analysis requires more data.")

with col_wlb:
    # Human-only after-hours rates
    human_mf = message_fact.filter(~pl.col("from_email").is_in(nh_list))
    if len(human_mf) > 0:
        human_ah = float(human_mf["is_after_hours"].mean())
        human_we = float(human_mf["is_weekend"].mean())
        w1, w2 = st.columns(2)
        with w1:
            st.metric("After-Hours (Human)", f"{human_ah:.1%}",
                      help="Staff sending email before 7AM or after 6PM")
        with w2:
            st.metric("Weekend (Human)", f"{human_we:.1%}",
                      help="Staff sending email on Saturday or Sunday")

        overall_ah = float(message_fact["is_after_hours"].mean())
        if overall_ah > human_ah + 0.02:
            st.caption(
                f"Overall after-hours rate is {overall_ah:.1%} (including automated systems). "
                f"Human staff rate of {human_ah:.1%} reflects actual work-life balance."
            )

st.divider()

# =========================================================================
# Section 5: Volume Trends & Concentration
# =========================================================================
col_left, col_right = st.columns(2)

with col_left:
    st.subheader("Weekly Message Volume")
    wa = weekly_agg.to_pandas()
    fig = px.line(wa, x="week_start", y="msg_count", title="Messages per Week")
    fig.update_layout(height=350)
    ev_weekly = st.plotly_chart(fig, width="stretch", on_select="rerun", key="p01_weekly")
    handle_plotly_week_click(ev_weekly, "p01_weekly", start_date, end_date)

with col_right:
    st.subheader("Sender Concentration")
    conc = _cached_sender_concentration(start_date, end_date)
    st.write(
        f"**Concentration score:** {conc['gini']:.2f} / 1.00 "
        f"({'highly concentrated' if conc['gini'] > 0.8 else 'moderate' if conc['gini'] > 0.5 else 'balanced'})"
    )
    st.write(f"**Top 10 senders** account for {conc['top_10_share']:.1%} of all message activity")

    top_senders = anon_df(conc["top_senders"].head(10))
    fig2 = px.bar(top_senders.to_pandas(), x="from_email", y="count", title="Top 10 Senders by Volume")
    fig2.update_layout(height=350, xaxis_tickangle=-45)
    ev_senders = st.plotly_chart(fig2, width="stretch", on_select="rerun", key="p01_senders")
    handle_plotly_person_click(ev_senders, "p01_senders", start_date, end_date)

st.divider()

# =========================================================================
# Section 6: External Dependencies
# =========================================================================
st.subheader("Top External Partners")
st.caption("External organizations your staff communicate with most.")

external_domains = (
    edge_fact.filter(~pl.col("to_email").str.split("@").list.last().is_in(
        [d.lower() for d in person_dim.filter(pl.col("is_internal"))["domain"].unique().to_list()]
    ))
    .with_columns(pl.col("to_email").str.split("@").list.last().alias("ext_domain"))
    .group_by("ext_domain")
    .agg(pl.len().alias("messages"))
    .sort("messages", descending=True)
    .head(10)
)

if len(external_domains) > 0:
    fig_ext = px.bar(
        external_domains.to_pandas(), x="ext_domain", y="messages",
        title="Top 10 External Domains by Message Volume",
        labels={"ext_domain": "External Organization", "messages": "Messages"},
    )
    fig_ext.update_layout(height=300, xaxis_tickangle=-45)
    st.plotly_chart(fig_ext, width="stretch")

st.divider()

# =========================================================================
# Section 7: Executive Narrative
# =========================================================================
st.subheader("Executive Narrative")
narrative = _cached_narrative(start_date, end_date)
st.markdown(narrative)

# =========================================================================
# Section 8: Export
# =========================================================================
st.divider()
col_export_a, col_export_b = st.columns(2)
with col_export_a:
    download_csv_button(weekly_agg, "weekly_aggregation.csv", "Download Weekly Data")
with col_export_b:
    org_name = st.text_input("Organization name (for report)", value="Organization", key="p01_org_name")
    if st.button("Generate Executive Report", type="primary"):
        report_html = generate_executive_report(
            message_fact, edge_fact, person_dim, weekly_agg, gm,
            narrative, start_date, end_date, org_name=org_name,
        )
        st.download_button(
            label="Download Report (HTML)",
            data=report_html.encode("utf-8"),
            file_name=f"email_analysis_{start_date}_{end_date}.html",
            mime="text/html",
        )
