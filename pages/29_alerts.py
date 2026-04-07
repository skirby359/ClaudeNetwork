"""Page 29: Alert Dashboard — Configurable thresholds with real-time flagging."""

import streamlit as st
import plotly.express as px
import polars as pl

from src.state import (
    render_date_filter, load_filtered_message_fact, load_filtered_edge_fact,
    load_filtered_graph_metrics, load_nonhuman_emails, load_person_dim,
)
from src.analytics.health_score import compute_health_score
from src.analytics.response_time import compute_reply_times
from src.analytics.compliance import detect_blackout_windows
from src.engagement import (
    evaluate_alerts, default_alert_rules,
)
from src.export import download_csv_button


@st.cache_data(show_spinner="Evaluating alerts...", ttl=3600)
def _cached_alerts(start_date, end_date, rules_json):
    """Run alert evaluation with current rules."""
    import json
    rules = json.loads(rules_json)

    mf = load_filtered_message_fact(start_date, end_date)
    ef = load_filtered_edge_fact(start_date, end_date)
    gm = load_filtered_graph_metrics(start_date, end_date)
    nonhuman = load_nonhuman_emails(start_date, end_date)
    nh_list = list(nonhuman)

    mf_human = mf.filter(~pl.col("from_email").is_in(nh_list))
    ef_human = ef.filter(
        ~pl.col("from_email").is_in(nh_list) & ~pl.col("to_email").is_in(nh_list)
    )

    # Health score
    reply_median = None
    try:
        rt = compute_reply_times(ef_human)
        if len(rt) > 0:
            reply_median = float(rt["median_reply_seconds"].median())
    except Exception:
        pass
    health = compute_health_score(mf_human, ef_human, gm, reply_median)

    # Blackouts
    blackouts = detect_blackout_windows(mf_human, min_gap_hours=24, min_historical_volume=3)

    # Bus factor (optional — may not be available)
    team_bf = None
    try:
        from src.analytics.bus_factor import compute_team_bus_factor
        from src.analytics.hierarchy import infer_reciprocal_teams
        from src.analytics.network import build_graph
        pd_dim = load_person_dim()
        G = build_graph(ef)
        teams = infer_reciprocal_teams(ef, pd_dim, exclude_emails=set(nh_list))
        if len(teams) > 0:
            team_bf = compute_team_bus_factor(teams, G)
    except Exception:
        pass

    alerts = evaluate_alerts(
        rules, mf_human, ef_human, gm,
        health_score=health,
        blackouts=blackouts,
        team_bus_factor=team_bf,
    )

    return alerts, health


st.set_page_config(page_title="Alert Dashboard", layout="wide")
st.title("Alert Dashboard")
st.caption("Configurable thresholds that flag communication risks across the organization.")

start_date, end_date = render_date_filter()

# --- Sidebar: Alert Rule Configuration ---
with st.sidebar:
    st.subheader("Alert Rules")

    # Initialize rules from session state or defaults
    if "_alert_rules" not in st.session_state:
        st.session_state._alert_rules = default_alert_rules()

    rules = st.session_state._alert_rules

    with st.expander("Configure Thresholds", expanded=False):
        updated_rules = []
        for i, rule in enumerate(rules):
            st.markdown(f"**{rule['name']}**")
            cols = st.columns([2, 1, 1])
            with cols[0]:
                st.caption(rule["description"])
            with cols[1]:
                new_thresh = st.number_input(
                    "Threshold",
                    value=float(rule["threshold"]),
                    key=f"alert_thresh_{i}",
                    label_visibility="collapsed",
                )
            with cols[2]:
                new_sev = st.selectbox(
                    "Severity",
                    ["critical", "warning", "info"],
                    index=["critical", "warning", "info"].index(rule.get("severity", "warning")),
                    key=f"alert_sev_{i}",
                    label_visibility="collapsed",
                )

            updated_rule = dict(rule)
            updated_rule["threshold"] = new_thresh
            updated_rule["severity"] = new_sev
            updated_rules.append(updated_rule)
            st.divider()

        st.session_state._alert_rules = updated_rules
        rules = updated_rules

    if st.button("Reset to Defaults", key="alert_reset"):
        st.session_state._alert_rules = default_alert_rules()
        st.rerun()

# Serialize rules for cache key
import json
rules_json = json.dumps(rules, default=str)

alerts, health = _cached_alerts(start_date, end_date, rules_json)

# --- KPI Row ---
st.divider()
critical_count = sum(1 for a in alerts if a.get("severity") == "critical")
warning_count = sum(1 for a in alerts if a.get("severity") == "warning")
info_count = sum(1 for a in alerts if a.get("severity") == "info")

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Total Alerts", len(alerts))
with col2:
    st.metric("Critical", critical_count, delta=None)
with col3:
    st.metric("Warnings", warning_count)
with col4:
    health_score_val = health.get("composite", 0) if health else 0
    st.metric("Health Score", f"{health_score_val:.0f}/100")

if not alerts:
    st.success("No alerts triggered with current thresholds. All clear.")
    st.stop()

# --- Critical Alerts ---
if critical_count > 0:
    st.divider()
    st.subheader("Critical Alerts")
    st.caption("These require immediate attention.")

    critical_alerts = [a for a in alerts if a.get("severity") == "critical"]
    for a in critical_alerts:
        st.error(f"**{a['name']}** \u2014 {a.get('entity', 'N/A')}: {a.get('detail', '')}")

# --- Warning Alerts ---
if warning_count > 0:
    st.divider()
    st.subheader("Warnings")

    warning_alerts = [a for a in alerts if a.get("severity") == "warning"]
    for a in warning_alerts[:20]:
        st.warning(f"**{a['name']}** \u2014 {a.get('entity', 'N/A')}: {a.get('detail', '')}")

    if len(warning_alerts) > 20:
        st.caption(f"... and {len(warning_alerts) - 20} more warnings")

# --- Info Alerts ---
if info_count > 0:
    st.divider()
    st.subheader("Informational")

    info_alerts = [a for a in alerts if a.get("severity") == "info"]
    for a in info_alerts[:10]:
        st.info(f"**{a['name']}** \u2014 {a.get('entity', 'N/A')}: {a.get('detail', '')}")

# --- Alert Summary Table ---
st.divider()
st.subheader("All Alerts")

if alerts:
    alert_df = pl.DataFrame([
        {
            "Severity": a.get("severity", "info"),
            "Rule": a.get("name", ""),
            "Entity": a.get("entity", ""),
            "Value": str(a.get("value", "")),
            "Detail": a.get("detail", ""),
        }
        for a in alerts
    ])

    # Color code by severity
    fig = px.histogram(
        alert_df.to_pandas(),
        x="Rule", color="Severity",
        color_discrete_map={"critical": "#e15759", "warning": "#f28e2b", "info": "#4e79a7"},
        title="Alerts by Rule",
    )
    fig.update_layout(height=300)
    st.plotly_chart(fig, use_container_width=True)

    st.dataframe(
        alert_df.to_pandas(),
        use_container_width=True,
        height=min(500, len(alerts) * 35 + 40),
    )
    download_csv_button(alert_df, "alerts.csv", "Download Alert Report")
