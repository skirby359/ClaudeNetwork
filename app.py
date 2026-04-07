"""Streamlit entry point for the Email Metadata Analytics Platform."""

import streamlit as st

st.set_page_config(
    page_title="Email Metadata Analytics",
    page_icon="📧",
    layout="wide",
    initial_sidebar_state="expanded",
)

from src.state import (
    get_config, run_full_pipeline, load_person_dim,
    render_date_filter, load_filtered_message_fact,
    render_dataset_selector, load_message_fact,
    _check_data_changed,
)
from src.analytics.data_quality import compute_quality_metrics, compute_per_file_stats
from src.export import download_csv_button, download_graphml_button, download_network_json_button
from src.anonymize import render_anonymize_toggle
from src.ingest.pipeline import get_last_ingestion_stats

# Detect data file changes before anything else
_check_data_changed()

st.title("Email Metadata Analytics Platform")
st.markdown("**Organizational communication pattern analysis from email headers**")

# Sidebar info
with st.sidebar:
    st.header("Dataset Info")

    # Multi-dataset selector
    render_dataset_selector()

    # Privacy toggle
    render_anonymize_toggle()

    # Global nonhuman filter
    st.toggle(
        "Exclude automated senders",
        value=st.session_state.get("exclude_nonhuman", True),
        key="_global_nonhuman_toggle",
        help="Hide copiers, bots, and system accounts from all analysis pages",
    )
    st.session_state.exclude_nonhuman = st.session_state._global_nonhuman_toggle

    config = get_config()
    csv_files = config.discover_csv_files()
    st.write(f"**{len(csv_files)} CSV file{'s' if len(csv_files) != 1 else ''}** in `data/`")
    with st.expander("Files"):
        for f in csv_files:
            size_mb = f.stat().st_size / (1024 * 1024)
            st.write(f"- {f.name} ({size_mb:.0f} MB)")
    user_domains = st.session_state.get("_internal_domains", [])
    if user_domains:
        st.write(f"**Internal domains:** {', '.join(user_domains)}")
    else:
        st.caption("Internal domains: auto-detected (configure in Settings)")
    st.page_link("pages/00_settings.py", label="Settings", icon=":material/settings:")

    with st.expander("Glossary"):
        st.markdown("""
        **Concentration Score** (Gini): 0 = everyone sends equally, 1 = one person sends everything.

        **Connector Score** (Betweenness): How many communication paths run through a person. High = critical bridge between groups.

        **Importance Score** (PageRank): Influence based on who communicates with you, not just volume. Being emailed by important people raises your score.

        **Communication Groups** (Communities): Natural clusters detected from who emails whom. People within a group communicate more with each other than with outsiders.

        **Automated Senders**: Copiers, scanners, alert systems, and mail infrastructure detected by name patterns and extreme send/receive ratios.

        **After-Hours**: Messages sent before 7 AM or after 6 PM on weekdays.

        **Reply Time**: Estimated from message timing patterns (A emails B, then B emails A within 24 hours).
        """)

    if st.button("Reload Pipeline", type="primary"):
        st.cache_resource.clear()
        st.cache_data.clear()
        st.session_state.pop("date_range", None)
        st.session_state.pop("_date_selection", None)
        st.session_state.pop("data_fingerprint", None)
        # Remove disk cache files
        for f in config.cache_dir.iterdir():
            if f.suffix in (".parquet", ".pickle"):
                f.unlink(missing_ok=True)
        st.rerun()

# Run pipeline on first load
try:
    run_full_pipeline()

    start_date, end_date = render_date_filter()
    message_fact = load_filtered_message_fact(start_date, end_date)
    person_dim = load_person_dim()

    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Messages", f"{len(message_fact):,}")
    with col2:
        st.metric("Unique People", f"{len(person_dim):,}")
    with col3:
        try:
            internal_count = len(person_dim.filter(person_dim["is_internal"]))
        except Exception:
            internal_count = 0
        st.metric("Internal People", f"{internal_count:,}")
    with col4:
        date_range = message_fact["timestamp"]
        try:
            min_date = date_range.min().strftime("%b %d, %Y")
            max_date = date_range.max().strftime("%b %d, %Y")
            st.metric("Date Range", f"{min_date} - {max_date}")
        except Exception:
            st.metric("Date Range", "N/A")

    # Data quality sidebar expander
    with st.sidebar:
        with st.expander("Data Quality"):
            full_mf = load_message_fact()
            quality = compute_quality_metrics(full_mf)
            st.write(f"**Total messages:** {quality['total_messages']:,}")
            st.write(f"**Zero-size:** {quality['zero_size_count']:,} ({quality['zero_size_pct']:.1%})")
            st.write(f"**Missing names:** {quality['missing_name_count']:,} ({quality['missing_name_pct']:.1%})")

            # Per-file ingestion stats
            ingestion_stats = get_last_ingestion_stats()
            if ingestion_stats:
                file_stats = compute_per_file_stats(ingestion_stats)
                st.write("**Per-file stats:**")
                st.dataframe(file_stats.to_pandas(), width="stretch")

        with st.expander("Export Data"):
            st.caption("Download data in various formats.")
            download_csv_button(message_fact, "message_fact.csv", "Download Messages (CSV)")
            download_csv_button(person_dim, "person_dim.csv", "Download People (CSV)")

            # PowerPoint export
            try:
                from src.export_pptx import generate_pptx
                from src.analytics.health_score import compute_health_score
                from src.analytics.response_time import compute_reply_times
                from src.analytics.silos import identify_bridges
                from src.analytics.network import build_graph
                from src.state import load_filtered_edge_fact, load_filtered_graph_metrics, load_nonhuman_emails

                if st.button("Generate PowerPoint Report"):
                    with st.spinner("Generating report..."):
                        ef_pptx = load_filtered_edge_fact(start_date, end_date)
                        gm_pptx = load_filtered_graph_metrics(start_date, end_date)
                        nonhuman = load_nonhuman_emails(start_date, end_date)
                        nh_list = list(nonhuman)

                        mf_human = message_fact.filter(~pl.col("from_email").is_in(nh_list))
                        ef_human = ef_pptx.filter(
                            ~pl.col("from_email").is_in(nh_list)
                            & ~pl.col("to_email").is_in(nh_list)
                        )

                        # Health score
                        reply_median = None
                        try:
                            rt = compute_reply_times(ef_human)
                            if len(rt) > 0:
                                reply_median = float(rt["median_reply_seconds"].median())
                        except Exception:
                            pass
                        health = compute_health_score(mf_human, ef_human, gm_pptx, reply_median)

                        # Bridges
                        G_pptx = build_graph(ef_pptx)
                        comm_lookup = dict(zip(gm_pptx["email"].to_list(), gm_pptx["community_id"].to_list()))
                        bridges_pptx = identify_bridges(G_pptx, comm_lookup)

                        pptx_bytes = generate_pptx(
                            message_fact=mf_human,
                            edge_fact=ef_human,
                            person_dim=person_dim,
                            graph_metrics=gm_pptx,
                            health_score=health,
                            bridges=bridges_pptx,
                            start_date=start_date,
                            end_date=end_date,
                        )

                    st.download_button(
                        label="Download PowerPoint",
                        data=pptx_bytes,
                        file_name="email_analysis_report.pptx",
                        mime="application/vnd.openxmlformats-officedocument.presentationml.presentation",
                    )
            except Exception as e:
                st.caption(f"PowerPoint export unavailable: {e}")

            # HTML report export
            try:
                from src.export_html import generate_html_report
                from src.analytics.narrative import generate_executive_narrative
                from src.state import load_filtered_weekly_agg

                if st.button("Generate HTML Report"):
                    with st.spinner("Generating HTML report..."):
                        ef_html = load_filtered_edge_fact(start_date, end_date)
                        gm_html = load_filtered_graph_metrics(start_date, end_date)
                        wa_html = load_filtered_weekly_agg(start_date, end_date)
                        nonhuman_h = load_nonhuman_emails(start_date, end_date)
                        nh_h = list(nonhuman_h)

                        mf_h = message_fact.filter(~pl.col("from_email").is_in(nh_h))
                        ef_h = ef_html.filter(
                            ~pl.col("from_email").is_in(nh_h) & ~pl.col("to_email").is_in(nh_h)
                        )

                        # Health score
                        reply_med = None
                        try:
                            rt_h = compute_reply_times(ef_h)
                            if len(rt_h) > 0:
                                reply_med = float(rt_h["median_reply_seconds"].median())
                        except Exception:
                            pass
                        health_h = compute_health_score(mf_h, ef_h, gm_html, reply_med)

                        narrative_text = generate_executive_narrative(mf_h, wa_html, ef_h, person_dim)

                        G_html = build_graph(ef_html)
                        cl = dict(zip(gm_html["email"].to_list(), gm_html["community_id"].to_list()))
                        bridges_h = identify_bridges(G_html, cl)

                        html_str = generate_html_report(
                            message_fact=mf_h, edge_fact=ef_h,
                            person_dim=person_dim, graph_metrics=gm_html,
                            health_score=health_h, narrative=narrative_text,
                            bridges=bridges_h, start_date=start_date, end_date=end_date,
                        )

                    st.download_button(
                        label="Download HTML Report",
                        data=html_str.encode("utf-8"),
                        file_name="email_analysis_report.html",
                        mime="text/html",
                    )
            except Exception as e:
                st.caption(f"HTML export unavailable: {e}")

            # Executive memo export
            try:
                from src.export_memo import generate_executive_memo
                from src.engagement import evaluate_alerts, default_alert_rules

                if st.button("Generate Executive Memo"):
                    with st.spinner("Generating executive memo..."):
                        ef_memo = load_filtered_edge_fact(start_date, end_date)
                        gm_memo = load_filtered_graph_metrics(start_date, end_date)
                        wa_memo = load_filtered_weekly_agg(start_date, end_date)
                        nonhuman_m = load_nonhuman_emails(start_date, end_date)
                        nh_m = list(nonhuman_m)

                        mf_m = message_fact.filter(~pl.col("from_email").is_in(nh_m))
                        ef_m = ef_memo.filter(
                            ~pl.col("from_email").is_in(nh_m) & ~pl.col("to_email").is_in(nh_m)
                        )

                        reply_med_m = None
                        try:
                            rt_m = compute_reply_times(ef_m)
                            if len(rt_m) > 0:
                                reply_med_m = float(rt_m["median_reply_seconds"].median())
                        except Exception:
                            pass
                        health_m = compute_health_score(mf_m, ef_m, gm_memo, reply_med_m)
                        narrative_m = generate_executive_narrative(mf_m, wa_memo, ef_m, person_dim)

                        # Evaluate alerts
                        rules = st.session_state.get("_alert_rules", default_alert_rules())
                        alerts_m = evaluate_alerts(
                            rules, mf_m, ef_m, gm_memo, health_score=health_m,
                        )

                        org = st.session_state.get("_org_name", "Organization")
                        memo_bytes = generate_executive_memo(
                            message_fact=mf_m, edge_fact=ef_m,
                            person_dim=person_dim, graph_metrics=gm_memo,
                            health_score=health_m, narrative=narrative_m,
                            alerts=alerts_m,
                            start_date=start_date, end_date=end_date,
                            org_name=org,
                        )

                    st.download_button(
                        label="Download Executive Memo",
                        data=memo_bytes,
                        file_name="executive_memo.pptx",
                        mime="application/vnd.openxmlformats-officedocument.presentationml.presentation",
                    )
            except Exception as e:
                st.caption(f"Executive memo unavailable: {e}")

            # Network export (GraphML + JSON)
            try:
                from src.state import load_filtered_edge_fact, load_filtered_graph_metrics
                ef_net = load_filtered_edge_fact(start_date, end_date)
                gm_net = load_filtered_graph_metrics(start_date, end_date)
                G_net = build_graph(ef_net)
                download_graphml_button(G_net, gm_net, "network.graphml", "Download GraphML")
                download_network_json_button(G_net, gm_net, "network.json", "Download Network JSON")
            except Exception as e:
                st.caption(f"Network export unavailable: {e}")

    st.divider()
    st.markdown("""
    ### Navigate the Analysis

    **Executive View**
    - **Executive Summary** — Key findings at a glance
    - **Risk Register** — Anomalies and flags
    - **Alert Dashboard** — Configurable threshold monitoring
    - **Narrative Insights** — Auto-generated analysis

    **Organizational Health**
    - **Time Norms** — When do people work?
    - **Response Time** — How fast do people respond?
    - **Data Quality** — Is the data complete?

    **People & Structure**
    - **Bottlenecks & Routing** — Who are the critical connectors?
    - **Hierarchy Inference** — Who leads?
    - **Silos & Bridges** — Where are the gaps?

    **Communication Patterns**
    - **Volume & Seasonality** — Message flow trends
    - **Broadcast & Attention** — Mass-send patterns
    - **Artifact vs Ping** — Message size patterns
    - **Dyads & Asymmetry** — Relationship analysis
    - **External Contacts** — External partners
    - **Search** — Look up any email address

    **Deep Dives**
    - **Network Map** — Interactive network visualization
    - **Coordination & Churn** — Community evolution
    - **Temporal Evolution** — How the network changes
    - **Size Forensics** — Size anomalies and templates
    - **Period Comparison** — Side-by-side analysis
    - **Automated Systems** — Machine traffic dashboard

    **Advanced Analytics**
    - **Community Detection v2** — Leiden multi-resolution analysis
    - **Structural Change** — Reorg detection from community shifts
    - **Compliance Patterns** — Blackouts, spikes, after-hours clusters
    - **Information Cascades** — Forwarding chains and amplifiers
    - **Bus Factor** — Key-person dependency and succession
    - **Person Comparison** — Side-by-side behavioral metrics

    **Setup:** Settings — Data upload, domains, department mapping, connectors
    """)

except Exception as e:
    st.error(f"Pipeline error: {e}")
    st.exception(e)
