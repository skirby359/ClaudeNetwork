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
from src.export import download_csv_button
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
            st.caption("Download the core datasets as CSV files.")
            download_csv_button(message_fact, "message_fact.csv", "Download Messages")
            ef = load_filtered_message_fact(start_date, end_date)
            download_csv_button(person_dim, "person_dim.csv", "Download People")

    st.divider()
    st.markdown("""
    ### Navigate the Analysis

    **Executive View**
    - **Executive Summary** — Key findings at a glance
    - **Risk Register** — Anomalies and flags
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

    **Setup:** Settings — Data upload, domains, connectors
    """)

except Exception as e:
    st.error(f"Pipeline error: {e}")
    st.exception(e)
