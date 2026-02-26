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
)

st.title("Email Metadata Analytics Platform")
st.markdown("**Organizational communication pattern analysis from email headers**")

# Sidebar info
with st.sidebar:
    st.header("Dataset Info")
    config = get_config()
    dataset = config.default_dataset
    csv_files = dataset.csv_paths
    st.write(f"**{len(csv_files)} CSV file{'s' if len(csv_files) != 1 else ''}** in `data/`")
    with st.expander("Files"):
        for f in csv_files:
            size_mb = f.stat().st_size / (1024 * 1024)
            st.write(f"- {f.name} ({size_mb:.0f} MB)")
    st.write(f"**Internal domains:** {', '.join(dataset.internal_domains)}")

    if st.button("Reload Pipeline", type="primary"):
        st.cache_data.clear()
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

    st.divider()
    st.markdown("""
    ### Navigate the Analysis

    Use the sidebar to explore the 10 analysis pages:

    1. **Executive Summary** — Key findings at a glance
    2. **Volume & Seasonality** — Message flow trends over time
    3. **Time Norms** — When do people communicate?
    4. **Broadcast & Attention** — Mass-send patterns and inbox load
    5. **Artifact vs Ping** — Message size and purpose patterns
    6. **Network Map** — Interactive communication network
    7. **Bottlenecks & Routing** — Who are the critical connectors?
    8. **Dyads & Asymmetry** — Bidirectional relationship analysis
    9. **Coordination & Churn** — Community structure evolution
    10. **Risk Register** — Anomalies and flags
    11. **External Contacts** — Top external addresses by volume
    12. **Search** — Look up any email address
    """)

except Exception as e:
    st.error(f"Pipeline error: {e}")
    st.exception(e)
