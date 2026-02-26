"""Streamlit session state management and data loaders."""

import datetime as dt

import streamlit as st
import polars as pl

from src.config import AppConfig, DatasetConfig
from src.cache_manager import read_parquet, read_pickle
from src.ingest.pipeline import run_ingestion
from src.transform.fact_tables import build_edge_fact, build_person_dim
from src.transform.weekly_agg import build_weekly_agg, compute_weekly_stats
from src.transform.timing import build_timing_metrics
from src.transform.broadcast import build_broadcast_metrics, compute_broadcast_stats
from src.analytics.network import (
    build_network_graph, compute_graph_metrics, compute_dyad_analysis,
    build_graph, compute_node_metrics, compute_dyads,
)


def get_config() -> AppConfig:
    """Get or create the AppConfig singleton."""
    if "config" not in st.session_state:
        st.session_state.config = AppConfig()
    return st.session_state.config


def get_dataset() -> DatasetConfig:
    """Get the current dataset configuration."""
    return get_config().default_dataset


# ---------------------------------------------------------------------------
# Date range filter
# ---------------------------------------------------------------------------

def render_date_filter() -> tuple[dt.date, dt.date]:
    """Render a date range slider in the sidebar. Returns (start, end) dates.

    Persists selection across pages via session_state.
    Defaults to the most recent 30 days in the dataset.
    """
    mf = load_message_fact()
    min_date = mf["timestamp"].min().date()
    max_date = mf["timestamp"].max().date()

    # Initialize session_state with last-30-days default on first visit
    if "date_range" not in st.session_state:
        default_start = max(min_date, max_date - dt.timedelta(days=30))
        st.session_state.date_range = (default_start, max_date)

    with st.sidebar:
        st.subheader("Date Range")
        start, end = st.slider(
            "Filter by date",
            min_value=min_date,
            max_value=max_date,
            format="MM/DD/YYYY",
            key="date_range",
        )

    return start, end


def apply_date_filter(
    df: pl.DataFrame,
    start_date: dt.date,
    end_date: dt.date,
    col: str = "timestamp",
) -> pl.DataFrame:
    """Filter a DataFrame by date range on the given column."""
    start_dt = dt.datetime.combine(start_date, dt.time.min)
    end_dt = dt.datetime.combine(end_date, dt.time.max)
    return df.filter(
        (pl.col(col) >= start_dt) & (pl.col(col) <= end_dt)
    )


# ---------------------------------------------------------------------------
# Full-dataset loaders — use @st.cache_resource to avoid deep-copying
# large DataFrames on every access (Fix 3)
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner="Loading message data...")
def load_message_fact() -> pl.DataFrame:
    config = get_config()
    return run_ingestion(config)


@st.cache_resource(show_spinner="Building edge fact table...")
def load_edge_fact() -> pl.DataFrame:
    config = get_config()
    message_fact = load_message_fact()
    return build_edge_fact(message_fact, config)


@st.cache_resource(show_spinner="Building person dimension...")
def load_person_dim() -> pl.DataFrame:
    config = get_config()
    message_fact = load_message_fact()
    edge_fact = load_edge_fact()
    return build_person_dim(edge_fact, message_fact, config)


@st.cache_resource(show_spinner="Computing weekly aggregations...")
def load_weekly_agg() -> pl.DataFrame:
    config = get_config()
    message_fact = load_message_fact()
    edge_fact = load_edge_fact()
    return build_weekly_agg(message_fact, edge_fact, config)


@st.cache_resource(show_spinner="Computing timing metrics...")
def load_timing_metrics() -> pl.DataFrame:
    config = get_config()
    message_fact = load_message_fact()
    return build_timing_metrics(message_fact, config)


@st.cache_resource(show_spinner="Computing broadcast metrics...")
def load_broadcast_metrics() -> pl.DataFrame:
    config = get_config()
    message_fact = load_message_fact()
    return build_broadcast_metrics(message_fact, config)


@st.cache_resource(show_spinner="Loading network graph...")
def load_network_graph():
    """Load NetworkX graph."""
    config = get_config()
    edge_fact = load_edge_fact()
    return build_network_graph(edge_fact, config)


@st.cache_resource(show_spinner="Computing graph metrics...")
def load_graph_metrics() -> pl.DataFrame:
    config = get_config()
    G = load_network_graph()
    return compute_graph_metrics(G, config)


@st.cache_resource(show_spinner="Analyzing communication pairs...")
def load_dyad_analysis() -> pl.DataFrame:
    config = get_config()
    edge_fact = load_edge_fact()
    return compute_dyad_analysis(edge_fact, config)


# ---------------------------------------------------------------------------
# Date-filtered cached loaders (Fix 1 + Fix 4)
# Keyed on (start_date, end_date) so same range across pages is instant.
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner=False, ttl=3600)
def load_filtered_message_fact(start_date: dt.date, end_date: dt.date) -> pl.DataFrame:
    return apply_date_filter(load_message_fact(), start_date, end_date)


@st.cache_data(show_spinner=False, ttl=3600)
def load_filtered_edge_fact(start_date: dt.date, end_date: dt.date) -> pl.DataFrame:
    return apply_date_filter(load_edge_fact(), start_date, end_date)


@st.cache_data(show_spinner=False, ttl=3600)
def load_filtered_weekly_agg(start_date: dt.date, end_date: dt.date) -> pl.DataFrame:
    mf = load_filtered_message_fact(start_date, end_date)
    ef = load_filtered_edge_fact(start_date, end_date)
    return compute_weekly_stats(mf, ef)


@st.cache_data(show_spinner=False, ttl=3600)
def load_filtered_broadcast(start_date: dt.date, end_date: dt.date) -> pl.DataFrame:
    mf = load_filtered_message_fact(start_date, end_date)
    return compute_broadcast_stats(mf)


@st.cache_data(show_spinner="Computing network for selected dates...", ttl=3600)
def load_filtered_graph_metrics(start_date: dt.date, end_date: dt.date) -> pl.DataFrame:
    """Shared cached graph metrics for date-filtered data. Used by pages 06, 07, 09."""
    ef = load_filtered_edge_fact(start_date, end_date)
    G = build_graph(ef)
    return compute_node_metrics(G)


@st.cache_data(show_spinner="Analyzing pairs for selected dates...", ttl=3600)
def load_filtered_dyads(start_date: dt.date, end_date: dt.date) -> pl.DataFrame:
    ef = load_filtered_edge_fact(start_date, end_date)
    return compute_dyads(ef)


def run_full_pipeline():
    """Run the full pipeline to populate all caches."""
    with st.spinner("Running data pipeline..."):
        load_message_fact()
        load_edge_fact()
        load_person_dim()
        load_weekly_agg()
        load_timing_metrics()
        load_broadcast_metrics()
