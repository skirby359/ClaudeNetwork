"""Streamlit session state management and data loaders."""

import datetime as dt
import hashlib

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


def _data_fingerprint(config: AppConfig) -> str:
    """Compute a fingerprint of the current CSV files in the data directory.

    Changes when files are added, removed, renamed, or modified.
    """
    csv_files = config.discover_csv_files()
    parts = []
    for p in csv_files:
        stat = p.stat()
        parts.append(f"{p.name}:{stat.st_size}:{stat.st_mtime}")
    return hashlib.md5("|".join(parts).encode()).hexdigest()


def _check_data_changed():
    """Detect if data files changed since last load. If so, clear all caches."""
    config = get_config()
    fingerprint = _data_fingerprint(config)

    if "data_fingerprint" in st.session_state and st.session_state.data_fingerprint != fingerprint:
        # Data files changed — nuke all caches
        st.cache_resource.clear()
        st.cache_data.clear()
        # Remove stale cache files on disk
        for f in config.cache_dir.iterdir():
            if f.suffix in (".parquet", ".pickle"):
                f.unlink(missing_ok=True)
        # Reset date range so it re-initializes from new data
        st.session_state.pop("date_range", None)
        st.session_state.pop("_date_selection", None)
        st.toast("Data files changed — caches cleared and reloading.")

    st.session_state.data_fingerprint = fingerprint


def get_config() -> AppConfig:
    """Get or create the AppConfig singleton."""
    if "config" not in st.session_state:
        st.session_state.config = AppConfig()
    return st.session_state.config


def get_dataset() -> DatasetConfig:
    """Get the current dataset configuration."""
    config = get_config()
    selected = st.session_state.get("_selected_dataset", "default")
    datasets = config.discover_datasets()
    if selected in datasets:
        return DatasetConfig(name=selected, csv_paths=datasets[selected])
    return config.default_dataset


# ---------------------------------------------------------------------------
# Multi-dataset selector
# ---------------------------------------------------------------------------

def render_dataset_selector() -> str:
    """Render a dataset selector in the sidebar. Returns the selected dataset name."""
    config = get_config()
    datasets = config.discover_datasets()
    if len(datasets) <= 1:
        return "default"

    names = list(datasets.keys())
    current = st.session_state.get("_selected_dataset", names[0])
    if current not in names:
        current = names[0]

    def _on_dataset_change():
        new_val = st.session_state._dataset_widget
        if st.session_state.get("_selected_dataset") != new_val:
            st.session_state._selected_dataset = new_val
            # Clear all caches on dataset switch
            st.cache_resource.clear()
            st.cache_data.clear()
            st.session_state.pop("date_range", None)
            st.session_state.pop("_date_selection", None)
            # Remove disk cache files
            for f in config.cache_dir.iterdir():
                if f.suffix in (".parquet", ".pickle"):
                    f.unlink(missing_ok=True)

    with st.sidebar:
        st.selectbox(
            "Dataset",
            options=names,
            index=names.index(current),
            key="_dataset_widget",
            on_change=_on_dataset_change,
        )

    return st.session_state.get("_selected_dataset", names[0])


# ---------------------------------------------------------------------------
# Date range filter
# ---------------------------------------------------------------------------

def render_date_filter() -> tuple[dt.date, dt.date]:
    """Render a date range slider in the sidebar. Returns (start, end) dates.

    Uses a separate session_state key ("_date_selection") as the source of truth,
    copied into the widget key ("date_range") before each render. This avoids
    Streamlit's multipage widget-state desync where navigating to a new page
    can reset widget values to their defaults.

    Includes quick-select preset buttons (7d, 30d, 90d, All).
    Defaults to the most recent 30 days in the dataset.
    Automatically detects data file changes and resets accordingly.
    """
    # Check for data file changes before loading
    _check_data_changed()

    mf = load_message_fact()
    min_date = mf["timestamp"].min().date()
    max_date = mf["timestamp"].max().date()

    # Clamp saved selection to actual data bounds
    if "_date_selection" in st.session_state:
        saved_start, saved_end = st.session_state._date_selection
        if saved_start > max_date or saved_end < min_date or saved_end > max_date or saved_start < min_date:
            del st.session_state._date_selection

    # Default to last 30 days on first visit
    if "_date_selection" not in st.session_state:
        default_start = max(min_date, max_date - dt.timedelta(days=30))
        st.session_state._date_selection = (default_start, max_date)

    # Push saved selection into the widget key before the slider renders
    st.session_state.date_range = st.session_state._date_selection

    def _on_date_change():
        """Callback: sync widget value back to our source-of-truth key."""
        st.session_state._date_selection = st.session_state.date_range

    with st.sidebar:
        st.subheader("Date Range")

        # Quick-select preset buttons
        preset_cols = st.columns(4)
        with preset_cols[0]:
            if st.button("7d", width="stretch"):
                st.session_state._date_selection = (max(min_date, max_date - dt.timedelta(days=7)), max_date)
                st.session_state.date_range = st.session_state._date_selection
                st.rerun()
        with preset_cols[1]:
            if st.button("30d", width="stretch"):
                st.session_state._date_selection = (max(min_date, max_date - dt.timedelta(days=30)), max_date)
                st.session_state.date_range = st.session_state._date_selection
                st.rerun()
        with preset_cols[2]:
            if st.button("90d", width="stretch"):
                st.session_state._date_selection = (max(min_date, max_date - dt.timedelta(days=90)), max_date)
                st.session_state.date_range = st.session_state._date_selection
                st.rerun()
        with preset_cols[3]:
            if st.button("All", width="stretch"):
                st.session_state._date_selection = (min_date, max_date)
                st.session_state.date_range = st.session_state._date_selection
                st.rerun()

        start, end = st.slider(
            "Filter by date",
            min_value=min_date,
            max_value=max_date,
            format="MM/DD/YYYY",
            key="date_range",
            on_change=_on_date_change,
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
# Comparison mode
# ---------------------------------------------------------------------------

def render_comparison_filter(min_date: dt.date, max_date: dt.date) -> tuple[bool, dt.date | None, dt.date | None]:
    """Render a comparison period filter. Returns (enabled, comp_start, comp_end)."""
    with st.sidebar:
        enabled = st.checkbox("Compare to previous period", key="_comparison_enabled")
        if not enabled:
            return False, None, None

        # Default comparison: same-length period immediately before current selection
        current_start, current_end = st.session_state.get("_date_selection", (min_date, max_date))
        period_days = (current_end - current_start).days
        default_comp_end = current_start - dt.timedelta(days=1)
        default_comp_start = max(min_date, default_comp_end - dt.timedelta(days=period_days))

        if "_comp_selection" not in st.session_state:
            st.session_state._comp_selection = (default_comp_start, default_comp_end)

        st.session_state._comp_range = st.session_state._comp_selection

        def _on_comp_change():
            st.session_state._comp_selection = st.session_state._comp_range

        comp_start, comp_end = st.slider(
            "Comparison period",
            min_value=min_date,
            max_value=max_date,
            format="MM/DD/YYYY",
            key="_comp_range",
            on_change=_on_comp_change,
        )

        return True, comp_start, comp_end


# ---------------------------------------------------------------------------
# Full-dataset loaders — use @st.cache_resource to avoid deep-copying
# large DataFrames on every access (Fix 3)
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner="Loading message data...")
def load_message_fact() -> pl.DataFrame:
    config = get_config()

    # Show progress bar during ingestion
    progress_bar = st.progress(0, text="Loading data...")

    def _progress_cb(fraction: float, filename: str):
        if fraction < 1.0:
            progress_bar.progress(fraction, text=f"Ingesting {filename}...")
        else:
            progress_bar.progress(1.0, text="Ingestion complete.")

    result = run_ingestion(config, progress_callback=_progress_cb)
    progress_bar.empty()
    return result


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
    config = get_config()
    cache_path = config.cache_path(config.message_fact_file)
    # Try lazy scan with predicate pushdown when parquet cache exists
    if cache_path.exists():
        try:
            start_dt = dt.datetime.combine(start_date, dt.time.min)
            end_dt = dt.datetime.combine(end_date, dt.time.max)
            return (
                pl.scan_parquet(cache_path)
                .filter(
                    (pl.col("timestamp") >= start_dt) & (pl.col("timestamp") <= end_dt)
                )
                .collect()
            )
        except Exception:
            pass
    return apply_date_filter(load_message_fact(), start_date, end_date)


@st.cache_data(show_spinner=False, ttl=3600)
def load_filtered_edge_fact(start_date: dt.date, end_date: dt.date) -> pl.DataFrame:
    config = get_config()
    cache_path = config.cache_path(config.edge_fact_file)
    if cache_path.exists():
        try:
            start_dt = dt.datetime.combine(start_date, dt.time.min)
            end_dt = dt.datetime.combine(end_date, dt.time.max)
            return (
                pl.scan_parquet(cache_path)
                .filter(
                    (pl.col("timestamp") >= start_dt) & (pl.col("timestamp") <= end_dt)
                )
                .collect()
            )
        except Exception:
            pass
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


@st.cache_data(show_spinner=False, ttl=3600)
def load_nonhuman_emails(start_date: dt.date, end_date: dt.date) -> frozenset:
    """Cached frozenset of nonhuman email addresses for the given date range."""
    from src.analytics.hierarchy import detect_nonhuman_addresses
    person_dim = load_person_dim()
    edge_fact = load_filtered_edge_fact(start_date, end_date)
    flagged = detect_nonhuman_addresses(person_dim, edge_fact)
    return frozenset(flagged.filter(pl.col("is_nonhuman"))["email"].to_list())


def run_full_pipeline():
    """Run the full pipeline to populate all caches."""
    with st.spinner("Running data pipeline..."):
        load_message_fact()
        load_edge_fact()
        load_person_dim()
        load_weekly_agg()
        load_timing_metrics()
        load_broadcast_metrics()
