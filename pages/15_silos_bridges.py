"""Page 15: Communication Silos & Bridges — Inter-community analysis."""

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import polars as pl

from src.page_logger import log_page_entry, log_page_error
from src.state import (
    load_person_dim,
    render_date_filter,
    load_filtered_edge_fact, load_filtered_graph_metrics,
    load_nonhuman_emails,
)
from src.analytics.network import build_graph
from src.analytics.silos import (
    compute_community_interaction_matrix, find_silent_community_pairs,
    identify_bridges, simulate_removal,
)
from src.export import download_csv_button
from src.drilldown import (
    handle_plotly_person_click, handle_dataframe_person_click,
    show_community_dialog, should_open_drilldown,
)


# ---------------------------------------------------------------------------
# Cached analytics
# ---------------------------------------------------------------------------

def _scope_edge_fact(start_date, end_date, scope):
    """Helper: return scoped edge_fact."""
    ef = load_filtered_edge_fact(start_date, end_date)
    pd_full = load_person_dim()
    internal_emails = set(pd_full.filter(pl.col("is_internal"))["email"].to_list())
    external_emails = set(pd_full.filter(~pl.col("is_internal"))["email"].to_list())

    if scope == "Internal only":
        return ef.filter(
            pl.col("from_email").is_in(list(internal_emails))
            & pl.col("to_email").is_in(list(internal_emails))
        )
    elif scope == "External only":
        return ef.filter(
            pl.col("from_email").is_in(list(external_emails))
            | pl.col("to_email").is_in(list(external_emails))
        )
    return ef


@st.cache_data(show_spinner="Computing community interactions...", ttl=3600)
def _cached_community_analysis(start_date, end_date, scope):
    """Cache interaction matrix and bridges together."""
    ef_scoped = _scope_edge_fact(start_date, end_date, scope)
    gm = load_filtered_graph_metrics(start_date, end_date)
    community_lookup = dict(zip(gm["email"].to_list(), gm["community_id"].to_list()))

    interaction_matrix = compute_community_interaction_matrix(ef_scoped, community_lookup)
    G = build_graph(ef_scoped)
    bridges = identify_bridges(G, community_lookup)
    return interaction_matrix, bridges


@st.cache_data(show_spinner=False, ttl=3600)
def _cached_simulate_removal(start_date, end_date, scope, email):
    """Cache removal simulation result."""
    ef_scoped = _scope_edge_fact(start_date, end_date, scope)
    G = build_graph(ef_scoped)
    return simulate_removal(G, email)


# ---------------------------------------------------------------------------
# Page layout
# ---------------------------------------------------------------------------

st.set_page_config(page_title="Silos & Bridges", layout="wide")
_page_log = log_page_entry("15_silos_bridges")
st.title("Communication Silos & Bridges")
st.caption("Analyzing how communities interact — and where they don't.")

start_date, end_date = render_date_filter()

edge_fact = load_filtered_edge_fact(start_date, end_date)
graph_metrics = load_filtered_graph_metrics(start_date, end_date)
person_dim = load_person_dim()

if len(edge_fact) == 0:
    st.warning("No edge data in selected date range.")
    st.stop()

# --- Nonhuman filter (global toggle) ---
nonhuman_emails = load_nonhuman_emails(start_date, end_date)
filter_nonhuman = st.session_state.get("exclude_nonhuman", True)
if filter_nonhuman and nonhuman_emails:
    nh_list = list(nonhuman_emails)
    graph_metrics = graph_metrics.filter(~pl.col("email").is_in(nh_list))

# Build community lookup and internal/external sets
community_lookup = dict(zip(
    graph_metrics["email"].to_list(),
    graph_metrics["community_id"].to_list(),
))

internal_emails = set(person_dim.filter(pl.col("is_internal"))["email"].to_list())

# Only consider communities with > 2 members
comm_counts = graph_metrics.group_by("community_id").agg(pl.len().alias("n"))
valid_communities = sorted(
    comm_counts.filter(pl.col("n") > 2)["community_id"].to_list()
)
n_communities = len(valid_communities)

# Map community IDs to readable labels with member counts
comm_size_map = dict(zip(
    comm_counts["community_id"].to_list(),
    comm_counts["n"].to_list(),
))
# Use auto-generated labels from community detection if available
if "community_label" in graph_metrics.columns:
    _label_lookup = dict(zip(
        graph_metrics["community_id"].to_list(),
        graph_metrics["community_label"].to_list(),
    ))
    comm_labels = {c: _label_lookup.get(c, f"Group {c} ({comm_size_map.get(c, 0)})") for c in valid_communities}
else:
    comm_labels = {c: f"Group {c} ({comm_size_map.get(c, 0)})" for c in valid_communities}

# --- Internal / External selector ---
scope_tab = st.radio(
    "Scope", ["All addresses", "Internal only", "External only"],
    horizontal=True, key="silos_scope",
)

ef_scoped = _scope_edge_fact(start_date, end_date, scope_tab)

if len(ef_scoped) == 0:
    st.info("No edges for the selected scope.")
    st.stop()

# Cached computation
interaction_matrix, bridges = _cached_community_analysis(start_date, end_date, scope_tab)

# Filter interaction matrix to valid communities
interaction_matrix = interaction_matrix.filter(
    pl.col("comm_from").is_in(valid_communities)
    & pl.col("comm_to").is_in(valid_communities)
)

# --- Section 1: Community interaction heatmap ---
st.divider()
st.subheader("Community-to-Community Message Flow")
st.caption(
    f"Showing {n_communities} communities with >2 members. "
    "Brighter cells = more messages between those communities."
)

if len(interaction_matrix) > 0:
    # Relabel for readability
    str_comm_labels = {str(k): v for k, v in comm_labels.items()}
    im_display = interaction_matrix.with_columns([
        pl.col("comm_from").cast(pl.Utf8).replace(str_comm_labels).alias("from_community"),
        pl.col("comm_to").cast(pl.Utf8).replace(str_comm_labels).alias("to_community"),
    ])
    pivot = im_display.to_pandas().pivot_table(
        index="from_community", columns="to_community", values="msg_count", fill_value=0
    )
    # Sort axes naturally
    sorted_labels = [comm_labels[c] for c in valid_communities if comm_labels[c] in pivot.index]
    pivot = pivot.reindex(index=sorted_labels, columns=sorted_labels, fill_value=0)

    fig = px.imshow(
        pivot,
        labels=dict(x="To Community", y="From Community", color="Messages"),
        title="Inter-Community Message Volume",
        aspect="auto",
        color_continuous_scale="Blues",
    )
    fig.update_layout(height=max(400, n_communities * 22 + 100))
    fig.update_xaxes(tickangle=-45, tickfont_size=10)
    fig.update_yaxes(tickfont_size=10)
    st.plotly_chart(fig, width="stretch")

# --- Section 1b: Community explorer ---
st.divider()
st.subheader("Community Explorer")
st.caption("Select a community to view its members and stats.")

selected_comm = st.selectbox(
    "Select Community",
    [comm_labels[c] for c in valid_communities],
    key="p15_comm_select",
)
# Extract community_id from the selected label
if selected_comm:
    # Reverse lookup: find comm_id whose label matches
    comm_id_from_label = next(
        (c for c in valid_communities if comm_labels.get(c) == selected_comm),
        valid_communities[0] if valid_communities else None,
    )
    if comm_id_from_label is not None and should_open_drilldown("p15_comm_viewer", comm_id_from_label):
        show_community_dialog(comm_id_from_label, start_date, end_date)

    # Inline member table
    if comm_id_from_label is not None:
        comm_members = (
            graph_metrics.filter(pl.col("community_id") == comm_id_from_label)
            .join(person_dim.select(["email", "display_name"]), on="email", how="left")
            .sort("pagerank", descending=True)
        )
        st.write(f"**{len(comm_members)} members** in {selected_comm}")
        st.dataframe(
            comm_members.select(["email", "display_name", "pagerank", "in_degree", "out_degree"])
            .head(30).to_pandas(),
            width="stretch", height=min(400, len(comm_members) * 35 + 40),
        )

# --- Section 2: Communication silos ---
st.divider()
st.subheader("Communication Silos")
st.caption("Community pairs with **zero** message flow between them.")

silent_pairs = find_silent_community_pairs(interaction_matrix, valid_communities)

if silent_pairs:
    st.warning(f"**{len(silent_pairs)}** community pairs have no communication at all.")
    silent_df = pl.DataFrame({
        "Community A": [comm_labels.get(p[0], str(p[0])) for p in silent_pairs[:50]],
        "Community B": [comm_labels.get(p[1], str(p[1])) for p in silent_pairs[:50]],
    })
    st.dataframe(silent_df.to_pandas(), width="stretch", height=min(400, len(silent_pairs) * 35 + 40))
else:
    st.success("No completely silent community pairs — all communities have some communication.")

# --- Section 3: Bridge people ---
st.divider()
st.subheader("Bridge People")
st.caption(
    "People whose contacts span multiple communities. "
    "Losing a bridge can isolate groups."
)

if len(bridges) > 0:
    # Enrich with internal/external label
    bridges_enriched = bridges.with_columns(
        pl.col("email").is_in(list(internal_emails)).alias("is_internal")
    )
    bridges_display = bridges_enriched.drop("external_communities").head(30)

    col_a, col_b = st.columns([3, 1])
    with col_a:
        fig_bridges = px.bar(
            bridges_display.to_pandas(),
            x="email", y="communities_bridged",
            color="is_internal",
            color_discrete_map={True: "#4e79a7", False: "#e15759"},
            title="Top Bridge People by Communities Bridged",
            labels={"communities_bridged": "Communities Bridged", "is_internal": "Internal?"},
        )
        fig_bridges.update_layout(height=400, xaxis_tickangle=-45)
        ev_bridges = st.plotly_chart(fig_bridges, width="stretch", on_select="rerun", key="p15_bridges")
        handle_plotly_person_click(ev_bridges, "p15_bridges", start_date, end_date)
    with col_b:
        n_internal_bridges = len(bridges_enriched.filter(pl.col("is_internal")))
        n_external_bridges = len(bridges_enriched.filter(~pl.col("is_internal")))
        st.metric("Internal Bridges", f"{n_internal_bridges:,}")
        st.metric("External Bridges", f"{n_external_bridges:,}")
        st.metric("Total Bridges", f"{len(bridges_enriched):,}")

    bridges_pd = bridges_display.to_pandas()
    ev_bridges_df = st.dataframe(bridges_pd, width="stretch", on_select="rerun", selection_mode="single-row", key="p15_bridges_df")
    handle_dataframe_person_click(ev_bridges_df, bridges_pd, "p15_bridges_df", "email", start_date, end_date)
    download_csv_button(bridges_display, "bridge_people.csv")
else:
    st.info("No bridge people identified.")

# --- Section 4: "What if?" simulator ---
st.divider()
st.subheader("What If? — Removal Impact Simulator")
st.caption("Select a person to see what happens to network connectivity if they leave.")

top_bridges = bridges.head(20)["email"].to_list() if len(bridges) > 0 else []
if top_bridges:
    selected = st.selectbox("Select a person to simulate removal:", top_bridges)
    if selected:
        result = _cached_simulate_removal(start_date, end_date, scope_tab, selected)
        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("Components Before", f"{result['before_components']:,}")
        with c2:
            st.metric(
                "Components After", f"{result['after_components']:,}",
                delta=f"+{result['component_increase']}" if result['component_increase'] > 0 else str(result['component_increase']),
                delta_color="inverse",
            )
        with c3:
            st.metric(
                "Largest Component",
                f"{result['before_largest']:,} -> {result['after_largest']:,}",
                delta=f"-{result['largest_decrease']}" if result['largest_decrease'] > 0 else str(-result['largest_decrease']),
                delta_color="inverse",
            )
        if result["component_increase"] > 0:
            st.warning(
                f"Removing **{selected.split('@')[0]}** would split the network into "
                f"{result['component_increase']} additional component(s)."
            )
        else:
            st.info(f"Removing **{selected.split('@')[0]}** would not fragment the network.")
else:
    st.info("Select a date range with enough data to identify bridge people.")
