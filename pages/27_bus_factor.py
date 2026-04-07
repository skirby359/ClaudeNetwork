"""Page 27: Key-Person Dependency / Bus Factor — Critical personnel risk analysis."""

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import polars as pl

from src.page_logger import log_page_entry, log_page_error
from src.state import (
    render_date_filter, load_filtered_edge_fact, load_filtered_graph_metrics,
    load_nonhuman_emails, load_person_dim,
)
from src.analytics.network import build_graph
from src.analytics.hierarchy import infer_reciprocal_teams
from src.analytics.silos import simulate_removal
from src.analytics.bus_factor import (
    find_articulation_points, compute_team_bus_factor,
    compute_succession_readiness, compute_dependency_risk_matrix,
)
from src.export import download_csv_button


@st.cache_data(show_spinner="Analyzing key-person dependencies...", ttl=3600)
def _cached_bus_factor(start_date, end_date):
    ef = load_filtered_edge_fact(start_date, end_date)
    gm = load_filtered_graph_metrics(start_date, end_date)
    pd_dim = load_person_dim()
    nonhuman = load_nonhuman_emails(start_date, end_date)

    G = build_graph(ef)
    ap = find_articulation_points(G)

    # Infer teams for bus factor calculation
    teams = infer_reciprocal_teams(ef, pd_dim, exclude_emails=set(nonhuman))
    team_bf = compute_team_bus_factor(teams, G)
    succession = compute_succession_readiness(G, ap)
    risk_matrix = compute_dependency_risk_matrix(G, gm, ap)

    return ap, teams, team_bf, succession, risk_matrix, G


st.set_page_config(page_title="Bus Factor", layout="wide")
_page_log = log_page_entry("27_bus_factor")
st.title("Key-Person Dependency / Bus Factor")
st.caption(
    "Identifying critical personnel whose departure would disrupt organizational communication. "
    "Articulation points are people whose removal disconnects the network."
)

start_date, end_date = render_date_filter()

ap, teams, team_bf, succession, risk_matrix, G = _cached_bus_factor(start_date, end_date)

# --- Nonhuman filter ---
nonhuman_emails = load_nonhuman_emails(start_date, end_date)
filter_nonhuman = st.session_state.get("exclude_nonhuman", True)
nh_list = list(nonhuman_emails) if filter_nonhuman and nonhuman_emails else []
if nh_list:
    risk_matrix = risk_matrix.filter(~pl.col("email").is_in(nh_list))
    ap = [p for p in ap if p not in set(nh_list)]

# --- Section 1: KPI Row ---
st.divider()
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Articulation Points", f"{len(ap):,}")
with col2:
    min_bf = team_bf["bus_factor"].min() if len(team_bf) > 0 else "N/A"
    st.metric("Lowest Team Bus Factor", min_bf)
with col3:
    avg_readiness = succession["readiness_score"].mean() if len(succession) > 0 else 0
    st.metric("Avg Succession Readiness", f"{avg_readiness:.0f}%")
with col4:
    critical_teams = len(team_bf.filter(pl.col("risk_level") == "critical")) if len(team_bf) > 0 else 0
    st.metric("Critical Teams", critical_teams)

# --- Section 2: Dependency Risk Matrix ---
st.divider()
st.subheader("Dependency Risk Matrix")
st.caption("People ranked by composite risk score (articulation point status, centrality, bridging role).")

if len(risk_matrix) > 0:
    top_risk = risk_matrix.head(30)

    fig_risk = px.scatter(
        top_risk.to_pandas(),
        x="betweenness", y="pagerank",
        size="risk_score", color="is_articulation_point",
        hover_name="email",
        color_discrete_map={True: "#e15759", False: "#4e79a7"},
        title="Dependency Risk: Centrality vs Influence",
        labels={
            "betweenness": "Connector Score (Betweenness)",
            "pagerank": "Importance (PageRank)",
            "is_articulation_point": "Articulation Point?",
        },
    )
    fig_risk.update_layout(height=450)
    st.plotly_chart(fig_risk, use_container_width=True)

    st.dataframe(
        top_risk.select([
            "email", "is_articulation_point", "risk_score",
            "betweenness", "pagerank", "communities_bridged",
        ]).to_pandas(),
        use_container_width=True,
    )
    download_csv_button(risk_matrix, "dependency_risk.csv")

# --- Section 3: Team Bus Factor ---
st.divider()
st.subheader("Team Bus Factor")
st.caption(
    "How many people need to leave before a team becomes disconnected. "
    "Lower = more fragile."
)

if len(team_bf) > 0:
    color_map = {"critical": "#e15759", "warning": "#f28e2b", "ok": "#59a14f"}
    fig_bf = px.bar(
        team_bf.to_pandas(),
        x="manager", y="bus_factor",
        color="risk_level",
        color_discrete_map=color_map,
        title="Bus Factor by Team",
        labels={"bus_factor": "Bus Factor", "manager": "Team Lead"},
    )
    fig_bf.update_layout(height=400, xaxis_tickangle=-45)
    st.plotly_chart(fig_bf, use_container_width=True)

    st.dataframe(
        team_bf.select(["manager", "team_size", "bus_factor", "risk_level"]).to_pandas(),
        use_container_width=True,
    )
    download_csv_button(team_bf.drop("critical_members"), "team_bus_factor.csv")
else:
    st.info("No teams with 3+ reciprocal members detected for bus factor analysis.")

# --- Section 4: Succession Readiness ---
st.divider()
st.subheader("Succession Readiness")
st.caption(
    "For each critical person (articulation point), who has the most overlapping contact network "
    "to serve as a backup?"
)

if len(succession) > 0:
    if nh_list:
        succession = succession.filter(~pl.col("critical_person").is_in(nh_list))

    fig_succ = px.bar(
        succession.head(20).to_pandas(),
        x="critical_person", y="readiness_score",
        color="contact_overlap_pct",
        color_continuous_scale="RdYlGn",
        title="Succession Readiness by Critical Person",
        labels={
            "readiness_score": "Readiness Score",
            "contact_overlap_pct": "Contact Overlap %",
        },
    )
    fig_succ.update_layout(height=400, xaxis_tickangle=-45)
    st.plotly_chart(fig_succ, use_container_width=True)

    st.dataframe(succession.head(30).to_pandas(), use_container_width=True)
    download_csv_button(succession, "succession_readiness.csv")
else:
    st.info("No articulation points found — the network has good redundancy.")

# --- Section 5: What-If Simulator ---
st.divider()
st.subheader("What-If Simulator")
st.caption("Select a person to see the network impact of their removal.")

if len(ap) > 0:
    # Show articulation points first, then other high-risk people
    options = ap[:30]
    selected = st.selectbox("Select a person:", options, key="bf_whatif")

    if selected:
        result = simulate_removal(G, selected)
        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("Components Before", f"{result['before_components']:,}")
        with c2:
            delta_str = f"+{result['component_increase']}" if result['component_increase'] > 0 else str(result['component_increase'])
            st.metric("Components After", f"{result['after_components']:,}",
                      delta=delta_str, delta_color="inverse")
        with c3:
            delta_str = f"-{result['largest_decrease']}" if result['largest_decrease'] > 0 else str(-result['largest_decrease'])
            st.metric("Largest Component",
                      f"{result['before_largest']:,} -> {result['after_largest']:,}",
                      delta=delta_str, delta_color="inverse")

        if result["component_increase"] > 0:
            st.warning(
                f"Removing **{selected.split('@')[0]}** would split the network into "
                f"{result['component_increase']} additional component(s)."
            )
        else:
            st.info(f"Removing **{selected.split('@')[0]}** would not fragment the network.")

        # Show succession candidate if available
        succ_row = succession.filter(pl.col("critical_person") == selected)
        if len(succ_row) > 0:
            row = succ_row.row(0, named=True)
            st.success(
                f"Best succession candidate: **{row['successor_candidate'].split('@')[0]}** "
                f"with {row['contact_overlap_pct']}% contact overlap "
                f"({row['shared_contacts']} shared contacts)."
            )
else:
    st.success("No articulation points found — the network is well-connected.")
