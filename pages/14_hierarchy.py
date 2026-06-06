"""Page 14: Organizational Hierarchy Inference — Leadership pattern detection."""

import streamlit as st
import plotly.express as px
import polars as pl

from src.page_logger import log_page_entry, log_page_error
from src.state import (
    load_person_dim,
    render_date_filter,
    load_filtered_edge_fact,
)
from src.analytics.hierarchy import (
    compute_hierarchy_score, detect_nonhuman_addresses,
    infer_reciprocal_teams, build_reporting_pairs_from_teams,
    infer_calculated_hierarchy,
)
import plotly.graph_objects as go
from src.export import download_csv_button
from src.drilldown import (
    handle_plotly_person_click, handle_scatter_person_click,
    handle_dataframe_person_click,
)


# ---------------------------------------------------------------------------
# Cached analytics
# ---------------------------------------------------------------------------

def _scope_data(start_date, end_date, scope):
    """Helper: return (ef_scoped, pd_scoped) for the given scope."""
    ef = load_filtered_edge_fact(start_date, end_date, scope="all")
    pd_full = load_person_dim()
    internal_emails = set(pd_full.filter(pl.col("is_internal"))["email"].to_list())
    external_emails = set(pd_full.filter(~pl.col("is_internal"))["email"].to_list())

    if scope == "Internal only":
        ef_scoped = ef.filter(
            pl.col("from_email").is_in(list(internal_emails))
            & pl.col("to_email").is_in(list(internal_emails))
        )
        pd_scoped = pd_full.filter(pl.col("is_internal"))
    elif scope == "External only":
        ef_scoped = ef.filter(
            pl.col("from_email").is_in(list(external_emails))
            | pl.col("to_email").is_in(list(external_emails))
        )
        pd_scoped = pd_full.filter(~pl.col("is_internal"))
    else:
        ef_scoped = ef
        pd_scoped = pd_full
    return ef_scoped, pd_scoped


@st.cache_data(show_spinner=False, ttl=3600)
def _cached_scoped_nonhuman(start_date, end_date, scope):
    """Auto-detected nonhuman emails for the given date range and scope."""
    ef_scoped, pd_scoped = _scope_data(start_date, end_date, scope)
    flagged = detect_nonhuman_addresses(pd_scoped, ef_scoped)
    return frozenset(flagged.filter(pl.col("is_nonhuman"))["email"].to_list())


@st.cache_data(show_spinner="Detecting reciprocal teams...", ttl=3600)
def _cached_teams(start_date, end_date, scope, exclude_emails, min_msgs, min_team):
    """Cache reciprocal teams. exclude_emails should be a frozenset."""
    ef_scoped, pd_scoped = _scope_data(start_date, end_date, scope)
    if exclude_emails:
        ef_filtered = ef_scoped.filter(
            ~pl.col("from_email").is_in(list(exclude_emails))
            & ~pl.col("to_email").is_in(list(exclude_emails))
        )
        pd_filtered = pd_scoped.filter(~pl.col("email").is_in(list(exclude_emails)))
    else:
        ef_filtered = ef_scoped
        pd_filtered = pd_scoped

    if len(ef_filtered) == 0:
        return pl.DataFrame()

    return infer_reciprocal_teams(
        ef_filtered, pd_filtered,
        min_msgs_per_direction=min_msgs,
        min_team_size=min_team,
        exclude_emails=set(exclude_emails) if exclude_emails else None,
    )


@st.cache_data(show_spinner="Computing hierarchy scores...", ttl=3600)
def _cached_hierarchy_score(start_date, end_date, scope, exclude_emails):
    """Cache hierarchy scores. exclude_emails should be a frozenset."""
    ef_scoped, pd_scoped = _scope_data(start_date, end_date, scope)
    if exclude_emails:
        ef_filtered = ef_scoped.filter(
            ~pl.col("from_email").is_in(list(exclude_emails))
            & ~pl.col("to_email").is_in(list(exclude_emails))
        )
        pd_filtered = pd_scoped.filter(~pl.col("email").is_in(list(exclude_emails)))
    else:
        ef_filtered = ef_scoped
        pd_filtered = pd_scoped

    if len(ef_filtered) == 0:
        return pl.DataFrame()

    return compute_hierarchy_score(ef_filtered, pd_filtered)


@st.cache_data(show_spinner="Inferring calculated hierarchy...", ttl=3600)
def _cached_calc_hierarchy(start_date, end_date, scope, exclude_emails, min_msgs):
    """Cache the inferred directional hierarchy."""
    ef_scoped, pd_scoped = _scope_data(start_date, end_date, scope)
    if exclude_emails:
        ef_scoped = ef_scoped.filter(
            ~pl.col("from_email").is_in(list(exclude_emails))
            & ~pl.col("to_email").is_in(list(exclude_emails))
        )
        pd_scoped = pd_scoped.filter(~pl.col("email").is_in(list(exclude_emails)))
    if len(ef_scoped) == 0:
        return pl.DataFrame()
    scores = compute_hierarchy_score(ef_scoped, pd_scoped)
    return infer_calculated_hierarchy(ef_scoped, scores, min_msgs=min_msgs)


# ---------------------------------------------------------------------------
# Page layout
# ---------------------------------------------------------------------------

st.set_page_config(page_title="Hierarchy Inference", layout="wide")
_page_log = log_page_entry("14_hierarchy")
st.title("Organizational Hierarchy Inference")

start_date, end_date = render_date_filter()

edge_fact = load_filtered_edge_fact(start_date, end_date, scope="all")
person_dim = load_person_dim()

if len(edge_fact) == 0:
    st.warning("No edge data in selected date range.")
    st.stop()

# --- Scope: Internal / External ---
scope_tab = st.radio(
    "Scope", ["Internal only", "External only", "All addresses"],
    horizontal=True, key="hierarchy_scope",
)

# --- Nonhuman filter ---
st.divider()
with st.expander("Nonhuman Address Filter", expanded=True):
    st.caption(
        "Copiers, system accounts, and automated senders are detected automatically "
        "(by name patterns and extreme send/receive ratios). "
        "You can also add addresses manually."
    )

    # Auto-detect nonhuman (cached)
    auto_nonhuman = _cached_scoped_nonhuman(start_date, end_date, scope_tab)

    col_left, col_right = st.columns([2, 1])
    with col_left:
        # Manual exclude text area
        manual_input = st.text_area(
            "Additional addresses to exclude (one per line)",
            value="",
            height=100,
            placeholder="e.g. fsprod@spokanecounty.org",
            key="hierarchy_manual_exclude",
        )
        manual_excludes = set()
        for line in manual_input.strip().split("\n"):
            line = line.strip()
            if line:
                manual_excludes.add(line)

    with col_right:
        st.write(f"**Auto-detected:** {len(auto_nonhuman)} nonhuman addresses")
        if auto_nonhuman:
            with st.popover("View auto-detected"):
                for addr in sorted(auto_nonhuman)[:50]:
                    st.text(addr)
                if len(auto_nonhuman) > 50:
                    st.caption(f"... and {len(auto_nonhuman) - 50} more")

    all_excludes = frozenset(auto_nonhuman | manual_excludes)
    filter_enabled = st.checkbox("Enable nonhuman filter", value=True, key="hierarchy_filter_on")

if filter_enabled:
    st.caption(f"Filtered out {len(all_excludes)} nonhuman addresses ({len(auto_nonhuman)} auto + {len(manual_excludes)} manual).")

effective_excludes = all_excludes if filter_enabled else frozenset()

# Check if enough data after filtering
ef_scoped, pd_scoped = _scope_data(start_date, end_date, scope_tab)
if filter_enabled:
    ef_check = ef_scoped.filter(
        ~pl.col("from_email").is_in(list(all_excludes))
        & ~pl.col("to_email").is_in(list(all_excludes))
    )
else:
    ef_check = ef_scoped

if len(ef_check) == 0:
    st.info("No edges remaining after filtering.")
    st.stop()

# =========================================================================
# TAB 1: Reciprocal Teams  |  TAB 2: Hierarchy Score (original)
# =========================================================================
tab_teams, tab_calc, tab_score = st.tabs([
    "Mutual Communication Groups (recommended)",
    "Calculated Hierarchy (inferred)",
    "Hierarchy Score (original)",
])

# --- Tab 1: Mutual communication groups ---
with tab_teams:
    st.subheader("Mutual Communication Groups")
    st.caption(
        "Finds people who **both send to and receive from** a consistent set of others. "
        "This naturally filters out copiers and bots (which only send, never receive replies). "
        "**These relationships are mutual (two-way), not supervisory** — they show who "
        "communicates closely with whom, *not* who reports to whom. A pair therefore "
        "appears from both sides (if A↔B, then B is in A's group and A is in B's)."
    )

    with st.sidebar:
        min_msgs = st.slider(
            "Min messages per direction", 3, 20, 5,
            help="A->B and B->A must each have at least this many messages",
            key="recip_min_msgs",
        )
        min_team = st.slider(
            "Min team size", 2, 10, 3,
            help="Minimum reciprocal contacts to qualify as a 'team'",
            key="recip_min_team",
        )

    # Cached team inference
    teams = _cached_teams(start_date, end_date, scope_tab, effective_excludes, min_msgs, min_team)

    if len(teams) == 0:
        st.info("No mutual communication groups found. Try lowering the minimum thresholds.")
    else:
        # KPIs
        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("People in Mutual Groups", f"{len(teams):,}")
        with c2:
            avg_team = teams["team_size"].mean()
            st.metric("Avg Mutual Contacts", f"{avg_team:.1f}")
        with c3:
            max_team = teams["team_size"].max()
            st.metric("Most Mutual Contacts", f"{max_team:,}")

        # Bar chart: top people by number of mutual contacts
        top_teams = teams.head(30).select([
            "manager", "display_name", "team_size",
            "total_sent_to_team", "total_recv_from_team",
        ])
        top_pd = top_teams.with_columns(
            pl.col("manager").str.split("@").list.first().alias("short_name")
        ).to_pandas()

        fig = px.bar(
            top_pd, x="short_name", y="team_size",
            hover_data=["manager", "display_name", "total_sent_to_team", "total_recv_from_team"],
            title="Top 30 by Number of Mutual Contacts",
            labels={"short_name": "Person", "team_size": "Mutual Contacts"},
            color="total_sent_to_team",
            color_continuous_scale="Blues",
        )
        fig.update_layout(height=450, xaxis_tickangle=-45)
        ev_teams = st.plotly_chart(fig, width="stretch", on_select="rerun", key="p14_teams")
        handle_scatter_person_click(ev_teams, "p14_teams", start_date, end_date, customdata_index=0)

        # Treemap — each person and the people they mutually communicate with.
        st.subheader("Mutual Contacts by Person")
        st.caption(
            "Each top-level box is a person; nested boxes are people they have "
            "**two-way** communication with. Relationships are mutual, so the same "
            "pair appears under both people — this is **not** a reporting chart."
        )
        pairs = build_reporting_pairs_from_teams(teams.head(20))
        if len(pairs) > 0:
            pairs_pd = pairs.rename({"manager": "person", "report": "mutual_contact"}).to_pandas()
            fig2 = px.treemap(
                pairs_pd, path=["person", "mutual_contact"], values="msg_count",
                title="Each Person's Mutual Contacts (two-way communication)",
            )
            fig2.update_layout(height=600)
            st.plotly_chart(fig2, width="stretch")

        # Table
        st.subheader("All Mutual Communication Groups")
        table_df = teams.select([
            pl.col("manager").alias("person"),
            "display_name",
            pl.col("team_size").alias("mutual_contacts"),
            pl.col("total_sent_to_team").alias("messages_sent"),
            pl.col("total_recv_from_team").alias("messages_received"),
        ])
        teams_pd = table_df.to_pandas()
        ev_teams_df = st.dataframe(teams_pd, width="stretch", on_select="rerun", selection_mode="single-row", key="p14_teams_df")
        handle_dataframe_person_click(ev_teams_df, teams_pd, "p14_teams_df", "person", start_date, end_date)
        download_csv_button(table_df, "mutual_communication_groups.csv")

# --- Tab: Calculated (inferred) hierarchy ---
with tab_calc:
    st.subheader("Calculated Hierarchy")
    st.warning(
        "**This is an algorithmic estimate inferred from email patterns — not an "
        "official org chart, and not evidence of actual reporting lines.** Each "
        "person is linked to the higher-'authority' colleague they communicate with "
        "most, where authority = sends to many / receives from few. Use it as a "
        "*hypothesis* about informal influence structure, never as an HR determination."
    )

    calc = _cached_calc_hierarchy(
        start_date, end_date, scope_tab, effective_excludes, min_msgs=5
    )

    if len(calc) == 0:
        st.info("Not enough data to infer a hierarchy in this scope.")
    else:
        n_roots = len(calc.filter(pl.col("inferred_manager") == ""))
        c1, c2 = st.columns(2)
        with c1:
            st.metric("People placed", f"{len(calc):,}")
        with c2:
            st.metric("Top-level (no inferred manager)", f"{n_roots:,}")

        # Org-chart-style treemap: parents = inferred manager, root = "" .
        calc_pd = calc.to_pandas()
        labels = [e.split("@")[0] for e in calc_pd["email"]]
        fig_calc = go.Figure(go.Treemap(
            ids=calc_pd["email"].tolist(),
            labels=labels,
            parents=calc_pd["inferred_manager"].tolist(),
            values=(calc_pd["total_volume"] + 1).tolist(),
            branchvalues="remainder",
            hovertext=[
                f"{e}<br>authority score: {s:.2f}<br>messages to inferred manager: {w}"
                for e, s, w in zip(calc_pd["email"], calc_pd["hierarchy_score"], calc_pd["msg_weight"])
            ],
            hoverinfo="text",
            root_color="lightgrey",
        ))
        fig_calc.update_layout(height=600, margin=dict(t=30, l=0, r=0, b=0))
        st.plotly_chart(fig_calc, width="stretch")

        st.subheader("Inferred Reporting Relationships")
        calc_table = calc.select([
            pl.col("email").alias("person"),
            "display_name",
            pl.col("inferred_manager").alias("inferred_reports_to"),
            pl.col("hierarchy_score").round(2).alias("authority_score"),
            pl.col("msg_weight").alias("messages_with_manager"),
        ])
        ev_calc = st.dataframe(
            calc_table.to_pandas(), width="stretch",
            on_select="rerun", selection_mode="single-row", key="p14_calc_df",
        )
        handle_dataframe_person_click(ev_calc, calc_table.to_pandas(), "p14_calc_df", "person", start_date, end_date)
        download_csv_button(calc_table, "calculated_hierarchy.csv")

# --- Tab 2: Original hierarchy score ---
with tab_score:
    st.subheader("Hierarchy Score")
    st.caption(
        "Original algorithm: score = unique_recipients / unique_senders. "
        "High score means sends to many, receives from few. "
        "Works best **with the nonhuman filter enabled** — otherwise copiers dominate."
    )

    # Cached hierarchy score
    hierarchy_scores = _cached_hierarchy_score(start_date, end_date, scope_tab, effective_excludes)

    if len(hierarchy_scores) == 0:
        st.info("Not enough data to compute hierarchy scores.")
    else:
        scatter_df = hierarchy_scores.filter(
            pl.col("total_sent").fill_null(0) + pl.col("total_received").fill_null(0) > 10
        ).with_columns(
            (pl.col("total_sent").fill_null(0) + pl.col("total_received").fill_null(0)).alias("total_volume")
        ).head(500)

        if len(scatter_df) > 0:
            scatter_pd = scatter_df.to_pandas()
            fig3 = px.scatter(
                scatter_pd, x="total_volume", y="hierarchy_score",
                hover_data=["email", "display_name", "domain"],
                color="is_internal",
                color_discrete_map={True: "#4e79a7", False: "#e15759"},
                title="Hierarchy Score vs Communication Volume",
                labels={"total_volume": "Total Messages", "hierarchy_score": "Hierarchy Score"},
            )
            fig3.update_layout(height=500)
            ev_hier_scat = st.plotly_chart(fig3, width="stretch", on_select="rerun", key="p14_hier_scat")
            handle_scatter_person_click(ev_hier_scat, "p14_hier_scat", start_date, end_date, customdata_index=0)

        top_scores = hierarchy_scores.select([
            "email", "display_name", "domain", "hierarchy_score",
            "unique_recipients", "unique_senders_to", "total_sent", "total_received",
        ]).head(50)
        hier_pd = top_scores.to_pandas()
        ev_hier_df = st.dataframe(hier_pd, width="stretch", on_select="rerun", selection_mode="single-row", key="p14_hier_df")
        handle_dataframe_person_click(ev_hier_df, hier_pd, "p14_hier_df", "email", start_date, end_date)
        download_csv_button(top_scores, "hierarchy_scores.csv")
