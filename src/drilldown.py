"""Drill-down dialog infrastructure for interactive data exploration."""

import datetime as dt

import streamlit as st
import plotly.express as px
import polars as pl


# ── Selection guard ──────────────────────────────────────────────────────

def should_open_drilldown(key: str, selection_value) -> bool:
    """Return True only for NEW selections (prevents dialog reopen on stale state)."""
    if selection_value is None:
        return False
    state_key = f"_last_sel_{key}"
    if st.session_state.get(state_key) == selection_value:
        return False
    st.session_state[state_key] = selection_value
    return True


# ── Extraction helpers ───────────────────────────────────────────────────

def extract_email_from_plotly(event, field: str = "x") -> str | None:
    """Extract value from plotly bar/line chart selection (x-axis field)."""
    try:
        points = event.selection.points
        if points:
            return points[0].get(field)
    except (AttributeError, TypeError, IndexError):
        pass
    return None


def extract_email_from_scatter(event, customdata_index: int = 0) -> str | None:
    """Extract email from scatter chart where email is in hover_data/customdata."""
    try:
        points = event.selection.points
        if points:
            cd = points[0].get("customdata")
            if cd and len(cd) > customdata_index:
                return cd[customdata_index]
    except (AttributeError, TypeError, IndexError):
        pass
    return None


def extract_email_from_dataframe(event, df_pandas, col: str) -> str | None:
    """Extract value from dataframe row selection."""
    try:
        rows = event.selection.rows
        if rows:
            row_idx = rows[0]
            if row_idx < len(df_pandas):
                return df_pandas.iloc[row_idx][col]
    except (AttributeError, TypeError, IndexError, KeyError):
        pass
    return None


def extract_community_from_plotly(event) -> int | None:
    """Extract community_id from community bar chart (x-axis)."""
    try:
        points = event.selection.points
        if points:
            val = points[0].get("x")
            if val is not None:
                return int(val)
    except (AttributeError, TypeError, IndexError, ValueError):
        pass
    return None


def extract_week_from_plotly(event) -> str | None:
    """Extract week_start from time-series chart (x-axis)."""
    try:
        points = event.selection.points
        if points:
            val = points[0].get("x")
            if val is not None:
                return str(val)
    except (AttributeError, TypeError, IndexError):
        pass
    return None


def extract_dyad_from_plotly(event, dyads_df_pandas) -> tuple[str, str] | None:
    """Extract (from_email, to_email) from dyad bar chart using pointIndex."""
    try:
        points = event.selection.points
        if points:
            idx = points[0].get("point_index", points[0].get("pointIndex", points[0].get("pointNumber")))
            if idx is not None and idx < len(dyads_df_pandas):
                row = dyads_df_pandas.iloc[idx]
                return (row["from_email"], row["to_email"])
    except (AttributeError, TypeError, IndexError, KeyError):
        pass
    return None


def extract_dyad_from_dataframe(event, df_pandas) -> tuple[str, str] | None:
    """Extract (from_email, to_email) from dataframe row selection."""
    try:
        rows = event.selection.rows
        if rows:
            row_idx = rows[0]
            if row_idx < len(df_pandas):
                row = df_pandas.iloc[row_idx]
                return (row["from_email"], row["to_email"])
    except (AttributeError, TypeError, IndexError, KeyError):
        pass
    return None


# ── Person dialog ────────────────────────────────────────────────────────

@st.dialog("Person Profile", width="large")
def show_person_dialog(email: str, start_date, end_date):
    """Condensed person profile popup."""
    from src.state import (
        load_person_dim, load_filtered_message_fact,
        load_filtered_edge_fact, load_filtered_graph_metrics,
    )

    pd_dim = load_person_dim()
    person = pd_dim.filter(pl.col("email") == email)

    if len(person) == 0:
        st.warning(f"No data found for **{email}**")
        return

    row = person.row(0, named=True)

    # Header
    st.markdown(f"**{row.get('display_name') or email}**")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Sent", f"{row['total_sent']:,}")
    with c2:
        st.metric("Received", f"{row['total_received']:,}")
    with c3:
        st.metric("Domain", row["domain"])
    with c4:
        st.metric("Type", "Internal" if row["is_internal"] else "External")

    # Top contacts
    ef = load_filtered_edge_fact(start_date, end_date)
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("**Top sent to:**")
        sent_to = (
            ef.filter(pl.col("from_email") == email)
            .group_by("to_email").agg(pl.len().alias("count"))
            .sort("count", descending=True).head(5)
        )
        if len(sent_to) > 0:
            st.dataframe(sent_to.to_pandas(), hide_index=True, width="stretch")
        else:
            st.caption("No sent messages in range")
    with col_b:
        st.markdown("**Top received from:**")
        recv_from = (
            ef.filter(pl.col("to_email") == email)
            .group_by("from_email").agg(pl.len().alias("count"))
            .sort("count", descending=True).head(5)
        )
        if len(recv_from) > 0:
            st.dataframe(recv_from.to_pandas(), hide_index=True, width="stretch")
        else:
            st.caption("No received messages in range")

    # Community + graph metrics
    try:
        gm = load_filtered_graph_metrics(start_date, end_date)
        person_gm = gm.filter(pl.col("email") == email)
        if len(person_gm) > 0:
            gm_row = person_gm.row(0, named=True)
            c1, c2, c3 = st.columns(3)
            with c1:
                st.metric("Community", gm_row["community_id"])
            with c2:
                st.metric("PageRank", f"{gm_row['pagerank']:.4f}")
            with c3:
                st.metric("Betweenness", f"{gm_row['betweenness_centrality']:.4f}")
    except Exception:
        pass

    # Behavioral
    mf = load_filtered_message_fact(start_date, end_date)
    person_msgs = mf.filter(pl.col("from_email") == email)
    if len(person_msgs) > 0:
        c1, c2 = st.columns(2)
        with c1:
            st.metric("After-Hours Rate", f"{person_msgs['is_after_hours'].mean():.1%}")
        with c2:
            st.metric("Weekend Rate", f"{person_msgs['is_weekend'].mean():.1%}")

    # Recent messages sent by this person
    st.divider()
    st.markdown("**Recent Messages Sent**")
    if len(person_msgs) > 0:
        msg_display = (
            person_msgs.sort("timestamp", descending=True)
            .head(50)
            .with_columns(pl.col("to_emails").list.join(", ").alias("recipients"))
            .select([
                pl.col("timestamp").dt.strftime("%Y-%m-%d %H:%M").alias("date"),
                "recipients",
                "n_recipients",
                (pl.col("size_bytes") / 1024).round(1).alias("size_kb"),
            ])
        )
        st.dataframe(msg_display.to_pandas(), hide_index=True, width="stretch", height=250)
        st.caption(f"Showing {min(50, len(person_msgs))} of {len(person_msgs):,} messages sent")
    else:
        st.caption("No sent messages in range")

    # Recent messages received
    ef_recv = ef.filter(pl.col("to_email") == email)
    if len(ef_recv) > 0:
        st.markdown("**Recent Messages Received**")
        recv_display = (
            ef_recv.sort("timestamp", descending=True)
            .head(50)
            .select([
                pl.col("timestamp").dt.strftime("%Y-%m-%d %H:%M").alias("date"),
                "from_email",
                (pl.col("size_bytes") / 1024).round(1).alias("size_kb"),
            ])
        )
        st.dataframe(recv_display.to_pandas(), hide_index=True, width="stretch", height=250)
        st.caption(f"Showing {min(50, len(ef_recv))} of {len(ef_recv):,} messages received")

    # Full profile link
    st.divider()
    st.page_link("pages/12_search.py", label="Open Full Profile",
                 icon=":material/open_in_new:", query_params={"email": email})


# ── Community dialog ─────────────────────────────────────────────────────

@st.dialog("Community Details", width="large")
def show_community_dialog(community_id: int, start_date, end_date):
    """Community members and stats popup."""
    from src.state import load_person_dim, load_filtered_graph_metrics

    gm = load_filtered_graph_metrics(start_date, end_date)
    members = gm.filter(pl.col("community_id") == community_id)
    pd_dim = load_person_dim()

    enriched = members.join(
        pd_dim.select(["email", "display_name", "is_internal"]),
        on="email", how="left",
    ).sort("pagerank", descending=True)

    st.markdown(f"### Community {community_id}")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Members", f"{len(enriched):,}")
    with c2:
        st.metric("Avg PageRank", f"{enriched['pagerank'].mean():.4f}")
    with c3:
        avg_deg = (enriched["in_degree"] + enriched["out_degree"]).mean()
        st.metric("Avg Degree", f"{avg_deg:.1f}")

    n_internal = len(enriched.filter(pl.col("is_internal")))
    st.caption(f"Internal: {n_internal} | External: {len(enriched) - n_internal}")

    st.dataframe(
        enriched.select(["email", "display_name", "pagerank", "in_degree", "out_degree"])
        .head(30).to_pandas(),
        hide_index=True, width="stretch",
    )


# ── Week dialog ──────────────────────────────────────────────────────────

@st.dialog("Week Breakdown", width="large")
def show_week_dialog(week_start_str: str, start_date, end_date):
    """Daily breakdown for a specific week."""
    from src.state import load_filtered_message_fact

    # Parse the week_start from the chart x-axis value
    try:
        if "T" in str(week_start_str):
            week_start = dt.datetime.fromisoformat(str(week_start_str).replace("Z", "")).date()
        else:
            week_start = dt.date.fromisoformat(str(week_start_str)[:10])
    except (ValueError, TypeError):
        st.warning(f"Could not parse date: {week_start_str}")
        return

    week_end = week_start + dt.timedelta(days=6)

    st.markdown(f"### Week of {week_start.strftime('%b %d, %Y')}")

    mf = load_filtered_message_fact(start_date, end_date)
    week_msgs = mf.filter(
        (pl.col("timestamp").dt.date() >= week_start)
        & (pl.col("timestamp").dt.date() <= week_end)
    )

    if len(week_msgs) == 0:
        st.info("No messages in this week.")
        return

    # Key stats
    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Total Messages", f"{len(week_msgs):,}")
    with c2:
        st.metric("Unique Senders", f"{week_msgs['from_email'].n_unique():,}")
    with c3:
        st.metric("After-Hours %", f"{week_msgs['is_after_hours'].mean():.1%}")

    # Daily breakdown
    daily = (
        week_msgs.with_columns(pl.col("timestamp").dt.date().alias("date"))
        .group_by("date").agg(pl.len().alias("count")).sort("date")
    )
    fig = px.bar(daily.to_pandas(), x="date", y="count", title="Daily Volume")
    fig.update_layout(height=250)
    st.plotly_chart(fig, width="stretch")

    # Top senders
    top_senders = (
        week_msgs.group_by("from_email")
        .agg(pl.len().alias("count"))
        .sort("count", descending=True).head(10)
    )
    st.markdown("**Top 10 senders this week:**")
    st.dataframe(top_senders.to_pandas(), hide_index=True, width="stretch")


# ── Dyad dialog ──────────────────────────────────────────────────────────

@st.dialog("Pair Details", width="large")
def show_dyad_dialog(email_a: str, email_b: str, start_date, end_date):
    """Exchange details between two people."""
    from src.state import load_filtered_edge_fact

    ef = load_filtered_edge_fact(start_date, end_date)

    a_to_b = ef.filter((pl.col("from_email") == email_a) & (pl.col("to_email") == email_b))
    b_to_a = ef.filter((pl.col("from_email") == email_b) & (pl.col("to_email") == email_a))

    name_a = email_a.split("@")[0]
    name_b = email_b.split("@")[0]

    st.markdown(f"### {name_a} <-> {name_b}")

    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric(f"{name_a} -> {name_b}", f"{len(a_to_b):,}")
    with c2:
        st.metric(f"{name_b} -> {name_a}", f"{len(b_to_a):,}")
    with c3:
        total = len(a_to_b) + len(b_to_a)
        if total > 0:
            asym = abs(len(a_to_b) - len(b_to_a)) / total
            st.metric("Asymmetry", f"{asym:.2f}")
        else:
            st.metric("Total", "0")

    # Weekly timeline
    parts = []
    if len(a_to_b) > 0:
        parts.append(a_to_b.select(["timestamp"]).with_columns(pl.lit(f"{name_a}->{name_b}").alias("direction")))
    if len(b_to_a) > 0:
        parts.append(b_to_a.select(["timestamp"]).with_columns(pl.lit(f"{name_b}->{name_a}").alias("direction")))

    if parts:
        all_msgs = pl.concat(parts)
        weekly = (
            all_msgs.with_columns(pl.col("timestamp").dt.truncate("1w").alias("week"))
            .group_by(["week", "direction"]).agg(pl.len().alias("count")).sort("week")
        )
        if len(weekly) > 0:
            fig = px.line(weekly.to_pandas(), x="week", y="count", color="direction",
                          title="Exchange Volume Over Time")
            fig.update_layout(height=250)
            st.plotly_chart(fig, width="stretch")


# ── Convenience wrappers (one-liners for pages) ─────────────────────────

def handle_plotly_person_click(event, key, start_date, end_date, field="x"):
    """Extract email from plotly chart selection and open person dialog if new."""
    email = extract_email_from_plotly(event, field=field)
    if should_open_drilldown(key, email):
        show_person_dialog(email, start_date, end_date)


def handle_scatter_person_click(event, key, start_date, end_date, customdata_index=0):
    """Extract email from scatter hover_data and open person dialog if new."""
    email = extract_email_from_scatter(event, customdata_index=customdata_index)
    if should_open_drilldown(key, email):
        show_person_dialog(email, start_date, end_date)


def handle_dataframe_person_click(event, df_pandas, key, email_col, start_date, end_date):
    """Extract email from dataframe row selection and open person dialog if new."""
    email = extract_email_from_dataframe(event, df_pandas, email_col)
    if should_open_drilldown(key, email):
        show_person_dialog(email, start_date, end_date)


def handle_plotly_community_click(event, key, start_date, end_date):
    """Extract community_id from chart and open community dialog if new."""
    comm_id = extract_community_from_plotly(event)
    if should_open_drilldown(key, comm_id):
        show_community_dialog(comm_id, start_date, end_date)


def handle_plotly_week_click(event, key, start_date, end_date):
    """Extract week_start from time-series chart and open week dialog if new."""
    week = extract_week_from_plotly(event)
    if should_open_drilldown(key, week):
        show_week_dialog(week, start_date, end_date)


def handle_dyad_chart_click(event, key, dyads_df_pandas, start_date, end_date):
    """Extract dyad from bar chart and open dyad dialog if new."""
    pair = extract_dyad_from_plotly(event, dyads_df_pandas)
    if pair and should_open_drilldown(key, pair):
        show_dyad_dialog(pair[0], pair[1], start_date, end_date)


def handle_dataframe_dyad_click(event, df_pandas, key, start_date, end_date):
    """Extract dyad from dataframe row and open dyad dialog if new."""
    pair = extract_dyad_from_dataframe(event, df_pandas)
    if pair and should_open_drilldown(key, pair):
        show_dyad_dialog(pair[0], pair[1], start_date, end_date)
