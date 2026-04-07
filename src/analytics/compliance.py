"""Compliance Pattern Detection: blackouts, external spikes, key dates, after-hours clusters."""

import networkx as nx
import community as community_louvain
import polars as pl
import numpy as np


def detect_blackout_windows(
    message_fact: pl.DataFrame,
    min_gap_hours: float = 48.0,
    min_historical_volume: int = 5,
) -> pl.DataFrame:
    """Find periods where an active sender goes suddenly silent.

    For each sender with at least min_historical_volume msgs/week average,
    find gaps >= min_gap_hours between consecutive messages.
    """
    if len(message_fact) == 0:
        return pl.DataFrame({
            "from_email": [], "gap_start": [], "gap_end": [],
            "gap_hours": [], "avg_weekly_volume": [],
        })

    # Per-sender message timestamps, sorted
    sender_msgs = (
        message_fact.select(["from_email", "timestamp"])
        .sort(["from_email", "timestamp"])
    )

    # Compute weekly volume per sender
    weekly_vol = (
        message_fact
        .with_columns(pl.col("timestamp").dt.truncate("1w").alias("week"))
        .group_by(["from_email", "week"])
        .agg(pl.len().alias("week_count"))
        .group_by("from_email")
        .agg(pl.col("week_count").mean().alias("avg_weekly_volume"))
        .filter(pl.col("avg_weekly_volume") >= min_historical_volume)
    )

    active_senders = set(weekly_vol["from_email"].to_list())
    sender_msgs = sender_msgs.filter(pl.col("from_email").is_in(list(active_senders)))

    # Compute gaps between consecutive messages per sender
    sender_msgs = sender_msgs.with_columns(
        pl.col("timestamp").shift(1).over("from_email").alias("prev_timestamp")
    )
    sender_msgs = sender_msgs.filter(pl.col("prev_timestamp").is_not_null())
    sender_msgs = sender_msgs.with_columns(
        ((pl.col("timestamp") - pl.col("prev_timestamp")).dt.total_seconds() / 3600.0)
        .alias("gap_hours")
    )

    # Filter to significant gaps
    gaps = sender_msgs.filter(pl.col("gap_hours") >= min_gap_hours)
    gaps = gaps.rename({"prev_timestamp": "gap_start", "timestamp": "gap_end"})

    # Join weekly volume
    gaps = gaps.join(weekly_vol, on="from_email", how="left")

    return (
        gaps.select(["from_email", "gap_start", "gap_end", "gap_hours", "avg_weekly_volume"])
        .sort("gap_hours", descending=True)
    )


def detect_external_spikes(
    edge_fact: pl.DataFrame,
    person_dim: pl.DataFrame,
    z_threshold: float = 2.5,
) -> pl.DataFrame:
    """Detect weeks where external contact count spikes for a person.

    Per-person weekly external unique contact count, z-scored against their own baseline.
    """
    if len(edge_fact) == 0:
        return pl.DataFrame({
            "from_email": [], "week_start": [], "external_contacts": [],
            "zscore": [], "is_spike": [],
        })

    # Identify external emails
    external_emails = set(
        person_dim.filter(~pl.col("is_internal"))["email"].to_list()
    ) if "is_internal" in person_dim.columns else set()

    if not external_emails:
        return pl.DataFrame({
            "from_email": [], "week_start": [], "external_contacts": [],
            "zscore": [], "is_spike": [],
        })

    # Filter to external-facing edges
    external_edges = edge_fact.filter(
        pl.col("to_email").is_in(list(external_emails))
    )

    # Weekly unique external contacts per sender
    weekly_ext = (
        external_edges
        .with_columns(pl.col("timestamp").dt.truncate("1w").alias("week_start"))
        .group_by(["from_email", "week_start"])
        .agg(pl.col("to_email").n_unique().alias("external_contacts"))
    )

    # Z-score per person against their own baseline
    weekly_ext = weekly_ext.with_columns([
        pl.col("external_contacts").mean().over("from_email").alias("mean_ext"),
        pl.col("external_contacts").std().over("from_email").alias("std_ext"),
    ])
    weekly_ext = weekly_ext.with_columns(
        pl.when(pl.col("std_ext") > 0)
        .then((pl.col("external_contacts") - pl.col("mean_ext")) / pl.col("std_ext"))
        .otherwise(0.0)
        .alias("zscore")
    )
    weekly_ext = weekly_ext.with_columns(
        (pl.col("zscore") >= z_threshold).alias("is_spike")
    )

    return (
        weekly_ext
        .select(["from_email", "week_start", "external_contacts", "zscore", "is_spike"])
        .sort("zscore", descending=True)
    )


def key_date_gap_analysis(
    message_fact: pl.DataFrame,
    key_dates: list[dict],
    window_days: int = 7,
) -> pl.DataFrame:
    """For each key date, analyze communication volume in +/- window.

    key_dates: [{"label": "Event Name", "date": datetime.date}, ...]
    """
    if len(message_fact) == 0 or not key_dates:
        return pl.DataFrame({
            "key_date": [], "label": [], "pre_window_volume": [],
            "post_window_volume": [], "volume_change_pct": [],
        })

    import datetime as dt

    records = []
    for kd in key_dates:
        date = kd["date"]
        label = kd["label"]

        pre_start = dt.datetime.combine(date - dt.timedelta(days=window_days), dt.time.min)
        pre_end = dt.datetime.combine(date - dt.timedelta(days=1), dt.time.max)
        post_start = dt.datetime.combine(date, dt.time.min)
        post_end = dt.datetime.combine(date + dt.timedelta(days=window_days), dt.time.max)

        pre_vol = len(message_fact.filter(
            (pl.col("timestamp") >= pre_start) & (pl.col("timestamp") <= pre_end)
        ))
        post_vol = len(message_fact.filter(
            (pl.col("timestamp") >= post_start) & (pl.col("timestamp") <= post_end)
        ))

        change_pct = ((post_vol - pre_vol) / pre_vol * 100) if pre_vol > 0 else 0.0

        records.append({
            "key_date": date,
            "label": label,
            "pre_window_volume": pre_vol,
            "post_window_volume": post_vol,
            "volume_change_pct": round(change_pct, 1),
        })

    return pl.DataFrame(records)


def detect_after_hours_clusters(
    edge_fact: pl.DataFrame,
    after_hours_start: int = 18,
    after_hours_end: int = 7,
    min_after_hours_msgs: int = 10,
    min_cluster_size: int = 2,
) -> pl.DataFrame:
    """Find groups who consistently communicate after hours.

    Builds a graph from after-hours-only edges and runs community detection.
    """
    if len(edge_fact) == 0:
        return pl.DataFrame({
            "cluster_id": [], "members": [], "n_members": [],
            "total_after_hours_msgs": [], "peak_hour": [],
        })

    # Filter to after-hours messages
    ah_edges = edge_fact.with_columns(
        pl.col("timestamp").dt.hour().alias("hour")
    )
    ah_edges = ah_edges.filter(
        (pl.col("hour") >= after_hours_start) | (pl.col("hour") < after_hours_end)
    )

    if len(ah_edges) == 0:
        return pl.DataFrame({
            "cluster_id": [], "members": [], "n_members": [],
            "total_after_hours_msgs": [], "peak_hour": [],
        })

    # Build after-hours graph
    ah_weights = (
        ah_edges.group_by(["from_email", "to_email"])
        .agg(pl.len().alias("weight"))
    )

    G = nx.Graph()
    for row in ah_weights.iter_rows(named=True):
        G.add_edge(row["from_email"], row["to_email"], weight=row["weight"])

    if G.number_of_nodes() < 2:
        return pl.DataFrame({
            "cluster_id": [], "members": [], "n_members": [],
            "total_after_hours_msgs": [], "peak_hour": [],
        })

    # Community detection on after-hours graph
    partition = community_louvain.best_partition(G, weight="weight", resolution=0.5)

    # Aggregate per cluster
    comm_members: dict[int, list[str]] = {}
    for node, cid in partition.items():
        comm_members.setdefault(cid, []).append(node)

    # Peak hour per person for cluster-level peak
    person_peak = (
        ah_edges.group_by(["from_email", "hour"])
        .agg(pl.len().alias("count"))
        .sort(["from_email", "count"], descending=[False, True])
        .group_by("from_email")
        .first()
        .select(["from_email", "hour"])
    )
    peak_lookup = dict(zip(
        person_peak["from_email"].to_list(),
        person_peak["hour"].to_list(),
    ))

    records = []
    for cid, members in comm_members.items():
        if len(members) < min_cluster_size:
            continue
        # Total after-hours messages within cluster
        member_set = set(members)
        cluster_msgs = len(ah_edges.filter(
            pl.col("from_email").is_in(list(member_set))
            & pl.col("to_email").is_in(list(member_set))
        ))
        if cluster_msgs < min_after_hours_msgs:
            continue

        # Mode peak hour across members
        member_peaks = [peak_lookup.get(m, 0) for m in members]
        peak_hour = max(set(member_peaks), key=member_peaks.count) if member_peaks else 0

        records.append({
            "cluster_id": cid,
            "members": members,
            "n_members": len(members),
            "total_after_hours_msgs": cluster_msgs,
            "peak_hour": peak_hour,
        })

    if not records:
        return pl.DataFrame({
            "cluster_id": [], "members": [], "n_members": [],
            "total_after_hours_msgs": [], "peak_hour": [],
        })

    return pl.DataFrame(records).sort("total_after_hours_msgs", descending=True)
