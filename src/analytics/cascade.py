"""Information Cascade Detection: forwarding chains, metrics, amplifier identification."""

import polars as pl
from collections import defaultdict


def detect_cascades(
    edge_fact: pl.DataFrame,
    max_delay_minutes: int = 60,
    min_chain_length: int = 3,
) -> pl.DataFrame:
    """Detect forwarding chains A->B->C within time windows.

    Uses join_asof to find potential forwarding: if A emails B at T1,
    then B emails C at T1+delta (within max_delay_minutes), that's a
    potential cascade step.

    Returns: cascade_id, step, from_email, to_email, timestamp, delay_seconds.
    """
    if len(edge_fact) == 0:
        return pl.DataFrame({
            "cascade_id": [], "step": [], "from_email": [],
            "to_email": [], "timestamp": [], "delay_seconds": [],
        })

    # Prepare: sorted messages
    msgs = (
        edge_fact.select(["from_email", "to_email", "timestamp"])
        .sort("timestamp")
        .with_row_index("msg_id")
    )

    # Find potential forwarding pairs using join_asof:
    # For each message B->C, find the most recent A->B within the time window
    # "incoming" = messages TO person B (A->B)
    incoming = msgs.select([
        pl.col("to_email").alias("person"),
        pl.col("from_email").alias("predecessor"),
        pl.col("timestamp").alias("in_time"),
        pl.col("msg_id").alias("in_msg_id"),
    ]).sort(["person", "in_time"])

    # "outgoing" = messages FROM person B (B->C)
    outgoing = msgs.select([
        pl.col("from_email").alias("person"),
        pl.col("to_email").alias("successor"),
        pl.col("timestamp").alias("out_time"),
        pl.col("msg_id").alias("out_msg_id"),
    ]).sort(["person", "out_time"])

    # For each outgoing message, find the latest incoming message to the same person
    # that arrived BEFORE the outgoing message (strategy="backward")
    joined = outgoing.join_asof(
        incoming,
        left_on="out_time",
        right_on="in_time",
        by="person",
        strategy="backward",
    )

    # Filter: incoming must exist and delay must be within window
    joined = joined.filter(
        pl.col("in_time").is_not_null()
        & (pl.col("predecessor") != pl.col("successor"))  # not echoing back
        & (pl.col("predecessor") != pl.col("person"))  # not self-reply
    )
    joined = joined.with_columns(
        ((pl.col("out_time") - pl.col("in_time")).dt.total_seconds()).alias("delay_seconds")
    )
    joined = joined.filter(
        (pl.col("delay_seconds") >= 0)
        & (pl.col("delay_seconds") <= max_delay_minutes * 60)
    )

    if len(joined) == 0:
        return pl.DataFrame({
            "cascade_id": [], "step": [], "from_email": [],
            "to_email": [], "timestamp": [], "delay_seconds": [],
        })

    # Build chains via BFS from seed edges
    # Represent each forwarding link as (predecessor->person, person->successor)
    edges = []
    for row in joined.iter_rows(named=True):
        edges.append({
            "from": row["predecessor"],
            "via": row["person"],
            "to": row["successor"],
            "time": row["out_time"],
            "delay": row["delay_seconds"],
            "msg_id": row["out_msg_id"],
        })

    # Group by "to" to build forward adjacency
    # Chain: if edge1.to == edge2.via, and edge2.time > edge1.time, they connect
    # Key by "to" so we can find edges where current_person received and then forwarded
    forward_adj = defaultdict(list)
    for e in edges:
        forward_adj[e["to"]].append(e)

    # Find chains using greedy forward chaining
    used_msgs = set()
    cascades = []
    cascade_id = 0

    # Sort edges by time to process chronologically
    edges.sort(key=lambda e: e["time"])

    for seed in edges:
        if seed["msg_id"] in used_msgs:
            continue

        # Try to build a chain starting from this edge
        chain = [{
            "from_email": seed["from"],
            "to_email": seed["via"],
            "timestamp": seed["time"],
            "delay_seconds": 0,
        }, {
            "from_email": seed["via"],
            "to_email": seed["to"],
            "timestamp": seed["time"],
            "delay_seconds": seed["delay"],
        }]
        used_msgs.add(seed["msg_id"])

        # Extend chain forward
        current_person = seed["to"]
        current_time = seed["time"]
        max_depth = 20  # prevent pathological cases

        while len(chain) < max_depth:
            # Find next hop: current_person sends to someone after current_time
            next_edges = [
                e for e in forward_adj.get(current_person, [])
                if e["msg_id"] not in used_msgs
                and e["time"] > current_time
                and (e["time"] - current_time).total_seconds() <= max_delay_minutes * 60
            ]
            if not next_edges:
                break

            # Take the earliest next hop
            next_edge = min(next_edges, key=lambda e: e["time"])
            delay = (next_edge["time"] - current_time).total_seconds()

            chain.append({
                "from_email": next_edge["via"],
                "to_email": next_edge["to"],
                "timestamp": next_edge["time"],
                "delay_seconds": delay,
            })
            used_msgs.add(next_edge["msg_id"])
            current_person = next_edge["to"]
            current_time = next_edge["time"]

        if len(chain) >= min_chain_length:
            for step_idx, link in enumerate(chain):
                link["cascade_id"] = cascade_id
                link["step"] = step_idx
            cascades.extend(chain)
            cascade_id += 1

    if not cascades:
        return pl.DataFrame({
            "cascade_id": [], "step": [], "from_email": [],
            "to_email": [], "timestamp": [], "delay_seconds": [],
        })

    return pl.DataFrame(cascades).select([
        "cascade_id", "step", "from_email", "to_email", "timestamp", "delay_seconds",
    ])


def compute_cascade_metrics(cascades: pl.DataFrame) -> pl.DataFrame:
    """Aggregate metrics per cascade."""
    if len(cascades) == 0:
        return pl.DataFrame({
            "cascade_id": [], "depth": [], "breadth": [],
            "velocity_seconds": [], "seed_sender": [],
            "duration_minutes": [],
        })

    metrics = (
        cascades.group_by("cascade_id")
        .agg([
            pl.col("step").max().alias("depth"),
            pl.col("to_email").n_unique().alias("breadth"),
            pl.col("delay_seconds").median().alias("velocity_seconds"),
            pl.col("from_email").first().alias("seed_sender"),
            pl.col("timestamp").min().alias("start_time"),
            pl.col("timestamp").max().alias("end_time"),
        ])
        .with_columns(
            ((pl.col("end_time") - pl.col("start_time")).dt.total_seconds() / 60.0)
            .alias("duration_minutes")
        )
        .drop(["start_time", "end_time"])
        .sort("depth", descending=True)
    )

    return metrics


def identify_amplifiers(cascades: pl.DataFrame) -> pl.DataFrame:
    """Find amplifier nodes: people with high fan-out within cascades."""
    if len(cascades) == 0:
        return pl.DataFrame({
            "email": [], "n_cascades": [], "avg_fanout": [],
            "total_downstream": [], "amplifier_score": [],
        })

    # For each person acting as a forwarder (from_email at step > 0)
    forwarders = cascades.filter(pl.col("step") > 0)

    # Fan-out per person per cascade
    fanout = (
        forwarders.group_by(["cascade_id", "from_email"])
        .agg(pl.col("to_email").n_unique().alias("fanout"))
    )

    # Aggregate across cascades
    amplifiers = (
        fanout.group_by("from_email")
        .agg([
            pl.len().alias("n_cascades"),
            pl.col("fanout").mean().alias("avg_fanout"),
            pl.col("fanout").sum().alias("total_downstream"),
        ])
        .rename({"from_email": "email"})
        .with_columns(
            (pl.col("n_cascades") * pl.col("avg_fanout")).alias("amplifier_score")
        )
        .sort("amplifier_score", descending=True)
    )

    return amplifiers
