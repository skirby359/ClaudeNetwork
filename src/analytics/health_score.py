"""Organizational Health Score: composite 0-100 metric from communication patterns."""

import datetime as dt

import polars as pl
import numpy as np

from src.analytics.volume import gini_coefficient


def compute_health_score(
    message_fact: pl.DataFrame,
    edge_fact: pl.DataFrame,
    graph_metrics: pl.DataFrame,
    reply_median_seconds: float | None = None,
) -> dict:
    """Compute a composite organizational health score (0-100).

    Sub-scores (each 0-100, higher = healthier):
    - Communication breadth: inverse of Gini (lower concentration = healthier)
    - Reciprocity rate: fraction of dyads with bidirectional communication
    - Work-life balance: inverse of after-hours rate
    - Response velocity: faster median reply = healthier (capped at 100)
    - Cross-group flow: ratio of inter-community to total edges
    - Resilience: inverse of max betweenness (no single point of failure)

    Returns dict with sub-scores, weights, and composite score.
    """
    sub_scores = {}

    # 1. Communication breadth (inverse Gini)
    if len(edge_fact) > 0:
        sender_counts = (
            edge_fact.group_by("from_email")
            .agg(pl.len().alias("count"))
        )["count"].to_numpy()
        gini = gini_coefficient(sender_counts)
        sub_scores["breadth"] = {
            "value": max(0, (1 - gini)) * 100,
            "label": "Communication Breadth",
            "detail": f"Gini: {gini:.2f} (lower = more distributed)",
        }
    else:
        sub_scores["breadth"] = {"value": 0, "label": "Communication Breadth", "detail": "No data"}

    # 2. Reciprocity rate
    if len(edge_fact) > 0:
        pair_counts = (
            edge_fact.group_by(["from_email", "to_email"])
            .agg(pl.len().alias("count"))
        )
        # Check for reverse edges
        forward = pair_counts.select(["from_email", "to_email"])
        reverse = pair_counts.select([
            pl.col("to_email").alias("from_email"),
            pl.col("from_email").alias("to_email"),
        ])
        bidirectional = forward.join(reverse, on=["from_email", "to_email"], how="inner")
        recip_rate = len(bidirectional) / max(len(pair_counts), 1)
        sub_scores["reciprocity"] = {
            "value": recip_rate * 100,
            "label": "Reciprocity",
            "detail": f"{recip_rate:.1%} of communication pairs are bidirectional",
        }
    else:
        sub_scores["reciprocity"] = {"value": 0, "label": "Reciprocity", "detail": "No data"}

    # 3. Work-life balance (inverse after-hours)
    if len(message_fact) > 0:
        ah_rate = float(message_fact["is_after_hours"].mean())
        # 0% after-hours = 100 score, 50%+ = 0 score
        wlb_score = max(0, (1 - ah_rate * 2)) * 100
        sub_scores["work_life"] = {
            "value": wlb_score,
            "label": "Work-Life Balance",
            "detail": f"{ah_rate:.1%} after-hours rate",
        }
    else:
        sub_scores["work_life"] = {"value": 50, "label": "Work-Life Balance", "detail": "No data"}

    # 4. Response velocity
    if reply_median_seconds is not None and reply_median_seconds > 0:
        # 15 min = 100, 4 hours = 50, 24 hours = 0
        minutes = reply_median_seconds / 60
        if minutes <= 15:
            resp_score = 100
        elif minutes <= 240:
            resp_score = 100 - (minutes - 15) / (240 - 15) * 50
        else:
            resp_score = max(0, 50 - (minutes - 240) / (1440 - 240) * 50)
        sub_scores["responsiveness"] = {
            "value": resp_score,
            "label": "Responsiveness",
            "detail": f"Median reply: {minutes:.0f} min",
        }
    else:
        sub_scores["responsiveness"] = {"value": 50, "label": "Responsiveness", "detail": "Insufficient reply data"}

    # 5. Cross-group flow (silo permeability)
    if len(graph_metrics) > 0 and "community_id" in graph_metrics.columns and len(edge_fact) > 0:
        # Use Polars joins instead of Python loop for cross-group calculation
        comm_df = graph_metrics.select(["email", "community_id"])
        sample = edge_fact.sample(min(100000, len(edge_fact))) if len(edge_fact) > 100000 else edge_fact
        with_comms = (
            sample.select(["from_email", "to_email"])
            .join(comm_df.rename({"email": "from_email", "community_id": "from_comm"}), on="from_email", how="inner")
            .join(comm_df.rename({"email": "to_email", "community_id": "to_comm"}), on="to_email", how="inner")
        )
        total = len(with_comms)
        cross = len(with_comms.filter(pl.col("from_comm") != pl.col("to_comm")))
        cross_rate = cross / max(total, 1)
        # 30-50% cross-group is healthy
        if cross_rate >= 0.3:
            flow_score = 100
        elif cross_rate >= 0.1:
            flow_score = 50 + (cross_rate - 0.1) / 0.2 * 50
        else:
            flow_score = cross_rate / 0.1 * 50
        sub_scores["cross_group"] = {
            "value": flow_score,
            "label": "Cross-Group Flow",
            "detail": f"{cross_rate:.1%} of messages cross group boundaries",
        }
    else:
        sub_scores["cross_group"] = {"value": 50, "label": "Cross-Group Flow", "detail": "No community data"}

    # 6. Resilience (inverse of max betweenness — no single point of failure)
    if len(graph_metrics) > 0 and "betweenness_centrality" in graph_metrics.columns:
        max_bc = float(graph_metrics["betweenness_centrality"].max())
        # max_bc of 0.01 = very resilient (100), 0.10 = fragile (0)
        resilience = max(0, (1 - max_bc * 10)) * 100
        sub_scores["resilience"] = {
            "value": resilience,
            "label": "Network Resilience",
            "detail": f"Max connector score: {max_bc:.4f} (lower = more resilient)",
        }
    else:
        sub_scores["resilience"] = {"value": 50, "label": "Network Resilience", "detail": "No graph data"}

    # Compute weighted composite
    weights = {
        "breadth": 0.15,
        "reciprocity": 0.20,
        "work_life": 0.15,
        "responsiveness": 0.20,
        "cross_group": 0.15,
        "resilience": 0.15,
    }

    composite = sum(
        sub_scores[k]["value"] * weights[k]
        for k in weights
    )

    return {
        "composite": composite,
        "sub_scores": sub_scores,
        "weights": weights,
    }


def compute_health_trend(
    message_fact: pl.DataFrame,
    edge_fact: pl.DataFrame,
    graph_metrics: pl.DataFrame,
) -> pl.DataFrame:
    """Compute health score per month for trend visualization.

    Returns DataFrame with month_id, composite, and each sub-score column.
    """
    if len(message_fact) == 0:
        return pl.DataFrame({"month_id": [], "composite": []})

    mf = message_fact.with_columns(
        pl.col("timestamp").dt.strftime("%Y-%m").alias("month_id")
    )
    ef = edge_fact.with_columns(
        pl.col("timestamp").dt.strftime("%Y-%m").alias("month_id")
    )
    months = sorted(mf["month_id"].unique().to_list())

    records = []
    for month in months:
        mf_m = mf.filter(pl.col("month_id") == month)
        ef_m = ef.filter(pl.col("month_id") == month)

        if len(mf_m) < 10 or len(ef_m) < 10:
            continue

        # Reply time for this month
        reply_median = None
        try:
            from src.analytics.response_time import compute_reply_times
            rt = compute_reply_times(ef_m)
            if len(rt) > 0:
                reply_median = float(rt["median_reply_seconds"].median())
        except Exception:
            pass

        health = compute_health_score(mf_m, ef_m, graph_metrics, reply_median)
        row = {"month_id": month, "composite": health["composite"]}
        for key, score in health["sub_scores"].items():
            row[key] = score["value"]
        records.append(row)

    if not records:
        return pl.DataFrame({"month_id": [], "composite": []})

    return pl.DataFrame(records)
