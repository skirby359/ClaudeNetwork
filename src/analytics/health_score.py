"""Organizational Health Score: composite 0-100 metric from communication patterns."""

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
        comm_lookup = dict(zip(
            graph_metrics["email"].to_list(),
            graph_metrics["community_id"].to_list(),
        ))
        # Sample edges for efficiency
        sample = edge_fact.head(min(100000, len(edge_fact)))
        cross = 0
        total = 0
        for row in sample.select(["from_email", "to_email"]).iter_rows():
            f_comm = comm_lookup.get(row[0])
            t_comm = comm_lookup.get(row[1])
            if f_comm is not None and t_comm is not None:
                total += 1
                if f_comm != t_comm:
                    cross += 1
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
