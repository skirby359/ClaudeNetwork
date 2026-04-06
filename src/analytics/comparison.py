"""Comparison analytics: period-over-period KPI deltas."""

import polars as pl


def compute_period_summary(mf: pl.DataFrame, ef: pl.DataFrame) -> dict:
    """Compute summary KPIs for a given period's message_fact and edge_fact."""
    total_msgs = len(mf)
    total_edges = len(ef)
    unique_senders = mf["from_email"].n_unique() if total_msgs > 0 else 0
    unique_recipients = ef["to_email"].n_unique() if total_edges > 0 else 0
    total_bytes = int(mf["size_bytes"].sum()) if total_msgs > 0 else 0
    avg_recipients = float(mf["n_recipients"].mean()) if total_msgs > 0 else 0.0
    after_hours_rate = float(mf["is_after_hours"].mean()) if total_msgs > 0 else 0.0
    weekend_rate = float(mf["is_weekend"].mean()) if total_msgs > 0 else 0.0

    return {
        "total_messages": total_msgs,
        "total_edges": total_edges,
        "unique_senders": unique_senders,
        "unique_recipients": unique_recipients,
        "total_bytes": total_bytes,
        "avg_recipients": avg_recipients,
        "after_hours_rate": after_hours_rate,
        "weekend_rate": weekend_rate,
    }


def compute_delta(current: dict, previous: dict) -> dict:
    """Compute absolute and percentage deltas between two period summaries."""
    result = {}
    for key in current:
        cur = current[key]
        prev = previous.get(key, 0)
        delta = cur - prev
        if isinstance(cur, float):
            pct = ((cur - prev) / prev * 100) if prev != 0 else 0.0
        else:
            pct = ((cur - prev) / prev * 100) if prev != 0 else 0.0
        result[key] = {"current": cur, "previous": prev, "delta": delta, "pct": pct}
    return result
