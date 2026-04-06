"""Auto-generated executive narrative from email analytics data."""

import polars as pl


def generate_executive_narrative(
    mf: pl.DataFrame,
    weekly_agg: pl.DataFrame,
    ef: pl.DataFrame,
    person_dim: pl.DataFrame,
) -> str:
    """Generate a markdown narrative summarizing the key patterns in the data."""
    parts = []
    total_msgs = len(mf)
    if total_msgs == 0:
        return "No messages in the selected date range."

    # Volume trend: first half vs second half
    n_weeks = len(weekly_agg)
    if n_weeks >= 4:
        half = n_weeks // 2
        sorted_wa = weekly_agg.sort("week_start")
        first_half = sorted_wa.head(half)
        second_half = sorted_wa.tail(n_weeks - half)
        avg_first = first_half["msg_count"].mean()
        avg_second = second_half["msg_count"].mean()
        if avg_first > 0:
            change_pct = (avg_second - avg_first) / avg_first * 100
            direction = "increased" if change_pct > 0 else "decreased"
            parts.append(
                f"**Volume trend:** Weekly message volume {direction} by "
                f"{abs(change_pct):.0f}% from the first half to the second half "
                f"of the selected period (avg {avg_first:.0f} → {avg_second:.0f} msgs/week)."
            )

    # After-hours rate
    ah_rate = float(mf["is_after_hours"].mean())
    we_rate = float(mf["is_weekend"].mean())
    parts.append(
        f"**Work patterns:** {ah_rate:.1%} of messages were sent outside business hours, "
        f"and {we_rate:.1%} on weekends."
    )

    # Network size
    n_people = len(person_dim)
    internal_count = len(person_dim.filter(person_dim["is_internal"])) if "is_internal" in person_dim.columns else 0
    external_count = n_people - internal_count
    parts.append(
        f"**Network size:** {n_people:,} unique people identified — "
        f"{internal_count:,} internal, {external_count:,} external."
    )

    # Concentration
    sender_counts = (
        ef.group_by("from_email")
        .agg(pl.len().alias("count"))
        .sort("count", descending=True)
    )
    total_edges = len(ef)
    if len(sender_counts) > 0 and total_edges > 0:
        top10_volume = int(sender_counts.head(10)["count"].sum())
        top10_share = top10_volume / total_edges
        parts.append(
            f"**Concentration:** The top 10 senders account for {top10_share:.1%} "
            f"of all message activity."
        )

    # Busiest week
    if n_weeks > 0:
        sorted_wa = weekly_agg.sort("msg_count", descending=True)
        busiest = sorted_wa.row(0, named=True)
        parts.append(
            f"**Peak week:** {busiest['week_start'].strftime('%b %d, %Y')} with "
            f"{busiest['msg_count']:,} messages."
        )

    return "\n\n".join(parts)
