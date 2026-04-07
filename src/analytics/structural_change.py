"""Structural Change Detection: community shifts, node switches, NMI alerts."""

import polars as pl
import numpy as np
from collections import Counter


def classify_community_shifts(
    snapshots: dict[str, pl.DataFrame],
    nmi_threshold: float = 0.7,
) -> pl.DataFrame:
    """Classify month-to-month community changes as stable/split/merge/reorg.

    Detection logic:
    - Stable: NMI > threshold
    - Split: one community in month N maps to 2+ in month N+1
    - Merge: 2+ communities in month N map to one in month N+1
    - Reorg: NMI < threshold but neither pure split nor merge
    """
    from src.analytics.temporal_network import _compute_nmi

    months = sorted(snapshots.keys())
    if len(months) < 2:
        return pl.DataFrame({
            "month_pair": [], "shift_type": [], "nmi": [],
            "detail": [],
        })

    records = []
    for i in range(len(months) - 1):
        m1, m2 = months[i], months[i + 1]
        df1 = snapshots[m1].select(["email", "community_id"]).rename({"community_id": "comm1"})
        df2 = snapshots[m2].select(["email", "community_id"]).rename({"community_id": "comm2"})

        common = df1.join(df2, on="email", how="inner")
        if len(common) < 10:
            records.append({
                "month_pair": f"{m1}/{m2}", "shift_type": "insufficient_data",
                "nmi": 0.0, "detail": f"Only {len(common)} common nodes",
            })
            continue

        labels1 = common["comm1"].to_numpy()
        labels2 = common["comm2"].to_numpy()
        nmi = float(_compute_nmi(labels1, labels2))

        if nmi > nmi_threshold:
            records.append({
                "month_pair": f"{m1}/{m2}", "shift_type": "stable",
                "nmi": nmi, "detail": "Communities largely unchanged",
            })
            continue

        # Analyze mapping: which old communities map to which new ones
        shift_type, detail = _classify_mapping(common)
        records.append({
            "month_pair": f"{m1}/{m2}", "shift_type": shift_type,
            "nmi": nmi, "detail": detail,
        })

    return pl.DataFrame(records)


def _classify_mapping(common: pl.DataFrame) -> tuple[str, str]:
    """Determine if the dominant pattern is split, merge, or reorg."""
    # For each old community, find distribution across new communities
    old_to_new: dict[int, Counter] = {}
    new_to_old: dict[int, Counter] = {}

    for row in common.iter_rows(named=True):
        c1, c2 = row["comm1"], row["comm2"]
        old_to_new.setdefault(c1, Counter())[c2] += 1
        new_to_old.setdefault(c2, Counter())[c1] += 1

    splits = 0
    merges = 0

    # Detect splits: old community maps to 2+ new communities with >30% each
    for old_c, new_dist in old_to_new.items():
        total = sum(new_dist.values())
        significant = [c for c, n in new_dist.items() if n / total > 0.3]
        if len(significant) >= 2:
            splits += 1

    # Detect merges: new community draws from 2+ old communities with >30% each
    for new_c, old_dist in new_to_old.items():
        total = sum(old_dist.values())
        significant = [c for c, n in old_dist.items() if n / total > 0.3]
        if len(significant) >= 2:
            merges += 1

    if splits > merges and splits > 0:
        return "split", f"{splits} community split(s) detected"
    elif merges > splits and merges > 0:
        return "merge", f"{merges} community merge(s) detected"
    elif splits > 0 or merges > 0:
        return "reorg", f"{splits} split(s), {merges} merge(s)"
    else:
        return "reorg", "General reorganization (no dominant split/merge pattern)"


def track_node_switches(snapshots: dict[str, pl.DataFrame]) -> pl.DataFrame:
    """Track per-person community changes across months.

    Returns: email, month_id, community_id, prev_community_id, switched (bool).
    """
    months = sorted(snapshots.keys())
    if len(months) < 2:
        return pl.DataFrame({
            "email": [], "month_id": [], "community_id": [],
            "prev_community_id": [], "switched": [],
        })

    records = []
    prev_lookup: dict[str, int] = {}

    for month in months:
        df = snapshots[month]
        curr_lookup = dict(zip(
            df["email"].to_list(), df["community_id"].to_list(),
        ))

        for email, comm in curr_lookup.items():
            prev_comm = prev_lookup.get(email)
            records.append({
                "email": email,
                "month_id": month,
                "community_id": comm,
                "prev_community_id": prev_comm,
                "switched": prev_comm is not None and prev_comm != comm,
            })

        prev_lookup = curr_lookup

    return pl.DataFrame(records)


def compute_switch_rates(node_switches: pl.DataFrame) -> pl.DataFrame:
    """Per-person switch frequency."""
    if len(node_switches) == 0:
        return pl.DataFrame({
            "email": [], "n_switches": [], "n_months_active": [], "switch_rate": [],
        })

    return (
        node_switches.group_by("email")
        .agg([
            pl.col("switched").sum().alias("n_switches"),
            pl.len().alias("n_months_active"),
        ])
        .with_columns(
            (pl.col("n_switches").cast(pl.Float64) / pl.col("n_months_active")).alias("switch_rate")
        )
        .sort("n_switches", descending=True)
    )


def nmi_drop_alerts(
    stability: pl.DataFrame,
    warning_threshold: float = 0.5,
    critical_threshold: float = 0.3,
) -> pl.DataFrame:
    """Flag month pairs where NMI drops below thresholds."""
    if len(stability) == 0 or "nmi" not in stability.columns:
        return pl.DataFrame({"month_pair": [], "nmi": [], "alert_level": []})

    alerts = stability.filter(pl.col("nmi") < warning_threshold)
    alerts = alerts.with_columns(
        pl.when(pl.col("nmi") < critical_threshold)
        .then(pl.lit("critical"))
        .otherwise(pl.lit("warning"))
        .alias("alert_level")
    )
    return alerts.select(["month_pair", "nmi", "alert_level"])


def build_community_flow(snapshots: dict[str, pl.DataFrame]) -> pl.DataFrame:
    """Build Sankey flow data: community membership transitions between months."""
    months = sorted(snapshots.keys())
    if len(months) < 2:
        return pl.DataFrame({
            "source": [], "target": [], "value": [],
        })

    records = []
    for i in range(len(months) - 1):
        m1, m2 = months[i], months[i + 1]
        df1 = snapshots[m1].select(["email", "community_id"]).rename({"community_id": "comm1"})
        df2 = snapshots[m2].select(["email", "community_id"]).rename({"community_id": "comm2"})

        common = df1.join(df2, on="email", how="inner")
        flows = (
            common.group_by(["comm1", "comm2"])
            .agg(pl.len().alias("count"))
        )

        for row in flows.iter_rows(named=True):
            records.append({
                "source": f"{m1}_C{row['comm1']}",
                "target": f"{m2}_C{row['comm2']}",
                "value": row["count"],
            })

    return pl.DataFrame(records)
