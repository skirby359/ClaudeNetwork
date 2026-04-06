"""Temporal network evolution: monthly snapshots, centrality trends, stability."""

import networkx as nx
import community as community_louvain
import polars as pl
import numpy as np


def build_monthly_snapshots(edge_fact: pl.DataFrame) -> dict[str, pl.DataFrame]:
    """Build per-month network snapshots with node metrics.

    Returns dict mapping month_id (YYYY-MM) to a DataFrame of node metrics.
    """
    # Add month_id column
    ef = edge_fact.with_columns(
        pl.col("timestamp").dt.strftime("%Y-%m").alias("month_id")
    )

    months = sorted(ef["month_id"].unique().to_list())
    snapshots = {}

    for month in months:
        month_edges = ef.filter(pl.col("month_id") == month)

        # Build graph for this month — extract columns as lists for speed
        G = nx.DiGraph()
        edge_weights = (
            month_edges.group_by(["from_email", "to_email"])
            .agg(pl.len().alias("weight"))
        )
        froms = edge_weights["from_email"].to_list()
        tos = edge_weights["to_email"].to_list()
        weights = edge_weights["weight"].to_list()
        for f, t, w in zip(froms, tos, weights):
            G.add_edge(f, t, weight=w)

        if G.number_of_nodes() < 2:
            continue

        # Compute metrics
        pagerank = nx.pagerank(G, weight="weight")
        in_deg = dict(G.in_degree(weight="weight"))
        out_deg = dict(G.out_degree(weight="weight"))

        # Community detection
        G_und = G.to_undirected()
        partition = community_louvain.best_partition(G_und, weight="weight", resolution=0.5)

        records = []
        for node in G.nodes():
            records.append({
                "email": node,
                "month_id": month,
                "pagerank": pagerank.get(node, 0.0),
                "in_degree": in_deg.get(node, 0),
                "out_degree": out_deg.get(node, 0),
                "community_id": partition.get(node, -1),
            })

        snapshots[month] = pl.DataFrame(records)

    return snapshots


def compute_centrality_trends(snapshots: dict[str, pl.DataFrame]) -> pl.DataFrame:
    """Compute per-person centrality (pagerank) across months."""
    if not snapshots:
        return pl.DataFrame({
            "email": [], "month_id": [], "pagerank": [],
            "in_degree": [], "out_degree": [],
        })

    dfs = list(snapshots.values())
    combined = pl.concat(dfs)
    return combined.select(["email", "month_id", "pagerank", "in_degree", "out_degree"])


def detect_rising_fading(centrality_trends: pl.DataFrame, min_months: int = 3) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Detect people with biggest centrality increase/decrease over time.

    Returns (rising_stars, fading_out) DataFrames.
    """
    if len(centrality_trends) == 0:
        empty = pl.DataFrame({"email": [], "first_pagerank": [], "last_pagerank": [], "change": []})
        return empty, empty

    months_sorted = sorted(centrality_trends["month_id"].unique().to_list())
    if len(months_sorted) < 2:
        empty = pl.DataFrame({"email": [], "first_pagerank": [], "last_pagerank": [], "change": []})
        return empty, empty

    # Get first-half and second-half averages per person
    half = len(months_sorted) // 2
    first_months = set(months_sorted[:half])
    second_months = set(months_sorted[half:])

    first_avg = (
        centrality_trends.filter(pl.col("month_id").is_in(list(first_months)))
        .group_by("email")
        .agg(pl.col("pagerank").mean().alias("first_pagerank"))
    )
    second_avg = (
        centrality_trends.filter(pl.col("month_id").is_in(list(second_months)))
        .group_by("email")
        .agg(pl.col("pagerank").mean().alias("last_pagerank"))
    )

    # People present in enough months
    person_months = (
        centrality_trends.group_by("email")
        .agg(pl.col("month_id").n_unique().alias("n_months"))
        .filter(pl.col("n_months") >= min_months)
    )

    merged = first_avg.join(second_avg, on="email", how="inner")
    merged = merged.join(person_months.select("email"), on="email", how="inner")
    merged = merged.with_columns(
        (pl.col("last_pagerank") - pl.col("first_pagerank")).alias("change")
    )

    rising = merged.sort("change", descending=True).head(20)
    fading = merged.sort("change", descending=False).head(20)

    return rising, fading


def compute_community_stability(snapshots: dict[str, pl.DataFrame]) -> pl.DataFrame:
    """Compute community membership stability between consecutive months.

    Uses Normalized Mutual Information (NMI) between consecutive partition assignments.
    """
    months = sorted(snapshots.keys())
    if len(months) < 2:
        return pl.DataFrame({"month_pair": [], "nmi": [], "n_communities": []})

    records = []
    for i in range(len(months) - 1):
        m1, m2 = months[i], months[i + 1]
        df1 = snapshots[m1].select(["email", "community_id"]).rename({"community_id": "comm1"})
        df2 = snapshots[m2].select(["email", "community_id"]).rename({"community_id": "comm2"})

        # Find common nodes
        common = df1.join(df2, on="email", how="inner")
        if len(common) < 10:
            records.append({"month_pair": f"{m1}/{m2}", "nmi": 0.0, "n_communities": 0})
            continue

        labels1 = common["comm1"].to_numpy()
        labels2 = common["comm2"].to_numpy()

        # Compute NMI manually (avoid sklearn dependency)
        nmi = _compute_nmi(labels1, labels2)
        n_comm = len(set(labels2.tolist()))

        records.append({"month_pair": f"{m1}/{m2}", "nmi": float(nmi), "n_communities": n_comm})

    return pl.DataFrame(records)


def _compute_nmi(labels1: np.ndarray, labels2: np.ndarray) -> float:
    """Compute Normalized Mutual Information between two label arrays.

    Uses a contingency matrix for O(n + k1*k2) instead of O(n*k1*k2).
    """
    n = len(labels1)
    if n == 0:
        return 0.0

    classes1 = np.unique(labels1)
    classes2 = np.unique(labels2)

    # Entropy H(X)
    def entropy(labels):
        _, counts = np.unique(labels, return_counts=True)
        probs = counts / n
        return -np.sum(probs * np.log(probs + 1e-12))

    h1 = entropy(labels1)
    h2 = entropy(labels2)

    if h1 == 0 or h2 == 0:
        return 0.0

    # Build contingency matrix in O(n) using index mapping
    c1_map = {c: i for i, c in enumerate(classes1)}
    c2_map = {c: i for i, c in enumerate(classes2)}
    contingency = np.zeros((len(classes1), len(classes2)), dtype=np.int64)
    for l1, l2 in zip(labels1, labels2):
        contingency[c1_map[l1], c2_map[l2]] += 1

    # Row/column marginals
    row_sums = contingency.sum(axis=1)
    col_sums = contingency.sum(axis=0)

    # Mutual information from contingency matrix
    mi = 0.0
    for i in range(len(classes1)):
        for j in range(len(classes2)):
            n_ij = contingency[i, j]
            if n_ij == 0:
                continue
            mi += (n_ij / n) * np.log((n * n_ij) / (row_sums[i] * col_sums[j]) + 1e-12)

    return mi / np.sqrt(h1 * h2)
