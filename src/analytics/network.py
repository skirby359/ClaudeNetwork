"""Network analytics: graph build, centrality, communities, dyads."""

import networkx as nx
import community as community_louvain
import polars as pl

from src.config import AppConfig
from src.cache_manager import cached_parquet, cached_pickle


# ---------------------------------------------------------------------------
# Pure computation functions (no file caching)
# ---------------------------------------------------------------------------

def build_graph(edge_fact: pl.DataFrame) -> nx.DiGraph:
    """Build a directed weighted graph from edge_fact."""
    G = nx.DiGraph()
    edge_weights = (
        edge_fact.group_by(["from_email", "to_email"])
        .agg([
            pl.len().alias("weight"),
            pl.col("size_bytes").sum().alias("total_bytes"),
        ])
    )
    froms = edge_weights["from_email"].to_list()
    tos = edge_weights["to_email"].to_list()
    weights = edge_weights["weight"].to_list()
    total_bytes = edge_weights["total_bytes"].to_list()
    for f, t, w, tb in zip(froms, tos, weights, total_bytes):
        G.add_edge(f, t, weight=w, total_bytes=tb)
    return G


def compute_node_metrics(
    G: nx.DiGraph,
    exclude_emails: set[str] | None = None,
    resolution: float = 0.5,
    min_community_size: int = 3,
) -> pl.DataFrame:
    """Compute per-node graph metrics from a NetworkX graph.

    Args:
        G: Directed weighted graph.
        exclude_emails: Emails to remove before community detection (e.g., nonhuman).
        resolution: Louvain resolution parameter. Lower = fewer, larger communities.
            Default 0.5 (less fragmentation than Louvain's default of 1.0).
        min_community_size: Merge communities smaller than this into nearest neighbor.
    """
    # Optionally remove nonhuman nodes for cleaner community detection
    if exclude_emails:
        G = G.copy()
        G.remove_nodes_from(exclude_emails & set(G.nodes()))

    if G.number_of_nodes() == 0:
        return pl.DataFrame({
            "email": [], "in_degree": [], "out_degree": [],
            "betweenness_centrality": [], "community_id": [],
            "pagerank": [], "community_label": [],
        })

    in_degree = dict(G.in_degree(weight="weight"))
    out_degree = dict(G.out_degree(weight="weight"))
    pagerank = nx.pagerank(G, weight="weight")

    if G.number_of_nodes() > 5000:
        betweenness = nx.betweenness_centrality(G, weight="weight", k=500)
    else:
        betweenness = nx.betweenness_centrality(G, weight="weight")

    G_undirected = G.to_undirected()
    partition = community_louvain.best_partition(
        G_undirected, weight="weight", resolution=resolution,
    )

    # Merge tiny communities into nearest neighbor
    partition = _merge_small_communities(partition, G, min_size=min_community_size)

    # Auto-label communities by most central person + dominant domain
    labels = _label_communities(partition, pagerank)

    records = []
    for node in G.nodes():
        comm_id = partition.get(node, -1)
        records.append({
            "email": node,
            "in_degree": in_degree.get(node, 0),
            "out_degree": out_degree.get(node, 0),
            "betweenness_centrality": betweenness.get(node, 0.0),
            "community_id": comm_id,
            "pagerank": pagerank.get(node, 0.0),
            "community_label": labels.get(comm_id, f"Group {comm_id}"),
        })
    return pl.DataFrame(records)


def _merge_small_communities(
    partition: dict[str, int],
    G: nx.DiGraph,
    min_size: int = 3,
) -> dict[str, int]:
    """Merge communities with fewer than min_size members into nearest neighbor."""
    # Count members per community
    comm_members: dict[int, list[str]] = {}
    for node, comm_id in partition.items():
        comm_members.setdefault(comm_id, []).append(node)

    small_comms = {c: nodes for c, nodes in comm_members.items() if len(nodes) < min_size}
    if not small_comms:
        return partition

    large_comm_ids = set(comm_members.keys()) - set(small_comms.keys())
    merged = dict(partition)

    for small_id, small_nodes in small_comms.items():
        # Find which large community they connect to most
        neighbor_comms: dict[int, int] = {}
        for node in small_nodes:
            for neighbor in set(G.predecessors(node)) | set(G.successors(node)):
                nc = partition.get(neighbor)
                if nc is not None and nc != small_id and nc in large_comm_ids:
                    neighbor_comms[nc] = neighbor_comms.get(nc, 0) + 1

        if neighbor_comms:
            target = max(neighbor_comms, key=neighbor_comms.get)
            for node in small_nodes:
                merged[node] = target

    return merged


def _label_communities(
    partition: dict[str, int],
    pagerank: dict[str, float],
) -> dict[int, str]:
    """Auto-label communities by most central person + dominant domain."""
    from collections import Counter

    comm_members: dict[int, list[str]] = {}
    for node, comm_id in partition.items():
        comm_members.setdefault(comm_id, []).append(node)

    labels = {}
    for comm_id, members in comm_members.items():
        # Find most central person
        central = max(members, key=lambda n: pagerank.get(n, 0))
        central_short = central.split("@")[0] if "@" in central else central

        # Find dominant domain
        domains = [m.split("@")[1] if "@" in m else "unknown" for m in members]
        top_domain = Counter(domains).most_common(1)[0][0] if domains else "unknown"

        labels[comm_id] = f"{central_short} ({top_domain}, {len(members)})"

    return labels


def compute_dyads(edge_fact: pl.DataFrame) -> pl.DataFrame:
    """Compute dyad analysis from edge_fact."""
    pairs = (
        edge_fact.group_by(["from_email", "to_email"])
        .agg([
            pl.len().alias("msg_count"),
            pl.col("size_bytes").sum().alias("total_bytes"),
        ])
    )

    pairs = pairs.with_columns([
        pl.min_horizontal("from_email", "to_email").alias("person_a"),
        pl.max_horizontal("from_email", "to_email").alias("person_b"),
    ])

    a_to_b = (
        pairs.filter(pl.col("from_email") == pl.col("person_a"))
        .select([
            pl.col("person_a"), pl.col("person_b"),
            pl.col("msg_count").alias("a_to_b_count"),
            pl.col("total_bytes").alias("a_to_b_bytes"),
        ])
    )

    b_to_a = (
        pairs.filter(pl.col("from_email") == pl.col("person_b"))
        .select([
            pl.col("person_a"), pl.col("person_b"),
            pl.col("msg_count").alias("b_to_a_count"),
            pl.col("total_bytes").alias("b_to_a_bytes"),
        ])
    )

    dyads = a_to_b.join(b_to_a, on=["person_a", "person_b"], how="full", coalesce=True)
    dyads = dyads.with_columns([
        pl.col("a_to_b_count").fill_null(0).cast(pl.Int64),
        pl.col("b_to_a_count").fill_null(0).cast(pl.Int64),
        pl.col("a_to_b_bytes").fill_null(0).cast(pl.Int64),
        pl.col("b_to_a_bytes").fill_null(0).cast(pl.Int64),
    ])

    dyads = dyads.filter(pl.col("person_a") != pl.col("person_b"))

    dyads = dyads.with_columns([
        (pl.col("a_to_b_count") + pl.col("b_to_a_count")).alias("total_pair_msgs"),
        (pl.col("a_to_b_bytes") + pl.col("b_to_a_bytes")).alias("total_pair_bytes"),
        (
            (pl.col("a_to_b_count") - pl.col("b_to_a_count")).abs().cast(pl.Float64)
            / (pl.col("a_to_b_count") + pl.col("b_to_a_count")).cast(pl.Float64)
        ).alias("asymmetry_ratio"),
    ])

    dyads = dyads.rename({"person_a": "from_email", "person_b": "to_email"})
    return dyads.sort("total_pair_msgs", descending=True)


# ---------------------------------------------------------------------------
# File-cached wrappers (used for full-dataset precomputation)
# ---------------------------------------------------------------------------

def build_network_graph(edge_fact: pl.DataFrame, config: AppConfig) -> nx.DiGraph:
    """Build graph with file caching."""
    cache_path = config.cache_path(config.network_graph_file)
    source_paths = [config.cache_path(config.edge_fact_file)]
    return cached_pickle(cache_path, source_paths, lambda: build_graph(edge_fact))


def compute_graph_metrics(G: nx.DiGraph, config: AppConfig) -> pl.DataFrame:
    """Compute graph metrics with file caching."""
    cache_path = config.cache_path(config.graph_metrics_file)
    source_paths = [config.cache_path(config.network_graph_file)]
    return cached_parquet(cache_path, source_paths, lambda: compute_node_metrics(G))


def compute_dyad_analysis(edge_fact: pl.DataFrame, config: AppConfig) -> pl.DataFrame:
    """Compute dyad analysis with file caching."""
    cache_path = config.cache_path(config.dyad_analysis_file)
    source_paths = [config.cache_path(config.edge_fact_file)]
    return cached_parquet(cache_path, source_paths, lambda: compute_dyads(edge_fact))
