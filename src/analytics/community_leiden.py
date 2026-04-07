"""Community Detection v2: Leiden algorithm with multi-resolution and type classification."""

import networkx as nx
import polars as pl
from collections import Counter

try:
    import igraph as ig
    import leidenalg
    HAS_LEIDEN = True
except ImportError:
    HAS_LEIDEN = False


def _nx_to_igraph(G: nx.DiGraph) -> "ig.Graph":
    """Convert NetworkX DiGraph to igraph, preserving weight edge attribute."""
    G_und = G.to_undirected()
    nodes = list(G_und.nodes())
    node_idx = {n: i for i, n in enumerate(nodes)}
    edges = [(node_idx[u], node_idx[v]) for u, v in G_und.edges()]
    weights = [G_und[u][v].get("weight", 1) for u, v in G_und.edges()]

    g = ig.Graph(n=len(nodes), edges=edges, directed=False)
    g.vs["name"] = nodes
    g.es["weight"] = weights
    return g


def detect_leiden_communities(
    G: nx.DiGraph,
    exclude_emails: set[str] | None = None,
    resolutions: dict[str, float] | None = None,
    min_community_size: int = 3,
) -> pl.DataFrame:
    """Run Leiden at multiple resolutions.

    Returns DataFrame with columns: email, community_coarse, community_medium,
    community_fine, community_label_coarse, community_label_medium,
    community_label_fine, community_type.

    Falls back to Louvain if leidenalg is not installed.
    """
    if resolutions is None:
        resolutions = {"coarse": 0.3, "medium": 0.7, "fine": 1.5}

    # Remove nonhuman nodes
    if exclude_emails:
        G = G.copy()
        G.remove_nodes_from(exclude_emails & set(G.nodes()))

    if G.number_of_nodes() == 0:
        return pl.DataFrame({
            "email": [], "community_coarse": [], "community_medium": [],
            "community_fine": [], "community_label_coarse": [],
            "community_label_medium": [], "community_label_fine": [],
            "community_type": [],
        })

    pagerank = nx.pagerank(G, weight="weight")

    if HAS_LEIDEN:
        partitions = _run_leiden(G, resolutions)
    else:
        partitions = _run_louvain_fallback(G, resolutions)

    # Merge small communities at each resolution
    for res_name in partitions:
        partitions[res_name] = _merge_small(partitions[res_name], G, min_community_size)

    # Label each resolution
    labels = {}
    for res_name, partition in partitions.items():
        labels[res_name] = _label_communities(partition, pagerank)

    # Classify community types at medium resolution
    type_map = _classify_communities(partitions["medium"])

    # Build result
    records = []
    for node in G.nodes():
        records.append({
            "email": node,
            "community_coarse": partitions["coarse"].get(node, -1),
            "community_medium": partitions["medium"].get(node, -1),
            "community_fine": partitions["fine"].get(node, -1),
            "community_label_coarse": labels["coarse"].get(
                partitions["coarse"].get(node, -1), "Unknown"
            ),
            "community_label_medium": labels["medium"].get(
                partitions["medium"].get(node, -1), "Unknown"
            ),
            "community_label_fine": labels["fine"].get(
                partitions["fine"].get(node, -1), "Unknown"
            ),
            "community_type": type_map.get(partitions["medium"].get(node, -1), "unknown"),
        })

    return pl.DataFrame(records)


def _run_leiden(G: nx.DiGraph, resolutions: dict[str, float]) -> dict[str, dict[str, int]]:
    """Run Leiden at each resolution level."""
    g = _nx_to_igraph(G)
    nodes = g.vs["name"]
    partitions = {}

    for res_name, res_val in resolutions.items():
        part = leidenalg.find_partition(
            g,
            leidenalg.RBConfigurationVertexPartition,
            weights="weight",
            resolution_parameter=res_val,
        )
        partitions[res_name] = {nodes[i]: part.membership[i] for i in range(len(nodes))}

    return partitions


def _run_louvain_fallback(G: nx.DiGraph, resolutions: dict[str, float]) -> dict[str, dict[str, int]]:
    """Fallback to Louvain when Leiden not available."""
    import community as community_louvain
    G_und = G.to_undirected()
    partitions = {}
    for res_name, res_val in resolutions.items():
        partitions[res_name] = community_louvain.best_partition(
            G_und, weight="weight", resolution=res_val,
        )
    return partitions


def _merge_small(
    partition: dict[str, int], G: nx.DiGraph, min_size: int,
) -> dict[str, int]:
    """Merge communities smaller than min_size into nearest large neighbor."""
    comm_members: dict[int, list[str]] = {}
    for node, cid in partition.items():
        comm_members.setdefault(cid, []).append(node)

    small = {c for c, m in comm_members.items() if len(m) < min_size}
    if not small:
        return partition

    large = set(comm_members.keys()) - small
    merged = dict(partition)

    for sid in small:
        neighbor_comms: dict[int, int] = {}
        for node in comm_members[sid]:
            for nb in set(G.predecessors(node)) | set(G.successors(node)):
                nc = partition.get(nb)
                if nc is not None and nc not in small:
                    neighbor_comms[nc] = neighbor_comms.get(nc, 0) + 1
        if neighbor_comms:
            target = max(neighbor_comms, key=neighbor_comms.get)
            for node in comm_members[sid]:
                merged[node] = target

    return merged


def _label_communities(
    partition: dict[str, int], pagerank: dict[str, float],
) -> dict[int, str]:
    """Label communities by most central person + dominant domain."""
    comm_members: dict[int, list[str]] = {}
    for node, cid in partition.items():
        comm_members.setdefault(cid, []).append(node)

    labels = {}
    for cid, members in comm_members.items():
        central = max(members, key=lambda n: pagerank.get(n, 0))
        central_short = central.split("@")[0] if "@" in central else central
        domains = [m.split("@")[1] if "@" in m else "unknown" for m in members]
        top_domain = Counter(domains).most_common(1)[0][0] if domains else "unknown"
        labels[cid] = f"{central_short} ({top_domain}, {len(members)})"

    return labels


def _classify_communities(partition: dict[str, int]) -> dict[int, str]:
    """Classify communities by size and domain composition."""
    comm_members: dict[int, list[str]] = {}
    for node, cid in partition.items():
        comm_members.setdefault(cid, []).append(node)

    type_map = {}
    for cid, members in comm_members.items():
        size = len(members)
        domains = [m.split("@")[1] if "@" in m else "unknown" for m in members]
        domain_counts = Counter(domains)
        top_domain_pct = domain_counts.most_common(1)[0][1] / size if size > 0 else 0

        if size <= 3:
            type_map[cid] = "pair"
        elif size <= 15:
            type_map[cid] = "team"
        elif top_domain_pct < 0.5:
            type_map[cid] = "cross-functional"
        else:
            type_map[cid] = "department"

    return type_map


def build_hierarchy_nesting(leiden_df: pl.DataFrame) -> pl.DataFrame:
    """Map fine -> medium -> coarse nesting for treemap visualization."""
    nesting = (
        leiden_df.group_by(["community_fine", "community_medium", "community_coarse"])
        .agg(pl.len().alias("n_members"))
        .sort(["community_coarse", "community_medium", "community_fine"])
    )
    return nesting


def compare_louvain_leiden(
    G: nx.DiGraph, exclude_emails: set[str] | None = None,
) -> dict:
    """Compare Louvain vs Leiden community counts and modularity."""
    import community as community_louvain

    if exclude_emails:
        G = G.copy()
        G.remove_nodes_from(exclude_emails & set(G.nodes()))

    G_und = G.to_undirected()

    # Louvain
    louvain_part = community_louvain.best_partition(G_und, weight="weight", resolution=0.7)
    louvain_comms = len(set(louvain_part.values()))
    louvain_mod = community_louvain.modularity(louvain_part, G_und, weight="weight")

    result = {
        "louvain_communities": louvain_comms,
        "louvain_modularity": round(louvain_mod, 4),
    }

    if HAS_LEIDEN:
        g = _nx_to_igraph(G)
        part = leidenalg.find_partition(
            g, leidenalg.RBConfigurationVertexPartition,
            weights="weight", resolution_parameter=0.7,
        )
        result["leiden_communities"] = len(set(part.membership))
        result["leiden_modularity"] = round(part.modularity, 4)
    else:
        result["leiden_communities"] = "N/A (not installed)"
        result["leiden_modularity"] = "N/A"

    return result
