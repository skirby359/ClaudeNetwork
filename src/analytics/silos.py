"""Communication silos and bridges: inter-community analysis."""

import networkx as nx
import polars as pl


def compute_community_interaction_matrix(
    edge_fact: pl.DataFrame,
    community_lookup: dict[str, int],
) -> pl.DataFrame:
    """Build a community-to-community message count matrix.

    Args:
        edge_fact: Edge fact table.
        community_lookup: dict mapping email -> community_id.
    """
    # Use Polars join instead of Python loop for community mapping
    lookup_df = pl.DataFrame({
        "email": list(community_lookup.keys()),
        "community_id": list(community_lookup.values()),
    })

    edges_with_comm = (
        edge_fact
        .join(
            lookup_df.rename({"email": "from_email", "community_id": "comm_from"}),
            on="from_email", how="left",
        )
        .join(
            lookup_df.rename({"email": "to_email", "community_id": "comm_to"}),
            on="to_email", how="left",
        )
    )

    # Filter out unknown communities
    edges_with_comm = edges_with_comm.filter(
        pl.col("comm_from").is_not_null() & pl.col("comm_to").is_not_null()
    )

    matrix = (
        edges_with_comm.group_by(["comm_from", "comm_to"])
        .agg(pl.len().alias("msg_count"))
        .sort(["comm_from", "comm_to"])
    )

    return matrix


def find_silent_community_pairs(
    interaction_matrix: pl.DataFrame,
    valid_communities: list[int] | set[int],
) -> list[tuple[int, int]]:
    """Find pairs of valid communities with zero communication.

    Args:
        interaction_matrix: Output of compute_community_interaction_matrix.
        valid_communities: Actual community IDs to consider (not a count).
    """
    # Build set of communicating pairs (normalized to min, max order)
    active_pairs = set()
    for row in interaction_matrix.iter_rows(named=True):
        a, b = row["comm_from"], row["comm_to"]
        active_pairs.add((min(a, b), max(a, b)))

    # Iterate only over actual community IDs, not 0..max_id
    comms = sorted(valid_communities)
    silent = []
    for i, a in enumerate(comms):
        for b in comms[i + 1:]:
            if (a, b) not in active_pairs:
                silent.append((a, b))
    return silent


def identify_bridges(
    G: nx.DiGraph,
    community_lookup: dict[str, int],
) -> pl.DataFrame:
    """Identify bridge people who connect 2+ communities."""
    bridge_records = []

    for node in G.nodes():
        if node not in community_lookup:
            continue
        home_comm = community_lookup[node]

        # Communities of neighbors
        neighbor_comms = set()
        for neighbor in set(G.predecessors(node)) | set(G.successors(node)):
            if neighbor in community_lookup:
                nc = community_lookup[neighbor]
                if nc != home_comm:
                    neighbor_comms.add(nc)

        if len(neighbor_comms) > 0:
            bridge_records.append({
                "email": node,
                "home_community": home_comm,
                "communities_bridged": len(neighbor_comms) + 1,  # including home
                "external_communities": sorted(list(neighbor_comms)),
            })

    if not bridge_records:
        return pl.DataFrame({
            "email": [], "home_community": [],
            "communities_bridged": [], "external_communities": [],
        })

    df = pl.DataFrame(bridge_records)
    return df.sort("communities_bridged", descending=True)


def simulate_removal(G: nx.DiGraph, person: str) -> dict:
    """Simulate removing a person from the network and measure impact."""
    G_undirected = G.to_undirected()

    # Before removal
    before_components = nx.number_connected_components(G_undirected)
    before_largest = len(max(nx.connected_components(G_undirected), key=len))

    # After removal
    G_removed = G_undirected.copy()
    if person in G_removed:
        G_removed.remove_node(person)

    if G_removed.number_of_nodes() == 0:
        after_components = 0
        after_largest = 0
    else:
        after_components = nx.number_connected_components(G_removed)
        after_largest = len(max(nx.connected_components(G_removed), key=len))

    return {
        "person": person,
        "before_components": before_components,
        "after_components": after_components,
        "component_increase": after_components - before_components,
        "before_largest": before_largest,
        "after_largest": after_largest,
        "largest_decrease": before_largest - after_largest,
    }
