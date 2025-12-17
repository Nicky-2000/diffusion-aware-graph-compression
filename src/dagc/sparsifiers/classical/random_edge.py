# src/dagc/compression.py

from typing import Dict, Any, Optional
import random

import networkx as nx


class CompressionResult:
    """
    Container for compressed graph information.

    Attributes:
        graph: The compressed graph.
        node_mapping: Optional mapping from original node -> compressed node.
                      For pure edge sparsification, this is the identity.
    """
    def __init__(self, graph: nx.Graph, node_mapping: Optional[Dict[Any, Any]] = None):
        self.graph = graph
        self.node_mapping = node_mapping if node_mapping is not None else {
            n: n for n in graph.nodes()
        }


def random_edge_sparsify(
    graph: nx.Graph,
    keep_ratio: float,
    rng_seed: int | None = None,
) -> CompressionResult:
    """
    Randomly keep a fraction of edges while preserving connectivity.

    Strategy:
        - Work on a copy of the original graph.
        - Compute the target number of edges:
              target_m = round(keep_ratio * |E|)
        - Enforce a lower bound: target_m >= |V| - 1
          (any connected simple graph on |V| nodes needs at least |V|-1 edges).
        - Shuffle the edge list.
        - For each edge in that random order:
            * tentatively remove it
            * if the graph is still connected, keep it removed
            * otherwise, add it back
        - Stop once we reach target_m edges or run out of removable edges.

    Args:
        graph:
            Original undirected, connected graph.
        keep_ratio:
            Fraction of edges to keep, in (0, 1].
        rng_seed:
            Optional random seed for reproducibility.

    Returns:
        CompressionResult with a graph that has the same nodes but fewer edges,
        and remains connected.
    """
    if keep_ratio <= 0.0 or keep_ratio > 1.0:
        raise ValueError(f"keep_ratio must be in (0, 1], got {keep_ratio}.")

    # Work on a copy so we don't mutate the original
    H = graph.copy()

    n = H.number_of_nodes()
    m = H.number_of_edges()

    # Trivial cases
    if n == 0 or m == 0:
        return CompressionResult(H)

    # Desired number of edges
    target_m = int(round(keep_ratio * m))

    # Never go below a spanning tree size
    min_edges = n - 1
    if target_m < min_edges:
        target_m = min_edges

    # If target_m >= m, nothing to do
    if target_m >= m:
        return CompressionResult(H)

    # RNG for reproducibility
    rng = random.Random(rng_seed)

    # Randomize edge removal order
    edges = list(H.edges())
    rng.shuffle(edges)

    for u, v in edges:
        if H.number_of_edges() <= target_m:
            break

        # Tentatively remove edge
        H.remove_edge(u, v)

        # If removal disconnects the graph, undo it
        if not nx.is_connected(H):
            print(f"Restoring edge ({u}, {v}) to preserve connectivity.")
            H.add_edge(u, v)

    return CompressionResult(H)
