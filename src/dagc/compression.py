# compression.py

from random import random
from typing import Callable, Dict, Any
import networkx as nx
import numpy as np

class CompressionResult:
    """
    Container for compressed graph information.

    Attributes:
        graph: The compressed graph.
        node_mapping: Optional mapping from original node -> compressed node.
                      For pure edge sparsification, this can just be identity.
    """
    def __init__(self, graph: nx.Graph, node_mapping: Dict[Any, Any] | None = None):
        self.graph = graph
        self.node_mapping = node_mapping if node_mapping is not None else {
            n: n for n in graph.nodes()
        }

def effective_resistance_matrix(G: nx.Graph):
    L = nx.laplacian_matrix(G).toarray()
    L_plus = np.linalg.pinv(L)  # (n × n)
    return L_plus


def graph_sparsify_effective_resistance(
    graph: nx.Graph,
    keep_ratio: float,
    rng_seed: int | None = None,
) -> CompressionResult:
    if keep_ratio <= 0.0 or keep_ratio > 1.0:
        raise ValueError(f"keep_ratio must be in (0, 1], got {keep_ratio}.")

    # Work on a copy
    G = graph.copy()
    n = G.number_of_nodes()
    m = G.number_of_edges()

    if n == 0 or m == 0:
        return CompressionResult(G)

    # Compute q (number of sampled edges)
    q = int(round(keep_ratio * m))
    if q < n - 1:
        q = n - 1  # spanning tree minimum

    # Step 1: compute L⁺
    L_plus = effective_resistance_matrix(G)
    weights = nx.get_edge_attributes(G, 'weight')
    if not weights:  
        weights = {e: 1.0 for e in G.edges()}

    # Step 2: compute effective resistance for each edge
    Re = {}
    for (u, v) in G.edges():
        chi = np.zeros(n)
        chi[u] = 1
        chi[v] = -1
        Re[(u, v)] = chi @ L_plus @ chi

    # Step 3: sampling probabilities p_e ∝ w_e · R_e
    total = sum(weights[e] * Re[e] for e in G.edges())
    pe = {e: (weights[e] * Re[e]) / total for e in G.edges()}

    # Step 4: sampling with replacement
    H = nx.Graph()
    H.add_nodes_from(G.nodes())
    rng = random.Random(rng_seed)
    edges = list(G.edges())

    for _ in range(q):
        e = rng.choices(edges, weights=[pe[e] for e in edges], k=1)[0]
        u, v = e
        new_w = weights[e] / (q * pe[e])  # Paper's rescaled weight

        if H.has_edge(u, v):
            H[u][v]['weight'] += new_w
        else:
            H.add_edge(u, v, weight=new_w)

    return CompressionResult(H)

def random_edge_sparsify(
    graph: nx.Graph,
    keep_ratio: float,
    rng_seed: int | None = None,
) -> CompressionResult:
    """
    Randomly keep a fraction of edges.

    Args:
        graph: Original graph.
        keep_ratio: Fraction of edges to keep, in (0, 1].
        rng_seed: Optional random seed.

    Returns:
        CompressionResult with a graph that has the same nodes but fewer edges.
    """
    edges=list(graph.edges())
    new_length=int(len(edges)*keep_ratio)
    edges_to_remove=random.sample(edges,len(edges)-new_length) #list of edges to remove
    for edge in edges_to_remove:
        graph.remove_edge(edge[0],edge[1])
    return CompressionResult(graph)

def compress_graph(
    graph: nx.Graph,
    method: str = "random_edge",
    **kwargs,
) -> CompressionResult:
    """
    Generic compression wrapper.

    Args:
        graph: Original graph.
        method: Name of compression method ("random_edge", "cut_sparsifier", etc.).
        kwargs: Method-specific arguments (e.g., keep_ratio).

    Returns:
        CompressionResult.
    """
    ...


def This_new_method(
    graph: nx.Graph,
    keep_ratio: float,
    rng_seed: int | None = None,
) -> CompressionResult:
    """
    Randomly keep a fraction of edges.

    Args:
        graph: Original graph.
        keep_ratio: Fraction of edges to keep, in (0, 1].
        rng_seed: Optional random seed.

    Returns:
        CompressionResult with a graph that has the same nodes but fewer edges.
    """
    ...
    

    
    
    

def paper_ABC_sparsity(
    graph: nx.Graph,
    keep_ratio: float,
    rng_seed: int | None = None,
) -> CompressionResult:
    """
    Randomly keep a fraction of edges.

    Args:
        graph: Original graph.
        keep_ratio: Fraction of edges to keep, in (0, 1].
        rng_seed: Optional random seed.

    Returns:
        CompressionResult with a graph that has the same nodes but fewer edges.
    """
    ...