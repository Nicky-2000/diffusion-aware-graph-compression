import random
from typing import Callable, Dict, Any
import networkx as nx
import numpy as np
from dagc.sparsifiers.classical.random_edge import CompressionResult

def effective_resistance_matrix(G: nx.Graph) -> tuple[np.ndarray, Dict[Any, int]]:
    """
    Returns:
      L_plus: (n x n) pseudoinverse of Laplacian
      node_to_idx: mapping from node label -> row/col index in L_plus
    """
    # Fix a stable node ordering
    nodes = list(G.nodes())
    node_to_idx = {u: i for i, u in enumerate(nodes)}

    L = nx.laplacian_matrix(G, nodelist=nodes).toarray()
    L_plus = np.linalg.pinv(L)  # (n × n)
    return L_plus, node_to_idx


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

    # 1) L⁺ and node indexing
    L_plus, node_to_idx = effective_resistance_matrix(G)

    # 2) edge weights (default 1.0)
    weights = nx.get_edge_attributes(G, "weight")
    if not weights:
        weights = {e: 1.0 for e in G.edges()}

    # 3) effective resistance for each edge
    Re: Dict[tuple[Any, Any], float] = {}
    for (u, v) in G.edges():
        i = node_to_idx[u]
        j = node_to_idx[v]

        chi = np.zeros(n)
        chi[i] = 1.0
        chi[j] = -1.0
        Re[(u, v)] = chi @ L_plus @ chi

    # 4) sampling probs p_e ∝ w_e · R_e
    total = sum(weights[e] * Re[e] for e in G.edges())
    pe = {e: (weights[e] * Re[e]) / total for e in G.edges()}

    # 5) sample with replacement
    H = nx.Graph()
    H.add_nodes_from(G.nodes())
    rng = random.Random(rng_seed)
    edges = list(G.edges())

    edge_probs = [pe[e] for e in edges]

    for _ in range(q):
        e = rng.choices(edges, weights=edge_probs, k=1)[0]
        u, v = e
        new_w = weights[e] / (q * pe[e])  # rescaled weight

        if H.has_edge(u, v):
            H[u][v]["weight"] += new_w
        else:
            H.add_edge(u, v, weight=new_w)

    return CompressionResult(H)
