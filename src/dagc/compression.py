# compression.py

from random import random
from typing import Callable, Dict, Any
import networkx as nx

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
        self.node_mapping = node_mapping if node_mapping is not None else {n: n for n in graph.nodes()}

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