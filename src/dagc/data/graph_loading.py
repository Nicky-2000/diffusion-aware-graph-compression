"""
Graph loading and seed set generation utilities.

This module provides:
- read_graph: load a NetworkX graph from disk.
- generate_seed_set: choose a set of seed nodes.
- generate_multiple_seed_sets: convenience wrapper for repeated experiments.
"""

from typing import List, Set, Any
import os
import pickle
import random

import networkx as nx


def read_graph(path: str, fmt: str = "edge_list") -> nx.Graph:
    """
    Load a graph from disk.

    Args:
        path: Path to the graph file.
        fmt: Format of the file. Supported options:
             - "edge_list": plain text edge list (e.g. produced by nx.write_edgelist)
             - "gpickle":  Python pickle of a NetworkX graph (our .gpickle files)

    Returns:
        A NetworkX graph (typically undirected).

    Raises:
        FileNotFoundError: if the path does not exist.
        ValueError: if the format is not supported.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Graph file not found: {path}")

    if fmt == "edge_list":
        # nodes will be read as strings by default; we can cast to int
        G = nx.read_edgelist(path, nodetype=int)
        return G
    elif fmt == "gpickle":
        with open(path, "rb") as f:
            G = pickle.load(f)
        if not isinstance(G, nx.Graph):
            raise ValueError(f"Loaded object from {path} is not a NetworkX graph.")
        return G
    else:
        raise ValueError(f"Unsupported graph format: {fmt}")


def generate_seed_set(graph: nx.Graph, k: int, strategy: str = "random") -> Set[Any]:
    """
    Generate a single seed set of size k.

    Args:
        graph: Input graph.
        k: Number of seeds to select (must be <= number of nodes in the graph).
        strategy: How to choose seeds. Supported options:
                  - "random": uniform random nodes
                  - "high_degree": top-k nodes by degree

    Returns:
        A set of node IDs to use as seeds.

    Raises:
        ValueError: if k is invalid or strategy is unsupported.
    """
    nodes = list(graph.nodes())
    n = len(nodes)

    if k <= 0:
        raise ValueError(f"k must be positive, got {k}.")
    if k > n:
        raise ValueError(f"k={k} is larger than number of nodes in graph (n={n}).")

    if strategy == "random":
        chosen = random.sample(nodes, k)
        return set(chosen)

    elif strategy == "high_degree":
        # Sort nodes by degree (descending) and take top-k
        # degree() returns (node, degree) pairs
        degrees = sorted(graph.degree, key=lambda x: x[1], reverse=True)
        top_k_nodes = [node for node, _deg in degrees[:k]]
        return set(top_k_nodes)

    else:
        raise ValueError(f"Unsupported seed selection strategy: {strategy}")


def generate_multiple_seed_sets(
    graph: nx.Graph,
    num_sets: int,
    k: int,
    strategy: str = "random",
) -> List[Set[Any]]:
    """
    Generate multiple seed sets for repeated experiments.

    Note:
        For strategy "high_degree", this will return the *same* seed set
        num_sets times. For "random", each seed set will typically differ.

    Args:
        graph: Input graph.
        num_sets: Number of seed sets to generate.
        k: Size of each seed set.
        strategy: Seed selection strategy passed through to generate_seed_set().

    Returns:
        A list of seed sets (each a set of node IDs).

    Raises:
        ValueError: if num_sets is not positive.
    """
    if num_sets <= 0:
        raise ValueError(f"num_sets must be positive, got {num_sets}.")

    seed_sets: List[Set[Any]] = []
    for _ in range(num_sets):
        seeds = generate_seed_set(graph, k=k, strategy=strategy)
        seed_sets.append(seeds)

    return seed_sets
