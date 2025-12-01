import random
import networkx as nx
from typing import List, Set, Any

def generate_seed_set(graph: nx.Graph, k: int, strategy: str = "random") -> Set[Any]:
    nodes = list(graph.nodes())
    n = len(nodes)

    if k <= 0 or k > n:
        raise ValueError(f"Invalid k={k} for n={n} nodes")

    if strategy == "random":
        return set(random.sample(nodes, k))

    elif strategy == "high_degree":
        degrees = sorted(graph.degree(), key=lambda x: x[1], reverse=True)
        return set([node for node, _ in degrees[:k]])

    else:
        raise ValueError(f"Unknown strategy: {strategy}")


def generate_multiple_seed_sets(
    graph: nx.Graph,
    num_sets: int,
    k: int,
    strategy: str = "random",
) -> List[Set[Any]]:
    if num_sets <= 0:
        raise ValueError("num_sets must be positive")

    return [generate_seed_set(graph, k, strategy) for _ in range(num_sets)]
