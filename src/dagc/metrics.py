# metrics.py

from typing import Set

def jaccard_similarity(set1: Set, set2: Set) -> float:
    """
    Jaccard similarity between two sets.

    = |A ∩ B| / |A ∪ B|, returns 0 if both sets are empty.
    """
    ...

def spread_difference(
    spread_original: float,
    spread_compressed: float,
    num_nodes_original: int,
) -> float:
    """
    Relative spread difference as in your proposal:

        |σ_G(S) - σ_G'(S)| / |V|

    where |V| is the number of nodes in the original graph.
    """
    ...
