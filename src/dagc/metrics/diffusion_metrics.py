# src/dagc/metrics.py

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Set

import networkx as nx

from dagc.diffusion.independent_cascade import DiffusionResult
from dagc.sparsifiers.gnn.gnn_sparsifier import GNNSparsifier
from dagc.sparsifiers.classical.random_edge import random_edge_sparsify
from dagc.sparsifiers.classical.effective_resistance import (
    graph_sparsify_effective_resistance,
)
from dagc.sparsifiers.gnn.utils import sparsify_graph_with_model
from dagc.graphs.seeds import generate_seed_set



# ---------------------------------------------------------------------------
# Simple spread difference helper (may be useful elsewhere)
# ---------------------------------------------------------------------------


def spread_difference(
    spread_original: float,
    spread_compressed: float,
    num_nodes_original: int,
) -> float:
    """
    Relative spread difference as in the proposal:

        |σ_G(S) - σ_G'(S)| / |V|

    where:
        - σ_G(S)   is expected spread on the original graph G
        - σ_G'(S)  is expected spread on the compressed graph G'
        - |V|      is the number of nodes in the original graph

    Args:
        spread_original: Expected spread on the original graph.
        spread_compressed: Expected spread on the compressed graph.
        num_nodes_original: Number of nodes in the original graph (|V|).

    Returns:
        A float (typically in [0, 1]).
    """
    if num_nodes_original <= 0:
        raise ValueError("num_nodes_original must be positive.")
    return abs(spread_original - spread_compressed) / num_nodes_original


# ---------------------------------------------------------------------------
# Diffusion summaries
# ---------------------------------------------------------------------------


@dataclass
class DiffusionSummary:
    """
    Collapsed summary of many diffusion runs on the same graph + seed set.

    Attributes:
        activation_prob:
            Mapping node -> estimated probability that the node
            is active at the end of diffusion.

        expected_spread:
            Expected number of activated nodes:
                expected_spread = sum_v activation_prob[v]
    """

    activation_prob: Dict[Any, float]
    expected_spread: float


def summarize_diffusion_results(
    results: List[DiffusionResult],
    all_nodes: Optional[Iterable[Any]] = None,
) -> DiffusionSummary:
    """
    Collapse multiple diffusion runs into activation probabilities per node
    and an overall expected spread.

    Args:
        results:
            List of DiffusionResult objects from repeated simulations
            (same graph + seed set + model parameters).
        all_nodes:
            Optional iterable of all nodes that should appear in the summary.
            If provided, nodes not activated in any run get probability 0.0.
            If None, only nodes that ever became active in any run appear.

    Returns:
        DiffusionSummary with:
            - activation_prob[v] for each node v
            - expected_spread
    """
    if not results:
        raise ValueError("summarize_diffusion_results: results list is empty.")

    if all_nodes is None:
        node_set: Set[Any] = set()
        for res in results:
            node_set |= res.all_activated
    else:
        node_set = set(all_nodes)

    # Count in how many runs each node was active.
    counts: Dict[Any, int] = {v: 0 for v in node_set}
    for res in results:
        for v in res.all_activated:
            if v in counts:
                counts[v] += 1
            else:
                # In case all_nodes didn't include some node
                counts[v] = 1
                node_set.add(v)

    num_runs = len(results)
    activation_prob: Dict[Any, float] = {
        v: counts.get(v, 0) / num_runs for v in node_set
    }

    expected_spread = sum(activation_prob.values())

    return DiffusionSummary(
        activation_prob=activation_prob,
        expected_spread=expected_spread,
    )


# ---------------------------------------------------------------------------
# Simple “diffusion-preservation” metrics between two summaries
# ---------------------------------------------------------------------------


def basic_diffusion_metrics(
    original: DiffusionSummary,
    compressed: DiffusionSummary,
) -> Dict[str, float]:
    """
    Compute the two main diffusion-preservation metrics:

      1) spread_abs_diff:
           |E[spread(G)] - E[spread(G')]|      (global behavior)

      2) prob_mse:
           (1/|V|) * sum_i (p_i^G - p_i^{G'})^2   (local / per-node behavior)
         where p_i^G is the activation probability of node i in G.

    Here |V| is taken as the size of the union of node sets
    from the two summaries, and missing nodes are treated as prob 0.0.
    """
    # 1) Expected spread difference (absolute)
    spread_abs_diff = abs(
        float(original.expected_spread) - float(compressed.expected_spread)
    )

    # 2) MSE over per-node activation probabilities
    nodes = set(original.activation_prob.keys()) | set(
        compressed.activation_prob.keys()
    )

    if not nodes:
        return {
            "spread_abs_diff": spread_abs_diff,
            "prob_mse": 0.0,
        }

    diffs = []
    for v in nodes:
        p_orig = float(original.activation_prob.get(v, 0.0))
        p_comp = float(compressed.activation_prob.get(v, 0.0))
        diffs.append((p_orig - p_comp) ** 2)

    prob_mse = sum(diffs) / len(diffs)

    return {
        "spread_abs_diff": spread_abs_diff,
        "prob_mse": prob_mse,
    }

