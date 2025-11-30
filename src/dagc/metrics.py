# src/dagc/metrics.py

"""
Metrics for comparing diffusion behavior on graphs.

This module provides:

- Basic set-based metrics:
    - jaccard_similarity(A, B)
    - spread_difference(σ_G, σ_G', |V|)

- Diffusion summaries:
    - DiffusionSummary: collapsed representation of many diffusion runs
    - summarize_diffusion_results(): list[DiffusionResult] -> DiffusionSummary

- Comparison metrics between two summaries (e.g., original vs compressed graph):
    - compare_diffusion_summaries(): spread + activation-probability errors
    - topk_jaccard_from_summaries(): Jaccard similarity of top-K most
      activated nodes (by activation probability).
"""

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Set

from .diffusion.independent_cascade import DiffusionResult


# ---------------------------------------------------------------------------
# Basic set / scalar metrics
# ---------------------------------------------------------------------------

def jaccard_similarity(set1: Set[Any], set2: Set[Any]) -> float:
    """
    Compute Jaccard similarity between two sets.

        J(A, B) = |A ∩ B| / |A ∪ B|

    Returns:
        A float in [0, 1].

    Note:
        If both sets are empty, this returns 0.0 by convention.
    """
    if not set1 and not set2:
        return 0.0

    intersection = len(set1 & set2)
    union = len(set1 | set2)

    if union == 0:
        return 0.0

    return intersection / union


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
# Comparing two summaries (original vs compressed)
# ---------------------------------------------------------------------------

def compare_diffusion_summaries(
    original: DiffusionSummary,
    compressed: DiffusionSummary,
    nodes: Optional[Iterable[Any]] = None,
) -> Dict[str, float]:
    """
    Compare two DiffusionSummary objects (original vs compressed graph).

    Metrics returned:
        - spread_original: expected spread on original graph
        - spread_compressed: expected spread on compressed graph
        - spread_abs_diff: spread_compressed - spread_original
        - spread_rel_error:
              |spread_compressed - spread_original| / spread_original
              (0 if spread_original == 0)
        - prob_mse: mean squared error over activation probabilities
        - prob_mae: mean absolute error over activation probabilities

    Args:
        original: Summary from the original graph.
        compressed: Summary from the compressed graph.
        nodes:
            Optional iterable of nodes over which to compute metrics.
            If None, use the intersection of nodes that appear in both
            summaries.

    Returns:
        A dict of scalar metrics.
    """
    if nodes is None:
        nodes_set = set(original.activation_prob.keys()) & set(
            compressed.activation_prob.keys()
        )
    else:
        nodes_set = set(nodes)

    if not nodes_set:
        raise ValueError("compare_diffusion_summaries: no overlapping nodes to compare.")

    # Basic spread metrics
    spread_orig = original.expected_spread
    spread_comp = compressed.expected_spread
    spread_diff = spread_comp - spread_orig
    if spread_orig > 0:
        spread_rel_error = abs(spread_diff) / spread_orig
    else:
        spread_rel_error = 0.0

    # Activation-probability error metrics
    sq_err = 0.0
    abs_err = 0.0
    n = 0

    for v in nodes_set:
        p_orig = original.activation_prob.get(v, 0.0)
        p_comp = compressed.activation_prob.get(v, 0.0)
        diff = p_comp - p_orig
        sq_err += diff * diff
        abs_err += abs(diff)
        n += 1

    prob_mse = sq_err / n
    prob_mae = abs_err / n

    return {
        "spread_original": spread_orig,
        "spread_compressed": spread_comp,
        "spread_abs_diff": spread_diff,
        "spread_rel_error": spread_rel_error,
        "prob_mse": prob_mse,
        "prob_mae": prob_mae,
    }


def topk_jaccard_from_summaries(
    original: DiffusionSummary,
    compressed: DiffusionSummary,
    k: int,
) -> float:
    """
    Compute Jaccard similarity between the top-K most activated nodes
    (by activation probability) in the original vs compressed graph.

    This measures how well the compressed graph preserves the identity of
    the most "influenced" / "reachable" nodes under diffusion.

    Args:
        original: DiffusionSummary for the original graph.
        compressed: DiffusionSummary for the compressed graph.
        k: Number of top nodes to consider in each summary.

    Returns:
        Jaccard similarity in [0, 1] between the two top-K sets.
    """
    if k <= 0:
        raise ValueError("k must be positive for topk_jaccard_from_summaries.")

    # Sort nodes by activation probability (descending)
    orig_sorted = sorted(
        original.activation_prob.items(),
        key=lambda x: x[1],
        reverse=True,
    )
    comp_sorted = sorted(
        compressed.activation_prob.items(),
        key=lambda x: x[1],
        reverse=True,
    )

    # Take top-k node IDs (handle case where there are fewer than k nodes)
    orig_topk = {node for node, _ in orig_sorted[:k]}
    comp_topk = {node for node, _ in comp_sorted[:k]}

    return jaccard_similarity(orig_topk, comp_topk)
