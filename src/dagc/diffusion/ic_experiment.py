# src/dagc/diffusion/ic_experiment.py

from typing import Iterable, Optional
import random
import networkx as nx

from dagc.diffusion.independent_cascade import run_ic_diffusion
from dagc.metrics.diffusion_metrics import summarize_diffusion_results


def run_ic_experiment(
    graph: nx.Graph,
    seed_set: Iterable,
    num_runs: int = 100,
    activation_prob: float = 0.1,
    base_seed: int = 12345,
    max_steps: Optional[int] = None,
):
    """
    Run Independent Cascade (IC) diffusion multiple times on a graph and
    return the summarized diffusion behavior.

    Args:
        graph: NetworkX graph.
        seed_set: initial active nodes.
        num_runs: number of Monte Carlo simulations.
        activation_prob: probability of activation along each edge.
        base_seed: base seed for reproducibility across graphs/methods.
        max_steps: optional cap on IC steps.

    Returns:
        A summary object from summarize_diffusion_results, e.g. with:
          - expected_spread
          - activation_prob_by_node
          - etc.
    """
    seed_set = list(seed_set)
    results = []

    for i in range(num_runs):
        rng = random.Random(base_seed + i)
        res = run_ic_diffusion(
            graph=graph,
            seed_set=seed_set,
            activation_prob=activation_prob,
            max_steps=max_steps,
            rng=rng,
        )
        results.append(res)

    summary = summarize_diffusion_results(results, all_nodes=graph.nodes())
    return summary
