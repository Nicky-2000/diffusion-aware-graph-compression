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
    edge_prob_mode: str = "weighted_attempts",   # "weighted_attempts" or "direct_prob"
    edge_prob_attr: str = "p",                  # used if direct_prob
    edge_weight_attr: str = "weight",           # used if weighted_attempts
):
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
            edge_prob_mode=edge_prob_mode,
            edge_prob_attr=edge_prob_attr,
            edge_weight_attr=edge_weight_attr,
        )
        results.append(res)

    summary = summarize_diffusion_results(results, all_nodes=graph.nodes())
    return summary
