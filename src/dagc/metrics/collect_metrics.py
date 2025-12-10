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
from dagc.diffusion.ic_experiment import run_ic_experiment
from dagc.metrics.diffusion_metrics import basic_diffusion_metrics

# ---------------------------------------------------------------------------
# End-to-end evaluation on a single graph
# ---------------------------------------------------------------------------


def evaluate_one_graph(
    G: nx.Graph,
    model: GNNSparsifier,
    keep_ratio: float = 0.5,
    seed_set_size: int = 10,
    activation_prob: float = 0.1,
    num_runs: int = 200,
    base_seed: int = 12345,
    device: str = "cpu",
):
    """
    For a single graph G:

      - choose a seed set S (same for all variants)
      - run IC on:
          * original graph G
          * random-sparsified graph H_rand
          * GNN-sparsified graph H_gnn
          * ER-sparsified graph H_eff
      - compute *simple* metrics for each vs original:

          1) spread_abs_diff:
               |E[spread(G)] - E[spread(G')]|
          2) prob_mse:
               MSE over node activation probabilities

    Returns:
      metrics_rand, metrics_gnn, metrics_eff,
      orig_summary, rand_summary, gnn_summary, eff_summary,
      H_rand, H_gnn, H_eff
    """
    # 1) Seed set (same seeds for all sparsifiers)
    seed_set = generate_seed_set(G, k=seed_set_size, strategy="high_degree")

    # 2) Original graph IC summary (ground truth diffusion behavior)
    orig_summary = run_ic_experiment(
        graph=G,
        seed_set=seed_set,
        num_runs=num_runs,
        activation_prob=activation_prob,
        base_seed=base_seed,
    )

    # 3) Random edge sparsifier
    rand_comp = random_edge_sparsify(
        graph=G,
        keep_ratio=keep_ratio,
        rng_seed=base_seed + 999,
    )
    H_rand = rand_comp.graph

    rand_summary = run_ic_experiment(
        graph=H_rand,
        seed_set=seed_set,
        num_runs=num_runs,
        activation_prob=activation_prob,
        base_seed=base_seed + 1000,
    )

    # 4) GNN sparsifier (your weighted GNN-based sparsified graph)
    H_gnn = sparsify_graph_with_model(
        model=model,
        G=G,
        keep_ratio=keep_ratio,
        device=device,
    )

    gnn_summary = run_ic_experiment(
        graph=H_gnn,
        seed_set=seed_set,
        num_runs=num_runs,
        activation_prob=activation_prob,
        base_seed=base_seed + 2000,
    )

    # 5) Effective resistance sparsifier
    eff_comp = graph_sparsify_effective_resistance(
        graph=G,
        keep_ratio=keep_ratio,
        rng_seed=base_seed + 3000,
    )
    H_eff = eff_comp.graph

    eff_summary = run_ic_experiment(
        graph=H_eff,
        seed_set=seed_set,
        num_runs=num_runs,
        activation_prob=activation_prob,
        base_seed=base_seed + 4000,
    )

    # 6) Simple metrics vs original for each sparsifier
    metrics_rand = basic_diffusion_metrics(
        original=orig_summary, compressed=rand_summary
    )
    metrics_gnn = basic_diffusion_metrics(
        original=orig_summary, compressed=gnn_summary
    )
    metrics_eff = basic_diffusion_metrics(
        original=orig_summary, compressed=eff_summary
    )

    return (
        metrics_rand,
        metrics_gnn,
        metrics_eff,
        orig_summary,
        rand_summary,
        gnn_summary,
        eff_summary,
        H_rand,
        H_gnn,
        H_eff,
    )
