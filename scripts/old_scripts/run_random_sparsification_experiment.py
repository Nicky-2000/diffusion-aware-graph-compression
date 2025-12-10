# scripts/run_random_sparsification_experiment.py

"""
Run a simple end-to-end experiment:

1. Load a base graph (e.g., BA(1000, 3)).
2. Choose a seed set S.
3. Run IC diffusion R times on the original graph.
4. Randomly sparsify the graph while preserving connectivity.
5. Run IC diffusion R times on the sparsified graph.
6. Summarize diffusion behavior and compare metrics.

This gives a baseline for:
    "How much does naive random edge sparsification distort diffusion?"
"""

import sys
import random
from pathlib import Path

import networkx as nx

# Make sure we can import the dagc package from src/
ROOT = Path(__file__).resolve().parents[1]
ROOT=ROOT.parent  # adjust if needed
sys.path.append(str(ROOT / "src"))

from dagc.data.read_graphs import read_graph, generate_seed_set
from dagc.diffusion.independent_cascade import run_ic_diffusion
from dagc.sparsifiers.classical.random_edge import random_edge_sparsify
from dagc.metrics.diffusion_metrics import (
    summarize_diffusion_results,
    compare_diffusion_summaries,
    topk_jaccard_from_summaries,
    spread_difference,
)


def main():
    # -----------------------------
    # Experiment configuration
    # -----------------------------
    graph_path = ROOT / "data" / "graphs" / "ba_1000_3.edgelist"

    num_runs = 100
    seed_set_size = 5
    activation_prob = 0.1
    keep_ratio = 0.5  # keep 50% of edges → remove 50%
    base_seed = 12345

    print("=== Random Edge Sparsification Experiment ===")
    print(f"Graph path:       {graph_path}")
    print(f"Num runs:         {num_runs}")
    print(f"Seed set size:    {seed_set_size}")
    print(f"Activation prob:  {activation_prob}")
    print(f"Keep ratio:       {keep_ratio} (≈ 10% edges removed)")
    print(f"Base RNG seed:    {base_seed}")
    print()

    # -----------------------------
    # Load original graph
    # -----------------------------
    print("Loading original graph...")
    G = read_graph(str(graph_path), fmt="edge_list")
    n = G.number_of_nodes()
    m = G.number_of_edges()
    print(f"Original graph: |V| = {n}, |E| = {m}")

    if not nx.is_connected(G):
        print("WARNING: Original graph is not connected!")

    # -----------------------------
    # Generate seed set
    # -----------------------------
    print(f"Generating seed set of size {seed_set_size} (high_degree)...")
    seed_set = generate_seed_set(G, k=seed_set_size, strategy="high_degree")
    print(f"Seed set: {sorted(seed_set)}")
    print()

    # -----------------------------
    # Run IC on original graph
    # -----------------------------
    print(f"Running IC diffusion on original graph for {num_runs} runs...")
    orig_results = []
    for i in range(num_runs):
        rng = random.Random(base_seed + i)
        res = run_ic_diffusion(
            graph=G,
            seed_set=seed_set,
            activation_prob=activation_prob,
            max_steps=None,
            rng=rng,
        )
        orig_results.append(res)

    orig_summary = summarize_diffusion_results(
        orig_results,
        all_nodes=G.nodes(),
    )
    print(f"Original graph: expected spread ≈ {orig_summary.expected_spread:.2f}")
    print()

    # -----------------------------
    # Sparsify graph
    # -----------------------------
    print(f"Sparsifying graph with keep_ratio={keep_ratio} ...")
    comp = random_edge_sparsify(
        graph=G,
        keep_ratio=keep_ratio,
        rng_seed=base_seed + 9999,
    )
    H = comp.graph
    n_H = H.number_of_nodes()
    m_H = H.number_of_edges()
    print(f"Sparsified graph: |V| = {n_H}, |E| = {m_H}")

    if not nx.is_connected(H):
        print("WARNING: Sparsified graph is not connected (this should not happen).")
    print()

    # -----------------------------
    # Run IC on sparsified graph
    # -----------------------------
    print(f"Running IC diffusion on sparsified graph for {num_runs} runs...")
    comp_results = []
    for i in range(num_runs):
        rng = random.Random(base_seed + 10_000 + i)
        res = run_ic_diffusion(
            graph=H,
            seed_set=seed_set,
            activation_prob=activation_prob,
            max_steps=None,
            rng=rng,
        )
        comp_results.append(res)

    comp_summary = summarize_diffusion_results(
        comp_results,
        all_nodes=G.nodes(),  # keep node universe aligned with original graph
    )
    print(f"Sparsified graph: expected spread ≈ {comp_summary.expected_spread:.2f}")
    print()

    # -----------------------------
    # Metrics: spread + activation probs
    # -----------------------------
    print("Comparing diffusion summaries (original vs sparsified)...")
    summary_metrics = compare_diffusion_summaries(
        original=orig_summary,
        compressed=comp_summary,
    )

    rel_spread_diff_by_V = spread_difference(
        spread_original=orig_summary.expected_spread,
        spread_compressed=comp_summary.expected_spread,
        num_nodes_original=n,
    )

    print("---- Spread Metrics ----")
    print(f"Expected spread (orig):       {summary_metrics['spread_original']:.3f}")
    print(f"Expected spread (sparse):     {summary_metrics['spread_compressed']:.3f}")
    print(f"Abs diff (sparse - orig):     {summary_metrics['spread_abs_diff']:.3f}")
    print(f"Rel error (vs orig σ):        {summary_metrics['spread_rel_error']:.4f}")
    print(f"Rel diff / |V|:               {rel_spread_diff_by_V:.4f}")
    print()

    print("---- Activation Probability Metrics ----")
    print(f"MSE over P(activated):        {summary_metrics['prob_mse']:.6f}")
    print(f"MAE over P(activated):        {summary_metrics['prob_mae']:.6f}")
    print()

    # -----------------------------
    # Top-K Jaccard of most activated nodes
    # -----------------------------
    for k in [10, 20, 50]:
        if k > n:
            break
        topk_jacc = topk_jaccard_from_summaries(
            original=orig_summary,
            compressed=comp_summary,
            k=k,
        )
        print(f"Top-{k} Jaccard (most activated nodes): {topk_jacc:.3f}")

    print()
    print("=== Done. ===")


if __name__ == "__main__":
    main()
