# scripts/run_gnn_sparsification_experiment.py

"""
Run an end-to-end experiment using a learned GNN-based sparsifier:

1. Load a base graph (e.g., BA(1000, 3)).
2. Choose a seed set S.
3. Run IC diffusion R times on the original graph.
4. Use a trained GNN sparsifier to score edges and keep the top-M edges.
5. Run IC diffusion R times on the GNN-sparsified graph.
6. Summarize diffusion behavior and compare metrics.

This evaluates:
    "How much does a learned GNN sparsifier preserve diffusion behavior
     compared to the original graph?"
"""

import sys
import random
from pathlib import Path

import networkx as nx

# Make sure we can import the dagc package from src/
ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

from dagc.data.graph_loading import read_graph, generate_seed_set
from dagc.diffusion.independent_cascade import run_ic_diffusion
from dagc.metrics import (
    summarize_diffusion_results,
    compare_diffusion_summaries,
    topk_jaccard_from_summaries,
    spread_difference,
)

# NEW: GNN-based sparsifier API (to be implemented)
from dagc.compression.learned_sparsify import (
    load_gnn_sparsifier,
    gnn_edge_sparsify,
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

    # Path to a trained GNN checkpoint (to be produced by your training script)
    model_checkpoint = ROOT / "checkpoints" / "gnn_sparsifier.pt"
    device = "cpu"  # or "cuda" if available

    print("=== GNN Edge Sparsification Experiment ===")
    print(f"Graph path:       {graph_path}")
    print(f"Num runs:         {num_runs}")
    print(f"Seed set size:    {seed_set_size}")
    print(f"Activation prob:  {activation_prob}")
    print(f"Keep ratio:       {keep_ratio}")
    print(f"Base RNG seed:    {base_seed}")
    print(f"Model checkpoint: {model_checkpoint}")
    print(f"Device:           {device}")
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
    print()

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
    # Load GNN sparsifier
    # -----------------------------
    print("Loading GNN sparsifier model...")
    if not model_checkpoint.exists():
        print(f"ERROR: Model checkpoint not found at: {model_checkpoint}")
        print("       Train the GNN sparsifier first, then re-run this script.")
        sys.exit(1)

    model = load_gnn_sparsifier(
        checkpoint_path=str(model_checkpoint),
        device=device,
    )
    print("GNN model loaded.")
    print()

    # -----------------------------
    # Sparsify graph using GNN
    # -----------------------------
    print(f"Sparsifying graph with keep_ratio={keep_ratio} using GNN...")
    comp = gnn_edge_sparsify(
        graph=G,
        keep_ratio=keep_ratio,
        model=model,
        device=device,
        ensure_connected=True,  # try to keep the largest component intact
    )

    # We assume `comp` is a small struct / namespace with at least `.graph`
    # (mirrors the random_edge_sparsify API)
    H = comp.graph
    n_H = H.number_of_nodes()
    m_H = H.number_of_edges()
    print(f"Sparsified graph: |V| = {n_H}, |E| = {m_H}")

    if not nx.is_connected(H):
        print("WARNING: Sparsified graph is not connected.")
    print()

    # -----------------------------
    # Run IC on GNN-sparsified graph
    # -----------------------------
    print(f"Running IC diffusion on GNN-sparsified graph for {num_runs} runs...")
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
    print(
        f"GNN-sparsified graph: expected spread ≈ "
        f"{comp_summary.expected_spread:.2f}"
    )
    print()

    # -----------------------------
    # Metrics: spread + activation probs
    # -----------------------------
    print("Comparing diffusion summaries (original vs GNN-sparsified)...")
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
    print(f"Expected spread (orig):          {summary_metrics['spread_original']:.3f}")
    print(
        f"Expected spread (GNN-sparse):    "
        f"{summary_metrics['spread_compressed']:.3f}"
    )
    print(
        f"Abs diff (GNN-sparse - orig):    "
        f"{summary_metrics['spread_abs_diff']:.3f}"
    )
    print(
        f"Rel error (vs orig σ):           "
        f"{summary_metrics['spread_rel_error']:.4f}"
    )
    print(f"Rel diff / |V|:                  {rel_spread_diff_by_V:.4f}")
    print()

    print("---- Activation Probability Metrics ----")
    print(f"MSE over P(activated):           {summary_metrics['prob_mse']:.6f}")
    print(f"MAE over P(activated):           {summary_metrics['prob_mae']:.6f}")
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
        print(
            f"Top-{k} Jaccard (most activated nodes): "
            f"{topk_jacc:.3f}"
        )

    print()
    print("=== Done. ===")


if __name__ == "__main__":
    main()
