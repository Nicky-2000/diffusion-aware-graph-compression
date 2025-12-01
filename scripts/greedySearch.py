def main():
    import sys
    from typing import Any, Iterable, List, Optional, Sequence, Set, Tuple
    import random
    import sys
    from pathlib import Path

    import networkx as nx

    # Make sure we can import the dagc package from src/
    ROOT = Path(__file__).resolve().parents[1]
    sys.path.append(str(ROOT / "src"))

    from dagc.graphs import read_graph, generate_seed_set
    from dagc.diffusion import run_ic_diffusion
    from dagc.compression import random_edge_sparsify
    from dagc.compression import graph_sparsify_effective_resistance
    from dagc.metrics import (
        summarize_diffusion_results,
        compare_diffusion_summaries,
        topk_jaccard_from_summaries,
        spread_difference,
    )


    def estimate_expected_spread_ic(
        graph: nx.Graph,
        seed_set: Iterable[Any],
        activation_prob: float = 0.1,
        num_runs: int = 100,
        base_seed: int = 0,
    ) -> float:
        """
        Estimate the expected spread σ(S) under the IC model
        via Monte Carlo simulations, using the same pattern as
        scripts/run_random_sparsification_experiment.py.

        Args:
            graph:
                NetworkX graph.
            seed_set:
                Initial active node set S.
            activation_prob:
                Edge activation probability in IC model.
            num_runs:
                Number of Monte Carlo runs.
            base_seed:
                Base RNG seed for reproducibility.

        Returns:
            Estimated expected spread σ(S).
        """
        seed_set = set(seed_set)

        results = []
        for i in range(num_runs):
            rng = random.Random(base_seed + i)
            res = run_ic_diffusion(
                graph=graph,
                seed_set=seed_set,
                activation_prob=activation_prob,
                max_steps=None,
                rng=rng,
            )
            results.append(res)

        summary = summarize_diffusion_results(
            results,
            all_nodes=graph.nodes(),
        )
        return summary.expected_spread


    def greedy_ic_influence_maximization(
        graph: nx.Graph,
        k: int,
        activation_prob: float = 0.1,
        num_runs_per_eval: int = 100,
        base_seed: int = 0,
        candidate_nodes: Optional[Sequence[Any]] = None,
        verbose: bool = True,
    ) -> Tuple[List[Any], List[float], List[float]]:
        """
        Greedy Approximation Algorithm for Influence Maximization under IC.

        Algorithm (matches your lecture pseudo-code):

            Start with A = ∅
            for i = 1 to k:
                choose v maximizing σ(A ∪ {v}) − σ(A)
                A ← A ∪ {v}

        Args:
            graph:
                NetworkX graph.
            k:
                Number of seeds to select.
            activation_prob:
                Edge activation probability in IC model.
            num_runs_per_eval:
                Monte Carlo runs used to estimate σ(·).
            base_seed:
                Base RNG seed to make the whole greedy run reproducible.
            candidate_nodes:
                Optional subset of nodes to pick seeds from.
                If None, use all nodes in graph.
            verbose:
                If True, print progress and marginal gains.

        Returns:
            (seed_list, spreads, marginal_gains)
                seed_list:      [v1, v2, ..., vk] in the order selected
                spreads:        [σ({v1}), σ({v1,v2}), ..., σ(A_k)]
                marginal_gains: [σ(A_1) − 0, σ(A_2) − σ(A_1), ..., σ(A_k) − σ(A_{k-1})]
        """
        if candidate_nodes is None:
            candidate_nodes = list(graph.nodes())
        else:
            candidate_nodes = list(candidate_nodes)

        selected: List[Any] = []
        selected_set: Set[Any] = set()

        spreads: List[float] = []
        marginal_gains: List[float] = []

        for i in range(k):
            sigma_A = estimate_expected_spread_ic(
                graph=graph,
                seed_set=selected_set,
                activation_prob=activation_prob,
                num_runs=num_runs_per_eval,
                base_seed=base_seed + i * 100_000,
            )

            best_v: Any = None
            best_gain: float = float("-inf")
            best_sigma_with_v: float = sigma_A

            for v in candidate_nodes:
                if v in selected_set:
                    continue

                sigma_with_v = estimate_expected_spread_ic(
                    graph=graph,
                    seed_set=selected_set | {v},
                    activation_prob=activation_prob,
                    num_runs=num_runs_per_eval,
                    base_seed=base_seed + i * 100_000 + 10_000,
                )
                gain = sigma_with_v - sigma_A

                if gain > best_gain:
                    best_gain = gain
                    best_v = v
                    best_sigma_with_v = sigma_with_v

            selected.append(best_v)
            selected_set.add(best_v)
            spreads.append(best_sigma_with_v)
            marginal_gains.append(best_gain)

            if verbose:
                print(
                    f"[iter {i+1}/{k}] chose v = {best_v}, "
                    f"marginal gain ≈ {best_gain:.3f}, "
                    f"σ(A) ≈ {best_sigma_with_v:.3f}"
                )

        return selected, spreads, marginal_gains
    graph_path = ROOT / "data" / "graphs" / "ba_1000_3.edgelist"
    G = read_graph(str(graph_path), fmt="edge_list")

    k = 5
    activation_prob = 0.1
    num_runs_per_eval = 50
    base_seed = 12345

    seeds, spreads, gains = greedy_ic_influence_maximization(
        graph=G,
        k=k,
        activation_prob=activation_prob,
        num_runs_per_eval=num_runs_per_eval,
        base_seed=base_seed,
        verbose=True,
    )

    print()
    print("Greedy seed set:", seeds)
    print("Final expected spread ≈", spreads[-1])

if __name__ == "__main__":
    main()