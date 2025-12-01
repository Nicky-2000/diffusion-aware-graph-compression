import networkx as nx
import torch

from dagc.sparsifiers.gnn.utils import (
    compute_transition_matrix_from_graph,
    build_temp_weighted_graph_from_probs,
)


def random_walk_preservation_loss(
    G: nx.Graph,
    edge_probs: torch.Tensor,  # [num_edges_dir]
    edge_index: torch.Tensor,  # [2, num_edges_dir]
    T_orig: torch.Tensor,  # [n, n] random-walk matrix of original graph
    keep_ratio: float,
    lambda_sparsity: float = 1.0,
    num_steps: int = 1,
) -> torch.Tensor:
    """
    Loss = ||T_tilde - T_orig||_F^2 + lambda_sparsity * (mean(p) - keep_ratio)^2

    where T_tilde is the k-step random-walk matrix built from a temporary weighted graph
    whose edge weights are given by edge_probs.
    """
    device = edge_probs.device

    # --- Build temporary weighted graph from probabilities ---
    G_tilde = build_temp_weighted_graph_from_probs(G, edge_probs, edge_index)

    # --- Compute learned k-step RW matrix using your helper ---
    T_tilde = compute_transition_matrix_from_graph(
        G_tilde,
        device=device,
        weight_attr="weight",
        num_steps=num_steps,
    )

    # --- RW preservation term (Frobenius norm squared) ---
    loss_rw = torch.sum((T_tilde - T_orig) ** 2)

    # --- Sparsity term: encourage mean edge prob ≈ keep_ratio ---
    mean_p = edge_probs.mean()
    loss_sparsity = (mean_p - keep_ratio) ** 2

    loss = loss_rw + lambda_sparsity * loss_sparsity
    return loss
