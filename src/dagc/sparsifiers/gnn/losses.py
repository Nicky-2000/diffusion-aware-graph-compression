import torch
import networkx as nx
from typing import Optional, Dict, Tuple

from .utils import build_temp_weighted_graph_from_probs, compute_transition_matrix_from_graph


def random_walk_preservation_loss(
    G: nx.Graph,
    edge_probs: torch.Tensor,
    edge_index: torch.Tensor,
    T_orig: torch.Tensor,
    keep_ratio: float,
    lambda_sparsity: float,
    num_steps: int = 1,
    return_components: bool = False,
) -> torch.Tensor | Tuple[torch.Tensor, Dict[str, float]]:
    """
    Total loss = RW loss + λ * sparsity loss.

    If return_components=True, returns (total_loss, {"rw": ..., "sparsity": ...}).
    """
    device = edge_probs.device

    # 1) Build weighted temp graph from probs
    G_tilde = build_temp_weighted_graph_from_probs(G, edge_probs, edge_index)

    # 2) Compute k-step transition matrix T_tilde
    T_tilde = compute_transition_matrix_from_graph(
        G_tilde,
        device=device,
        weight_attr="weight",
        num_steps=num_steps,
    )

    # 3) Random-walk preservation term: ||T_tilde - T_orig||_F^2
    diff = T_tilde - T_orig
    rw_loss = torch.mean(diff * diff)  # Frobenius norm squared

    # 4) Sparsity term: encourage average prob ≈ keep_ratio
    avg_prob = torch.mean(edge_probs)
    sparsity_loss = (avg_prob - keep_ratio) ** 2

    total_loss = rw_loss + lambda_sparsity * sparsity_loss

    if not return_components:
        return total_loss

    comps = {
        "rw_loss": float(rw_loss.detach().cpu()),
        "sparsity_loss": float(sparsity_loss.detach().cpu()),
    }
    return total_loss, comps
