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


### TRYING THIS OUT

import torch
import networkx as nx
from typing import Iterable, Optional, Dict, Any

from .utils import build_temp_weighted_graph_from_probs  # you already have this

def _weight_to_ic_prob(weight: torch.Tensor, base_p: float) -> torch.Tensor:
    """
    Map a nonnegative 'weight' w to an IC-style edge activation probability:
        p_eff = 1 - (1 - base_p) ** w

    This is exactly your "effective number of parallel edges" interpretation.
    """
    # Clamp to avoid numerical issues
    w_clamped = torch.clamp(weight, min=0.0)
    return 1.0 - (1.0 - base_p) ** w_clamped


def _mean_field_ic(
    P: torch.Tensor,          # [n, n] edge activation probabilities (row = source, col = target)
    seed_indices: torch.Tensor,  # [k] indices of seed nodes
    num_steps: int = 3,
) -> torch.Tensor:
    """
    Differentiable mean-field IC propagation.

    Returns:
        a_T: [n] tensor, approximate activation probability for each node
             after num_steps steps.
    """
    device = P.device
    n = P.size(0)

    # a^(0): seeds are 1, others 0
    a = torch.zeros(n, device=device)
    a[seed_indices] = 1.0

    for _ in range(num_steps):
        # M[u, v] = a_u * P[u, v]
        M = a.unsqueeze(1) * P  # [n, n]

        # fail[v] = product over u of (1 - M[u, v])
        fail = torch.prod(1.0 - M, dim=0)  # [n]

        # New activation probability at this step
        new = (1.0 - a) * (1.0 - fail)

        # Update a^(t+1)
        a = a + new

    return a  # [n]

def _graph_to_ic_prob_matrix_original(
    G: nx.Graph,
    base_p: float = 0.1,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """
    Build an [n, n] activation-probability matrix P for the *original* graph G,
    assuming each undirected edge {u, v} has IC probability base_p.
    """
    if device is None:
        device = torch.device("cpu")

    nodes = list(G.nodes())
    node_to_idx: Dict[Any, int] = {u: i for i, u in enumerate(nodes)}
    n = len(nodes)

    P = torch.zeros((n, n), dtype=torch.float32, device=device)

    for u, v in G.edges():
        i = node_to_idx[u]
        j = node_to_idx[v]
        P[i, j] = base_p
        P[j, i] = base_p

    return P, nodes, node_to_idx


def _graph_to_ic_prob_matrix_gnn(
    G: nx.Graph,
    edge_probs: torch.Tensor,
    edge_index: torch.Tensor,
    base_p: float = 0.1,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """
    Build an [n, n] activation-probability matrix P for the *GNN-sparsified*
    graph defined by (G, edge_probs, edge_index).

    Steps:
      1) Build temp weighted undirected graph G_tilde from edge_probs.
      2) Convert weights to IC probabilities via p_eff = 1 - (1 - base_p)^w.
    """
    if device is None:
        device = torch.device("cpu")

    # 1) Build temp weighted graph from probs (you already use this in RW loss)
    G_tilde = build_temp_weighted_graph_from_probs(G, edge_probs, edge_index)

    nodes = list(G_tilde.nodes())
    node_to_idx: Dict[Any, int] = {u: i for i, u in enumerate(nodes)}
    n = len(nodes)

    # 2) Build P matrix from edge weights
    P = torch.zeros((n, n), dtype=torch.float32, device=device)

    for u, v, data in G_tilde.edges(data=True):
        w = float(data.get("weight", 0.0))
        # convert weight -> IC prob
        p_eff = _weight_to_ic_prob(torch.tensor(w, device=device), base_p=base_p)
        i = node_to_idx[u]
        j = node_to_idx[v]
        P[i, j] = p_eff
        P[j, i] = p_eff

    return P, nodes, node_to_idx



def ic_preservation_loss(
    G: nx.Graph,
    edge_probs: torch.Tensor,
    edge_index: torch.Tensor,
    seed_nodes: Iterable[Any],
    base_p: float,
    keep_ratio: float,
    lambda_sparsity: float,
    num_steps: int = 3,
    device: Optional[torch.device] = None,
    return_components: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, Dict[str, float]]:
    """
    IC-style diffusion preservation loss (mean-field, differentiable).

    Total loss = IC_MSE + λ * sparsity_loss, where:

      IC_MSE =
        (1 / |V|) * sum_v (a_gnn(v) - a_orig(v))^2

    and a_orig, a_gnn are mean-field IC activation probabilities
    after `num_steps` steps, starting from the same seed set.

    Args:
        G: original NetworkX graph (unweighted or weighted separately).
        edge_probs: [num_edges_dir] GNN edge scores (logits/probs).
        edge_index: [2, num_edges_dir] directed edge indices.
        seed_nodes: iterable of node labels used as seeds.
        base_p: base IC probability on the original graph edges.
        keep_ratio: target average keep probability for sparsity regularization.
        lambda_sparsity: weight on sparsity loss term.
        num_steps: number of mean-field IC steps.
        device: torch.device.
        return_components: if True, return (loss, {"ic_mse": ..., "sparsity": ...}).

    Returns:
        total_loss or (total_loss, components_dict)
    """
    if device is None:
        device = edge_probs.device

    # ----- 1) Original graph mean-field IC -----
    P_orig, nodes_orig, node_to_idx_orig = _graph_to_ic_prob_matrix_original(
        G, base_p=base_p, device=device
    )

    seed_indices_orig = torch.tensor(
        [node_to_idx_orig[u] for u in seed_nodes],
        dtype=torch.long,
        device=device,
    )

    a_orig = _mean_field_ic(
        P=P_orig,
        seed_indices=seed_indices_orig,
        num_steps=num_steps,
    )  # [n]

    # ----- 2) GNN graph mean-field IC -----
    P_gnn, nodes_gnn, node_to_idx_gnn = _graph_to_ic_prob_matrix_gnn(
        G,
        edge_probs=edge_probs,
        edge_index=edge_index,
        base_p=base_p,
        device=device,
    )

    # Assume node sets are identical; if not, we'd need alignment logic
    seed_indices_gnn = torch.tensor(
        [node_to_idx_gnn[u] for u in seed_nodes],
        dtype=torch.long,
        device=device,
    )

    a_gnn = _mean_field_ic(
        P=P_gnn,
        seed_indices=seed_indices_gnn,
        num_steps=num_steps,
    )  # [n]

    # ----- 3) IC-MSE loss -----
    ic_mse = torch.mean((a_gnn - a_orig) ** 2)

    # ----- 4) Sparsity term -----
    avg_prob = torch.mean(edge_probs)
    sparsity_loss = (avg_prob - keep_ratio) ** 2

    total_loss = ic_mse + lambda_sparsity * sparsity_loss

    if not return_components:
        return total_loss

    comps = {
        "ic_mse": float(ic_mse.detach().cpu()),
        "sparsity_loss": float(sparsity_loss.detach().cpu()),
    }
    return total_loss, comps

