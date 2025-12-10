from typing import Optional
import networkx as nx
import torch

from dagc.data.graph_loading import graph_to_tensors
from dagc.sparsifiers.gnn.gnn_sparsifier import GNNSparsifier


def compute_transition_matrix_from_graph(
    G: nx.Graph,
    device: Optional[torch.device] = None,
    weight_attr: Optional[str] = None,
    num_steps: int = 1,
) -> torch.Tensor:
    """
    Build the k-step random walk matrix form an (optionally weighted) undirected graph
    - If weight_attr is None:
        Treat all edges as weight 1.
    - If weight_attr is not None:
        Use G[u][v][weight_attr] as the (symmetric) weight for edge (u,v)

    Random-walk matrix T is row-stochastic:
        T[i,j] = P(next node = j| current node = i)

    if num_steps = 1:
        Returns T
    if num_steps > 1:
    returns T^k (k-step transition probabilities)

    """

    if device is None:
        device = torch.device("cpu")

    n = G.number_of_nodes()
    T = torch.zeros((n, n), dtype=torch.float32, device=device)

    # Build (possibly weighted) adjacency into T
    for u, v, data in G.edges(data=True):
        if weight_attr is None:
            w = 1.0
        else:
            w = float(data.get(weight_attr, 1.0))
        T[u, v] = w
        T[v, u] = w

    # Row-normalize to get 1-step random walk matrix
    row_sums = T.sum(dim=1, keepdim=True) + 1e-8  # [n,1]
    T = T / row_sums

    if num_steps <= 1:
        return T

    # Compute k-step transition: T^k
    # For modest n and small k this is fine
    T_k = T.clone()
    for _ in range(num_steps - 1):
        # print(f"Multiplying for step {_ + 2}...")
        T_k = T_k @ T  # Matrix multiplication

    return T_k


def build_temp_weighted_graph_from_probs(
    G: nx.Graph,
    edge_probs: torch.Tensor,
    edge_index: torch.Tensor,
) -> nx.Graph:
    """
    Build a temporary weighted undirected graph G_tilde from:
      - the original node set of G
      - directed edge_index and per-edge probabilities edge_probs

    We:
      - keep the same nodes as G
      - add an undirected edge {u, v} with attribute 'weight' = max(p(u->v), p(v->u)).
        (edge_index is directed, but G_tilde is undirected.)
    """
    # Start from an empty graph with the same nodes
    G_tilde = nx.Graph()
    G_tilde.add_nodes_from(G.nodes())

    # Move to CPU / numpy for easy iteration
    src_np = edge_index[0].detach().cpu().numpy()
    dst_np = edge_index[1].detach().cpu().numpy()
    p_np = edge_probs.detach().cpu().numpy()

    seen = set()  # keep track of undirected pairs we've added

    for u, v, p in zip(src_np, dst_np, p_np):
        if u == v:
            continue  # ignore self-loops if any
        key = (min(u, v), max(u, v))
        if key in seen:
            # take the max weight over both directions
            old_w = G_tilde[key[0]][key[1]]["weight"]
            if p > old_w:
                G_tilde[key[0]][key[1]]["weight"] = float(p)
        else:
            G_tilde.add_edge(key[0], key[1], weight=float(p))
            seen.add(key)

    return G_tilde

def sparsify_graph_with_model(
    model: GNNSparsifier,
    G: nx.Graph,
    keep_ratio: float = 0.5,
    device="cpu"
):
    """
    Use a trained GNNSparsifier model to produce a new sparsified graph H
    """
    
    # Step 1: Convert to tensors 
    x, edge_index = graph_to_tensors(G, device=device)
    
    # Step 2: Model forward pass
    model.eval()
    with torch.no_grad():
        out = model(x, edge_index)
        probs = out.edge_probs # [num_edges_dir]
    
    # Step 3: Convert directed probs back to undirected edge scores
    undirected_scores = _collapse_directed_edge_probs(edge_index, probs)
    
    # Step 4: Decide how many edges to keep 
    num_edges_original = G.number_of_edges()
    k = int(num_edges_original * keep_ratio)
    k = max(1, k)
    
    # Step 5: Keep top-k edges by score
    keep_edges = sorted(
        undirected_scores.items(), 
        key=lambda kv: kv[1],
        reverse=True,
    )[:k]
    
    # If scores can be negative (e.g. raw logits), shift them so weights ≥ 0
    min_score = min(score for (_, score) in keep_edges)
    shift = -min_score if min_score < 0 else 0.0
    print(f"Shifting edge scores by {shift:.4f} to ensure non-negativity.")
    
    # Build sparsified graph H
    H = nx.Graph()
    H.add_nodes_from(G.nodes())
    
    for (u,v), score in keep_edges:
        w = float(score + shift)
        H.add_edge(u,v, weight=w)
    
    return H
    
    
    
    
def _collapse_directed_edge_probs(edge_index, probs):
    """
    Convert directed edges (u→v and v→u) into a single undirected score.
    """
    src, dst = edge_index
    scores = {}
    for i in range(src.size(0)):
        u, v = int(src[i]), int(dst[i])
        key = (u, v) if u < v else (v, u)
        scores.setdefault(key, []).append(float(probs[i]))

    # average or max (we will pick max — usually more stable)
    return {k: max(v_list) for k, v_list in scores.items()}
