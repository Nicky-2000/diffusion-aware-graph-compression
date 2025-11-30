from typing import Optional
import networkx as nx
import torch


def graph_to_tensors(
    G: nx.Graph, device: Optional[torch.device] = None
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Turns a NetworkX Undirected graph G into:
    - x: [num_nodes, in_dim] which is node features for each node
    - edge_index: [2, num_edges_dir] (basically we have a src and dst list for each edge)

    This assuems that the nodes are labeled 0..n-1
    """

    if device is None:
        device = torch.device("cpu")

    n = G.number_of_nodes()

    # Simple Node features [degree, 1]. This can be expanded later
    degrees = torch.tensor(
        [G.degree(i) for i in range(n)], dtype=torch.float32, device=device
    ).unsqueeze(
        -1
    )  # [n, 1]
    ones = torch.ones((n, 1), dtype=torch.float32, device=device)
    x = torch.cat([degrees, ones], dim=-1)  # [n, 2] (two features per node right now)

    # Build directed edges
    src_list = []
    dst_list = []
    for u, v in G.edges():
        # Add one edge representing u->v
        src_list.append(u)
        dst_list.append(v)
        # Add another one representing v->u
        src_list.append(v)
        dst_list.append(u)

    edge_index = torch.tensor(
        [src_list, dst_list], dtype=torch.long, device=device
    )  # [2, num_edges_dir]

    return x, edge_index


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
