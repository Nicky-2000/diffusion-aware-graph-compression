# src/dagc/data/graph_loading.py

import os
import pickle
import networkx as nx
from typing import Optional
import torch


def load_graphs_from_dir(dir_path: str) -> list[nx.Graph]:
    """
    Load all .gpickle graphs from a directory and return them as a list.
    Ignores non-gpickle files.

    Args:
        dir_path: directory containing *.gpickle graphs

    Returns:
        list of NetworkX Graph objects
    """
    graphs = []
    for fname in os.listdir(dir_path):
        if fname.endswith(".gpickle"):
            fpath = os.path.join(dir_path, fname)
            with open(fpath, "rb") as f:
                graphs.append(pickle.load(f))
    return graphs


# def graph_to_tensors(
#     G: nx.Graph, device: Optional[torch.device] = None
# ) -> tuple[torch.Tensor, torch.Tensor]:
#     """
#     Turns a NetworkX Undirected graph G into:
#     - x: [num_nodes, in_dim] which is node features for each node
#     - edge_index: [2, num_edges_dir] (basically we have a src and dst list for each edge)

#     This assuems that the nodes are labeled 0..n-1
#     """

#     if device is None:
#         device = torch.device("cpu")

#     n = G.number_of_nodes()

#     # Simple Node features [degree, 1]. This can be expanded later
#     degrees = torch.tensor(
#         [G.degree(i) for i in range(n)], dtype=torch.float32, device=device
#     ).unsqueeze(
#         -1
#     )  # [n, 1]
#     ones = torch.ones((n, 1), dtype=torch.float32, device=device)
#     x = torch.cat([degrees, ones], dim=-1)  # [n, 2] (two features per node right now)

#     # Build directed edges
#     src_list = []
#     dst_list = []
#     for u, v in G.edges():
#         # Add one edge representing u->v
#         src_list.append(u)
#         dst_list.append(v)
#         # Add another one representing v->u
#         src_list.append(v)
#         dst_list.append(u)

#     edge_index = torch.tensor(
#         [src_list, dst_list], dtype=torch.long, device=device
#     )  # [2, num_edges_dir]

#     return x, edge_index

def compute_node_features(
    G: nx.Graph, device: Optional[torch.device] = None
) -> torch.Tensor:
    """
    Compute a feature vector for every node.

    Current features (all scalar per node):
      1. degree(v)
      2. 1.0 (bias term)
      3. clustering coefficient(v)
      4. core number(v)

    You can add/remove features here as you like.
    """
    if device is None:
        device = torch.device("cpu")

    n = G.number_of_nodes()
    nodes = list(range(n))  # assumes nodes are 0..n-1

    # 1) Degree
    deg = torch.tensor(
        [G.degree(v) for v in nodes],
        dtype=torch.float32,
        device=device,
    ).unsqueeze(-1)  # [n, 1]

    # 2) Bias term (all ones)
    bias = torch.ones((n, 1), dtype=torch.float32, device=device)  # [n, 1]

    # 3) Clustering coefficient
    clustering_dict = nx.clustering(G)
    clustering = torch.tensor(
        [clustering_dict[v] for v in nodes],
        dtype=torch.float32,
        device=device,
    ).unsqueeze(-1)  # [n, 1]

    # 4) k-core number
    core_dict = nx.core_number(G)
    core = torch.tensor(
        [core_dict[v] for v in nodes],
        dtype=torch.float32,
        device=device,
    ).unsqueeze(-1)  # [n, 1]

    # Concatenate all feature columns -> [n, in_dim]
    x = torch.cat([deg, bias, clustering, core], dim=-1)
    # x = torch.cat([deg, bias], dim=-1)
    return x


def graph_to_tensors(
    G: nx.Graph, device: Optional[torch.device] = None
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Turns a NetworkX Undirected graph G into:
      - x: [num_nodes, in_dim] node features
      - edge_index: [2, num_edges_dir] directed edge index

    Assumes nodes are labeled 0..n-1.
    """
    if device is None:
        device = torch.device("cpu")

    n = G.number_of_nodes()

    # --- Node features: just call the helper ---
    x = compute_node_features(G, device=device)  # [n, in_dim]

    # --- Directed edge_index construction (unchanged) ---
    src_list = []
    dst_list = []
    for u, v in G.edges():
        src_list.append(u)
        dst_list.append(v)
        src_list.append(v)
        dst_list.append(u)

    edge_index = torch.tensor(
        [src_list, dst_list],
        dtype=torch.long,
        device=device,
    )  # [2, num_edges_dir]

    return x, edge_index