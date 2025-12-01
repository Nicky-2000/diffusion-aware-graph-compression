# src/dagc/data/dataset_gnn.py

from dataclasses import dataclass

import os
import pickle
import networkx as nx
import torch
from torch.utils.data import Dataset

from dagc.data.graph_loading import graph_to_tensors


@dataclass
class GraphSample:
    """
    A single graph sample for GNN training.

    Attributes:
        graph: the original NetworkX graph (used e.g. for T_orig computation)
        x: node features [num_nodes, in_dim]
        edge_index: directed edge index [2, num_edges_dir]
    """

    graph: nx.Graph
    x: torch.Tensor
    edge_index: torch.Tensor


class GraphDirectoryDataset(Dataset):
    """
    Dataset that loads .gpickle graphs from a directory on demand,
    and converts each to (graph, x, edge_index) using graph_to_tensors.
    """

    def __init__(self, dir_path: str, device: str = "cpu"):
        self.dir_path = dir_path
        self.device = torch.device(device)

        # Pre-index all .gpickle files
        self.files: list[str] = [
            os.path.join(dir_path, f)
            for f in os.listdir(dir_path)
            if f.endswith(".gpickle")
        ]
        self.files.sort()  # deterministic order

        if not self.files:
            raise ValueError(f"No .gpickle files found in {dir_path}")

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> GraphSample:
        path = self.files[idx]
        with open(path, "rb") as f:
            G: nx.Graph = pickle.load(f)

        x, edge_index = graph_to_tensors(G, device=self.device)
        return GraphSample(graph=G, x=x, edge_index=edge_index)
