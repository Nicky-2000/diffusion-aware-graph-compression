import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleGCNLayer(nn.Module):
    """This is a very simple GCN-like Layer. Graph convolutional neural network
    h_new(v) = ReLU(W-self h(v) + W_neigh * mean_{u in N(v)} h(u))
    """

    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.W_self = nn.Linear(in_dim, out_dim, bias=True)
        self.W_neigh = nn.Linear(in_dim, out_dim, bias=False)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        x : [num_nodes, in_dim]
        edge_index: [2, num_edges_dir] with (src, dst) per column
        """
        num_nodes = x.size(0)
        src, dst = edge_index

        # Gather source node embeddings for each edge
        # Basically this is the "node features" of the "source node" of each edge.
        # So the 0th element of x_src will be the node features of the source node of the 0th edge
        # this is probably the node features of node 0!
        x_src = x[src]  # [num_edges_dir, in_dim]

        # Initialize neighbour aggregation with zeros
        neigh_agg = torch.zeros_like(x)  # [num_nodes, in_dim]
        # neigh_agg[v] will store the sum of features received from all neighbors of v.
        # Aggregate (sum) messages into destination nodes
        neigh_agg.index_add_(0, dst, x_src)  # Sum messages into neigh_add[dst]

        # Compute degree for normalization (numbwer of incoming edges)
        deg = torch.zeros(num_nodes, device=x.device, dtype=x.dtype)
        one_vec = torch.ones_like(src, dtype=x.dtype)
        deg.index_add_(0, dst, one_vec)
        # Avoid division by zero
        deg = torch.clamp(deg, min=1.0).unsqueeze(-1)  # [num_nodes, 1]

        neigh_mean = neigh_agg / deg  # Mean Neighbor features

        out = self.W_self(x) + self.W_neigh(neigh_mean)
        return F.relu(out)


class GNNEncoder(nn.Module):
    """
    Stack of simple GCNLayers
    h^(0) = x
    h^(l+1) = SimpleGCNLayer_l(h^(l), edge_index)
    """

    def __init__(self, in_dim: int, hidden_dim: int, num_layers: int):
        super().__init__()
        layers = []

        # First layer: in_dim -> hidden_dim
        layers.append(SimpleGCNLayer(in_dim, hidden_dim))

        # Remaining layers hidden_dim -> hidden_dim
        for _ in range(num_layers - 1):
            layers.append(SimpleGCNLayer(hidden_dim, hidden_dim))

        self.layers = nn.ModuleList(layers)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        x: [num_nodes, in_dim] (in_dim is the dimension of the features for node i)
        edge_index: [2, num_edges_dir]
        returns: H [num_nodes, hidden_dim]
        """
        h = x
        for layer in self.layers:
            h = layer(h, edge_index)
        return h
