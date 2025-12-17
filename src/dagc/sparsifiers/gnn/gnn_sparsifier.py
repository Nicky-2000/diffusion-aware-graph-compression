from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F
from dagc.sparsifiers.gnn.layers import GNNEncoder


@dataclass
class GNNSparsifierOutput:
    node_embeddings: torch.Tensor  # [num_nodes, hidden_dim]
    edge_logits: torch.Tensor  # [num_edges_dir]
    edge_probs: torch.Tensor  # [num_edges_dir]


class GNNSparsifier(nn.Module):
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int = 64,
        num_layers: int = 2,
        edge_mlp_hidden_dim: int = 64,
    ):
        super().__init__()
        self.encoder = GNNEncoder(
            in_dim=in_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
        )

        # Edge feature dimension: [h_u, h_v, h_u * h_v, |h_u - h_v|]
        edge_feat_dim = 4 * hidden_dim

        self.edge_mlp = nn.Sequential(
            nn.Linear(edge_feat_dim, edge_mlp_hidden_dim),
            nn.ReLU(),
            nn.Linear(edge_mlp_hidden_dim, 1),  # scalar logit per edge
        )

    def forward(
        self,
        x: torch.Tensor,  # [num_nodes, in_dim]
        edge_index: torch.Tensor,  # [2, num_edges_dir]
    ) -> GNNSparsifierOutput:
        # Step 1) Node embeddings from GNN
        H = self.encoder(x, edge_index)  # [num_nodes, hidden_dim]

        # Step 2) Build per-edge features
        src, dst = edge_index

        # Learned embedding for the source / dest node of each edge
        h_src = H[src]  # [num_edges_dir, hidden_dim]
        h_dst = H[dst]  # [num_edges_dir, hidden_dim]

        edge_feat = torch.cat(
            [h_src, h_dst, h_src * h_dst, torch.abs(h_src - h_dst)],
            dim=-1,
        )  # [num_edges_dir, 4 * hidden_dim]

        # Step 3) Edge logits + probabilities
        edge_logits = self.edge_mlp(edge_feat).squeeze(-1)  # [num_edges_dir]
        edge_probs = torch.sigmoid(edge_logits)  # [num_edges_dir]

        return GNNSparsifierOutput(
            node_embeddings=H,
            edge_logits=edge_logits,
            edge_probs=edge_probs,
        )
