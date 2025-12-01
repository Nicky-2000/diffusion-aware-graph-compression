# src/dagc/sparsifiers/gnn/train.py

import torch

from dagc.data.dataset_gnn import GraphDirectoryDataset, GraphSample
from dagc.sparsifiers.gnn.gnn_sparsifier import GNNSparsifier
from dagc.sparsifiers.gnn.losses import random_walk_preservation_loss
from dagc.sparsifiers.gnn.utils import compute_transition_matrix_from_graph


def train_gnn_sparsifier_on_dataset(
    train_dir: str,
    keep_ratio: float = 0.5,
    num_steps: int = 1,
    hidden_dim: int = 64,
    num_layers: int = 2,
    edge_mlp_hidden_dim: int = 64,
    num_epochs: int = 5,
    learning_rate: float = 1e-3,
    lambda_sparsity: float = 1.0,
    device: str = "cpu",
    print_every: int = 1,
) -> GNNSparsifier:
    """
    Train a GNNSparsifier on a directory of .gpickle graphs (BA train set).

    Each graph is treated as a separate training example. For each graph:
      - compute x, edge_index (already done by the dataset)
      - compute T_orig (k-step random walk)
      - forward through model
      - compute RW + sparsity loss
      - backprop + optimizer step
    """
    dev = torch.device(device)

    # 1) Build dataset (no DataLoader for now)
    dataset = GraphDirectoryDataset(train_dir, device=device)

    # 2) Infer in_dim from first sample
    first_sample: GraphSample = dataset[0]
    in_dim = first_sample.x.size(1)

    # 3) Init model + optimizer
    model = GNNSparsifier(
        in_dim=in_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        edge_mlp_hidden_dim=edge_mlp_hidden_dim,
    ).to(dev)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        num_graphs = 0

        # iterate directly over graphs
        for idx in range(len(dataset)):
            sample: GraphSample = dataset[idx]
            G = sample.graph
            x = sample.x.to(dev)
            edge_index = sample.edge_index.to(dev)

            num_graphs += 1

            # Precompute T_orig for this graph
            T_orig = compute_transition_matrix_from_graph(
                G,
                device=dev,
                weight_attr=None,
                num_steps=num_steps,
            )

            # Forward
            optimizer.zero_grad()
            out = model(x, edge_index)

            loss = random_walk_preservation_loss(
                G=G,
                edge_probs=out.edge_probs,
                edge_index=edge_index,
                T_orig=T_orig,
                keep_ratio=keep_ratio,
                lambda_sparsity=lambda_sparsity,
                num_steps=num_steps,
            )

            # Backprop
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / max(1, num_graphs)
        if (epoch + 1) % print_every == 0:
            print(
                f"[Epoch {epoch+1}/{num_epochs}] "
                f"avg_loss = {avg_loss:.6f} over {num_graphs} graphs"
            )

    return model
