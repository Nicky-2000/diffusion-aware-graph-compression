# src/dagc/sparsifiers/gnn/train.py

import os
import torch

from dagc.data.dataset_gnn import GraphDirectoryDataset, GraphSample
from dagc.sparsifiers.gnn.gnn_sparsifier import GNNSparsifier
from dagc.sparsifiers.gnn.losses import random_walk_preservation_loss
from dagc.sparsifiers.gnn.utils import compute_transition_matrix_from_graph


def _select_device(device: str | torch.device) -> torch.device:
    if isinstance(device, torch.device):
        return device
    if device == "auto":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    return torch.device(device)


def _eval_on_dataset(
    model: GNNSparsifier,
    dataset: GraphDirectoryDataset,
    keep_ratio: float,
    num_steps: int,
    lambda_sparsity: float,
    device: torch.device,
) -> tuple[float, float, float]:
    """
    Evaluate model on a dataset (no grad).
    Returns:
        avg_total_loss, avg_rw_loss, avg_sparsity_loss
    """
    model.eval()
    total_loss_sum = 0.0
    rw_loss_sum = 0.0
    sparsity_loss_sum = 0.0
    num_graphs = 0

    with torch.no_grad():
        for idx in range(len(dataset)):
            sample: GraphSample = dataset[idx]
            G = sample.graph
            x = sample.x.to(device)
            edge_index = sample.edge_index.to(device)

            T_orig = compute_transition_matrix_from_graph(
                G,
                device=device,
                weight_attr=None,
                num_steps=num_steps,
            )

            out = model(x, edge_index)

            total_loss, comps = random_walk_preservation_loss(
                G=G,
                edge_probs=out.edge_probs,
                edge_index=edge_index,
                T_orig=T_orig,
                keep_ratio=keep_ratio,
                lambda_sparsity=lambda_sparsity,
                num_steps=num_steps,
                return_components=True,
            )

            total_loss_sum += float(total_loss.item())
            rw_loss_sum += comps["rw_loss"]
            sparsity_loss_sum += comps["sparsity_loss"]
            num_graphs += 1

    denom = max(1, num_graphs)
    return (
        total_loss_sum / denom,
        rw_loss_sum / denom,
        sparsity_loss_sum / denom,
    )


def train_gnn_sparsifier_on_dataset(
    train_dir: str,
    val_dir: str,
    keep_ratio: float = 0.8,
    num_steps: int = 1,
    hidden_dim: int = 128,
    num_layers: int = 2,
    edge_mlp_hidden_dim: int = 128,
    num_epochs: int = 20,
    learning_rate: float = 1e-3,
    lambda_sparsity: float = 1.0,
    device: str | torch.device = "cpu",
    print_every: int = 1,
    checkpoint_path: str | None = None,
) -> tuple[GNNSparsifier, dict]:
    """
    Train a GNNSparsifier and evaluate on a validation set.

    Returns:
        model, history
    where history = {
        "epoch": [...],
        "train_loss": [...],
        "val_loss": [...],
        "train_rw_loss": [...],
        "train_sparsity_loss": [...],
        "val_rw_loss": [...],
        "val_sparsity_loss": [...],
    }
    """
    dev = _select_device(device)
    print(f"[train] Using device: {dev}")

    train_dataset = GraphDirectoryDataset(train_dir, device=dev)
    val_dataset = GraphDirectoryDataset(val_dir, device=dev)
    print(
        f"[train] Loaded {len(train_dataset)} train graphs and {len(val_dataset)} val graphs"
    )

    # Infer in_dim from a sample
    first_sample: GraphSample = train_dataset[0]
    in_dim = first_sample.x.size(1)

    model = GNNSparsifier(
        in_dim=in_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        edge_mlp_hidden_dim=edge_mlp_hidden_dim,
    ).to(dev)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    history = {
        "epoch": [],
        "train_loss": [],
        "val_loss": [],
        "train_rw_loss": [],
        "train_sparsity_loss": [],
        "val_rw_loss": [],
        "val_sparsity_loss": [],
        "train_avg_prob": []
    }

    best_val_loss = float("inf")
    best_state_dict = None

    for epoch in range(1, num_epochs + 1):
        model.train()
        train_total_sum = 0.0
        train_rw_sum = 0.0
        train_sparsity_sum = 0.0
        avg_prob_sum = 0.0
        num_graphs = 0

        for idx in range(len(train_dataset)):
            sample: GraphSample = train_dataset[idx]
            G = sample.graph
            x = sample.x.to(dev)
            edge_index = sample.edge_index.to(dev)

            T_orig = compute_transition_matrix_from_graph(
                G,
                device=dev,
                weight_attr=None,
                num_steps=num_steps,
            )

            optimizer.zero_grad()
            out = model(x, edge_index)

            loss, comps = random_walk_preservation_loss(
                G=G,
                edge_probs=out.edge_probs,
                edge_index=edge_index,
                T_orig=T_orig,
                keep_ratio=keep_ratio,
                lambda_sparsity=lambda_sparsity,
                num_steps=num_steps,
                return_components=True,
            )
            avg_prob_sum += float(out.edge_probs.mean().detach().cpu())

            loss.backward()
            optimizer.step()

            train_total_sum += float(loss.item())
            train_rw_sum += comps["rw_loss"]
            train_sparsity_sum += comps["sparsity_loss"]
            num_graphs += 1

        denom = max(1, num_graphs)
        train_loss = train_total_sum / denom
        train_rw_loss = train_rw_sum / denom
        train_sparsity_loss = train_sparsity_sum / denom

        # Validation
        val_loss, val_rw_loss, val_sparsity_loss = _eval_on_dataset(
            model=model,
            dataset=val_dataset,
            keep_ratio=keep_ratio,
            num_steps=num_steps,
            lambda_sparsity=lambda_sparsity,
            device=dev,
        )

        history["epoch"].append(epoch)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_rw_loss"].append(train_rw_loss)
        history["train_sparsity_loss"].append(train_sparsity_loss)
        history["val_rw_loss"].append(val_rw_loss)
        history["val_sparsity_loss"].append(val_sparsity_loss)
        history["train_avg_prob"].append(avg_prob_sum / num_graphs)


        if epoch % print_every == 0:
            print(
                f"[Epoch {epoch}/{num_epochs}] "
                f"train_loss = {train_loss:.6f} (rw={train_rw_loss:.4f}, sparse={train_sparsity_loss:.4f}) "
                f"val_loss = {val_loss:.6f} (rw={val_rw_loss:.4f}, sparse={val_sparsity_loss:.4f})"
            )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state_dict = model.state_dict()

    # Restore best
    if best_state_dict is not None:
        print(f"[train] Restoring best model with val_loss = {best_val_loss:.6f}")
        model.load_state_dict(best_state_dict)

   # Save checkpoint (wrapped dict instead of raw state_dict)
    if checkpoint_path is not None:
        os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)

        checkpoint = {
            "model_state_dict": model.state_dict(),
            "config": {
                "in_dim": in_dim,
                "hidden_dim": hidden_dim,
                "num_layers": num_layers,
                "edge_mlp_hidden_dim": edge_mlp_hidden_dim,
                "keep_ratio": keep_ratio,
                "num_steps": num_steps,
                "lambda_sparsity": lambda_sparsity,
                "learning_rate": learning_rate,
                "train_dir": train_dir,
                "val_dir": val_dir,
            },
            "history": history,
        }

        torch.save(checkpoint, checkpoint_path)
        print(f"[train] Saved checkpoint to: {checkpoint_path}")

    return model, history


def load_gnn_sparsifier(
    checkpoint_path: str,
    device: str | torch.device = "auto",
) -> GNNSparsifier:
    """
    Load a GNNSparsifier from a checkpoint saved by train_gnn_sparsifier_on_dataset.

    Returns the model on the requested device, in eval() mode.
    """
    dev = _select_device(device)
    ckpt = torch.load(checkpoint_path, map_location=dev)

    config = ckpt.get("config", {})
    state_dict = ckpt["model_state_dict"]

    in_dim = config.get("in_dim")
    if in_dim is None:
        raise ValueError(
            "Checkpoint is missing 'in_dim' in 'config'. "
            "Did you save it with the updated train function?"
        )

    model = GNNSparsifier(
        in_dim=in_dim,
        hidden_dim=config.get("hidden_dim", 64),
        num_layers=config.get("num_layers", 2),
        edge_mlp_hidden_dim=config.get("edge_mlp_hidden_dim", 64),
    ).to(dev)

    model.load_state_dict(state_dict)
    model.eval()

    print(f"[load] Loaded GNNSparsifier from {checkpoint_path} on device {dev}")
    return model
