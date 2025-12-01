# src/dagc/sparsifiers/gnn/train.py

import os
import torch
from pathlib import Path

from dagc.data.dataset_gnn import GraphDirectoryDataset, GraphSample
from dagc.sparsifiers.gnn.gnn_sparsifier import GNNSparsifier
from dagc.sparsifiers.gnn.losses import random_walk_preservation_loss
from dagc.sparsifiers.gnn.utils import compute_transition_matrix_from_graph


def _get_device(device: str | torch.device | None) -> torch.device:
    """
    Small helper to resolve device, with Apple MPS support.

    Usage:
      - device="cpu"
      - device="cuda"
      - device="mps"
      - device="auto"  -> prefers mps, then cuda, else cpu
    """
    if isinstance(device, torch.device):
        return device

    if device is None or device == "auto":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")

    return torch.device(device)


def _compute_dataset_loss(
    dataset: GraphDirectoryDataset,
    model: GNNSparsifier,
    dev: torch.device,
    keep_ratio: float,
    lambda_sparsity: float,
    num_steps: int,
) -> float:
    """
    Evaluate random-walk preservation loss over all graphs in a dataset
    (no gradient, model in eval mode).
    """
    model.eval()
    total_loss = 0.0
    num_graphs = 0

    with torch.no_grad():
        for idx in range(len(dataset)):
            sample: GraphSample = dataset[idx]
            G = sample.graph
            x = sample.x.to(dev)
            edge_index = sample.edge_index.to(dev)

            T_orig = compute_transition_matrix_from_graph(
                G,
                device=dev,
                weight_attr=None,
                num_steps=num_steps,
            )

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

            total_loss += loss.item()
            num_graphs += 1

    if num_graphs == 0:
        return float("nan")
    return total_loss / num_graphs


def train_gnn_sparsifier_on_dataset(
    train_dir: str,
    val_dir: str | None = None,
    keep_ratio: float = 0.5,
    num_steps: int = 1,
    hidden_dim: int = 64,
    num_layers: int = 2,
    edge_mlp_hidden_dim: int = 64,
    num_epochs: int = 5,
    learning_rate: float = 1e-3,
    lambda_sparsity: float = 1.0,
    device: str | torch.device = "auto",
    print_every: int = 1,
    checkpoint_path: str | None = None,
    save_best: bool = True,
) -> GNNSparsifier:
    """
    Train a GNNSparsifier on a directory of .gpickle graphs (BA train set).

    If val_dir is provided:
      - compute validation loss each epoch
      - keep the best model (lowest val loss) in memory
      - optionally save that best model to checkpoint_path

    If val_dir is None:
      - only track / print train loss
      - optionally save the final model at the end
    """
    dev = _get_device(device)
    print(f"[train] Using device: {dev}")

    # 1) Build datasets
    train_dataset = GraphDirectoryDataset(train_dir, device=dev)

    val_dataset = None
    if val_dir is not None:
        val_dataset = GraphDirectoryDataset(val_dir, device=dev)
        print(
            f"[train] Loaded {len(train_dataset)} train graphs and "
            f"{len(val_dataset)} val graphs"
        )
    else:
        print(f"[train] Loaded {len(train_dataset)} train graphs (no val set).")

    # 2) Infer in_dim from first sample
    first_sample: GraphSample = train_dataset[0]
    in_dim = first_sample.x.size(1)

    # 3) Init model + optimizer
    model = GNNSparsifier(
        in_dim=in_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        edge_mlp_hidden_dim=edge_mlp_hidden_dim,
    ).to(dev)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # We'll store this config in the checkpoint so we can reconstruct the model later
    config = {
        "in_dim": in_dim,
        "hidden_dim": hidden_dim,
        "num_layers": num_layers,
        "edge_mlp_hidden_dim": edge_mlp_hidden_dim,
        "keep_ratio": keep_ratio,
        "num_steps": num_steps,
        "lambda_sparsity": lambda_sparsity,
    }
    
    history = {"epoch": [], "train_loss": [], "val_loss": []}
    best_val_loss = float("inf")
    best_state_dict = None
    
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)


    # 4) Training loop
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        num_graphs = 0

        for idx in range(len(train_dataset)):
            sample: GraphSample = train_dataset[idx]
            G = sample.graph
            x = sample.x.to(dev)
            edge_index = sample.edge_index.to(dev)

            num_graphs += 1

            T_orig = compute_transition_matrix_from_graph(
                G,
                device=dev,
                weight_attr=None,
                num_steps=num_steps,
            )

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

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        avg_train_loss = epoch_loss / max(1, num_graphs)
        avg_val_loss = float("nan")

        # 5) Validation loss (if we have a val set)
        if val_dataset is not None:
            avg_val_loss = _compute_dataset_loss(
                dataset=val_dataset,
                model=model,
                dev=dev,
                keep_ratio=keep_ratio,
                lambda_sparsity=lambda_sparsity,
                num_steps=num_steps,
            )

            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_state_dict = {
                    k: v.detach().cpu().clone() for k, v in model.state_dict().items()
                }

            if (epoch + 1) % print_every == 0:
                print(
                    f"[Epoch {epoch+1}/{num_epochs}] "
                    f"train_loss = {avg_train_loss:.6f} "
                    f"val_loss = {avg_val_loss:.6f}"
                )
        else:
            if (epoch + 1) % print_every == 0:
                print(
                    f"[Epoch {epoch+1}/{num_epochs}] "
                    f"train_loss = {avg_train_loss:.6f}"
                )
        
        history["epoch"].append(epoch + 1)
        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(avg_val_loss)

    # 6) Restore best model (if val set used)
    if val_dataset is not None and best_state_dict is not None:
        print(f"[train] Restoring best model with val_loss = {best_val_loss:.6f}")
        model.load_state_dict(best_state_dict)

    # 7) Save checkpoint if requested
    if checkpoint_path is not None:
        ckpt_path = Path(checkpoint_path)
        ckpt_path.parent.mkdir(parents=True, exist_ok=True)

        # "best" = best val model, else final model
        state = (
            best_state_dict if (save_best and best_state_dict is not None) else model.state_dict()
        )

        torch.save(
            {
                "model_state_dict": state,
                "config": config,
            },
            ckpt_path,
        )
        print(f"[train] Saved checkpoint to: {ckpt_path}")

    return model, history


def load_gnn_sparsifier(
    checkpoint_path: str,
    device: str | torch.device = "auto",
) -> GNNSparsifier:
    """
    Load a GNNSparsifier from a checkpoint saved by train_gnn_sparsifier_on_dataset.

    Returns the model on the requested device, in eval() mode.
    """
    dev = _get_device(device)
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
