# scripts/generate_ba_dataset.py

"""
Generate a BA graph dataset for GNN training / validation / testing.

Creates:
  data/synthetic_graphs/ba/train/ba_{i}.gpickle
  data/synthetic_graphs/ba/val/ba_{i}.gpickle
  data/synthetic_graphs/ba/test/ba_{i}.gpickle
"""

import os
import pickle
import argparse
from pathlib import Path

import networkx as nx
from tqdm import tqdm


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def generate_ba_graph(n: int, m: int, seed: int) -> nx.Graph:
    """
    Generate a single Barabási–Albert graph.
    """
    return nx.barabasi_albert_graph(n=n, m=m, seed=seed)


def save_graph_gpickle(G: nx.Graph, path: Path) -> None:
    with open(path, "wb") as f:
        pickle.dump(G, f, protocol=pickle.HIGHEST_PROTOCOL)


def main():
    parser = argparse.ArgumentParser(description="Generate BA train/val/test datasets.")
    parser.add_argument("--num-train", type=int, default=120)
    parser.add_argument("--num-val", type=int, default=40)
    parser.add_argument("--num-test", type=int, default=40)

    parser.add_argument("--n", type=int, default=1000, help="Number of nodes per graph.")
    parser.add_argument("--m", type=int, default=3, help="Edges to attach per new node.")

    parser.add_argument(
        "--base-dir",
        type=str,
        default=None,
        help="Base directory for synthetic BA graphs (default: <repo_root>/data/synthetic_graphs/ba)",
    )

    args = parser.parse_args()

    # Figure out repo root = parent of scripts/
    ROOT = Path(__file__).resolve().parents[1]

    if args.base_dir is None:
        base_dir = ROOT / "data" / "synthetic_graphs" / "ba"
    else:
        base_dir = Path(args.base_dir)

    train_dir = base_dir / "train"
    val_dir = base_dir / "val"
    test_dir = base_dir / "test"

    for d in [train_dir, val_dir, test_dir]:
        ensure_dir(d)

    cfg_str = f"(n={args.n}, m={args.m})"
    print(f"Saving BA graphs under {base_dir} {cfg_str}")

    # We’ll just use a global counter to derive seeds deterministically
    seed_counter = 42

    def generate_split(split_name: str, num_graphs: int, target_dir: Path):
        nonlocal seed_counter
        if num_graphs <= 0:
            return
        print(f"\nGenerating {num_graphs} graphs for split='{split_name}'...")
        for i in tqdm(range(num_graphs)):
            seed = seed_counter
            seed_counter += 1

            G = generate_ba_graph(n=args.n, m=args.m, seed=seed)
            out_path = target_dir / f"ba_{i}.gpickle"
            save_graph_gpickle(G, out_path)

    generate_split("train", args.num_train, train_dir)
    generate_split("val", args.num_val, val_dir)
    generate_split("test", args.num_test, test_dir)

    print("\nDone generating BA dataset!")


if __name__ == "__main__":
    main()
