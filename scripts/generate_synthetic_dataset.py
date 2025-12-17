"""
Generate a mixed synthetic graph dataset for GNN training / validation / testing.

Creates:
  data/synthetic_graphs/multi/train/<type>_<i>.gpickle
  data/synthetic_graphs/multi/val/<type>_<i>.gpickle
  data/synthetic_graphs/multi/test/<type>_<i>.gpickle

Supported types:
  - ba    : Barabasi-Albert
  - er    : Erdos-Renyi
  - ws    : Watts-Strogatz
  - sbm   : Stochastic Block Model
  - rgg   : Random Geometric Graph
  - hk    : Powerlaw Cluster (Holme-Kim)
"""

import os
import pickle
import argparse
from pathlib import Path
from typing import Optional, Dict, Any

import networkx as nx
from tqdm import tqdm


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def save_graph_gpickle(G: nx.Graph, path: Path) -> None:
    with open(path, "wb") as f:
        pickle.dump(G, f, protocol=pickle.HIGHEST_PROTOCOL)


def _relabel_to_0n(G: nx.Graph) -> nx.Graph:
    """Ensure node labels are 0..n-1 (your pipeline assumes this)."""
    return nx.convert_node_labels_to_integers(G, first_label=0, ordering="sorted")


# -------- Graph generators --------

def gen_ba(n: int, m: int, seed: int) -> nx.Graph:
    return nx.barabasi_albert_graph(n=n, m=m, seed=seed)

def gen_er(n: int, p: float, seed: int) -> nx.Graph:
    return nx.erdos_renyi_graph(n=n, p=p, seed=seed)

def gen_ws(n: int, k: int, p: float, seed: int) -> nx.Graph:
    # k must be even for WS
    if k % 2 == 1:
        k += 1
    return nx.watts_strogatz_graph(n=n, k=k, p=p, seed=seed)

def gen_sbm(n: int, num_blocks: int, p_in: float, p_out: float, seed: int) -> nx.Graph:
    # equal-sized blocks (last block absorbs remainder)
    base = n // num_blocks
    sizes = [base] * num_blocks
    sizes[-1] += n - sum(sizes)

    # block probability matrix
    probs = [[p_out] * num_blocks for _ in range(num_blocks)]
    for i in range(num_blocks):
        probs[i][i] = p_in

    G = nx.stochastic_block_model(sizes, probs, seed=seed)
    return G

def gen_rgg(n: int, radius: float, seed: int) -> nx.Graph:
    # random geometric in unit square
    G = nx.random_geometric_graph(n=n, radius=radius, seed=seed)
    return G

def gen_hk(n: int, m: int, p: float, seed: int) -> nx.Graph:
    # Holme–Kim / powerlaw cluster (triangles)
    return nx.powerlaw_cluster_graph(n=n, m=m, p=p, seed=seed)


def main():
    parser = argparse.ArgumentParser(description="Generate mixed synthetic graph dataset.")
    parser.add_argument("--num-train", type=int, default=120)
    parser.add_argument("--num-val", type=int, default=40)
    parser.add_argument("--num-test", type=int, default=40)

    parser.add_argument("--n", type=int, default=1000, help="Number of nodes per graph.")

    # Which graph families to include (comma-separated)
    parser.add_argument(
        "--types",
        type=str,
        default="ba,er,ws,sbm,rgg,hk",
        help="Comma-separated list: ba,er,ws,sbm,rgg,hk",
    )

    # BA / HK params
    parser.add_argument("--m", type=int, default=3, help="BA/HK: edges attached per new node.")
    parser.add_argument("--hk-triad-p", type=float, default=0.3, help="HK: triangle formation prob.")

    # ER params
    parser.add_argument("--er-p", type=float, default=0.006, help="ER: edge probability p.")

    # WS params
    parser.add_argument("--ws-k", type=int, default=12, help="WS: each node connected to k nearest neighbors.")
    parser.add_argument("--ws-beta", type=float, default=0.1, help="WS: rewiring probability.")

    # SBM params
    parser.add_argument("--sbm-blocks", type=int, default=5, help="SBM: number of blocks.")
    parser.add_argument("--sbm-p-in", type=float, default=0.02, help="SBM: within-block prob.")
    parser.add_argument("--sbm-p-out", type=float, default=0.002, help="SBM: cross-block prob.")

    # RGG params
    parser.add_argument("--rgg-radius", type=float, default=0.05, help="RGG: connection radius in unit square.")

    parser.add_argument(
        "--base-dir",
        type=str,
        default=None,
        help="Base directory (default: <repo_root>/data/synthetic_graphs/multi)",
    )

    args = parser.parse_args()

    ROOT = Path(__file__).resolve().parents[1]
    base_dir = Path(args.base_dir) if args.base_dir is not None else (ROOT / "data" / "synthetic_graphs" / "multi")

    train_dir = base_dir / "train"
    val_dir = base_dir / "val"
    test_dir = base_dir / "test"
    for d in [train_dir, val_dir, test_dir]:
        ensure_dir(d)

    types = [t.strip() for t in args.types.split(",") if t.strip()]
    valid = {"ba", "er", "ws", "sbm", "rgg", "hk"}
    for t in types:
        if t not in valid:
            raise ValueError(f"Unknown type '{t}'. Valid: {sorted(valid)}")

    # deterministic seed counter
    seed_counter = 42

    def make_graph(gtype: str, seed: int) -> nx.Graph:
        if gtype == "ba":
            G = gen_ba(n=args.n, m=args.m, seed=seed)
        elif gtype == "er":
            G = gen_er(n=args.n, p=args.er_p, seed=seed)
        elif gtype == "ws":
            G = gen_ws(n=args.n, k=args.ws_k, p=args.ws_beta, seed=seed)
        elif gtype == "sbm":
            G = gen_sbm(n=args.n, num_blocks=args.sbm_blocks, p_in=args.sbm_p_in, p_out=args.sbm_p_out, seed=seed)
        elif gtype == "rgg":
            G = gen_rgg(n=args.n, radius=args.rgg_radius, seed=seed)
        elif gtype == "hk":
            G = gen_hk(n=args.n, m=args.m, p=args.hk_triad_p, seed=seed)
        else:
            raise ValueError(gtype)

        # Ensure simple undirected graph with 0..n-1 labels
        G = _relabel_to_0n(G)

        # Some generators can produce isolates; that’s fine, but ensure undirected simple graph
        if isinstance(G, nx.MultiGraph) or isinstance(G, nx.MultiDiGraph):
            G = nx.Graph(G)

        return G

    def generate_split(split_name: str, num_graphs: int, target_dir: Path):
        nonlocal seed_counter
        if num_graphs <= 0:
            return

        per_type = max(1, num_graphs // len(types))
        leftover = num_graphs - per_type * len(types)

        print(f"\nGenerating {num_graphs} graphs for split='{split_name}' into {target_dir}")
        idx = 0
        for t in types:
            count = per_type + (1 if leftover > 0 else 0)
            leftover = max(0, leftover - 1)

            for _ in tqdm(range(count), desc=f"{split_name}:{t}"):
                seed = seed_counter
                seed_counter += 1
                G = make_graph(t, seed=seed)
                out_path = target_dir / f"{t}_{idx}.gpickle"
                save_graph_gpickle(G, out_path)
                idx += 1

    generate_split("train", args.num_train, train_dir)
    generate_split("val", args.num_val, val_dir)
    generate_split("test", args.num_test, test_dir)

    print(f"\nDone! Saved dataset to: {base_dir}")


if __name__ == "__main__":
    main()
