"""
Download and preprocess the SNAP email-Eu-core network.

Steps:
  - Download compressed edge list
  - Convert to undirected graph
  - Relabel nodes to 0..n-1
  - Save as .gpickle for use with existing pipeline

Output:
  data/real_graphs/email_eu_core/email_eu_core_undirected.gpickle
"""

import os
import gzip
import pickle
import urllib.request
import ssl
from pathlib import Path

import networkx as nx

ssl._create_default_https_context = ssl._create_unverified_context



URL = "https://snap.stanford.edu/data/email-Eu-core.txt.gz"


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def main():
    ROOT = Path(__file__).resolve().parents[1]
    out_dir = ROOT / "data" / "real_graphs" / "email_eu_core"
    ensure_dir(out_dir)

    gz_path = out_dir / "email-Eu-core.txt.gz"
    txt_path = out_dir / "email-Eu-core.txt"
    gp_path = out_dir / "email_eu_core_undirected.gpickle"

    print("[download] Fetching email-Eu-core...")
    urllib.request.urlretrieve(URL, gz_path)

    print("[extract] Decompressing...")
    with gzip.open(gz_path, "rt") as fin, open(txt_path, "w") as fout:
        for line in fin:
            if line.startswith("#") or not line.strip():
                continue
            fout.write(line)

    print("[build] Loading edge list...")
    G = nx.read_edgelist(
        txt_path,
        nodetype=int,
        create_using=nx.DiGraph(),
    )

    print("[process] Converting to undirected + relabeling...")
    G = G.to_undirected()
    # Remove self-loops (required for nx.core_number)
    G.remove_edges_from(nx.selfloop_edges(G))
    # Keep only the largest connected component (LCC)
    if not nx.is_connected(G):
        lcc_nodes = max(nx.connected_components(G), key=len)
        G = G.subgraph(lcc_nodes).copy()
    
    G = nx.convert_node_labels_to_integers(G, first_label=0)
    G = G.to_undirected()

    print(
        f"[stats] n={G.number_of_nodes()}, m={G.number_of_edges()}"
    )

    with open(gp_path, "wb") as f:
        pickle.dump(G, f, protocol=pickle.HIGHEST_PROTOCOL)

    print("[save] Saved to:", gp_path)


if __name__ == "__main__":
    main()
