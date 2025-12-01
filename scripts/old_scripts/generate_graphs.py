# scripts/generate_graphs.py

"""
Graph Generation Script
=======================

This script generates and saves a reproducible Barabási–Albert (BA) scale-free network
using NetworkX. These graphs are widely used in diffusion, influence propagation,
and network science research because they naturally form *hubs* through the process
of **preferential attachment**.

Why Barabási–Albert Graphs?
---------------------------
A BA(n, m) graph is constructed by starting with a small seed network and then
adding nodes one at a time. Each new node connects to `m` existing nodes with
probability proportional to their degree ("the rich get richer"). This produces:

- A **scale-free degree distribution** (~ k^(-3))
- Highly connected **hub nodes**
- Short average path lengths
- Structure similar to social, citation, and communication networks

These structural properties make BA graphs an excellent testbed for diffusion
models such as Independent Cascade (IC) and Linear Threshold (LT) because
activation tends to propagate through hubs.

Further Reading
---------------
- NetworkX BA generator docs:
  https://networkx.org/documentation/stable/reference/generated/networkx.generators.random_graphs.barabasi_albert_graph.html

- Barabási, A.-L. & Albert, R. (1999). *Emergence of Scaling in Random Networks.*
  Science, 286(5439), 509–512.
  https://barabasi.com/f/2013/04/19/science99.pdf

- Barabási's free online book **Network Science**:
  https://networksciencebook.com/

- “Scale-Free Networks (Yale Open Course)”:
  https://www.youtube.com/watch?v=mKXi1EoI1Vo

This script should be run *once* to generate a stable, reproducible dataset
stored in the `data/graphs/` directory.
"""

import os
import pickle
import networkx as nx


def ensure_dir(path: str):
    """
    Ensure that a directory exists, creating it if necessary.

    Args:
        path: Directory path to create if missing.
    """
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def generate_ba_graph(n: int = 1000, m: int = 3, seed: int = 42) -> nx.Graph:
    """
    Generate a Barabási–Albert scale-free graph using NetworkX.

    Args:
        n: Number of nodes in the graph.
        m: Number of edges to attach from each new node to existing nodes.
        seed: Random seed for reproducibility.

    Returns:
        A NetworkX Graph object representing the BA(n, m) network.
    """
    print(f"Generating BA graph with n={n}, m={m}, seed={seed} ...")
    return nx.barabasi_albert_graph(n=n, m=m, seed=seed)


def save_graph(G: nx.Graph, name: str):
    """
    Save a graph in both edge list and gpickle-like (pickle) formats.

    Args:
        G: The NetworkX graph object to save.
        name: Base filename (without extension) for saving.

    Output files:
        - data/graphs/{name}.edgelist  (human readable)
        - data/graphs/{name}.gpickle   (binary pickle, fast load)
    """
    base_dir = "data/graphs"
    ensure_dir(base_dir)

    edgelist_path = os.path.join(base_dir, f"{name}.edgelist")
    gpickle_path = os.path.join(base_dir, f"{name}.gpickle")

    print(f"Saving edge list to {edgelist_path} ...")
    nx.write_edgelist(G, edgelist_path, data=False)

    print(f"Saving pickle to {gpickle_path} ...")
    with open(gpickle_path, "wb") as f:
        pickle.dump(G, f, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"Done! Saved graph as:\n  {edgelist_path}\n  {gpickle_path}")


if __name__ == "__main__":
    G = generate_ba_graph(n=1000, m=3, seed=42)
    save_graph(G, "ba_1000_3")
