import os
import pickle
import networkx as nx

def read_graph(path: str, fmt: str = "edge_list") -> nx.Graph:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Graph file not found: {path}")

    if fmt == "edge_list":
        # cast nodes to int
        G = nx.read_edgelist(path, nodetype=int)
        return G

    elif fmt == "gpickle":
        with open(path, "rb") as f:
            G = pickle.load(f)
        if not isinstance(G, nx.Graph):
            raise ValueError("Loaded object is not a NetworkX graph.")
        return G

    else:
        raise ValueError(f"Unsupported graph format: {fmt}")
