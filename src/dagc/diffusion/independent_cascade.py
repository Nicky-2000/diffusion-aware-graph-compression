# src/dagc/diffusion.py

"""
Diffusion models (IC, LT, etc.).

For now we implement:
- Independent Cascade (IC) diffusion for a single simulation run.
"""

from typing import List, Set, Any, Optional
import random

import networkx as nx


class DiffusionResult:
    """
    Container for the result of a single diffusion run.

    Attributes:
        activated_by_step: list of sets, where activated_by_step[t]
            is the set of nodes newly activated at step t (t=0 is the seeds).
        all_activated: set of all nodes that were ever activated.
    """

    def __init__(self, activated_by_step: List[Set[Any]]):
        self.activated_by_step: List[Set[Any]] = activated_by_step
        all_nodes: Set[Any] = set()
        for s in activated_by_step:
            all_nodes |= s
        self.all_activated: Set[Any] = all_nodes

    def num_activated(self) -> int:
        """Total number of activated nodes."""
        return len(self.all_activated)

from typing import List, Set, Any, Optional, Literal
import random
import math
import networkx as nx


def run_ic_diffusion(
    graph: nx.Graph,
    seed_set: Set[Any],
    activation_prob: float = 0.1,
    max_steps: Optional[int] = None,
    rng: Optional[random.Random] = None,
    edge_prob_mode: Literal["weighted_attempts", "direct_prob"] = "weighted_attempts",
    edge_prob_attr: str = "p",          # used when edge_prob_mode="direct_prob"
    edge_weight_attr: str = "weight",   # used when edge_prob_mode="weighted_attempts"
    clamp_eps: float = 1e-9,
) -> DiffusionResult:
    """
    Run one Independent Cascade (IC) diffusion simulation.

    Two supported modes:
      1) weighted_attempts (default):
           p_uv = 1 - (1 - activation_prob) ** w_uv
         where w_uv is read from edge attribute `edge_weight_attr` (default "weight").

      2) direct_prob:
           p_uv = graph[u][v][edge_prob_attr]
         (fallback to activation_prob if missing). This is the mode for the GNN,
         where the model predicts per-edge IC probabilities.

    Args:
        activation_prob: Base probability p0 (used directly in direct_prob only as fallback,
                         and in weighted_attempts as the base attempt prob).
    """
    if rng is None:
        rng = random.Random()

    seed_set = set(seed_set)

    activated_by_step: List[Set[Any]] = [set(seed_set)]
    ever_active: Set[Any] = set(seed_set)

    step = 0
    while True:
        if max_steps is not None and step >= max_steps:
            break

        frontier = activated_by_step[-1]
        if not frontier:
            break

        newly_active: Set[Any] = set()

        for u in frontier:
            for v in graph.neighbors(u):
                if v in ever_active:
                    continue

                if edge_prob_mode == "direct_prob":
                    # GNN: edge stores p_uv directly (in [0,1])
                    # p_uv = graph[u][v].get(edge_prob_attr, activation_prob)
                    p_uv = activation_prob

                elif edge_prob_mode == "weighted_attempts":
                    # Effective resistance / weighted IC: use the "attempts" mapping
                    w_uv = graph[u][v].get(edge_weight_attr, 1.0)
                    # clamp for numerical sanity
                    p0 = min(max(activation_prob, clamp_eps), 1.0 - clamp_eps)
                    w_uv = max(float(w_uv), 0.0)
                    p_uv = 1.0 - (1.0 - p0) ** w_uv

                else:
                    raise ValueError(f"Unknown edge_prob_mode={edge_prob_mode}")

                # final clamp just in case
                p_uv = min(max(float(p_uv), 0.0), 1.0)

                if rng.random() < p_uv:
                    newly_active.add(v)

        if not newly_active:
            break

        activated_by_step.append(newly_active)
        ever_active |= newly_active
        step += 1

    return DiffusionResult(activated_by_step)
