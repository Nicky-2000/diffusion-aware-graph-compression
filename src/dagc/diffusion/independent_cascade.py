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


def run_ic_diffusion(
    graph: nx.Graph,
    seed_set: Set[Any],
    activation_prob: float = 0.1,
    max_steps: Optional[int] = None,
    rng: Optional[random.Random] = None,
) -> DiffusionResult:
    """
    Run one Independent Cascade (IC) diffusion simulation.

    Model:
        - Start with seed_set active at step 0.
        - At each step t, nodes that became active at step t
          get *one chance* to activate each currently inactive neighbor
          with probability `activation_prob`.
        - Process continues until no new activations occur, or max_steps is reached.

    Args:
        graph: Input NetworkX graph.
        seed_set: Initial active nodes.
        activation_prob: Probability of activation along each edge.
        max_steps: Optional max number of steps to run. If None, run until no change.
        rng: Optional random.Random instance for reproducibility.

    Returns:
        DiffusionResult containing activation by step and total activated nodes.
    """
    
    # for edge in graph.edges():
    weight_max = 0.0
    for u, v in graph.edges():
        # find the greatest weight in the graph
        if "weight" not in graph[u][v]:
            graph[u][v]["weight"] = 1.0
        weight_max = max(weight_max, graph[u][v]["weight"])
            
    if rng is None:
        rng = random.Random()

    # copy seed set so we don't mutate input
    seed_set = set(seed_set)

    # step 0 = seeds
    activated_by_step: List[Set[Any]] = [set(seed_set)]
    ever_active: Set[Any] = set(seed_set)

    step = 0
    while True:
        if max_steps is not None and step >= max_steps:
            break

        frontier = activated_by_step[-1]
        if not frontier:
            break  # nothing left to spread from

        newly_active: Set[Any] = set()

        for u in frontier:
            for v in graph.neighbors(u):
                if v in ever_active:
                    continue  # already active before
                if rng.random() < 1-(1-activation_prob)**graph[u][v].get("weight", 1.0):
                    newly_active.add(v)

        if not newly_active:
            break

        activated_by_step.append(newly_active)
        ever_active |= newly_active
        step += 1

    return DiffusionResult(activated_by_step)


