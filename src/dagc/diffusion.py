# src/dagc/diffusion.py

"""
Diffusion models (IC, LT, etc.).

For now we implement:
- Independent Cascade (IC) diffusion
- A helper to estimate expected spread via Monte Carlo
"""

from typing import List, Set, Any, Optional, Literal
import random

import networkx as nx



class DiffusionResult:
    """
    Container for the result of a single diffusion run.

    Attributes:
        activated_by_step: list of sets, where activated_by_step[t]
            is the set of nodes newly activated at step t (t starts at 0 for seeds).
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
    if rng is None:
        rng = random.Random()

    # Ensure we don't mutate the original seed set
    current_active: Set[Any] = set(seed_set)
    activated_by_step: List[Set[Any]] = [set(seed_set)]

    # Track which nodes have *ever* been active
    ever_active: Set[Any] = set(seed_set)

    step = 0
    while True:
        if max_steps is not None and step >= max_steps:
            break

        newly_active: Set[Any] = set()

        # Only nodes that were newly activated in the previous step
        # get to try activating neighbors now.
        frontier = activated_by_step[-1]
        if not frontier:
            break  # no one to spread from

        for u in frontier:
            for v in graph.neighbors(u):
                if v in ever_active:
                    continue  # already active from a previous step
                if rng.random() < activation_prob:
                    newly_active.add(v)

        if not newly_active:
            break

        activated_by_step.append(newly_active)
        ever_active |= newly_active
        step += 1

    return DiffusionResult(activated_by_step)


def estimate_spread(
    graph: nx.Graph,
    seed_set: Set[Any],
    model: Literal["IC"] = "IC",
    num_runs: int = 50,
    activation_prob: float = 0.1,
    max_steps: Optional[int] = None,
    rng_seed: Optional[int] = None,
) -> float:
    """
    Estimate expected spread (expected number of activated nodes)
    under a given diffusion model via Monte Carlo.

    Currently supported:
        - model="IC": Independent Cascade

    Args:
        graph: Input graph.
        seed_set: Seed nodes.
        model: Diffusion model name.
        num_runs: Number of Monte Carlo runs.
        activation_prob: IC activation probability.
        max_steps: Optional limit on number of steps.
        rng_seed: Optional seed for reproducibility.

    Returns:
        Estimated expected number of activated nodes (float).
    """
    if model != "IC":
        raise ValueError(f"Unsupported diffusion model: {model}")

    rng = random.Random(rng_seed)
    total = 0.0

    for _ in range(num_runs):
        # Use a separate RNG state per run to be explicit
        run_rng = random.Random(rng.random())
        result = run_ic_diffusion(
            graph,
            seed_set,
            activation_prob=activation_prob,
            max_steps=max_steps,
            rng=run_rng,
        )
        total += result.num_activated()

    return total / num_runs


# We should check to see if the optimal seed set for diffusion is the same? or how different is it? 
# - Seed set of what? 
# - Optimal 5 node seed set.. Optial 10 node seed set? 
#   - Need a way to calculate optimal seed set...
#

# Random Notes; 
#  seed = [0,1,2,3]
    
#     [
        
#         0
#         ..
#         n nodes
#     ]
    
#     Orginal graph ..
    
#     original_graph_results = [
#         [1,1,1,1,0,0,0,0...n] = timestep 1
#         [1,1,1,1,0,0,0,1...n] = timestep 2 -- after 1 round
#         [1,1,1,1,0,0,0,1...n] = timestep 2 -- after 1 round
#         [1,1,1,1,0,0,0,1...n] = timestep 2 -- after 1 round
#         [1,1,1,1,0,0,0,1...n] = timestep 2 -- after 1 round
#         [1,1,1,1,0,0,0,1...n] = timestep 2 -- after 1 round
#         ..
#         [1,1,1,1,0,0,0,1...n] = timestep 2 -- after 100 rounds
#     ] * 100 times. 
    
#     []
    
#     Graph 
#     Number of simulations  = 100 
    
#     for number in number_of_simulations:
#         results_of_one_sim = run_ic_diffusion(Graph, num_steps =100, probabilty_of_activation=0.1, seed = number)
#         all_results += results_of_one_sim
    
#     all_results == Combine... 

        
    
#     sparse graph = Sparsify(orignal_grapm)
    
    
#     original_graph_results = [
#         [1,1,1,1,0,0,0,0...n] = timestep 1
#         [1,1,1,1,0,1,0,0...n] = timestep 2 -- after 1 round
#         [1,1,1,1,0,0,0,1...n] = timestep 2 -- after 1 round
#         [1,1,1,1,0,0,0,1...n] = timestep 2 -- after 1 round
#         [1,1,1,1,0,0,0,1...n] = timestep 2 -- after 1 round
#         [1,1,1,1,0,0,0,1...n] = timestep 2 -- after 1 round
#         ..
#         [1,1,1,1,0,0,0,1...n] = timestep 2 -- after 100 rounds
#     ]

#     Simulation with sparse Graphs: 
        
#     sparse graph_results = .... 
    
#     calcualte_results = claculate metrics( original_graph_results, sparse_graph_results)
    
#     - a few numbers... 
    