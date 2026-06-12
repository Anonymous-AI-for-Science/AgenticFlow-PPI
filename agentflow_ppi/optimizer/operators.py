"""Physical operator abstraction for the AgentFlow-PPI optimizer.

A query is answered by a PLAN: an ordered pipeline of physical operators, each with
a parameterization, a measurable execution cost, and a contribution to answer
quality. This replaces the single 1-D admission test with a real plan space the
optimizer searches over.

Operators:
  * Expand(depth)        typed neighborhood expansion to `depth` hops (cost grows
                         with frontier size; sets the candidate pool)
  * ReachFilter          exact SHRC reachability prune of the candidate pool
                         (cheap; removes unreachable candidates -- never hurts quality)
  * SymbolicRank         exact path-score ranking (cheap; baseline quality)
  * NeuralRerank         learned reranker over the candidate pool (expensive; may
                         help or hurt depending on the query)
  * Aggregate(k)         take top-k as the answer

A Plan fixes: expansion depth, whether the reranker is used, and the frontier
budget at which expansion is truncated. The cost of a plan on a query is the sum of
its operator costs; the quality is the F1@k of the ranking the plan produces.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

# Per-operator cost coefficients (ms). These are the cost-model parameters the
# optimizer reasons about; they are explicit and editable rather than hidden.
COST = {
    "expand_per_cand_ms": 0.01,     # typed expansion, per candidate touched
    "reach_per_cand_ms": 0.004,     # SHRC reachability filter, per candidate
    "symbolic_per_cand_ms": 0.05,   # exact path-score ranking, per candidate
    "rerank_fixed_ms": 0.9,         # neural reranker fixed overhead (configurable)
    "rerank_per_cand_ms": 0.02,     # neural reranker, per candidate scored
    "aggregate_ms": 0.01,
}


@dataclass(frozen=True)
class Plan:
    """A concrete physical plan for one query."""
    expand_depth: int          # 2 or 3
    use_rerank: bool           # admit the neural reranker?
    frontier_budget: int       # truncate the candidate pool to this size
    k: int = 2                 # top-k aggregation

    def label(self) -> str:
        return f"d{self.expand_depth}/{'R' if self.use_rerank else 'S'}/b{self.frontier_budget}"


def plan_cost_ms(plan: Plan, frontier_size: int, rerank_fixed_ms: float = None) -> float:
    """Analytic cost of executing `plan` on a query whose expanded frontier has
    `frontier_size` candidates (after truncation to the plan's budget)."""
    n = min(frontier_size, plan.frontier_budget)
    rf = COST["rerank_fixed_ms"] if rerank_fixed_ms is None else rerank_fixed_ms
    cost = (COST["expand_per_cand_ms"] * frontier_size      # expansion touches the full frontier
            + COST["reach_per_cand_ms"] * n
            + COST["symbolic_per_cand_ms"] * n
            + COST["aggregate_ms"])
    if plan.use_rerank:
        cost += rf + COST["rerank_per_cand_ms"] * n
    return cost


def enumerate_plans(depths=(2, 3), budgets=(20, 50, 200), ks=(2,)) -> List[Plan]:
    """Bounded plan space: depth x rerank{on,off} x frontier budget x k.
    Returns the candidate plans the optimizer chooses among per query."""
    plans = []
    for d in depths:
        for b in budgets:
            for use_r in (False, True):
                for k in ks:
                    plans.append(Plan(expand_depth=d, use_rerank=use_r, frontier_budget=b, k=k))
    return plans
