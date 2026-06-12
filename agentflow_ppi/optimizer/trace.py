"""Execute plans on queries and collect (cost, quality) traces.

For each query and each candidate plan we execute the pipeline and record:
  * frontier_size       candidates after expansion (before budget truncation)
  * cost_ms             analytic plan cost (operators.plan_cost_ms)
  * quality             F1@k of the plan's ranking against the gold positives
  * plan + query features for the cost/quality models

These per-(query,plan) traces are the supervised signal for the learned cost and
quality models, and the ground truth for the oracle policy.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from agentflow_ppi.eval.harness import (symbolic_order, rerank, f1_at_k, path_score)
from .operators import Plan, plan_cost_ms, enumerate_plans


@dataclass
class Trace:
    qid: int
    plan: Plan
    frontier_size: int
    cost_ms: float
    quality: float
    informative: bool


def _expand_pool(h, p, depth: int) -> List[int]:
    """Typed expansion to `depth` hops from the source, intersected with the query's
    candidate pool (which already encodes reachability)."""
    # the harness pool is the depth-3 reachable set; emulate shallower depth by a
    # BFS truncation over typed_adj
    seen = {p.s}; frontier = [p.s]; reached = set()
    for _ in range(depth):
        nxt = []
        for u in frontier:
            for v, _m, _c in h.typed_adj.get(u, []):
                if v not in seen:
                    seen.add(v); reached.add(v); nxt.append(v)
        frontier = nxt
    return [c for c in p.cands if c in reached]


def execute_plan(h, model, qid: int, plan: Plan, rerank_fixed_ms: float = None) -> Trace:
    p = h.pools[qid]
    pool = _expand_pool(h, p, plan.expand_depth)
    if not pool:
        pool = list(p.cands)  # depth-2 may miss; fall back to the full reachable pool
    frontier_size = len(pool)
    cand = pool[:plan.frontier_budget]
    if plan.use_rerank and model is not None:
        ranked = rerank(h, model, p.s, p.t, p.modality, cand)
    else:
        ranked = sorted(cand, key=lambda v: path_score(h.typed_adj, p.s, v, p.t), reverse=True)
    quality = f1_at_k(ranked, p.positives, k=plan.k)
    cost = plan_cost_ms(plan, frontier_size, rerank_fixed_ms=rerank_fixed_ms)
    informative = bool(h.informative and qid in h.informative)
    return Trace(qid=qid, plan=plan, frontier_size=frontier_size,
                 cost_ms=cost, quality=quality, informative=informative)


def collect_traces(h, model, qids: List[int], plans: Optional[List[Plan]] = None,
                   rerank_fixed_ms: float = None) -> List[Trace]:
    plans = plans or enumerate_plans()
    out: List[Trace] = []
    for qid in qids:
        for plan in plans:
            out.append(execute_plan(h, model, qid, plan, rerank_fixed_ms=rerank_fixed_ms))
    return out


def plan_query_features(h, qid: int, plan: Plan) -> np.ndarray:
    """Features for the cost/quality models: query-level signal + plan parameters."""
    from agentflow_ppi.eval.harness import query_gain_features
    qf = query_gain_features(h, h.pools[qid])
    pf = np.array([plan.expand_depth, 1.0 if plan.use_rerank else 0.0,
                   plan.frontier_budget, plan.k], dtype=np.float64)
    return np.concatenate([qf, pf])
