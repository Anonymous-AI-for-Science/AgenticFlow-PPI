"""Dispatch policies over the plan space, including the oracle and the learned
cost-aware optimizer.

Each policy maps a query to a chosen Plan. We evaluate the chosen plan's REALIZED
cost and quality (from the traces) so policies are compared on ground truth.

Policies:
  * OraclePolicy        per-query best plan by realized quality s.t. realized
                        cost <= budget (the unattainable upper bound)
  * NeverRerankPolicy   cheapest symbolic plan
  * AlwaysRerankPolicy  always use the reranker (largest budget)
  * FixedOrderPolicy    one fixed plan for all queries
  * ThresholdPolicy     the prior 1-D admission rule (rerank iff predicted gain>0
                        and frontier<=budget) -- the baseline this phase improves on
  * LearnedOptimizer    enumerate plans, predict cost+quality, pick argmax predicted
                        quality subject to predicted cost <= budget (a real plan
                        search, not a binary threshold)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

import numpy as np

from .operators import Plan, enumerate_plans
from .trace import Trace, plan_query_features


def _trace_table(traces: List[Trace]) -> Dict[int, Dict[str, Trace]]:
    tab: Dict[int, Dict[str, Trace]] = {}
    for t in traces:
        tab.setdefault(t.qid, {})[t.plan.label()] = t
    return tab


class Policy:
    name = "policy"
    def choose(self, qid: int, plans: List[Plan]) -> Plan:
        raise NotImplementedError


@dataclass
class OraclePolicy(Policy):
    table: Dict[int, Dict[str, Trace]]
    budget_ms: float
    name: str = "oracle"
    def choose(self, qid, plans):
        cands = [self.table[qid][p.label()] for p in plans if p.label() in self.table[qid]]
        feasible = [t for t in cands if t.cost_ms <= self.budget_ms] or cands
        best = max(feasible, key=lambda t: (t.quality, -t.cost_ms))
        return best.plan


@dataclass
class FixedPlanPolicy(Policy):
    plan: Plan
    name: str = "fixed"
    def choose(self, qid, plans):
        return self.plan


@dataclass
class NeverRerankPolicy(Policy):
    name: str = "never-rerank"
    def choose(self, qid, plans):
        syms = [p for p in plans if not p.use_rerank]
        return min(syms, key=lambda p: (p.frontier_budget, p.expand_depth))


@dataclass
class AlwaysRerankPolicy(Policy):
    name: str = "always-rerank"
    def choose(self, qid, plans):
        rer = [p for p in plans if p.use_rerank]
        return max(rer, key=lambda p: p.frontier_budget)


@dataclass
class ThresholdPolicy(Policy):
    """The prior 1-D admission rule, lifted into the plan space: pick a fixed
    moderate plan, and flip the reranker on iff a gain predictor says >0 and the
    frontier is within budget. This is the baseline the learned optimizer improves on."""
    gain_fn: Callable[[int], float]
    frontier_fn: Callable[[int], int]
    frontier_budget: int
    base_depth: int = 3
    base_budget: int = 50
    name: str = "threshold-1d"
    def choose(self, qid, plans):
        use_r = self.gain_fn(qid) > 0.0 and self.frontier_fn(qid) <= self.frontier_budget
        cand = [p for p in plans if p.use_rerank == use_r and p.expand_depth == self.base_depth
                and p.frontier_budget == self.base_budget]
        return cand[0] if cand else plans[0]


@dataclass
class LearnedOptimizer(Policy):
    """Real plan-space optimizer: for each candidate plan, predict cost and quality;
    choose the plan maximizing predicted quality subject to predicted cost <= budget.
    Falls back to the cheapest plan if none is predicted feasible."""
    h: object
    cost_model: object
    quality_model: object
    budget_ms: float
    name: str = "learned-optimizer"
    def choose(self, qid, plans):
        scored = []
        for p in plans:
            x = plan_query_features(self.h, qid, p)
            c = self.cost_model.predict(x); q = self.quality_model.predict(x)
            scored.append((p, c, q))
        feasible = [(p, c, q) for (p, c, q) in scored if c <= self.budget_ms]
        if not feasible:
            return min(scored, key=lambda t: t[1])[0]
        return max(feasible, key=lambda t: (t[2], -t[1]))[0]


def evaluate_policy(policy: Policy, table: Dict[int, Dict[str, Trace]], qids: List[int],
                    plans: List[Plan], budget_ms: float = None,
                    violation_penalty: bool = True) -> Dict[str, List[float]]:
    """Return realized quality and cost for the policy's chosen plan per query.

    Unified hard-budget scoring (design rationale): when a budget is given and the chosen
    plan exceeds it, the query is counted as a budget violation AND its scored quality
    is the budget-feasible fallback's quality (the best plan whose cost <= budget),
    i.e. the system must fall back to stay within budget. This makes the oracle a true
    upper bound and regret well-posed: no policy can outscore the oracle by violating
    the budget, because violations are charged the feasible-fallback quality.
    """
    quals, costs, raw_quals, violated = [], [], [], []
    for qid in qids:
        plan = policy.choose(qid, plans)
        tr = table[qid].get(plan.label())
        if tr is None:
            continue
        raw_quals.append(tr.quality)
        if budget_ms is not None and tr.cost_ms > budget_ms and violation_penalty:
            # must fall back to the best plan that fits the budget
            feas = [t for t in table[qid].values() if t.cost_ms <= budget_ms]
            fb = max(feas, key=lambda t: (t.quality, -t.cost_ms)) if feas else tr
            quals.append(fb.quality); costs.append(fb.cost_ms); violated.append(1)
        else:
            quals.append(tr.quality); costs.append(tr.cost_ms)
            violated.append(1 if (budget_ms is not None and tr.cost_ms > budget_ms) else 0)
    return {"quality": quals, "cost_ms": costs, "raw_quality": raw_quals, "violated": violated}
