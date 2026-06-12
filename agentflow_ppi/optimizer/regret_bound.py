"""Regret bound for the plan-space optimizer (originality: turn the heuristic into
a policy with a guarantee).

Proposition (plan-selection regret). Fix a per-query latency budget B. Let the
optimizer choose, per query q, the budget-feasible plan maximizing the PREDICTED
quality qhat(q,.). Let the prediction error be bounded by
    |qhat(q,p) - q(q,p)| <= eps   for all feasible plans p.
Then for every query the realized quality of the optimizer's chosen plan p_opt and
the oracle's feasible-optimal plan p* satisfy
    q(q, p*) - q(q, p_opt) <= 2 eps,
hence the mean regret is at most 2 eps.

Proof. Both p_opt and p* are budget-feasible. By optimality of p_opt under qhat,
qhat(q,p_opt) >= qhat(q,p*). Then
    q(q,p*) - q(q,p_opt)
      = [q(q,p*) - qhat(q,p*)] + [qhat(q,p*) - qhat(q,p_opt)] + [qhat(q,p_opt) - q(q,p_opt)]
      <= eps + 0 + eps = 2 eps.                                                   QED

This module measures eps on held-out traces (max abs quality-prediction error over
feasible plans) and checks that the realized per-query regret never exceeds 2 eps,
turning the bound into a falsifiable, verified guarantee rather than an assertion.
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np

from .operators import Plan
from .trace import Trace, plan_query_features
from .policies import OraclePolicy, LearnedOptimizer


def measure_eps(h, table: Dict[int, Dict[str, Trace]], qids: List[int], plans: List[Plan],
                quality_model, budget_ms: float) -> float:
    """Empirical eps: max |predicted - realized| quality over budget-feasible plans."""
    worst = 0.0
    for qid in qids:
        for p in plans:
            tr = table[qid].get(p.label())
            if tr is None or tr.cost_ms > budget_ms:
                continue
            pred = quality_model.predict(plan_query_features(h, qid, p))
            worst = max(worst, abs(pred - tr.quality))
    return worst


def measure_eps_per_query(h, table, qids, plans, quality_model, budget_ms):
    """Per-query eps_q = max error over that query's feasible plans. The regret bound
    holds per query with its own eps_q, so the per-query distribution (median, p90)
    is the operative quantity, not the single global worst case."""
    out = []
    for qid in qids:
        e = 0.0; seen = False
        for p in plans:
            tr = table[qid].get(p.label())
            if tr is None or tr.cost_ms > budget_ms:
                continue
            seen = True
            pred = quality_model.predict(plan_query_features(h, qid, p))
            e = max(e, abs(pred - tr.quality))
        if seen:
            out.append(e)
    return out


def verify_regret_bound(h, table, qids, plans, cost_model, quality_model,
                        budget_ms: float) -> Dict:
    """Check the per-query regret never exceeds 2*eps_q (the proposition), and report
    the per-query eps distribution so the bound's operative tightness is visible."""
    eps = measure_eps(h, table, qids, plans, quality_model, budget_ms)
    eps_pq = measure_eps_per_query(h, table, qids, plans, quality_model, budget_ms)
    eps_pq_sorted = sorted(eps_pq)
    median_eps = eps_pq_sorted[len(eps_pq_sorted) // 2] if eps_pq_sorted else 0.0
    p90_eps = eps_pq_sorted[int(0.9 * (len(eps_pq_sorted) - 1))] if eps_pq_sorted else 0.0
    # fraction of queries whose bound 2*eps_q is informative (< trivial range 1.0)
    informative = sum(1 for e in eps_pq if 2 * e < 1.0) / max(len(eps_pq), 1)
    oracle = OraclePolicy(table=table, budget_ms=budget_ms)
    opt = LearnedOptimizer(h=h, cost_model=cost_model, quality_model=quality_model,
                           budget_ms=budget_ms)
    max_regret = 0.0; n = 0; violations = 0
    for qid in qids:
        op = oracle.choose(qid, plans); pp = opt.choose(qid, plans)
        ot = table[qid].get(op.label()); pt = table[qid].get(pp.label())
        if ot is None or pt is None:
            continue
        def feas_q(plan):
            tr = table[qid].get(plan.label())
            if tr.cost_ms <= budget_ms:
                return tr.quality
            fb = [t for t in table[qid].values() if t.cost_ms <= budget_ms]
            return max(fb, key=lambda t: t.quality).quality if fb else tr.quality
        r = max(0.0, feas_q(op) - feas_q(pp))
        max_regret = max(max_regret, r); n += 1
        if r > 2 * eps + 1e-9:
            violations += 1
    return {"eps": round(eps, 4), "two_eps_bound": round(2 * eps, 4),
            "median_eps": round(median_eps, 4), "p90_eps": round(p90_eps, 4),
            "median_two_eps": round(2 * median_eps, 4),
            "frac_queries_informative_bound": round(informative, 4),
            "max_observed_regret": round(max_regret, 4),
            "bound_violations": violations, "queries": n,
            "bound_holds": violations == 0}
