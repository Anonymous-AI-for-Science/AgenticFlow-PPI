"""Optimizer evaluation metrics: regret, budget violation, Pareto frontier.

  * regret          per-query quality gap between the oracle's chosen plan and the
                    policy's chosen plan, averaged (lower is better; oracle = 0).
  * budget_violation fraction of queries whose chosen plan's realized cost exceeds
                    the latency budget (lower is better).
  * pareto_frontier the (cost, quality) operating points of all policies, with the
                    non-dominated set marked, so the cost/quality trade-off is shown
                    rather than a single threshold.
"""

from __future__ import annotations

import statistics
from typing import Dict, List, Tuple

from .policies import Policy, evaluate_policy, OraclePolicy


def _feasible_quality(table, qid, plan, budget_ms):
    """Quality the system actually realizes for `plan` on `qid` under a hard budget:
    if the plan fits, its quality; otherwise the best budget-feasible fallback's."""
    tr = table[qid].get(plan.label())
    if tr is None:
        return None
    if budget_ms is None or tr.cost_ms <= budget_ms:
        return tr.quality
    feas = [t for t in table[qid].values() if t.cost_ms <= budget_ms]
    return (max(feas, key=lambda t: (t.quality, -t.cost_ms)).quality) if feas else tr.quality


def regret_vs_oracle(policy: Policy, oracle: OraclePolicy, table, qids, plans,
                     budget_ms: float = None) -> float:
    """Mean per-query quality gap between the oracle and the policy, BOTH scored under
    the same hard budget (reviewer W1). The oracle chooses the best budget-feasible
    plan, so its feasible quality is an upper bound and regret is always >= 0."""
    bm = budget_ms if budget_ms is not None else oracle.budget_ms
    total = 0.0; n = 0
    for qid in qids:
        oq = _feasible_quality(table, qid, oracle.choose(qid, plans), bm)
        pq = _feasible_quality(table, qid, policy.choose(qid, plans), bm)
        if oq is None or pq is None:
            continue
        total += max(0.0, oq - pq); n += 1
    return round(total / n, 4) if n else 0.0


def budget_violation_rate(res: Dict[str, List[float]], budget_ms: float) -> float:
    costs = res["cost_ms"]
    if not costs:
        return 0.0
    return round(sum(1 for c in costs if c > budget_ms) / len(costs), 4)


def summarize_policy(name: str, res: Dict[str, List[float]], budget_ms: float,
                     regret: float = None) -> Dict:
    q = res["quality"]; c = res["cost_ms"]
    out = {
        "policy": name,
        "mean_quality": round(statistics.mean(q), 4) if q else 0.0,
        "mean_cost_ms": round(statistics.mean(c), 4) if c else 0.0,
        "budget_violation": budget_violation_rate(res, budget_ms),
    }
    if regret is not None:
        out["regret_vs_oracle"] = regret
    return out


def pareto_frontier(points: List[Dict]) -> List[Dict]:
    """Mark non-dominated (cost down, quality up) operating points."""
    for pt in points:
        dominated = False
        for other in points:
            if other is pt:
                continue
            if (other["mean_cost_ms"] <= pt["mean_cost_ms"]
                    and other["mean_quality"] >= pt["mean_quality"]
                    and (other["mean_cost_ms"] < pt["mean_cost_ms"]
                         or other["mean_quality"] > pt["mean_quality"])):
                dominated = True; break
        pt["pareto_optimal"] = not dominated
    return points
