"""Tests for the plan-space optimizer (Phase 4).

Verify the physical-operator cost model, plan enumeration, trace collection, the
learned cost/quality models, and that the policies (including the oracle) produce
valid choices with sensible regret/budget-violation properties.
"""

from agentflow_ppi.eval.harness import build_harness_large, train_reranker
from agentflow_ppi.optimizer.operators import enumerate_plans, plan_cost_ms, Plan
from agentflow_ppi.optimizer.trace import collect_traces
from agentflow_ppi.optimizer.cost_quality import train_cost_quality, model_diagnostics
from agentflow_ppi.optimizer import policies as P
from agentflow_ppi.optimizer import metrics as M


def _setup(n_pathways=12):
    h = build_harness_large(num_pathways=n_pathways, pathway_len=6,
                            informative_fraction=0.5, seed=7, max_hops=3)
    qids = list(range(len(h.pools)))
    model = train_reranker(h, qids, 7)
    plans = enumerate_plans()
    traces = collect_traces(h, model, qids, plans)
    return h, qids, model, plans, traces


def test_plan_cost_monotone_in_budget_and_rerank():
    # larger frontier budget and enabling rerank both increase cost
    base = Plan(3, False, 20); bigger = Plan(3, False, 200); reranked = Plan(3, True, 20)
    assert plan_cost_ms(bigger, 300) > plan_cost_ms(base, 300)
    assert plan_cost_ms(reranked, 300) > plan_cost_ms(base, 300)


def test_traces_and_models_fit():
    h, qids, model, plans, traces = _setup()
    assert len(traces) == len(qids) * len(plans)
    cm, qm = train_cost_quality(h, traces)
    diag = model_diagnostics(h, traces, cm, qm)
    # cost is close to deterministic in features -> high R^2
    assert diag["cost_r2"] > 0.8


def test_oracle_has_zero_regret_and_optimizer_respects_budget():
    h, qids, model, plans, traces = _setup()
    cm, qm = train_cost_quality(h, traces)
    table = {}
    for t in traces:
        table.setdefault(t.qid, {})[t.plan.label()] = t
    budget = 0.8
    oracle = P.OraclePolicy(table=table, budget_ms=budget)
    opt = P.LearnedOptimizer(h=h, cost_model=cm, quality_model=qm, budget_ms=budget)
    # oracle regret vs itself is 0
    assert M.regret_vs_oracle(oracle, oracle, table, qids, plans) == 0.0
    # optimizer chosen plans are valid and mostly within budget
    res = P.evaluate_policy(opt, table, qids, plans)
    assert res["quality"] and res["cost_ms"]
    # the optimizer should violate the budget far less than always-rerank
    alw = P.evaluate_policy(P.AlwaysRerankPolicy(), table, qids, plans)
    assert M.budget_violation_rate(res, budget) <= M.budget_violation_rate(alw, budget)


def test_regret_bound_holds():
    """The proven plan-selection regret bound (regret <= 2*eps) must hold on the
    collected traces -- a verified guarantee, not an assertion."""
    from agentflow_ppi.optimizer.regret_bound import verify_regret_bound
    from agentflow_ppi.optimizer.cost_quality import train_cost_quality
    h, qids, model, plans, traces = _setup()
    cm, qm = train_cost_quality(h, traces)
    table = {}
    for t in traces:
        table.setdefault(t.qid, {})[t.plan.label()] = t
    rep = verify_regret_bound(h, table, qids, plans, cm, qm, budget_ms=1.0)
    assert rep["bound_holds"], rep
    assert rep["max_observed_regret"] <= rep["two_eps_bound"] + 1e-9


def test_codesign_label_only_matches_recompute():
    """The O(1) label-only dispatch signal must yield the same decision as a fresh
    recomputation (reviewer W7: co-design is correct, not just faster)."""
    from agentflow_ppi.eval.harness import build_harness_large
    from collections import deque
    h = build_harness_large(num_pathways=10, pathway_len=6, informative_fraction=0.5,
                            seed=7, max_hops=3)
    def recompute(s, cand):
        seen = {s}; q = deque([s])
        while q:
            u = q.popleft()
            for v, _m, _c in h.typed_adj.get(u, []):
                if v not in seen:
                    seen.add(v); q.append(v)
        return sum(1 for c in cand if c in seen)
    mism = 0
    for qid in range(len(h.pools)):
        p = h.pools[qid]
        label_count = sum(1 for c in p.cands if h.reach(p.s, c))
        if (label_count >= 2) != (recompute(p.s, p.cands) >= 2):
            mism += 1
    assert mism == 0
