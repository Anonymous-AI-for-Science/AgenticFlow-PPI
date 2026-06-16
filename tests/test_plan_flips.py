"""Test the plan-flip analysis (R1-O3.A): the cost objective must select plans that a
fixed order would not, on a non-trivial fraction of queries, and the flips must be
realized (cost/quality come from executed traces)."""

import importlib.util
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


def _load(name):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    return mod


def test_objective_selects_different_plan_than_fixed_order():
    pf = _load("benchmark_plan_flips")
    import numpy as np
    from agentflow_ppi.optimizer.operators import enumerate_plans
    from agentflow_ppi.optimizer.trace import collect_traces
    from agentflow_ppi.optimizer.cost_quality import train_cost_quality
    from agentflow_ppi.optimizer import policies as P
    from agentflow_ppi.eval.harness import build_harness_large, train_reranker

    h = build_harness_large(num_pathways=20, pathway_len=8, informative_fraction=0.5,
                            seed=7, max_hops=3)
    plans = enumerate_plans()
    qids = list(range(len(h.pools)))
    model = train_reranker(h, qids, 7)
    traces = collect_traces(h, model, qids, plans)
    cost_model, quality_model = train_cost_quality(h, traces)
    table = pf._trace_table(traces)

    fixed = P.FixedPlanPolicy(plan=pf.FIXED_PLAN)
    opt = P.LearnedOptimizer(h=h, cost_model=cost_model, quality_model=quality_model,
                             budget_ms=pf.BUDGET_MS)
    flips = sum(1 for qid in qids
                if opt.choose(qid, plans).label() != fixed.choose(qid, plans).label())
    # the objective must disagree with the fixed order on a substantial fraction
    assert flips / len(qids) > 0.3
    # and the chosen plans must exist in the executed trace table (realized, not predicted)
    for qid in qids[:20]:
        assert opt.choose(qid, plans).label() in table[qid]
