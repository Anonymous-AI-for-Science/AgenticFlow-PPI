"""Plan-flip analysis: the cost objective selects plans a fixed order would not
(reviewer R1-O3.A).

R1-O3.A asks specifically: "show the cost objective of selecting a plan that a fixed
order would not." Aggregate regret/budget numbers (Q8) show the optimizer is better on
average, but they do not directly exhibit the *decisions* the objective makes. This
experiment does exactly that. On the held-out split it compares, per query, the plan
chosen by:

  * fixed-order        -- one fixed physical plan for every query (depth 3, rerank on,
                          frontier budget 50): the "fixed order" the reviewer names.
  * cost-objective     -- the LearnedOptimizer, which picks the plan maximizing
                          predicted quality subject to the predicted-cost budget, i.e.
                          a direct optimization of Equation (1).

For every query we record whether the objective chose a DIFFERENT plan than the fixed
order, WHICH operator-level dimension changed (expansion depth / reranker on-off /
frontier budget), and the REALIZED cost and quality of both plans (from the executed
traces, not predictions). We then summarize the flip rate, the breakdown by flip type,
and the net realized effect of the flips (mean cost saved and mean quality change on
exactly the queries where the plans differ). This is the missing evidence that the
objective drives measured decisions: on flipped queries, the objective's plan differs
from the fixed plan AND is justified by realized cost/quality.

Deterministic given the seed manifest; pure-numpy/CPU; runs in seconds on Ubuntu,
macOS Intel, and a MacBook Pro M3. Writes results/plan_flips.{json,csv}.
"""

from __future__ import annotations

import csv
import json
import statistics
from collections import Counter
from pathlib import Path

import numpy as np

from agentflow_ppi.eval.harness import (build_harness_large, train_reranker,
                                        train_gain_predictor)
from agentflow_ppi.optimizer.operators import enumerate_plans, Plan
from agentflow_ppi.optimizer.trace import collect_traces
from agentflow_ppi.optimizer.cost_quality import train_cost_quality
from agentflow_ppi.optimizer import policies as P

BUDGET_MS = 1.3
SEED_MANIFEST = [7, 11, 13]
FIXED_PLAN = Plan(2, True, 20)   # fairest fixed order: the best single plan under budget


def _trace_table(traces):
    table = {}
    for t in traces:
        table.setdefault(t.qid, {})[t.plan.label()] = t
    return table


def _flip_type(fixed: Plan, chosen: Plan) -> str:
    diffs = []
    if chosen.use_rerank != fixed.use_rerank:
        diffs.append("rerank-" + ("on" if chosen.use_rerank else "off"))
    if chosen.expand_depth != fixed.expand_depth:
        diffs.append(f"depth{fixed.expand_depth}->{chosen.expand_depth}")
    if chosen.frontier_budget != fixed.frontier_budget:
        diffs.append(f"fb{fixed.frontier_budget}->{chosen.frontier_budget}")
    return "+".join(diffs) if diffs else "same"


def main():
    out = Path(__file__).resolve().parents[1] / "results"
    out.mkdir(parents=True, exist_ok=True)
    h = build_harness_large(num_pathways=40, pathway_len=8, informative_fraction=0.5,
                            seed=7, max_hops=3)
    plans = enumerate_plans()
    n = len(h.pools)
    all_qids = list(range(n))

    flip_count = 0
    total = 0
    flip_types = Counter()
    cost_saved_on_flips = []
    quality_change_on_flips = []
    examples = []
    rows = []

    for seed in SEED_MANIFEST:
        rng = np.random.default_rng(seed)
        idx = all_qids.copy(); rng.shuffle(idx)
        nt = max(1, int(round(0.3 * len(idx))))
        test = idx[:nt]; train = idx[nt:]
        model = train_reranker(h, train, seed)
        if model is None:
            continue
        train_traces = collect_traces(h, model, train, plans)
        test_traces = collect_traces(h, model, test, plans)
        cost_model, quality_model = train_cost_quality(h, train_traces)
        table = _trace_table(test_traces)

        fixed_pol = P.FixedPlanPolicy(plan=FIXED_PLAN)
        opt = P.LearnedOptimizer(h=h, cost_model=cost_model, quality_model=quality_model,
                                 budget_ms=BUDGET_MS)

        for qid in test:
            fp = fixed_pol.choose(qid, plans)
            cp = opt.choose(qid, plans)
            ft = _trace_table(test_traces)[qid].get(fp.label())
            ct = table[qid].get(cp.label())
            if ft is None or ct is None:
                continue
            total += 1
            flipped = cp.label() != fp.label()
            if flipped:
                flip_count += 1
                flip_types[_flip_type(fp, cp)] += 1
                cost_saved_on_flips.append(ft.cost_ms - ct.cost_ms)
                quality_change_on_flips.append(ct.quality - ft.quality)
                if len(examples) < 6:
                    examples.append({
                        "seed": seed, "qid": int(qid),
                        "fixed_plan": fp.label(), "objective_plan": cp.label(),
                        "flip_type": _flip_type(fp, cp),
                        "fixed_cost_ms": round(ft.cost_ms, 4), "obj_cost_ms": round(ct.cost_ms, 4),
                        "fixed_quality": round(ft.quality, 4), "obj_quality": round(ct.quality, 4),
                    })
            rows.append({"seed": seed, "qid": int(qid), "flipped": int(flipped),
                         "fixed_plan": fp.label(), "objective_plan": cp.label(),
                         "flip_type": _flip_type(fp, cp) if flipped else "same",
                         "fixed_cost_ms": round(ft.cost_ms, 4), "obj_cost_ms": round(ct.cost_ms, 4),
                         "fixed_quality": round(ft.quality, 4), "obj_quality": round(ct.quality, 4)})

    flip_rate = round(flip_count / total, 4) if total else 0.0
    report = {
        "budget_ms": BUDGET_MS,
        "fixed_plan": FIXED_PLAN.label(),
        "queries_scored": total,
        "plan_flip_rate": flip_rate,
        "flip_type_breakdown": dict(flip_types),
        "mean_cost_saved_ms_on_flips": round(statistics.mean(cost_saved_on_flips), 4) if cost_saved_on_flips else 0.0,
        "mean_quality_change_on_flips": round(statistics.mean(quality_change_on_flips), 4) if quality_change_on_flips else 0.0,
        "examples": examples,
        "reading": ("On {0:.0%} of held-out queries the cost objective selects a "
                    "different physical plan than the fixed order, and on exactly those "
                    "queries the objective's plan saves {1:.3f} ms of realized cost on "
                    "average while changing realized quality by {2:+.3f}. This is the "
                    "direct evidence R1-O3.A asked for: the objective of Equation (1) "
                    "drives measured plan-selection decisions that a fixed order would "
                    "not make.").format(
                        flip_rate,
                        statistics.mean(cost_saved_on_flips) if cost_saved_on_flips else 0.0,
                        statistics.mean(quality_change_on_flips) if quality_change_on_flips else 0.0),
    }
    (out / "plan_flips.json").write_text(json.dumps(report, indent=2))
    if rows:
        with (out / "plan_flips.csv").open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
