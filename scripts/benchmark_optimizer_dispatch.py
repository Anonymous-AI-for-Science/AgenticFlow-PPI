"""Phase 4 -- large-scale optimizer dispatch experiment.

Turns the 1-D admission threshold into a real plan-space optimizer and evaluates it
against the oracle and policy baselines on a large query workload, reporting regret,
budget violation, and the cost/quality Pareto frontier.

Pipeline:
  1. build a large mixed-sign workload (hundreds of pathway-grounded queries)
  2. enumerate physical plans (depth x rerank x frontier budget)
  3. collect (query, plan) -> (cost, quality) traces by executing every plan
  4. learn cost and quality models from a train split
  5. on the test split, run policies (oracle, never/always-rerank, fixed,
     1-D threshold, learned optimizer) and compute realized quality/cost
  6. report regret vs oracle, budget-violation rate, and the Pareto frontier

Writes results/optimizer_dispatch_summary.json and optimizer_pareto.csv.
"""

from __future__ import annotations

import csv
import json
import statistics
from pathlib import Path

import numpy as np

from agentflow_ppi.eval.harness import (build_harness_large, train_reranker,
                                        train_gain_predictor, predict_gain, SEED_MANIFEST)
from agentflow_ppi.optimizer.operators import enumerate_plans, Plan
from agentflow_ppi.optimizer.trace import collect_traces
from agentflow_ppi.optimizer.cost_quality import train_cost_quality, model_diagnostics
from agentflow_ppi.optimizer import policies as P
from agentflow_ppi.optimizer import metrics as M

BUDGET_SWEEP = [0.4, 0.6, 0.9, 1.3, 2.0, 3.0]  # per-query latency budgets (ms)
BUDGET_MS = 1.3  # headline budget for the regret/violation table


def _trace_table(traces):
    tab = {}
    for t in traces:
        tab.setdefault(t.qid, {})[t.plan.label()] = t
    return tab


def main():
    out = Path(__file__).resolve().parents[1] / "results"; out.mkdir(parents=True, exist_ok=True)
    h = build_harness_large(num_pathways=40, pathway_len=8, informative_fraction=0.5,
                            seed=7, max_hops=3)
    plans = enumerate_plans()
    n = len(h.pools)

    agg_rows = {}
    diagnostics = []
    all_qids = list(range(n))

    # aggregate over seeds for stable estimates
    pol_quality = {name: [] for name in
                   ["oracle", "never-rerank", "always-rerank", "fixed", "threshold-1d", "learned-optimizer"]}
    pol_cost = {name: [] for name in pol_quality}
    pol_regret = {name: [] for name in pol_quality}
    pol_viol = {name: [] for name in pol_quality}

    for seed in SEED_MANIFEST:
        rng = np.random.default_rng(seed)
        idx = all_qids.copy(); rng.shuffle(idx)
        nt = max(1, int(round(0.3 * len(idx))))
        test = idx[:nt]; train = idx[nt:]
        model = train_reranker(h, train, seed)
        gain_pred = train_gain_predictor(h, train, seed)
        if model is None or gain_pred is None:
            continue

        # collect traces on ALL queries (train traces fit the models; test traces score policies)
        train_traces = collect_traces(h, model, train, plans)
        test_traces = collect_traces(h, model, test, plans)
        cost_model, quality_model = train_cost_quality(h, train_traces)
        diagnostics.append(model_diagnostics(h, test_traces, cost_model, quality_model))

        table = _trace_table(test_traces)

        oracle = P.OraclePolicy(table=table, budget_ms=BUDGET_MS)
        gain_fn = lambda qid: predict_gain(h, gain_pred, h.pools[qid])
        frontier_fn = lambda qid: len(h.pools[qid].cands)
        pols = {
            "oracle": oracle,
            "never-rerank": P.NeverRerankPolicy(),
            "always-rerank": P.AlwaysRerankPolicy(),
            "fixed": P.FixedPlanPolicy(plan=Plan(3, True, 50)),
            "threshold-1d": P.ThresholdPolicy(gain_fn=gain_fn, frontier_fn=frontier_fn,
                                              frontier_budget=50),
            "learned-optimizer": P.LearnedOptimizer(h=h, cost_model=cost_model,
                                                    quality_model=quality_model, budget_ms=BUDGET_MS),
        }
        for name, pol in pols.items():
            res = P.evaluate_policy(pol, table, test, plans, budget_ms=BUDGET_MS)
            if res["quality"]:
                pol_quality[name].append(statistics.mean(res["quality"]))
                pol_cost[name].append(statistics.mean(res["cost_ms"]))
                pol_regret[name].append(M.regret_vs_oracle(pol, oracle, table, test, plans, budget_ms=BUDGET_MS))
                pol_viol[name].append(sum(res["violated"]) / max(len(res["violated"]), 1))

    # build summary rows + Pareto
    points = []
    for name in pol_quality:
        if not pol_quality[name]:
            continue
        points.append({
            "policy": name,
            "mean_quality": round(statistics.mean(pol_quality[name]), 4),
            "mean_cost_ms": round(statistics.mean(pol_cost[name]), 4),
            "regret_vs_oracle": round(statistics.mean(pol_regret[name]), 4),
            "budget_violation": round(statistics.mean(pol_viol[name]), 4),
            "raw_quality_no_budget": round(statistics.mean(pol_quality[name]), 4),
        })
    points = M.pareto_frontier(points)

    # Regret bound verification (originality: the optimizer is a policy with a
    # guarantee, not a heuristic). On each seed's held-out split, measure eps (max
    # quality-prediction error over feasible plans) and check per-query regret <= 2 eps.
    from agentflow_ppi.optimizer.regret_bound import verify_regret_bound
    bound_checks = []
    for seed in SEED_MANIFEST:
        rng = np.random.default_rng(seed)
        idx = all_qids.copy(); rng.shuffle(idx)
        nt = max(1, int(round(0.3 * len(idx))))
        test = idx[:nt]; train = idx[nt:]
        model = train_reranker(h, train, seed)
        if model is None:
            continue
        tr_tr = collect_traces(h, model, train, plans); te_tr = collect_traces(h, model, test, plans)
        cm, qm = train_cost_quality(h, tr_tr); table = _trace_table(te_tr)
        bound_checks.append(verify_regret_bound(h, table, test, plans, cm, qm, BUDGET_MS))
    regret_bound = {
        "mean_eps": round(float(np.mean([b["eps"] for b in bound_checks])), 4),
        "mean_two_eps_bound": round(float(np.mean([b["two_eps_bound"] for b in bound_checks])), 4),
        "mean_median_eps": round(float(np.mean([b["median_eps"] for b in bound_checks])), 4),
        "mean_median_two_eps": round(float(np.mean([b["median_two_eps"] for b in bound_checks])), 4),
        "mean_p90_eps": round(float(np.mean([b["p90_eps"] for b in bound_checks])), 4),
        "mean_frac_informative": round(float(np.mean([b["frac_queries_informative_bound"] for b in bound_checks])), 4),
        "mean_max_regret": round(float(np.mean([b["max_observed_regret"] for b in bound_checks])), 4),
        "bound_holds_all_seeds": all(b["bound_holds"] for b in bound_checks),
        "total_bound_violations": sum(b["bound_violations"] for b in bound_checks),
    }

    # Budget sweep: at each per-query latency budget, compare the learned optimizer
    # (re-planned for that budget) against always-rerank and the 1-D threshold. This
    # makes the budget constraint active and shows the optimizer respects tight
    # budgets where always-rerank violates them, which a 1-D threshold cannot adapt to.
    sweep_rows = []
    for budget in BUDGET_SWEEP:
        oq, ov, tq, tv, aq, av = [], [], [], [], [], []
        for seed in SEED_MANIFEST:
            rng = np.random.default_rng(seed)
            idx = all_qids.copy(); rng.shuffle(idx)
            nt = max(1, int(round(0.3 * len(idx))))
            test = idx[:nt]; train = idx[nt:]
            model = train_reranker(h, train, seed); gp = train_gain_predictor(h, train, seed)
            if model is None or gp is None:
                continue
            tr_tr = collect_traces(h, model, train, plans); te_tr = collect_traces(h, model, test, plans)
            cm, qm = train_cost_quality(h, tr_tr); table = _trace_table(te_tr)
            opt = P.LearnedOptimizer(h=h, cost_model=cm, quality_model=qm, budget_ms=budget)
            thr = P.ThresholdPolicy(gain_fn=lambda qid: predict_gain(h, gp, h.pools[qid]),
                                    frontier_fn=lambda qid: len(h.pools[qid].cands), frontier_budget=50)
            alw = P.AlwaysRerankPolicy()
            ro = P.evaluate_policy(opt, table, test, plans, budget_ms=budget)
            rt = P.evaluate_policy(thr, table, test, plans, budget_ms=budget)
            ra = P.evaluate_policy(alw, table, test, plans, budget_ms=budget)
            oq.append(statistics.mean(ro["quality"])); ov.append(round(sum(ro["violated"])/max(len(ro["violated"]),1),4))
            tq.append(statistics.mean(rt["quality"])); tv.append(round(sum(rt["violated"])/max(len(rt["violated"]),1),4))
            aq.append(statistics.mean(ra["quality"])); av.append(round(sum(ra["violated"])/max(len(ra["violated"]),1),4))
        sweep_rows.append({
            "budget_ms": budget,
            "optimizer_quality": round(statistics.mean(oq), 4),
            "optimizer_violation": round(statistics.mean(ov), 4),
            "threshold_quality": round(statistics.mean(tq), 4),
            "threshold_violation": round(statistics.mean(tv), 4),
            "always_quality": round(statistics.mean(aq), 4),
            "always_violation": round(statistics.mean(av), 4),
        })
    with (out / "optimizer_budget_sweep.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(sweep_rows[0].keys())); w.writeheader(); w.writerows(sweep_rows)

    summary = {
        "workload": {"queries": n, "informative": len(h.informative or []),
                     "plans_per_query": len(plans), "budget_ms": BUDGET_MS},
        "model_fit": {"cost_r2": round(statistics.mean([d["cost_r2"] for d in diagnostics]), 3),
                      "quality_r2": round(statistics.mean([d["quality_r2"] for d in diagnostics]), 3)},
        "policies": points,
        "regret_bound": regret_bound,
        "budget_sweep": sweep_rows,
        "reading": ("The learned optimizer searches the plan space via predicted "
                    "cost/quality under a budget; it should achieve lower regret than "
                    "the 1-D threshold and respect the budget better than always-rerank, "
                    "and lie on the cost/quality Pareto frontier."),
    }
    (out / "optimizer_dispatch_summary.json").write_text(json.dumps(summary, indent=2))
    with (out / "optimizer_pareto.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["policy", "mean_quality", "mean_cost_ms",
                                          "regret_vs_oracle", "budget_violation", "raw_quality_no_budget", "pareto_optimal"])
        w.writeheader(); w.writerows(points)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
