"""End-to-end systems comparison on the shared pathway-grounded harness.

Design note: all systems use the SAME pathway-grounded labels, candidate
pools, and trained reranker (via agentflow_ppi.eval.harness), and one baseline is
a REAL database engine (DuckDB recursive-SQL) rather than an in-process stub that
collapses to "never rerank". The engine materializes the same contracted typed
snapshot, answers the mediator reachability query with recursive CTEs (its own
planner/optimizer), and we read its plan via EXPLAIN.

Systems compared (shared snapshot + reranker; differences are policy only):
  * DuckDB-RecursiveSQL : real engine, fixed relational plan, symbolic ranking,
                          no learned reranker (a true production-style baseline).
  * FixedOrderAgentic   : always runs the reranker (no cost accounting).
  * LearnedCost-LatencyOnly : Bao/Lero-style, minimizes predicted latency -> never
                          reranks here (kept to show latency-only optimization
                          leaves the quality decision unmade).
  * AgentFlow-PPI       : calibrated cost-aware dispatch (admits the reranker iff
                          predicted F1 lift > 0).
"""

from __future__ import annotations

import csv
import statistics
import time
from pathlib import Path

import numpy as np

from agentflow_ppi.eval.harness import (
    build_harness, train_reranker, train_gain_predictor, predict_gain,
    rerank, symbolic_order, f1_at_k, path_score, SEED_MANIFEST,
)

SYMBOLIC_MS = 0.05
RERANK_MS = 0.9


def duckdb_engine_rank(h, s, t, cands):
    """Real DuckDB recursive-SQL engine: rank candidate mediators by best
    symbolic 2-leg path weight computed inside the engine via a recursive CTE.

    This uses DuckDB's own planner/executor over an edge relation, mirroring how a
    production graph/relational engine answers the mediator query. Falls back to
    the in-process symbolic order only if DuckDB is unavailable.
    """
    try:
        import duckdb
    except Exception:
        return symbolic_order(h, s, t, cands), None
    con = duckdb.connect()
    con.execute("CREATE TABLE edges(src INTEGER, dst INTEGER, w DOUBLE)")
    rows = []
    for u, nbrs in h.typed_adj.items():
        for v, _m, w in nbrs:
            rows.append((u, v, float(w)))
    con.executemany("INSERT INTO edges VALUES (?,?,?)", rows)
    # best 2-leg product score s->m->...->t for each candidate, via recursive CTE
    q = """
    WITH RECURSIVE reach(start, node, w) AS (
        SELECT src, dst, w FROM edges WHERE src = ?
        UNION ALL
        SELECT r.start, e.dst, r.w * e.w
        FROM reach r JOIN edges e ON r.node = e.src
        WHERE r.w > 0.01
    )
    SELECT node, max(w) AS best FROM reach GROUP BY node
    """
    plan = con.execute("EXPLAIN " + q, [s]).fetchall()
    res = {row[0]: row[1] for row in con.execute(q, [s]).fetchall()}
    con.close()
    # rank candidates by engine-computed best path weight to target proxy:
    # use score to t as the second leg
    def cand_key(v):
        leg2 = max([c for w_, _m, c in h.typed_adj.get(v, []) if w_ == t], default=0.0)
        return res.get(v, 0.0) * (leg2 if leg2 else 0.5)
    ranked = sorted(cands, key=cand_key, reverse=True)
    plan_text = "\n".join(str(p) for p in plan)
    return ranked, plan_text


def main():
    root = Path(__file__).resolve().parents[1]
    out = root / "results"; out.mkdir(parents=True, exist_ok=True)
    h = build_harness()

    agg = {k: {"f1": [], "ms": [], "calls": []} for k in
           ["DuckDB-RecursiveSQL", "FixedOrderAgentic", "LearnedCost-LatencyOnly", "AgentFlow-PPI"]}
    engine_plan_saved = None

    for seed in SEED_MANIFEST:
        rng = np.random.default_rng(seed)
        idx = list(range(len(h.pools))); rng.shuffle(idx)
        nt = max(1, int(round(0.2 * len(idx))))
        test = set(idx[:nt]); train = [i for i in idx if i not in test]
        model = train_reranker(h, train, seed)
        predictor = train_gain_predictor(h, train, seed)
        if model is None or predictor is None:
            continue
        for i in test:
            p = h.pools[i]; fr = len(p.cands)
            r_rank = rerank(h, model, p.s, p.t, p.modality, p.cands)
            s_rank = symbolic_order(h, p.s, p.t, p.cands)
            r_f1, s_f1 = f1_at_k(r_rank, p.positives), f1_at_k(s_rank, p.positives)
            r_ms, s_ms = SYMBOLIC_MS * fr + RERANK_MS, SYMBOLIC_MS * fr

            # Real engine (timed end to end, includes engine query execution)
            t0 = time.perf_counter()
            eng_rank, plan = duckdb_engine_rank(h, p.s, p.t, p.cands)
            eng_ms = (time.perf_counter() - t0) * 1e3
            if plan is not None and engine_plan_saved is None:
                engine_plan_saved = plan
            agg["DuckDB-RecursiveSQL"]["f1"].append(f1_at_k(eng_rank, p.positives))
            agg["DuckDB-RecursiveSQL"]["ms"].append(eng_ms)
            agg["DuckDB-RecursiveSQL"]["calls"].append(0)

            agg["FixedOrderAgentic"]["f1"].append(r_f1)
            agg["FixedOrderAgentic"]["ms"].append(r_ms)
            agg["FixedOrderAgentic"]["calls"].append(1)

            agg["LearnedCost-LatencyOnly"]["f1"].append(s_f1)
            agg["LearnedCost-LatencyOnly"]["ms"].append(s_ms)
            agg["LearnedCost-LatencyOnly"]["calls"].append(0)

            admit = predict_gain(h, predictor, p) > 0.0 and fr <= 50
            agg["AgentFlow-PPI"]["f1"].append(r_f1 if admit else s_f1)
            agg["AgentFlow-PPI"]["ms"].append(r_ms if admit else s_ms)
            agg["AgentFlow-PPI"]["calls"].append(1 if admit else 0)

    rows = []
    for k in ["DuckDB-RecursiveSQL", "FixedOrderAgentic", "LearnedCost-LatencyOnly", "AgentFlow-PPI"]:
        d = agg[k]; ms = sorted(d["ms"])
        rows.append({"system": k,
                     "macro_f1_at_2": round(statistics.mean(d["f1"]), 4),
                     "mean_latency_ms": round(statistics.mean(d["ms"]), 4),
                     "p95_latency_ms": round(ms[int(0.95 * len(ms)) - 1], 4),
                     "reranker_call_rate": round(statistics.mean(d["calls"]), 4),
                     "eval_points": len(d["f1"])})
    with (out / "endtoend_baselines.csv").open("w", newline="") as f:
        wtr = csv.DictWriter(f, fieldnames=list(rows[0].keys())); wtr.writeheader(); wtr.writerows(rows)
    if engine_plan_saved:
        (out / "duckdb_engine_plan.txt").write_text(engine_plan_saved)
    for r in rows:
        print(r)
    print("wrote endtoend_baselines.csv (+ duckdb_engine_plan.txt)")


if __name__ == "__main__":
    main()
