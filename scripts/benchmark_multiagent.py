"""End-to-end multi-agent execution measurement.

Answers reviewer concern O1.A (R1) and O1 (R2): the multi-agent layer named in
the title is now instantiated AND measured. For every biological query we run the
full Planner -> Executor -> Reachability -> Executor(dispatch) -> Aggregator flow
through the orchestrator and record:

  * the decomposition output (number and type of operators emitted by the planner),
  * the number of inter-agent messages on the bus,
  * per-agent wall-clock time,
  * the cost-aware dispatch decision (admitted/suppressed) and its reason,
  * a fully serialized message trace for one representative query (for the paper's
    worked example, answering O3.B).

All numbers are measured at runtime; nothing is hard-coded.
"""

from __future__ import annotations

import csv
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from agentflow_ppi.agents import MultiAgentOrchestrator, OrchestratorConfig, QueryRequest
from agentflow_ppi.agents.executor_agent import TypedGraph
from agentflow_ppi.data.cycle_handling import condense_to_dag
from agentflow_ppi.reachability import SHRCIndex


def load_named_graph(tsv_path: Path):
    import csv as _csv
    names: Dict[str, int] = {}
    typed_adj: Dict[int, List[Tuple[int, str, float]]] = defaultdict(list)
    edges: List[Tuple[int, int]] = []

    def nid(name: str) -> int:
        if name not in names:
            names[name] = len(names)
        return names[name]

    with tsv_path.open() as f:
        for row in _csv.DictReader(f, delimiter="\t"):
            u, v = nid(row["source"]), nid(row["target"])
            typed_adj[u].append((v, row["modality"], float(row["score"])))
            edges.append((u, v))
    return names, {i: n for n, i in names.items()}, typed_adj, edges


def main():
    root = Path(__file__).resolve().parents[1]
    data_root = root / "examples" / "biological_queries"
    names, id_to_name, typed_adj, edges = load_named_graph(data_root / "named_ppi_edges.tsv")
    n = len(names)
    queries = json.loads((data_root / "real_bio_queries.json").read_text())

    # Condense and build SHRC on the component DAG; remap node ids to components
    # so the orchestrator's reachability agent runs on the acyclic snapshot.
    cond = condense_to_dag(n, edges)
    comp = cond.component_of
    shrc = SHRCIndex.from_edges(num_nodes=cond.num_components, edges=cond.dag_edges).build()

    # Wrap the component-DAG SHRC so the orchestrator can call .reachable on
    # ORIGINAL node ids transparently.
    class CompSHRC:
        def __init__(self, base, comp):
            self.base = base; self.comp = comp
        def reachable(self, a, b):
            return self.comp[a] == self.comp[b] or self.base.reachable(self.comp[a], self.comp[b])
    comp_index = CompSHRC(shrc, comp)

    typed_graph = TypedGraph(num_nodes=n, adjacency=[typed_adj.get(i, []) for i in range(n)])

    # Deterministic modality-aware reranker stub: scores a candidate by its
    # modality agreement with the query (mirrors the learned reranker's signal).
    def make_reranker(modality):
        def rr(cands):
            out = {}
            for v in cands:
                e = typed_adj.get(v, [])
                out[v] = (sum(1 for _w, m, _c in e if m == modality) / len(e)) if e else 0.0
            return out
        return rr

    def symbolic_scorer(cands):
        # path-score-like symbolic order (best incoming confidence)
        out = {}
        for v in cands:
            out[v] = max([c for nbrs in typed_adj.values() for (_w, _m, c) in [] ], default=0.0)
        # simpler: incoming max confidence
        for v in cands:
            inc = [c for u in range(n) for (w, _m, c) in typed_adj.get(u, []) if w == v]
            out[v] = max(inc, default=0.0)
        return out

    config = OrchestratorConfig()
    rows = []
    msg_counts = []
    op_counter = Counter()
    agent_time = defaultdict(list)
    admitted = 0
    rep_trace = None

    from agentflow_ppi.data.pathway_ground_truth import query_is_pathway_grounded
    from agentflow_ppi.eval.harness import build_harness as _bh
    _harness_keys = {p.key for p in _bh().pools}
    for q in queries:
        if q["source"] not in names or q["target"] not in names:
            continue
        # Restrict to exactly the pathway-grounded, label-evaluable query set used
        # by every quality experiment so the reported cardinality is consistent
        # across the whole paper (reviewer E1): N = 17.
        if f"{q['source']}->{q['target']}" not in _harness_keys:
            continue
        modality = q["modality"]
        orch = MultiAgentOrchestrator(
            typed_graph=typed_graph, shrc_index=comp_index, id_to_name=id_to_name,
            reranker=make_reranker(modality), symbolic_scorer=symbolic_scorer, config=config,
        )
        req = QueryRequest(query_id=f"{q['source']}->{q['target']}", source=q["source"],
                           target=q["target"], modality=modality,
                           min_confidence=q.get("min_score", 0.7), top_k=2)
        # Expected gain proxy: reranking helps most when the reachable mediator
        # frontier is modality-AMBIGUOUS, i.e. candidates disagree on whether they
        # match the query modality. We estimate it from the candidate pool's
        # modality-agreement variance. Homogeneous pools (all match / none match)
        # get low gain (symbolic order suffices); mixed pools get high gain.
        s_id = names[q["source"]]; t_id = names[q["target"]]
        probe = orch.executor.typed_expand(s_id, modality, req.max_hops)
        pool = [v for v in probe if v not in (s_id, t_id)
                and comp_index.reachable(s_id, v) and comp_index.reachable(v, t_id)]
        if len(pool) >= 2:
            agree = []
            for v in pool:
                e = typed_adj.get(v, [])
                agree.append((sum(1 for _w, m, _c in e if m == modality) / len(e)) if e else 0.0)
            mean_a = sum(agree) / len(agree)
            ambiguity = sum(abs(a - mean_a) for a in agree) / len(agree)  # mean abs deviation
            expected_gain = float(np.clip(0.6 * (ambiguity * 4.0), 0.0, 0.6))
        else:
            expected_gain = 0.02  # too few candidates to benefit from reranking
        res = orch.execute(req, expected_gain=expected_gain)
        msg_counts.append(res.message_count())
        op_counter.update([s.operator for s in res.plan.steps])
        for rec in res.agent_records:
            agent_time[rec.agent_name].append(rec.wall_time_s * 1e3)  # ms
        admitted += int(res.dispatch.admitted)
        rows.append({
            "query": res.query_id,
            "plan_ops": len(res.plan.steps),
            "messages": res.message_count(),
            "post_frontier": res.dispatch.frontier_size,
            "expected_gain": round(res.dispatch.expected_gain, 3),
            "reranker_admitted": int(res.dispatch.admitted),
            "dispatch_reason": res.dispatch.reason,
            "total_ms": round(res.total_wall_time_s * 1e3, 4),
            "returned": ";".join(res.ranked_candidates),
        })
        if res.query_id == "EGFR->STAT3":
            rep_trace = [
                {"sender": m.sender, "recipient": m.recipient, "stage": m.stage, "payload": m.payload}
                for m in res.message_trace
            ]

    out_dir = root / "results"; out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "multiagent_execution.csv").open("w", newline="") as f:
        wtr = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        wtr.writeheader(); wtr.writerows(rows)

    summary = {
        "num_queries": len(rows),
        "avg_messages_per_query": round(statistics.mean(msg_counts), 2),
        "plan_operator_histogram": dict(op_counter),
        "reranker_admitted": admitted,
        "reranker_suppressed": len(rows) - admitted,
        "avg_agent_ms": {a: round(statistics.mean(t), 4) for a, t in agent_time.items()},
        "avg_total_ms": round(statistics.mean([r["total_ms"] for r in rows]), 4),
    }
    (out_dir / "multiagent_summary.json").write_text(json.dumps(summary, indent=2))
    if rep_trace is not None:
        (out_dir / "multiagent_worked_example.json").write_text(json.dumps(rep_trace, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
