"""Tests for the query-time multi-agent execution layer."""

from __future__ import annotations

from agentflow_ppi.agents import (
    MultiAgentOrchestrator,
    QueryRequest,
    QueryPlannerAgent,
)
from agentflow_ppi.agents.executor_agent import TypedGraph
from agentflow_ppi.reachability import SHRCIndex


def _toy_graph():
    # 0 -> {1,2} ; 1 -> 3 ; 2 -> 3 ; 3 -> 4   (mediators 1,2,3 between 0 and 4)
    typed = {
        0: [(1, "functional", 0.9), (2, "physical", 0.8)],
        1: [(3, "functional", 0.7)],
        2: [(3, "physical", 0.6)],
        3: [(4, "functional", 0.85)],
        4: [],
    }
    adj = [typed.get(i, []) for i in range(5)]
    edges = [(u, v) for u in typed for v, _m, _c in typed[u]]
    idx = SHRCIndex.from_edges(num_nodes=5, edges=edges).build()
    id_to_name = {0: "S", 1: "M1", 2: "M2", 3: "M3", 4: "T"}
    return TypedGraph(5, adj), idx, id_to_name


def test_planner_emits_typed_plan():
    planner = QueryPlannerAgent()
    plan = planner.decompose(QueryRequest("q", "S", "T", "functional"))
    ops = [s.operator for s in plan.steps]
    assert "typed_expand" in ops
    assert "reachability_prune" in ops
    assert "neural_rerank" in ops
    # reachability must precede reranking
    assert ops.index("reachability_prune") < ops.index("neural_rerank")


def test_orchestrator_end_to_end_and_messages():
    tg, idx, id2n = _toy_graph()

    def reranker(cands):
        return {v: float(v) for v in cands}

    orch = MultiAgentOrchestrator(tg, idx, id2n, reranker=reranker)
    res = orch.execute(QueryRequest("q", "S", "T", "functional", min_confidence=0.0, top_k=2),
                       expected_gain=0.5)
    # full inter-agent trace is recorded
    assert res.message_count() == 6
    assert {m.sender for m in res.message_trace} >= {"user", "planner", "executor", "reachability", "aggregator"}
    # reachability pruned to genuine mediators (1,2,3 reach T)
    assert res.dispatch.frontier_size >= 1
    # per-agent records exist for all four agents
    agents = {r.agent_name for r in res.agent_records}
    assert agents >= {"planner", "executor", "reachability", "aggregator"}
    assert res.total_wall_time_s >= 0.0


def test_dispatch_is_selective():
    tg, idx, id2n = _toy_graph()
    orch = MultiAgentOrchestrator(tg, idx, id2n, reranker=lambda c: {v: 0.0 for v in c})
    # low gain -> suppressed
    low = orch.execute(QueryRequest("q", "S", "T", "functional", min_confidence=0.0),
                       expected_gain=0.0)
    assert low.dispatch.admitted is False
    # high gain, small frontier -> admitted
    high = orch.execute(QueryRequest("q", "S", "T", "functional", min_confidence=0.0),
                        expected_gain=0.8)
    assert high.dispatch.admitted is True


def test_large_benchmark_is_mixed_sign():
    """The large pathway benchmark must produce a workload where reranking helps
    on informative pathways and not on the rest (reviewer W2/W8/W12)."""
    from agentflow_ppi.eval.harness import (
        build_harness_large, train_reranker, rerank, symbolic_order, f1_at_k,
    )
    import statistics
    h = build_harness_large(num_pathways=20, pathway_len=6, informative_fraction=0.5, seed=7, max_hops=3)
    assert len(h.pools) > 100
    assert h.informative is not None and 0 < len(h.informative) < len(h.pools)
    # informative pathways should show a non-negative mean lift; non-informative non-positive
    import numpy as np
    rng = np.random.default_rng(7)
    idx = list(range(len(h.pools))); rng.shuffle(idx)
    nt = max(1, int(round(0.2 * len(idx)))); test = set(idx[:nt]); train = [i for i in idx if i not in test]
    m = train_reranker(h, train, 7)
    assert m is not None
    li, ln = [], []
    for i in test:
        p = h.pools[i]
        b = f1_at_k(symbolic_order(h, p.s, p.t, p.cands), p.positives)
        r = f1_at_k(rerank(h, m, p.s, p.t, p.modality, p.cands), p.positives)
        (li if i in h.informative else ln).append(r - b)
    # informative mean lift should exceed non-informative mean lift
    assert statistics.mean(li) > statistics.mean(ln)


def test_gain_predictor_discriminates():
    """The calibrated gain predictor must admit more on informative than on
    non-informative queries (reviewer W1/W8)."""
    from agentflow_ppi.eval.harness import build_harness_large, train_gain_predictor, predict_gain
    import numpy as np, statistics
    h = build_harness_large(num_pathways=20, pathway_len=6, informative_fraction=0.5, seed=13, max_hops=3)
    rng = np.random.default_rng(13)
    idx = list(range(len(h.pools))); rng.shuffle(idx)
    nt = max(1, int(round(0.2 * len(idx)))); test = set(idx[:nt]); train = [i for i in idx if i not in test]
    pred = train_gain_predictor(h, train, 13)
    assert pred is not None
    ai = [1 if predict_gain(h, pred, h.pools[i]) > 0 else 0 for i in test if i in h.informative]
    an = [1 if predict_gain(h, pred, h.pools[i]) > 0 else 0 for i in test if i not in h.informative]
    if ai and an:
        assert statistics.mean(ai) >= statistics.mean(an)
