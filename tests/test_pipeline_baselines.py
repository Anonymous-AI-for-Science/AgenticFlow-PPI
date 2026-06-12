"""Tests for the system-level pipeline baselines and published-index runner (design rationale)."""

import importlib.util
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


def _load(name):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    return mod


def test_three_pipelines_same_answer_space():
    """All three pipelines answer-check against the same BFS-exact reachable set, so
    F1 is comparable; AgentFlow-PPI must not lose quality vs the baselines."""
    pb = _load("benchmark_pipeline_baselines")
    n, adj = pb.build_snapshot(5000, pb.SEED)
    import numpy as np
    rng = np.random.default_rng(pb.SEED)
    fams = pb.make_families(n, adj, rng)
    assert len(fams) >= 10
    import numpy as np
    f1s = {}
    rerank = {}
    for name, fn in pb.PIPELINES.items():
        tot_f1 = 0.0; tot_r = 0
        for q in fams:
            r = fn(q, adj, np.random.default_rng(1))
            tot_f1 += r["f1"]; tot_r += r["reranker_calls"]
        f1s[name] = tot_f1 / len(fams); rerank[name] = tot_r
    # quality preserved (within float noise) ...
    assert abs(f1s["agentflow-ppi"] - f1s["fixed-order"]) < 1e-6
    # ... at strictly lower reranker cost than always-on fixed-order
    assert rerank["agentflow-ppi"] < rerank["fixed-order"]


def test_published_index_agrees_with_bfs():
    pi = _load("benchmark_published_index")
    res = pi.run_published_or_faithful(n_nodes=2000, seed=3, n_queries=200)
    assert res["grail_agreement_vs_bfs"] == 1.0
    assert res["pll_agreement_vs_bfs"] == 1.0
    assert res["backend"] in ("published-cxx:grail", "faithful-reimpl")
