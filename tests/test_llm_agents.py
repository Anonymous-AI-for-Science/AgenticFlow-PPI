"""Tests for the Ollama-backed LLM multi-agent layer (reviewer R1-O1.A).

These run fully offline (force_offline=True) so they are deterministic and need no
Ollama server, while still exercising the real prompt rendering, JSON parsing,
schema repair, and the authority constraints (LLM never changes exact numbers)."""

from agentflow_ppi.agents.llm import OllamaClient, LLMMultiAgentCollaboration, LLMAgentTrace


def _collab():
    return LLMMultiAgentCollaboration(OllamaClient(model="llama3.1:8b", force_offline=True))


def test_prompts_exist_and_render():
    from agentflow_ppi.agents.llm.ollama_client import _render_prompt, PROMPTS_DIR
    for name in ["planner", "reachability", "executor", "aggregator"]:
        assert (PROMPTS_DIR / f"{name}.md").exists()
    txt = _render_prompt("executor", {"frontier_size": 5, "selectivity": 0.5,
        "expected_gain": -0.1, "predicted_symbolic_cost": 0.1,
        "predicted_reranker_cost": 0.9, "objective_symbolic": 0.8, "objective_reranked": 0.7})
    assert "{frontier_size}" not in txt and "5" in txt


def test_four_agents_produce_valid_schema():
    collab = _collab(); trace = LLMAgentTrace()
    class R: pass
    req = R(); req.query_id="A->B"; req.source="A"; req.target="B"
    req.modality="functional"; req.max_hops=3; req.min_confidence=0.7; req.top_k=2
    collab.plan(req, [{"name":"x","operator":"typed_expand","inputs":["source"],"params":{}}], trace)
    collab.reachability_report("A", 9, 5, trace)
    collab.dispatch(5, 0.556, -0.02, 0.1, 0.9, 0.83, 0.81, trace)
    collab.aggregate("A->B","A","B",[{"id":"M1","alias":"M1","score":0.9,"path_evidence":"A-M1-B"},
                                      {"id":"M2","alias":"M2","score":0.8,"path_evidence":"A-M2-B"}], 2, trace)
    s = trace.summary()
    assert s["llm_turns"] == 4
    assert s["schema_valid_rate"] == 1.0
    assert s["agents"] == ["planner","reachability","executor","aggregator"]


def test_llm_never_changes_exact_reachability():
    """Authority constraint: reachable count/selectivity come from SHRC, not the LLM."""
    collab = _collab(); trace = LLMAgentTrace()
    rep = collab.reachability_report("EGFR", raw_frontier=9, reachable=5, trace=trace)
    assert rep["reachable_count"] == 5
    assert rep["selectivity"] == round(5/9, 3)


def test_dispatch_follows_cost_objective():
    """Executor admits iff objective_reranked > objective_symbolic, regardless of LLM."""
    collab = _collab(); trace = LLMAgentTrace()
    decline = collab.dispatch(5, 0.5, -0.02, 0.1, 0.9, 0.83, 0.81, trace)
    admit = collab.dispatch(5, 0.5, 0.11, 0.1, 0.9, 0.71, 0.80, trace)
    assert decline["admit_reranker"] is False
    assert admit["admit_reranker"] is True
