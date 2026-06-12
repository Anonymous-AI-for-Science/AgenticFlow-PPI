"""Run the Ollama LLM multi-agent flow on a single query, printing the full
inter-agent trace. Usage:

    # with a real model (requires `ollama serve` + `ollama pull llama3.1:8b`)
    python scripts/run_llm_query.py --source EGFR --target STAT3 --model llama3.1:8b

    # offline deterministic (no server needed)
    python scripts/run_llm_query.py --source EGFR --target STAT3 --offline
"""
from __future__ import annotations

import argparse
import json

from agentflow_ppi.agents.llm import LLMAgentTrace, LLMMultiAgentCollaboration, OllamaClient


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True)
    ap.add_argument("--target", required=True)
    ap.add_argument("--model", default="llama3.1:8b")
    ap.add_argument("--offline", action="store_true")
    a = ap.parse_args()

    client = OllamaClient(model=a.model, force_offline=a.offline)
    collab = LLMMultiAgentCollaboration(client)
    trace = LLMAgentTrace()

    class R: pass
    req = R(); req.query_id = f"{a.source}->{a.target}"; req.source = a.source
    req.target = a.target; req.modality = "functional"; req.max_hops = 3
    req.min_confidence = 0.7; req.top_k = 2

    default = [{"name": "expand", "operator": "typed_expand", "inputs": ["source"], "params": {}},
               {"name": "prune", "operator": "reachability_prune", "inputs": ["expand"], "params": {}},
               {"name": "rerank", "operator": "neural_rerank", "inputs": ["prune"], "params": {"optional": True}},
               {"name": "agg", "operator": "aggregate", "inputs": ["rerank"], "params": {}}]

    print(f"# Ollama available: {client.available()} (model={a.model})\n")
    plan = collab.plan(req, default, trace)
    print("PLANNER ->", json.dumps(plan.get("steps", []))[:200])
    rep = collab.reachability_report(a.source, raw_frontier=9, reachable=5, trace=trace)
    print("REACHABILITY ->", rep)
    disp = collab.dispatch(5, rep["selectivity"], -0.02, 0.1, 0.9, 0.83, 0.81, trace)
    print("EXECUTOR (dispatch) ->", disp)
    agg = collab.aggregate(req.query_id, a.source, a.target,
                           [{"id": "JAK1", "alias": "JAK1", "score": 0.91, "path_evidence": f"{a.source}-JAK1-{a.target}"},
                            {"id": "SRC", "alias": "SRC", "score": 0.82, "path_evidence": f"{a.source}-SRC-{a.target}"}],
                           2, trace)
    print("AGGREGATOR ->", json.dumps(agg)[:200])
    print("\n# TRACE SUMMARY:", json.dumps(trace.summary(), indent=2))


if __name__ == "__main__":
    main()
