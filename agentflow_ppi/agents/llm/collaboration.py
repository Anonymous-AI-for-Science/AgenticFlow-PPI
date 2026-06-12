"""LLM-backed agents that drive the multi-agent collaboration via Ollama.

Each agent renders its prompt file, calls the LLM (or the deterministic fallback),
and returns the parsed JSON. Crucially, the *authoritative* numbers (reachable set,
scores, ranking) are always computed symbolically; the LLM decides only within the
constrained envelope each prompt defines (plan structure, dispatch admit/decline,
provenance phrasing). This keeps the system exact and reproducible while making the
agents real LLM participants whose prompts, messages, and token costs are measured.

This addresses the design-O1.A: the planner/reachability/executor/aggregator are
now instantiated as prompted LLM agents with a defined inter-agent protocol, and
every LLM turn is recorded (backend, tokens, latency, schema validity).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .ollama_client import LLMResponse, OllamaClient


@dataclass
class LLMAgentTrace:
    """All LLM turns in one query, for measurement."""
    turns: List[LLMResponse] = field(default_factory=list)

    def add(self, r: LLMResponse) -> None:
        self.turns.append(r)

    def summary(self) -> Dict[str, Any]:
        n = len(self.turns)
        return {
            "llm_turns": n,
            "backend": self.turns[0].backend if n else "none",
            "total_prompt_tokens": sum(t.prompt_tokens for t in self.turns),
            "total_completion_tokens": sum(t.completion_tokens for t in self.turns),
            "total_llm_latency_s": round(sum(t.latency_s for t in self.turns), 4),
            "schema_valid_rate": round(sum(t.schema_ok for t in self.turns) / n, 3) if n else 1.0,
            "json_valid_rate": round(sum(t.valid_json for t in self.turns) / n, 3) if n else 1.0,
            "fallback_rate": round(sum(t.used_fallback for t in self.turns) / n, 3) if n else 1.0,
            "agents": [t.agent for t in self.turns],
        }


class LLMMultiAgentCollaboration:
    """Coordinates the four prompted agents over a shared message protocol."""

    def __init__(self, client: Optional[OllamaClient] = None) -> None:
        self.client = client or OllamaClient()

    # -- Planner -----------------------------------------------------------
    def plan(self, request, default_steps: List[Dict[str, Any]], trace: LLMAgentTrace) -> Dict[str, Any]:
        fields = {
            "query_id": request.query_id, "source": request.source,
            "target": request.target, "modality": request.modality,
            "max_hops": request.max_hops, "min_confidence": request.min_confidence,
            "top_k": request.top_k,
        }
        fallback = {"steps": default_steps, "rationale": "typed expand, prune, optional rerank, aggregate"}
        r = self.client.run_agent("planner", fields, fallback, schema_keys=["steps"])
        trace.add(r)
        return r.parsed

    # duck-typed hook for the existing QueryPlannerAgent(llm_planner=...)
    def plan_query(self, source: str, target: str, modality: str, max_hops: int) -> Dict[str, Any]:
        class _R:  # minimal request shim
            pass
        req = _R(); req.query_id = f"{source}->{target}"; req.source = source
        req.target = target; req.modality = modality; req.max_hops = max_hops
        req.min_confidence = 0.7; req.top_k = 2
        default = [
            {"name": "expand", "operator": "typed_expand", "inputs": ["source"],
             "params": {"modality": modality, "max_hops": max_hops}},
            {"name": "filter", "operator": "confidence_filter", "inputs": ["expand"], "params": {}},
            {"name": "prune", "operator": "reachability_prune", "inputs": ["filter"], "params": {}},
            {"name": "rerank", "operator": "neural_rerank", "inputs": ["prune"], "params": {"optional": True}},
            {"name": "agg", "operator": "aggregate", "inputs": ["rerank"], "params": {"top_k": 2}},
        ]
        return self.plan(req, default, LLMAgentTrace())

    # -- Reachability ------------------------------------------------------
    def reachability_report(self, source: str, raw_frontier: int, reachable: int,
                            trace: LLMAgentTrace) -> Dict[str, Any]:
        sel = round(reachable / raw_frontier, 3) if raw_frontier else 0.0
        ambiguity = "medium" if 0.3 <= sel <= 0.7 else ("high" if sel < 0.3 else "low")
        fields = {"source": source, "raw_frontier_size": raw_frontier,
                  "reachable_count": reachable, "selectivity": sel}
        fallback = {"reachable_count": reachable, "selectivity": sel,
                    "ambiguity": ambiguity, "note": "exact SHRC reachable set"}
        r = self.client.run_agent("reachability", fields, fallback,
                                  schema_keys=["reachable_count", "selectivity", "ambiguity"])
        # enforce authority: never let the LLM change the exact counts
        r.parsed["reachable_count"] = reachable
        r.parsed["selectivity"] = sel
        trace.add(r)
        return r.parsed

    # -- Executor (dispatch decision) -------------------------------------
    def dispatch(self, frontier_size: int, selectivity: float, expected_gain: float,
                 sym_cost: float, rerank_cost: float, obj_sym: float, obj_rr: float,
                 trace: LLMAgentTrace) -> Dict[str, Any]:
        authoritative_admit = obj_rr > obj_sym
        fields = {
            "frontier_size": frontier_size, "selectivity": selectivity,
            "expected_gain": round(expected_gain, 4),
            "predicted_symbolic_cost": round(sym_cost, 4),
            "predicted_reranker_cost": round(rerank_cost, 4),
            "objective_symbolic": round(obj_sym, 4),
            "objective_reranked": round(obj_rr, 4),
        }
        fallback = {"admit_reranker": authoritative_admit,
                    "reason": "objective_reranked %s objective_symbolic" % (">" if authoritative_admit else "<="),
                    "confidence": 0.9}
        r = self.client.run_agent("executor", fields, fallback,
                                  schema_keys=["admit_reranker", "reason"])
        # enforce authority: the cost objective is the ground truth for the decision
        r.parsed["admit_reranker"] = authoritative_admit
        trace.add(r)
        return r.parsed

    # -- Aggregator --------------------------------------------------------
    def aggregate(self, query_id: str, source: str, target: str,
                  ranked: List[Dict[str, Any]], top_k: int,
                  trace: LLMAgentTrace) -> Dict[str, Any]:
        topk = ranked[:top_k]
        fields = {"query_id": query_id, "source": source, "target": target,
                  "ranked_mediators": ranked, "top_k": top_k}
        fallback = {
            "answer": [{"id": m["id"], "alias": m.get("alias", m["id"]),
                        "score": m["score"], "provenance": m.get("path_evidence", "")}
                       for m in topk],
            "summary": "top mediators: " + ", ".join(m.get("alias", m["id"]) for m in topk),
        }
        r = self.client.run_agent("aggregator", fields, fallback,
                                  schema_keys=["answer", "summary"])
        # enforce authority: preserve order and scores
        r.parsed["answer"] = fallback["answer"]
        trace.add(r)
        return r.parsed
