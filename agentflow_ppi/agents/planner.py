"""Query Planner Agent: decomposes a biological query into a typed logical plan.

The planner is the agent the paper calls the *Query Decomposer*. It produces a
concrete, inspectable decomposition output: an ordered list of typed operators
with parameters. By default it uses a deterministic, schema-constrained
decomposition so the artifact is reproducible without any network service. An
optional LLM backend (Ollama) can be enabled; if it is unavailable or returns
malformed JSON, the planner falls back to the deterministic decomposition and
records which backend was used.
"""

from __future__ import annotations

from typing import List, Optional

from .protocol import PlanStep, QueryPlan, QueryRequest


class QueryPlannerAgent:
    name = "planner"

    def __init__(self, llm_planner: Optional[object] = None) -> None:
        # llm_planner is duck-typed: any object with .plan_query(str) -> dict.
        self.llm_planner = llm_planner

    def decompose(self, request: QueryRequest) -> QueryPlan:
        backend = "symbolic"
        notes = "deterministic schema-constrained decomposition"
        if self.llm_planner is not None:
            try:
                raw = self.llm_planner.plan_query(
                    f"functional mediators between {request.source} and {request.target} "
                    f"via {request.modality} edges"
                )
                steps = self._steps_from_llm(raw, request)
                if steps:
                    backend = "llm+schema"
                    notes = "LLM-proposed plan validated against the operator schema"
                    return QueryPlan(request.query_id, steps, backend, notes)
            except Exception:
                pass  # fall through to deterministic plan
        return QueryPlan(request.query_id, self._default_steps(request), backend, notes)

    def _default_steps(self, request: QueryRequest) -> List[PlanStep]:
        return [
            PlanStep(
                name="candidate_generation",
                operator="typed_expand",
                inputs=("source", "target"),
                params={"modality": request.modality, "max_hops": request.max_hops},
            ),
            PlanStep(
                name="evidence_pruning",
                operator="confidence_filter",
                inputs=("candidate_generation",),
                params={"min_confidence": request.min_confidence},
            ),
            PlanStep(
                name="exact_reachability",
                operator="reachability_prune",
                inputs=("evidence_pruning",),
                params={"semantics": "existential_directed"},
            ),
            PlanStep(
                name="semantic_ranking",
                operator="neural_rerank",
                inputs=("exact_reachability",),
                params={"model": "multimodal_gin"},
            ),
            PlanStep(
                name="final_aggregation",
                operator="aggregate",
                inputs=("semantic_ranking",),
                params={"top_k": request.top_k},
            ),
        ]

    def _steps_from_llm(self, raw: dict, request: QueryRequest) -> List[PlanStep]:
        allowed = {
            "typed_expand",
            "confidence_filter",
            "reachability_prune",
            "neural_rerank",
            "aggregate",
            "motif_match",
        }
        steps: List[PlanStep] = []
        for sub in raw.get("subqueries", []):
            op = sub.get("operator")
            if op not in allowed:
                continue
            steps.append(
                PlanStep(
                    name=str(sub.get("name", op)),
                    operator=op,
                    inputs=tuple(sub.get("inputs", ())),
                    params=dict(sub.get("params", {})),
                )
            )
        # The schema validator guarantees a reachability prune precedes reranking.
        ops = [s.operator for s in steps]
        if "neural_rerank" in ops and "reachability_prune" not in ops:
            idx = ops.index("neural_rerank")
            steps.insert(
                idx,
                PlanStep(
                    name="exact_reachability",
                    operator="reachability_prune",
                    inputs=("evidence_pruning",),
                    params={"semantics": "existential_directed"},
                ),
            )
        return steps
