"""Multi-Agent Orchestrator: the measured end-to-end query flow.

This module wires the four query-time agents into one execution flow and records
*every* inter-agent message, per-agent wall time, the decomposition output, and
the cost-aware dispatch decision. It is the concrete artifact behind the title
phrase "Multi-Agent Decomposition and Execution": running ``execute`` on a query
produces a :class:`QueryResult` whose ``message_trace`` and ``agent_records`` can
be serialized and measured.

The flow:

    user --(request)--> planner
    planner --(plan)--> executor          # typed_expand + confidence_filter
    executor --(frontier)--> reachability  # exact SHRC pruning -> selectivity
    reachability --(pruned frontier + selectivity)--> executor  # cost-aware dispatch
    executor --(ranked)--> aggregator      # top-k + provenance
    aggregator --(result)--> user

Each arrow is a real :class:`SubgoalMessage` appended to the trace.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable, Dict, List, Mapping, Optional, Sequence

from agentflow_ppi.execution.cost_dispatcher import CostBasedDispatcher, DispatchConfig
from agentflow_ppi.reachability import SHRCIndex

from .aggregator_agent import AggregatorAgent
from .executor_agent import ExecutorAgent, TypedGraph
from .planner import QueryPlannerAgent
from .protocol import (
    AgentRecord,
    QueryPlan,
    QueryRequest,
    QueryResult,
    SubgoalMessage,
)
from .reachability_agent import ReachabilityAgent


@dataclass(slots=True)
class OrchestratorConfig:
    dispatch: DispatchConfig = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.dispatch is None:
            self.dispatch = DispatchConfig()


class MultiAgentOrchestrator:
    def __init__(
        self,
        typed_graph: TypedGraph,
        shrc_index: SHRCIndex,
        id_to_name: Mapping[int, str],
        reranker: Optional[Callable[[Sequence[int]], Dict[int, float]]] = None,
        symbolic_scorer: Optional[Callable[[Sequence[int]], Dict[int, float]]] = None,
        llm_planner: Optional[object] = None,
        config: Optional[OrchestratorConfig] = None,
    ) -> None:
        self.config = config or OrchestratorConfig()
        dispatcher = CostBasedDispatcher(self.config.dispatch)
        self.planner = QueryPlannerAgent(llm_planner=llm_planner)
        self.reachability = ReachabilityAgent(shrc_index)
        self.executor = ExecutorAgent(typed_graph, dispatcher=dispatcher, reranker=reranker)
        self.aggregator = AggregatorAgent(id_to_name)
        # A symbolic scorer used when the reranker is suppressed (path-score order).
        self.symbolic_scorer = symbolic_scorer
        self.name_to_id = {name: idx for idx, name in id_to_name.items()}

    def execute(self, request: QueryRequest, expected_gain: float) -> QueryResult:
        t_start = time.perf_counter()
        trace: List[SubgoalMessage] = []
        records: List[AgentRecord] = []

        src = self.name_to_id.get(request.source, -1)
        tgt = self.name_to_id.get(request.target, -1)

        # 1. Planner: decompose.
        t0 = time.perf_counter()
        trace.append(SubgoalMessage("user", "planner", "request", request.query_id,
                                    {"source": request.source, "target": request.target}))
        plan: QueryPlan = self.planner.decompose(request)
        trace.append(SubgoalMessage("planner", "executor", "plan", request.query_id,
                                    {"steps": [s.operator for s in plan.steps], "backend": plan.planner_backend}))
        records.append(AgentRecord(self.planner.name, "decompose", time.perf_counter() - t0,
                                   1, 1, f"emitted {len(plan.steps)}-operator plan",
                                   {"num_steps": float(len(plan.steps))}))

        # 2. Executor: typed expansion + confidence filter.
        t0 = time.perf_counter()
        raw = self.executor.typed_expand(src, request.modality, request.max_hops) if src >= 0 else []
        filtered = self.executor.confidence_filter(src, raw, request.min_confidence)
        trace.append(SubgoalMessage("executor", "reachability", "frontier", request.query_id,
                                    {"raw_frontier": len(raw), "filtered_frontier": len(filtered)}))
        records.append(AgentRecord(self.executor.name, "expand_filter", time.perf_counter() - t0,
                                   1, 1, f"expanded {len(raw)} -> filtered {len(filtered)}",
                                   {"raw_frontier": float(len(raw)), "filtered_frontier": float(len(filtered))}))

        # 3. Reachability agent: exact SHRC pruning + selectivity evidence.
        t0 = time.perf_counter()
        report = self.reachability.prune(src, tgt, filtered) if (src >= 0 and tgt >= 0) else None
        post = report.post_frontier if report else 0
        kept = report.kept if report else []
        trace.append(SubgoalMessage("reachability", "executor", "pruned", request.query_id,
                                    {"post_frontier": post,
                                     "selectivity": round(report.selectivity, 4) if report else 0.0}))
        records.append(AgentRecord(self.reachability.name, "shrc_prune", time.perf_counter() - t0,
                                   1, 1, f"pruned to {post} exact mediators",
                                   {"post_frontier": float(post),
                                    "selectivity": float(report.selectivity) if report else 0.0}))

        # 4. Executor: cost-aware dispatch decision (compare two concrete plans).
        t0 = time.perf_counter()
        decision = self.executor.decide_dispatch(post, expected_gain)
        if decision.admitted and self.executor.reranker is not None:
            ranked = self.executor.rerank(kept)
            applied = ["typed_expand", "confidence_filter", "reachability_prune", "neural_rerank"]
        else:
            # symbolic-only ordering when the reranker is suppressed
            if self.symbolic_scorer is not None and kept:
                scores = self.symbolic_scorer(kept)
                ranked = sorted(kept, key=lambda v: scores.get(v, 0.0), reverse=True)
            else:
                ranked = list(kept)
            applied = ["typed_expand", "confidence_filter", "reachability_prune"]
        trace.append(SubgoalMessage("executor", "aggregator", "ranked", request.query_id,
                                    {"admitted": decision.admitted, "reason": decision.reason,
                                     "num_ranked": len(ranked)}))
        records.append(AgentRecord(self.executor.name, "dispatch_rank", time.perf_counter() - t0,
                                   1, 1, f"reranker {'admitted' if decision.admitted else 'suppressed'}: {decision.reason}",
                                   {"admitted": 1.0 if decision.admitted else 0.0}))

        # 5. Aggregator: top-k + provenance.
        t0 = time.perf_counter()
        agg = self.aggregator.aggregate(ranked, request.top_k, applied)
        trace.append(SubgoalMessage("aggregator", "user", "result", request.query_id,
                                    {"returned": agg["provenance"]["num_returned"]}))
        records.append(AgentRecord(self.aggregator.name, "aggregate", time.perf_counter() - t0,
                                   1, 1, f"returned top-{request.top_k}", {}))

        total = time.perf_counter() - t_start
        return QueryResult(
            query_id=request.query_id,
            plan=plan,
            ranked_candidates=agg["ranked_names"],
            dispatch=decision,
            agent_records=records,
            message_trace=trace,
            provenance=agg["provenance"],
            total_wall_time_s=total,
        )
