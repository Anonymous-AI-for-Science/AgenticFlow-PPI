"""Typed messages and records for the query-time multi-agent layer.

These dataclasses make the inter-agent protocol explicit and serializable so
that the decomposition output and message trace can be logged, measured, and
audited. Nothing here depends on an LLM service; the planner has an optional
LLM backend but defaults to a deterministic symbolic decomposition so that the
artifact is exactly reproducible.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


@dataclass(slots=True)
class QueryRequest:
    """A biological query handed to the orchestrator."""

    query_id: str
    source: str
    target: str
    modality: str = "functional"
    max_hops: int = 3
    min_confidence: float = 0.7
    top_k: int = 2


@dataclass(slots=True)
class PlanStep:
    """One typed operator in the logical plan emitted by the planner."""

    name: str
    operator: str  # typed_expand | confidence_filter | reachability_prune | neural_rerank | aggregate
    inputs: Tuple[str, ...]
    params: Dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class QueryPlan:
    """A logical plan: an ordered list of typed operators plus metadata."""

    query_id: str
    steps: List[PlanStep]
    planner_backend: str = "symbolic"
    notes: str = ""


@dataclass(slots=True)
class SubgoalMessage:
    """An inter-agent message carrying a subgoal and its evolving state."""

    sender: str
    recipient: str
    stage: str
    query_id: str
    payload: Dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class DispatchDecision:
    """The cost-aware decision about whether the reranker is admitted."""

    admitted: bool
    frontier_size: int
    expected_gain: float
    predicted_symbolic_cost: float
    predicted_reranker_cost: float
    objective_symbolic: float
    objective_reranked: float
    reason: str


@dataclass(slots=True)
class AgentRecord:
    """Per-agent execution record used for measurement."""

    agent_name: str
    stage: str
    wall_time_s: float
    messages_in: int
    messages_out: int
    summary: str
    metrics: Dict[str, float] = field(default_factory=dict)


@dataclass(slots=True)
class QueryResult:
    """The materialized end-to-end result for one query."""

    query_id: str
    plan: QueryPlan
    ranked_candidates: List[str]
    dispatch: DispatchDecision
    agent_records: List[AgentRecord]
    message_trace: List[SubgoalMessage]
    provenance: Dict[str, Any] = field(default_factory=dict)
    total_wall_time_s: float = 0.0

    def message_count(self) -> int:
        return len(self.message_trace)
