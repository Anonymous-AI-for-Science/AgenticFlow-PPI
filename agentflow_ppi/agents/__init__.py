"""Query-time multi-agent decomposition and execution layer for AgentFlow-PPI.

This package instantiates the four agents named in the paper as concrete,
message-passing query-execution components rather than as research-meta stubs:

* :class:`PlannerAgent` decomposes a biological query into a typed logical plan
  (the Query Decomposer).
* :class:`ReachabilityAgent` runs exact SHRC pruning and emits selectivity
  evidence (the structural / "theory" agent).
* :class:`ExecutorAgent` runs typed expansion, confidence filtering, and -- only
  when admitted by cost-aware dispatch -- the learned reranker (the Multi-modal
  Executor / "engineering" agent).
* :class:`AggregatorAgent` merges candidates, restores aliases, and attaches
  provenance (the Result Aggregator / "evaluation" agent).

Every agent records the inter-agent messages it sends and receives, so that the
decomposition output, message trace, per-agent latency, and dispatch decision
can be measured end to end. This is the artifact backing the title phrase
"Multi-Agent Decomposition and Execution".
"""

from .protocol import (
    AgentRecord,
    DispatchDecision,
    PlanStep,
    QueryPlan,
    QueryRequest,
    QueryResult,
    SubgoalMessage,
)
from .planner import QueryPlannerAgent
from .reachability_agent import ReachabilityAgent
from .executor_agent import ExecutorAgent
from .aggregator_agent import AggregatorAgent
from .orchestrator import MultiAgentOrchestrator, OrchestratorConfig

__all__ = [
    "AgentRecord",
    "DispatchDecision",
    "PlanStep",
    "QueryPlan",
    "QueryRequest",
    "QueryResult",
    "SubgoalMessage",
    "QueryPlannerAgent",
    "ReachabilityAgent",
    "ExecutorAgent",
    "AggregatorAgent",
    "MultiAgentOrchestrator",
    "OrchestratorConfig",
]
