"""Cost-based dispatcher for AgentFlow-PPI.

This module estimates subquery cost from predicted compute, memory, network,
and neural inference overhead, then dispatches subqueries to workers while
respecting load balancing constraints. The released artifact also exposes the
paper's explicit reranker suppression parameters so that the gating policy is
not buried in prose: neural reranking is admitted only when the predicted
post-pruning frontier is within budget and the expected gain is large enough to
justify the extra operator. The released artifact treats the frontier budget as
an engineering capacity guardrail and the gain threshold as the validation-tuned
selectivity knob.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple


@dataclass(slots=True)
class WorkerState:
    worker_id: str
    load: float = 0.0
    active_subqueries: int = 0


@dataclass(slots=True)
class SubquerySpec:
    name: str
    operator: str
    predicted_compute: float
    predicted_memory: float
    predicted_network: float
    predicted_neural: float
    expected_gain: float
    partition_overlap: float
    metadata: Dict[str, float] = field(default_factory=dict)


@dataclass(slots=True)
class DispatchConfig:
    alpha: float = 1.0
    beta: float = 0.25
    gamma: float = 0.5
    delta: float = 0.75
    lambda_imbalance: float = 0.4
    mu_gain: float = 0.6
    max_worker_load: float = 1.0
    reranker_frontier_budget: int = 50
    # Admission uses a CALIBRATED gain predictor (admit iff predicted F1 lift > 0);
    # see agentflow_ppi/eval/harness.train_gain_predictor. The scalar threshold
    # below is the legacy uncalibrated heuristic floor on the modality-ambiguity
    # proxy, retained only for the dispatch ablation's "gain-only (uncalibrated)"
    # row. Experiments that report AgentFlow-PPI use the calibrated predictor.
    reranker_gain_threshold: float = 0.0
    calibrated_admission: bool = True


class CostBasedDispatcher:
    def __init__(self, config: DispatchConfig | None = None) -> None:
        self.config = config or DispatchConfig()

    def estimate_cost(self, subquery: SubquerySpec) -> float:
        return (
            self.config.alpha * subquery.predicted_compute
            + self.config.beta * subquery.predicted_memory
            + self.config.gamma * subquery.predicted_network
            + self.config.delta * subquery.predicted_neural
        )

    def dispatch_score(self, subquery: SubquerySpec, worker: WorkerState) -> float:
        imbalance = max(0.0, worker.load + subquery.partition_overlap - self.config.max_worker_load)
        return (
            self.estimate_cost(subquery)
            + self.config.lambda_imbalance * imbalance
            - self.config.mu_gain * subquery.expected_gain
        )

    def choose_worker(self, subquery: SubquerySpec, workers: List[WorkerState]) -> Tuple[WorkerState, float]:
        scored = [(worker, self.dispatch_score(subquery, worker)) for worker in workers]
        return min(scored, key=lambda item: item[1])

    def should_execute_neural(self, frontier_size: int, expected_gain: float) -> bool:
        """Return whether neural reranking should be admitted.

        The released paper uses two explicit thresholds:
        - predicted post-pruning frontier <= 50 candidates (capacity guardrail)
        - expected gain >= 0.05 (validation-selected selectivity threshold)
        """
        if frontier_size > self.config.reranker_frontier_budget:
            return False
        if expected_gain < self.config.reranker_gain_threshold:
            return False
        return True

    def schedule(self, subqueries: List[SubquerySpec], workers: List[WorkerState]) -> List[Tuple[str, str, float]]:
        ordered = sorted(subqueries, key=self.estimate_cost)
        assignments: List[Tuple[str, str, float]] = []
        for subquery in ordered:
            worker, score = self.choose_worker(subquery, workers)
            worker.load += min(1.0, subquery.partition_overlap + 0.05 * subquery.predicted_compute)
            worker.active_subqueries += 1
            assignments.append((subquery.name, worker.worker_id, score))
        return assignments
