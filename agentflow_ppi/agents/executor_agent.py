"""Executor Agent: typed expansion, confidence filtering, and gated reranking.

This agent owns the Multi-modal Executor stage. It (1) expands typed neighbors
of the source up to ``max_hops`` to form the raw candidate frontier, (2) applies
the confidence filter, and (3) consults the cost-aware dispatcher to decide
whether the learned reranker is admitted as a physical operator. Crucially, the
dispatch decision is computed by *comparing the optimizer objective of two
concrete plans* (symbolic-only vs. symbolic+reranker), so the cost objective
drives a measured route choice rather than being a decorative equation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from agentflow_ppi.execution.cost_dispatcher import CostBasedDispatcher, DispatchConfig
from .protocol import DispatchDecision


@dataclass(slots=True)
class TypedGraph:
    """A lightweight typed multimodal graph view used by the executor."""

    num_nodes: int
    # adjacency[u] -> list of (v, modality, confidence)
    adjacency: List[List[Tuple[int, str, float]]]


class ExecutorAgent:
    name = "executor"

    def __init__(
        self,
        graph: TypedGraph,
        dispatcher: Optional[CostBasedDispatcher] = None,
        reranker: Optional[Callable[[Sequence[int]], Dict[int, float]]] = None,
    ) -> None:
        self.graph = graph
        self.dispatcher = dispatcher or CostBasedDispatcher(DispatchConfig())
        self.reranker = reranker

    def typed_expand(self, source: int, modality: str, max_hops: int) -> List[int]:
        """Breadth-limited expansion from the source.

        Modality is treated as a *ranking* signal rather than a hard traversal
        filter: mechanistic mediators are often reached through edges of a
        different modality than the query intent (e.g. a functional cross-talk
        query whose mediators sit on physical-binding legs). Restricting
        traversal to a single modality would discard those mediators before the
        reachability and reranking stages can score them, so expansion explores
        all modalities and the modality-agreement feature is applied downstream.
        """
        seen = {source}
        frontier = [source]
        out: List[int] = []
        for _ in range(max_hops):
            nxt: List[int] = []
            for u in frontier:
                if u >= self.graph.num_nodes:
                    continue
                for v, _m, _conf in self.graph.adjacency[u]:
                    if v not in seen:
                        seen.add(v)
                        out.append(v)
                        nxt.append(v)
            frontier = nxt
        return out

    def confidence_filter(self, source: int, candidates: Sequence[int], min_confidence: float) -> List[int]:
        if source >= self.graph.num_nodes:
            return list(candidates)
        conf_of: Dict[int, float] = {}
        for v, _m, conf in self.graph.adjacency[source]:
            conf_of[v] = max(conf_of.get(v, 0.0), conf)
        # Keep candidates either directly above threshold or with no direct edge
        # (multi-hop mediators), preserving recall for the reachability stage.
        return [v for v in candidates if conf_of.get(v, 1.0) >= min_confidence]

    def decide_dispatch(self, frontier_size: int, expected_gain: float) -> DispatchDecision:
        """Compare the optimizer objective of two concrete plans and choose one.

        Both objectives are expressed as expected *regret* (lower is better) on a
        common scale where 1.0 is the cost of one full reranker invocation:

            obj_symbolic = quality_regret(no rerank)
                         = mu_gain * expected_gain        # forgone quality
            obj_rerank   = neural_cost(frontier)          # paid compute/latency

        The reranker is admitted iff it is in the capacity budget, its gain clears
        the selectivity floor, and paying its cost is cheaper than forgoing the
        quality it would recover. This is the measured realization of Eq. (1):
        a route the dispatcher selects only when the gain justifies the cost.
        """
        cfg = self.dispatcher.config
        # Neural cost grows mildly with frontier (more candidates to score).
        neural_cost = cfg.delta * (0.10 + 0.004 * max(frontier_size, 1))
        sym_cost = cfg.alpha * (0.02 * max(frontier_size, 1))
        obj_sym = sym_cost + cfg.mu_gain * expected_gain      # symbolic-only forgoes the gain
        obj_rerank = sym_cost + neural_cost                   # reranked pays compute, recovers gain
        capacity_ok = frontier_size <= cfg.reranker_frontier_budget
        gain_ok = expected_gain >= cfg.reranker_gain_threshold
        admitted = capacity_ok and gain_ok and (obj_rerank < obj_sym)
        if not capacity_ok:
            reason = "frontier exceeds capacity budget"
        elif not gain_ok:
            reason = "expected gain below selectivity threshold"
        elif obj_rerank < obj_sym:
            reason = "reranker objective lower than symbolic-only"
        else:
            reason = "reranker cost exceeds recoverable gain"
        return DispatchDecision(
            admitted=admitted,
            frontier_size=frontier_size,
            expected_gain=expected_gain,
            predicted_symbolic_cost=sym_cost,
            predicted_reranker_cost=neural_cost,
            objective_symbolic=obj_sym,
            objective_reranked=obj_rerank,
            reason=reason,
        )

    def rerank(self, candidates: Sequence[int]) -> List[int]:
        if self.reranker is None or not candidates:
            return list(candidates)
        scores = self.reranker(candidates)
        return sorted(candidates, key=lambda v: scores.get(v, 0.0), reverse=True)
