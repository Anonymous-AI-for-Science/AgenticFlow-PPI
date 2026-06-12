"""End-to-end execution wrapper for AgentFlow-PPI."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch

from agentflow_ppi.data.string_loader import STRINGLoader
from agentflow_ppi.execution.cost_dispatcher import CostBasedDispatcher, DispatchConfig, SubquerySpec, WorkerState
from agentflow_ppi.models.multimodal_gin import MultiModalGIN, MultiModalGINConfig
from agentflow_ppi.planner.ollama_planner import OllamaPlanner
from agentflow_ppi.reachability import SHRCIndex


@dataclass(slots=True)
class PipelineConfig:
    workers: int = 3
    top_k: int = 10


class AgentFlowPipeline:
    """Glue code that mirrors the paper's four-stage execution flow.

    The pipeline now exposes an SHRC-backed reachability service for typed path
    predicates. In the intended use case, a regulatory or pathway DAG is passed
    separately from the dense multimodal PPI association graph; the SHRC index is
    then used as an exact pruning oracle before expensive neural reranking.
    """

    def __init__(
        self,
        planner: Optional[OllamaPlanner] = None,
        dispatcher: Optional[CostBasedDispatcher] = None,
        model: Optional[MultiModalGIN] = None,
        config: Optional[PipelineConfig] = None,
    ) -> None:
        self.planner = planner or OllamaPlanner()
        self.dispatcher = dispatcher or CostBasedDispatcher(DispatchConfig())
        self.model = model or MultiModalGIN(MultiModalGINConfig(graph_dim=1, sequence_dim=1, structure_dim=1))
        self.config = config or PipelineConfig()
        self.reachability_index: Optional[SHRCIndex] = None

    def make_subqueries(self, plan: Dict[str, Any]) -> List[SubquerySpec]:
        specs: List[SubquerySpec] = []
        for index, subquery in enumerate(plan.get("subqueries", [])):
            operator = subquery.get("operator", "typed_expand")
            predicted_neural = 0.6 if operator == "neural_rerank" else 0.0
            expected_gain = 0.8 if operator == "neural_rerank" else 0.3
            if operator == "constrained_path":
                expected_gain = 0.55
            specs.append(
                SubquerySpec(
                    name=subquery["name"],
                    operator=operator,
                    predicted_compute=0.3 + 0.15 * index,
                    predicted_memory=0.2 + 0.05 * index,
                    predicted_network=0.1 + 0.08 * index,
                    predicted_neural=predicted_neural,
                    expected_gain=expected_gain,
                    partition_overlap=0.1 + 0.05 * index,
                )
            )
        return specs

    def build_reachability_index(self, dag_edges: Sequence[Tuple[int, int]], num_nodes: int) -> SHRCIndex:
        """Construct the sparsity-driven hybrid reachability index.

        The preprocessing cost is dominated by the core 2-hop construction on the
        reduced core. Query evaluation then becomes either an O(1) interval check
        on the peeled forest or an exact 2-hop lookup on the augmented core.
        """
        self.reachability_index = SHRCIndex.from_edges(num_nodes=num_nodes, edges=dag_edges).build()
        return self.reachability_index

    def execute(
        self,
        query: str,
        data_root: str,
        dag_edges: Optional[Sequence[Tuple[int, int]]] = None,
        dag_num_nodes: Optional[int] = None,
        reachability_probe: Optional[Tuple[int, int]] = None,
    ) -> Dict[str, Any]:
        plan = self.planner.plan_query(query)
        subqueries = self.make_subqueries(plan)
        workers = [WorkerState(worker_id=f"worker-{i}") for i in range(self.config.workers)]
        schedule = self.dispatcher.schedule(subqueries, workers)

        loader = STRINGLoader(data_root)
        graph = loader.build_pyg_graph(max_chunks=1)
        manifest_json = loader.manifest.to_json()

        reachability_summary = None
        reachability_trace = None
        if dag_edges is not None and dag_num_nodes is not None:
            index = self.build_reachability_index(dag_edges=dag_edges, num_nodes=dag_num_nodes)
            reachability_summary = index.summary()
            if reachability_probe is not None:
                reachability_trace = index.explain(*reachability_probe).__dict__

        if graph.x.size(1) == 3:
            scores = graph.x[:, 0]
        else:
            with torch.no_grad():
                self.model.eval()
                scores = self.model.score_candidates(graph)

        topk = min(self.config.top_k, scores.numel())
        top_indices = torch.topk(scores, k=topk).indices.tolist()

        return {
            "plan": plan,
            "schedule": schedule,
            "top_indices": top_indices,
            "manifest": manifest_json,
            "reachability_summary": reachability_summary,
            "reachability_trace": reachability_trace,
        }


