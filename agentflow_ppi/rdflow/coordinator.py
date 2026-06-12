"""Coordinator for the Apple-Silicon-aware asynchronous research flow."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Dict, List, Sequence

from .agents import EngineeringAgent, EvaluationAgent, PlannerAgent, TheoryAgent
from .bus import AsyncMessageBus
from .device import DeviceManager, MPSExecutionConfig
from .messages import AgentMessage, AgentResult, WorkItem, WorkType
from .router import PriorityRouter, RouterConfig


@dataclass(slots=True)
class RDFlowConfig:
    """Configuration of the single-host asynchronous agent system."""

    batch_size: int = 8
    router_temperature: float = 0.5
    prefer_mps: bool = True
    synchronize_on_barrier: bool = False


class RDFlowCoordinator:
    """Orchestrate a research-and-development flow across specialized agents.

    The coordinator uses three mechanisms for efficiency:
    1. a single shared device manager so all neural routing happens on one
       accelerator context,
    2. batched router inference to amortize MPS kernel-launch overhead, and
    3. queue isolation so slow agents do not block unrelated work.
    """

    def __init__(self, config: RDFlowConfig | None = None) -> None:
        self.config = config or RDFlowConfig()
        self.bus = AsyncMessageBus()
        self.device_manager = DeviceManager(
            MPSExecutionConfig(
                prefer_mps=self.config.prefer_mps,
                synchronize_on_barrier=self.config.synchronize_on_barrier,
            )
        )
        self.agents = {
            "planner": PlannerAgent("planner", self.bus),
            "theory": TheoryAgent("theory", self.bus),
            "engineering": EngineeringAgent("engineering", self.bus),
            "evaluation": EvaluationAgent("evaluation", self.bus),
        }
        self.router = PriorityRouter(
            agent_names=["theory", "engineering", "evaluation"],
            config=RouterConfig(temperature=self.config.router_temperature),
        )
        self.device_manager.move_module(self.router)

    async def execute(self, request: str) -> Dict[str, object]:
        root = WorkItem(
            item_id="root",
            work_type=WorkType.PLANNING,
            title="Top-level research request",
            payload={"request": request},
            priority=1.0,
        )
        await self.bus.publish(AgentMessage(sender="user", recipient="planner", work_item=root, stage="request"))

        planner_result = await self.agents["planner"].run_once(timeout=1.0)
        if planner_result is None:
            raise RuntimeError("Planner did not produce a result in time.")

        subtasks = list(planner_result.artifacts["subtasks"])
        routes = self._route_subtasks(subtasks)
        for work_item, recipient in zip(subtasks, routes):
            await self.bus.publish(AgentMessage(sender="planner", recipient=recipient, work_item=work_item, stage="subtask"))

        worker_results = await asyncio.gather(
            self.agents["theory"].run_once(timeout=1.0),
            self.agents["engineering"].run_once(timeout=1.0),
            self.agents["evaluation"].run_once(timeout=1.0),
        )
        finalized = self._synthesize([planner_result, *[r for r in worker_results if r is not None]])
        self.device_manager.barrier()
        return finalized

    def _route_subtasks(self, subtasks: Sequence[WorkItem]) -> List[str]:
        # Batch routing is O(B * d * h) for batch size B and tiny network width h.
        # On Apple Silicon, performing this in one MPS forward pass is far cheaper
        # than scoring each subtask separately because kernel-launch overhead is amortized.
        return self.router.route(list(subtasks), self.device_manager)

    def _synthesize(self, results: Sequence[AgentResult]) -> Dict[str, object]:
        summaries = {result.agent_name: result.summary for result in results}
        artifacts: Dict[str, object] = {}
        metrics: Dict[str, float] = {}
        for result in results:
            artifacts[result.agent_name] = result.artifacts
            for key, value in result.metrics.items():
                metrics[f"{result.agent_name}.{key}"] = value
        return {
            "device": self.device_manager.report(),
            "summaries": summaries,
            "artifacts": artifacts,
            "metrics": metrics,
        }


