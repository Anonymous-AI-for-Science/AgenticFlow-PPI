"""Planning agent for research automation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List

from ..messages import AgentMessage, AgentResult, WorkItem, WorkType
from .base import BaseAgent


@dataclass(slots=True)
class PlannerConfig:
    theory_keywords: tuple[str, ...] = ("theorem", "lemma", "proof", "bound")
    engineering_keywords: tuple[str, ...] = ("system", "implementation", "mps", "latency")
    evaluation_keywords: tuple[str, ...] = ("experiment", "benchmark", "ablation", "metric")


class PlannerAgent(BaseAgent):
    """Decompose a top-level request into theory, engineering, and evaluation subtasks."""

    def __init__(self, name: str, bus, config: PlannerConfig | None = None) -> None:
        super().__init__(name, bus)
        self.config = config or PlannerConfig()

    async def handle(self, message: AgentMessage) -> AgentResult:
        request = str(message.work_item.payload.get("request", ""))
        subtasks = self._decompose(request, message.work_item.item_id)
        return AgentResult(
            agent_name=self.name,
            item_id=message.work_item.item_id,
            summary=f"Decomposed request into {len(subtasks)} specialized tasks.",
            artifacts={"subtasks": subtasks},
            metrics={"subtask_count": float(len(subtasks))},
        )

    def _decompose(self, request: str, root_id: str) -> List[WorkItem]:
        lowered = request.lower()
        subtasks: list[WorkItem] = [
            WorkItem(
                item_id=f"{root_id}-theory",
                work_type=WorkType.THEORY,
                title="Mathematical guarantees",
                payload={"focus": "proofs", "request": request},
                priority=0.95 if any(token in lowered for token in self.config.theory_keywords) else 0.65,
                dependencies=(root_id,),
            ),
            WorkItem(
                item_id=f"{root_id}-engineering",
                work_type=WorkType.ENGINEERING,
                title="MPS-aware implementation",
                payload={"focus": "system", "request": request},
                priority=0.90 if any(token in lowered for token in self.config.engineering_keywords) else 0.70,
                dependencies=(root_id,),
            ),
            WorkItem(
                item_id=f"{root_id}-evaluation",
                work_type=WorkType.EVALUATION,
                title="Validation and reporting",
                payload={"focus": "evaluation", "request": request},
                priority=0.85 if any(token in lowered for token in self.config.evaluation_keywords) else 0.60,
                dependencies=(root_id,),
            ),
        ]
        return subtasks


