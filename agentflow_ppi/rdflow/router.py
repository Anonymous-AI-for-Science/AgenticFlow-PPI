"""Neural priority router for asynchronous learned-operator task dispatch."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import torch
from torch import nn

from .device import DeviceManager
from .messages import WorkItem, work_item_to_features


@dataclass(slots=True)
class RouterConfig:
    input_dim: int = 9
    hidden_dim: int = 32
    temperature: float = 0.5
    seed: int = 7


class PriorityRouter(nn.Module):
    """Tiny MLP that scores work items for a fixed roster of agents.

    The model is intentionally small so that it benefits from MPS acceleration
    even on laptop hardware. Forward passes are performed in batches to amortize
    kernel-launch overhead.
    """

    def __init__(self, agent_names: Sequence[str], config: RouterConfig | None = None) -> None:
        super().__init__()
        self.agent_names = list(agent_names)
        self.config = config or RouterConfig()
        generator = torch.Generator(device="cpu")
        generator.manual_seed(self.config.seed)
        self.network = nn.Sequential(
            nn.Linear(self.config.input_dim, self.config.hidden_dim),
            nn.GELU(),
            nn.Linear(self.config.hidden_dim, len(self.agent_names)),
        )
        for parameter in self.network.parameters():
            if parameter.ndim > 1:
                nn.init.xavier_uniform_(parameter, gain=0.7, generator=generator)
            else:
                nn.init.zeros_(parameter)

    def encode_batch(self, work_items: Iterable[WorkItem], manager: DeviceManager) -> torch.Tensor:
        rows = [work_item_to_features(item) for item in work_items]
        if not rows:
            return manager.tensor([], dtype=torch.float32).reshape(0, self.config.input_dim)
        return manager.tensor(rows, dtype=torch.float32)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        logits = self.network(features)
        return logits / self.config.temperature

    @torch.no_grad()
    def route_probabilities(self, work_items: Sequence[WorkItem], manager: DeviceManager) -> torch.Tensor:
        features = self.encode_batch(work_items, manager)
        logits = self.forward(features)
        return torch.softmax(logits, dim=-1)

    @torch.no_grad()
    def route(self, work_items: Sequence[WorkItem], manager: DeviceManager) -> list[str]:
        probs = self.route_probabilities(work_items, manager)
        indices = torch.argmax(probs, dim=-1).tolist()
        manager.barrier()
        return [self.agent_names[index] for index in indices]


