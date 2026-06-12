"""Base classes for asynchronous R&D agents."""

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from typing import Optional

from ..bus import AsyncMessageBus
from ..messages import AgentMessage, AgentResult


class BaseAgent(ABC):
    """Abstract async worker.

    Each worker consumes messages from a dedicated queue and emits a structured
    AgentResult. The base loop is written once so that specialization remains
    focused on the domain logic.
    """

    def __init__(self, name: str, bus: AsyncMessageBus) -> None:
        self.name = name
        self.bus = bus
        self._shutdown = asyncio.Event()

    async def stop(self) -> None:
        self._shutdown.set()

    async def run_once(self, timeout: Optional[float] = None) -> AgentResult | None:
        if self._shutdown.is_set():
            return None
        try:
            message = await asyncio.wait_for(self.bus.receive(self.name), timeout=timeout)
        except asyncio.TimeoutError:
            return None
        try:
            result = await self.handle(message)
            return result
        finally:
            self.bus.task_done(self.name)

    @abstractmethod
    async def handle(self, message: AgentMessage) -> AgentResult:
        raise NotImplementedError


