"""Async message bus for single-node learned-operator collaboration."""

from __future__ import annotations

import asyncio
from collections import defaultdict
from typing import DefaultDict

from .messages import AgentMessage


class AsyncMessageBus:
    """Named-queue message bus backed by asyncio.Queue.

    Time complexity:
        * publish: O(1)
        * receive: O(1) amortized
    Space complexity is linear in the number of in-flight messages.
    """

    def __init__(self) -> None:
        self._queues: DefaultDict[str, asyncio.Queue[AgentMessage]] = defaultdict(asyncio.Queue)

    async def publish(self, message: AgentMessage) -> None:
        await self._queues[message.recipient].put(message)

    async def receive(self, recipient: str) -> AgentMessage:
        return await self._queues[recipient].get()

    def task_done(self, recipient: str) -> None:
        self._queues[recipient].task_done()

    async def join_all(self) -> None:
        await asyncio.gather(*(queue.join() for queue in self._queues.values()))


