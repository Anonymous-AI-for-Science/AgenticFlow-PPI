"""Typed messages for the asynchronous research automation flow."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from time import time
from typing import Any, Dict, Mapping, Tuple


class WorkType(str, Enum):
    """Logical task categories handled by the agent swarm."""

    PLANNING = "planning"
    THEORY = "theory"
    ENGINEERING = "engineering"
    EVALUATION = "evaluation"
    SYNTHESIS = "synthesis"


@dataclass(slots=True)
class WorkItem:
    """A research subproblem routed through the agent bus."""

    item_id: str
    work_type: WorkType
    title: str
    payload: Dict[str, Any]
    priority: float = 0.5
    dependencies: Tuple[str, ...] = ()
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class AgentMessage:
    """Envelope used by agents to exchange tasks and results."""

    sender: str
    recipient: str
    work_item: WorkItem
    stage: str
    created_at: float = field(default_factory=time)


@dataclass(slots=True)
class AgentResult:
    """Materialized output produced by an agent."""

    agent_name: str
    item_id: str
    summary: str
    artifacts: Dict[str, Any]
    metrics: Dict[str, float] = field(default_factory=dict)


def work_item_to_features(work_item: WorkItem) -> list[float]:
    """Encode a work item as a dense numeric feature vector.

    The vector is intentionally compact so that a batch of work items can be
    scored efficiently on MPS. The feature design mixes normalized priority and
    simple lexical statistics with one-hot task indicators.
    """
    payload_text = " ".join(f"{k}:{v}" for k, v in sorted(work_item.payload.items()))
    title_len = len(work_item.title.split())
    payload_len = len(payload_text.split())
    dep_count = len(work_item.dependencies)
    type_flags = [1.0 if work_item.work_type is value else 0.0 for value in WorkType]
    return [
        float(work_item.priority),
        min(title_len / 32.0, 1.0),
        min(payload_len / 128.0, 1.0),
        min(dep_count / 8.0, 1.0),
        *type_flags,
    ]


def merge_metadata(base: Mapping[str, Any], extra: Mapping[str, Any]) -> Dict[str, Any]:
    merged = dict(base)
    merged.update(extra)
    return merged


