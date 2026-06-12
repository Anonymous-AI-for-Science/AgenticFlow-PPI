"""Engine adapter interface and the in-process SHRC reference engine.

Every baseline implements the same interface so the harness can measure them
uniformly: load the canonical export, then answer each query's gold mediators with
cold and warm timing. Adapters that need an external server (Neo4j, PostgreSQL,
TigerGraph) raise EngineUnavailable when the server/driver is absent, so the
harness can skip them gracefully and still run the in-process engine offline.
"""

from __future__ import annotations

import time
import tracemalloc
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set


class EngineUnavailable(RuntimeError):
    """Raised when a baseline's server/driver/container is not available."""


@dataclass
class QueryTiming:
    qid: int
    answer: Set[int]
    cold_ms: float
    warm_ms: float


@dataclass
class EngineResult:
    engine: str
    available: bool
    load_seconds: float = 0.0
    peak_mem_mb: float = 0.0
    timings: List[QueryTiming] = field(default_factory=list)
    timed_out: int = 0
    note: str = ""

    def answers(self) -> Dict[int, Set[int]]:
        return {t.qid: t.answer for t in self.timings}


class BaseEngine:
    name = "base"

    def load(self, export_dir: Path) -> None:
        raise NotImplementedError

    def mediators(self, source: int, target: int, gold: List[int]) -> Set[int]:
        raise NotImplementedError

    def close(self) -> None:
        pass


# --------------------------- in-process SHRC ------------------------------- #

class InProcessSHRCEngine(BaseEngine):
    """The paper's own engine: SHRC exact reachability + gold-mediator filter,
    entirely in-process. Always available; serves as the correctness reference and
    the latency floor."""
    name = "shrc-inproc"

    def load(self, export_dir: Path) -> None:
        from agentflow_ppi.engines.canonical_export import load_export
        from agentflow_ppi.reachability import SHRCIndex
        from agentflow_ppi.data.cycle_handling import condense_to_dag
        self.nodes, edges, self.queries = load_export(export_dir)
        n = len(self.nodes)
        # directed edges (expand undirected to both directions)
        dir_edges = []
        for s, d, _m, _sc, directed in edges:
            dir_edges.append((s, d))
            if not directed:
                dir_edges.append((d, s))
        cond = condense_to_dag(n, dir_edges)
        self.comp = cond.component_of
        self.shrc = SHRCIndex.from_edges(num_nodes=cond.num_components,
                                         edges=cond.dag_edges).build()

    def _reach(self, a: int, b: int) -> bool:
        return self.comp[a] == self.comp[b] or self.shrc.reachable(self.comp[a], self.comp[b])

    def mediators(self, source: int, target: int, gold: List[int]) -> Set[int]:
        out = set()
        for m in gold:
            if m in (source, target):
                continue
            if self._reach(source, m) and self._reach(m, target):
                out.add(m)
        return out
