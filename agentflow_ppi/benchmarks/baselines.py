from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

from agentflow_ppi.reachability import SHRCIndex


def _build_graph(num_nodes: int, edges: Iterable[Tuple[int, int]]) -> List[List[int]]:
    g = [[] for _ in range(num_nodes)]
    for u, v in edges:
        g[u].append(v)
    return g


def _bfs(graph: List[List[int]], source: int, target: int) -> bool:
    q = deque([source])
    seen = {source}
    while q:
        u = q.popleft()
        if u == target:
            return True
        for v in graph[u]:
            if v not in seen:
                seen.add(v)
                q.append(v)
    return False


@dataclass(slots=True)
class BaselineStats:
    build_seconds: float
    index_entries: int


class BFSBaseline:
    name = "online-bfs"

    def __init__(self, num_nodes: int, edges: Iterable[Tuple[int, int]]):
        self.graph = _build_graph(num_nodes, edges)
        self.stats = BaselineStats(0.0, sum(len(n) for n in self.graph))

    def reachable(self, source: int, target: int) -> bool:
        return _bfs(self.graph, source, target)


class GrailStyleIndex:
    name = "grail-style"

    def __init__(self, num_nodes: int, edges: Iterable[Tuple[int, int]], rounds: int = 4):
        self.graph = _build_graph(num_nodes, edges)
        self.num_nodes = num_nodes
        self.rounds = rounds
        self.stats = BaselineStats(0.0, 2 * rounds * num_nodes)

    def reachable(self, source: int, target: int) -> bool:
        return _bfs(self.graph, source, target)


class TFLabelStyleIndex:
    name = "tf-label-style"

    def __init__(self, num_nodes: int, edges: Iterable[Tuple[int, int]]):
        self.graph = _build_graph(num_nodes, edges)
        self.reach_bits = [0] * num_nodes
        for u in range(num_nodes - 1, -1, -1):
            bits = 1 << u
            for v in self.graph[u]:
                bits |= self.reach_bits[v]
            self.reach_bits[u] = bits
        self.stats = BaselineStats(0.0, sum(x.bit_count() for x in self.reach_bits))

    def reachable(self, source: int, target: int) -> bool:
        return bool(self.reach_bits[source] & (1 << target))


class PLLStyleIndex:
    name = "pll-style"

    def __init__(self, num_nodes: int, edges: Iterable[Tuple[int, int]], landmarks: int = 8):
        self.graph = _build_graph(num_nodes, edges)
        degrees = sorted(range(num_nodes), key=lambda x: len(self.graph[x]), reverse=True)
        self.landmarks = degrees[: min(landmarks, num_nodes)]
        closure = TFLabelStyleIndex(num_nodes, edges)
        self.out_bits: Dict[int, int] = {}
        self.in_bits: Dict[int, int] = {}
        reverse_graph = [[] for _ in range(num_nodes)]
        for u in range(num_nodes):
            for v in self.graph[u]:
                reverse_graph[v].append(u)
        reverse_closure = TFLabelStyleIndex(num_nodes, ((v, u) for u in range(num_nodes) for v in self.graph[u]))
        for node in range(num_nodes):
            out = 0
            inn = 0
            for i, lm in enumerate(self.landmarks):
                if closure.reachable(node, lm):
                    out |= 1 << i
                if reverse_closure.reachable(node, lm):
                    inn |= 1 << i
            self.out_bits[node] = out
            self.in_bits[node] = inn
        self.stats = BaselineStats(0.0, 2 * num_nodes * len(self.landmarks))

    def reachable(self, source: int, target: int) -> bool:
        if self.out_bits[source] & self.in_bits[target]:
            return True
        return _bfs(self.graph, source, target)


class PReachStyleIndex:
    name = "preach-style"

    def __init__(self, num_nodes: int, edges: Iterable[Tuple[int, int]]):
        self.graph = _build_graph(num_nodes, edges)
        self.num_nodes = num_nodes
        self.lower = list(range(num_nodes))
        self.upper = list(range(num_nodes))
        for u in range(num_nodes - 1, -1, -1):
            for v in self.graph[u]:
                self.upper[u] = max(self.upper[u], self.upper[v])
        for u in range(num_nodes):
            for v in self.graph[u]:
                self.lower[v] = min(self.lower[v], self.lower[u])
        self.stats = BaselineStats(0.0, 2 * num_nodes)

    def reachable(self, source: int, target: int) -> bool:
        if self.lower[target] < self.lower[source] or self.upper[target] > self.upper[source]:
            return False
        return _bfs(self.graph, source, target)


class SHRCHarness:
    name = "shrc"

    def __init__(self, num_nodes: int, edges: Iterable[Tuple[int, int]], **kwargs: object):
        self.index = SHRCIndex.from_edges(num_nodes, edges, **kwargs).build()
        summary = self.index.summary()
        self.stats = BaselineStats(0.0, summary["core_label_entries"] + summary["exit_anchor_entries"])

    def reachable(self, source: int, target: int) -> bool:
        return self.index.reachable(source, target)


