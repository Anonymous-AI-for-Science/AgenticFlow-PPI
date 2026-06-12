"""Regression tests for the SHRC reachability index."""

from __future__ import annotations

from collections import deque

from agentflow_ppi.reachability import SHRCIndex


def _bfs_reachable(num_nodes: int, edges: list[tuple[int, int]], source: int, target: int) -> bool:
    graph = [[] for _ in range(num_nodes)]
    for src, dst in edges:
        graph[src].append(dst)
    queue = deque([source])
    seen = {source}
    while queue:
        node = queue.popleft()
        if node == target:
            return True
        for child in graph[node]:
            if child not in seen:
                seen.add(child)
                queue.append(child)
    return False


def test_shrc_matches_bfs_on_hybrid_dag() -> None:
    num_nodes = 9
    edges = [(0, 1), (1, 2), (2, 3), (1, 4), (4, 5), (2, 6), (5, 7), (6, 7), (7, 8)]
    index = SHRCIndex.from_edges(num_nodes, edges).build()

    for source in range(num_nodes):
        for target in range(num_nodes):
            assert index.reachable(source, target) == _bfs_reachable(num_nodes, edges, source, target)


def test_shrc_handles_tree_core_reentry() -> None:
    num_nodes = 10
    edges = [(0, 6), (0, 8), (1, 2), (1, 3), (1, 5), (2, 4), (2, 5), (4, 8)]
    index = SHRCIndex.from_edges(num_nodes, edges).build()

    assert index.reachable(1, 8)
    assert index.explain(1, 8).route == "core-2hop"
    assert not index.reachable(3, 8)


