from __future__ import annotations

import random
from typing import Iterable, List, Tuple

Edge = Tuple[int, int]


def random_sparse_dag(num_nodes: int, edge_prob: float, seed: int = 7) -> List[Edge]:
    rng = random.Random(seed)
    edges: List[Edge] = []
    for u in range(num_nodes):
        for v in range(u + 1, num_nodes):
            if rng.random() < edge_prob:
                edges.append((u, v))
    return edges


def layered_biclique_core(layers: int = 4, width: int = 8) -> Tuple[int, List[Edge]]:
    edges: List[Edge] = []
    num_nodes = layers * width
    for l in range(layers - 1):
        left = range(l * width, (l + 1) * width)
        right = range((l + 1) * width, (l + 2) * width)
        for u in left:
            for v in right:
                edges.append((u, v))
    return num_nodes, edges


def diamond_fan_core(depth: int = 5, fanout: int = 4) -> Tuple[int, List[Edge]]:
    edges: List[Edge] = []
    node = 0
    current = [node]
    node += 1
    for _ in range(depth):
        nxt = list(range(node, node + fanout))
        for u in current:
            for v in nxt:
                edges.append((u, v))
        current = nxt
        node += fanout
    sink = node
    for u in current:
        edges.append((u, sink))
    return sink + 1, edges


def sparse_periphery_with_core(core_nodes: int = 24, tree_depth: int = 4, branch: int = 2) -> Tuple[int, List[Edge]]:
    num_core, core_edges = layered_biclique_core(layers=3, width=max(2, core_nodes // 3))
    edges = list(core_edges)
    node = num_core
    frontier = [0, num_core - 1]
    for attach in frontier:
        parents = [attach]
        for _ in range(tree_depth):
            children = []
            for p in parents:
                for _ in range(branch):
                    child = node
                    node += 1
                    edges.append((p, child))
                    children.append(child)
            parents = children
    return node, edges


def iter_query_pairs(num_nodes: int, max_pairs: int, seed: int = 7) -> Iterable[Tuple[int, int]]:
    rng = random.Random(seed)
    pairs = set()
    while len(pairs) < max_pairs:
        u = rng.randrange(num_nodes)
        v = rng.randrange(num_nodes)
        if u < v:
            pairs.add((u, v))
    return sorted(pairs)


