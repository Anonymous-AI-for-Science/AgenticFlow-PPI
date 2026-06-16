from __future__ import annotations

"""Cycle handling utilities for directional biological graphs.

The main SHRC index expects an acyclic directional graph. Real regulatory graphs may
contain feedback loops, so the artifact exposes cycle contraction as an explicit,
auditable preprocessing step instead of treating it as an implicit side effect.
"""

from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Tuple

Edge = Tuple[int, int]


@dataclass(slots=True)
class CondensationResult:
    num_original_nodes: int
    num_components: int
    component_of: List[int]
    dag_edges: List[Edge]
    removed_intra_component_edges: List[Edge]
    component_sizes: Dict[int, int]


def condense_to_dag(num_nodes: int, edges: Sequence[Edge]) -> CondensationResult:
    """Contract strongly connected components and return a condensation DAG.

    Complexity:
        O(|V| + |E|) time and O(|V| + |E|) space via Tarjan's SCC algorithm.
    """

    graph = [[] for _ in range(num_nodes)]
    for u, v in edges:
        graph[u].append(v)

    index = 0
    indices = [-1] * num_nodes
    low = [0] * num_nodes
    stack: List[int] = []
    on_stack = [False] * num_nodes
    comp_of = [-1] * num_nodes
    comp_count = 0

    # Iterative Tarjan SCC: an explicit work stack replaces Python recursion so that
    # graphs with long directed chains (real STRING-scale snapshots have paths far
    # deeper than the default recursion limit of ~1000) do not raise RecursionError.
    # Each work item is (vertex, next_neighbor_index); the algorithm is identical to
    # the textbook recursive form, only the call stack is made explicit.
    for root in range(num_nodes):
        if indices[root] != -1:
            continue
        work: List[List[int]] = [[root, 0]]
        while work:
            v, pi = work[-1]
            if pi == 0:
                indices[v] = low[v] = index
                index += 1
                stack.append(v)
                on_stack[v] = True
            recursed = False
            i = pi
            while i < len(graph[v]):
                w = graph[v][i]
                if indices[w] == -1:
                    # "recurse" into w: resume v at the next neighbor afterwards
                    work[-1][1] = i + 1
                    work.append([w, 0])
                    recursed = True
                    break
                elif on_stack[w]:
                    low[v] = min(low[v], indices[w])
                i += 1
            if recursed:
                continue
            # all neighbors processed: finish v (root-of-SCC check + popping)
            if low[v] == indices[v]:
                while True:
                    w = stack.pop()
                    on_stack[w] = False
                    comp_of[w] = comp_count
                    if w == v:
                        break
                comp_count += 1
            work.pop()
            if work:  # propagate low-link to the parent that recursed into v
                parent = work[-1][0]
                low[parent] = min(low[parent], low[v])

    dag = set()
    removed: List[Edge] = []
    sizes: Dict[int, int] = {c: 0 for c in range(comp_count)}
    for node in range(num_nodes):
        sizes[comp_of[node]] += 1
    for u, v in edges:
        cu, cv = comp_of[u], comp_of[v]
        if cu == cv:
            removed.append((u, v))
        else:
            dag.add((cu, cv))
    return CondensationResult(
        num_original_nodes=num_nodes,
        num_components=comp_count,
        component_of=comp_of,
        dag_edges=sorted(dag),
        removed_intra_component_edges=removed,
        component_sizes=sizes,
    )
