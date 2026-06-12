"""Tests for modality-partitioned 2-hop labeling (fallback-free LCR)."""

import math
import random
from collections import defaultdict, deque

from agentflow_ppi.benchmarks.modality_labeling import ModalityPartitionedLabeling


def _brute(adj, s, t, allowed):
    if s == t:
        return True
    seen = {s}; q = deque([s])
    while q:
        u = q.popleft()
        for v, l in adj[u]:
            if l in allowed and v not in seen:
                if v == t:
                    return True
                seen.add(v); q.append(v)
    return False


def _graph(n, ne, mods, seed):
    rng = random.Random(seed); edges = set()
    while len(edges) < ne:
        a, b = rng.randrange(n), rng.randrange(n)
        if a != b:
            edges.add((a, b))
    return [(u, v, rng.choice(mods)) for u, v in edges]


def test_mpl_exact_label_constrained():
    mods = ["a", "b", "c", "d"]
    labeled = _graph(80, 240, mods, seed=3)
    adj = defaultdict(list)
    for u, v, l in labeled:
        adj[u].append((v, l))
    idx = ModalityPartitionedLabeling(80, labeled)
    rng = random.Random(5)
    mism = 0
    for _ in range(500):
        s, t = rng.randrange(80), rng.randrange(80)
        A = set(rng.sample(mods, rng.randint(1, 4)))
        if idx.reachable(s, t, A) != _brute(adj, s, t, A):
            mism += 1
    assert mism == 0


def test_mpl_antichain_within_sperner_bound():
    """The per-(node,hub) antichain must not exceed the Sperner bound C(m, m//2)."""
    for m in range(2, 7):
        mods = [f"m{i}" for i in range(m)]
        labeled = _graph(100, 300, mods, seed=7)
        idx = ModalityPartitionedLabeling(100, labeled)
        assert idx.stats.max_antichain <= math.comb(m, m // 2), \
            f"m={m}: antichain {idx.stats.max_antichain} > Sperner {math.comb(m, m // 2)}"


def test_mpl_fallback_free_no_traversal_state():
    """The query must not depend on adjacency (fallback-free): after construction we
    can answer with only the label stores."""
    mods = ["a", "b"]
    labeled = _graph(40, 120, mods, seed=1)
    idx = ModalityPartitionedLabeling(40, labeled)
    # drop adjacency to prove the query needs only labels
    idx.fwd = None; idx.bwd = None
    # should still answer without error (no traversal)
    _ = idx.reachable(0, 5, {"a", "b"})
