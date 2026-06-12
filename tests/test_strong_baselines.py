"""Tests for the strong reachability baselines (Phase 3).

Verify each faithful baseline (GRAIL, PReaCH, PLL/2-hop, LCR) is EXACT against BFS
on a random cyclic graph (cycles exercise the SCC-condensation path), and that the
reference-implementation downloader exposes the canonical repositories.
"""

import random
from collections import defaultdict, deque

from agentflow_ppi.benchmarks.strong import GrailIndex, PReaChIndex, PLLIndex, LCRIndex
from agentflow_ppi.benchmarks.external_impls.download_refs import REF_IMPLS


def _ref(n, edges):
    g = defaultdict(list)
    for u, v in edges:
        g[u].append(v)
    def reach(s, t):
        if s == t:
            return True
        seen = {s}; q = deque([s])
        while q:
            u = q.popleft()
            for w in g[u]:
                if w == t:
                    return True
                if w not in seen:
                    seen.add(w); q.append(w)
        return False
    return reach


def _random_graph(n=120, m=380, seed=5, allow_cycles=True):
    rng = random.Random(seed); edges = set()
    while len(edges) < m:
        a, b = rng.randrange(n), rng.randrange(n)
        if a == b:
            continue
        if not allow_cycles and a > b:
            a, b = b, a
        edges.add((a, b))
    return n, list(edges)


def test_strong_baselines_exact_on_cyclic_graph():
    n, edges = _random_graph(allow_cycles=True)
    ref = _ref(n, edges)
    for cls in (GrailIndex, PReaChIndex, PLLIndex):
        idx = cls(n, edges)
        mism = sum(1 for s in range(0, n, 5) for t in range(n)
                   if s != t and idx.reachable(s, t) != ref(s, t))
        assert mism == 0, f"{cls.name} had {mism} mismatches"


def test_lcr_label_constrained_exact():
    n, plain = _random_graph(n=100, m=300, seed=9, allow_cycles=False)
    rng = random.Random(1)
    labeled = [(u, v, rng.choice(["a", "b", "c"])) for u, v in plain]
    lcr = LCRIndex(n, labeled)
    gl = defaultdict(list)
    for u, v, l in labeled:
        gl[u].append((v, l))
    allowed = {"a", "b"}
    def ref_l(s, t):
        if s == t:
            return True
        seen = {s}; q = deque([s])
        while q:
            u = q.popleft()
            for w, l in gl[u]:
                if l in allowed and w not in seen:
                    if w == t:
                        return True
                    seen.add(w); q.append(w)
        return False
    mism = sum(1 for s in range(0, n, 7) for t in range(n)
               if s != t and lcr.reachable(s, t, allowed) != ref_l(s, t))
    assert mism == 0


def test_ref_impl_manifest_present():
    assert {"grail", "pll", "oreach"} <= set(REF_IMPLS)
    for impl in REF_IMPLS.values():
        assert impl.zip_urls() and all(u.startswith("https://codeload.github.com/") for u in impl.zip_urls())
