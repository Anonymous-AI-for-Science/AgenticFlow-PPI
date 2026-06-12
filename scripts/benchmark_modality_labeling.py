"""Benchmark: modality-partitioned 2-hop labeling (fallback-free LCR).

Measures, on synthetic typed graphs at several modality counts:
  * exactness vs a brute-force label-restricted BFS (must be 0 mismatches),
  * the maximum per-(node,hub) antichain size vs the Sperner bound C(m, floor(m/2)),
    which is the empirical check of the space-complexity proposition,
  * fallback-free query latency vs a per-query label-restricted BFS (the old path),
  * total index entries.

Writes results/modality_labeling.json and modality_labeling.csv.
"""

from __future__ import annotations

import csv
import json
import math
import random
import time
from collections import defaultdict, deque
from pathlib import Path

from agentflow_ppi.benchmarks.modality_labeling import ModalityPartitionedLabeling


def _brute_lcr(adj, s, t, allowed):
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


def _random_typed_graph(n, m_edges, mods, seed):
    rng = random.Random(seed)
    edges = set()
    while len(edges) < m_edges:
        a, b = rng.randrange(n), rng.randrange(n)
        if a != b:
            edges.add((a, b))
    return [(u, v, rng.choice(mods)) for u, v in edges]


def main():
    out = Path(__file__).resolve().parents[1] / "results"; out.mkdir(parents=True, exist_ok=True)
    rows = []
    for m in range(2, 7):
        mods = [f"mod{i}" for i in range(m)]
        n, ne = 120, 360
        labeled = _random_typed_graph(n, ne, mods, seed=7)
        adj = defaultdict(list)
        for u, v, l in labeled:
            adj[u].append((v, l))

        t0 = time.perf_counter()
        idx = ModalityPartitionedLabeling(n, labeled)
        build_ms = (time.perf_counter() - t0) * 1000.0

        rng = random.Random(11)
        queries = []
        for _ in range(400):
            s, t = rng.randrange(n), rng.randrange(n)
            k = rng.randint(1, m)
            queries.append((s, t, set(rng.sample(mods, k))))

        # exactness + fallback-free latency
        mism = 0
        t0 = time.perf_counter()
        for s, t, A in queries:
            if idx.reachable(s, t, A) != _brute_lcr(adj, s, t, A):
                mism += 1
        mpl_us = (time.perf_counter() - t0) / len(queries) * 1e6

        # old path: per-query label-restricted BFS
        t0 = time.perf_counter()
        for s, t, A in queries:
            _brute_lcr(adj, s, t, A)
        bfs_us = (time.perf_counter() - t0) / len(queries) * 1e6

        sperner = math.comb(m, m // 2)
        rows.append({
            "modalities": m,
            "nodes": n, "edges": ne,
            "index_entries": idx.stats.index_entries,
            "max_antichain": idx.stats.max_antichain,
            "sperner_bound": sperner,
            "antichain_within_bound": idx.stats.max_antichain <= sperner,
            "build_ms": round(build_ms, 2),
            "mpl_query_us": round(mpl_us, 3),
            "bfs_fallback_query_us": round(bfs_us, 3),
            "speedup": round(bfs_us / mpl_us, 2) if mpl_us else None,
            "exact_mismatches": mism,
        })

    with (out / "modality_labeling.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)

    # scaling crossover: MPL query is near-constant in graph size while a label-
    # restricted BFS grows, so MPL overtakes BFS as the graph grows (m fixed = 4).
    scale_rows = []
    mods4 = [f"mod{i}" for i in range(4)]
    for n, ne in [(120, 360), (400, 1200), (1000, 3000), (2000, 6000)]:
        labeled = _random_typed_graph(n, ne, mods4, seed=7)
        adj = defaultdict(list)
        for u, v, l in labeled:
            adj[u].append((v, l))
        idx = ModalityPartitionedLabeling(n, labeled)
        rng = random.Random(13)
        q = [(rng.randrange(n), rng.randrange(n), set(rng.sample(mods4, rng.randint(1, 4))))
             for _ in range(300)]
        mism = sum(1 for s, t, A in q if idx.reachable(s, t, A) != _brute_lcr(adj, s, t, A))
        t0 = time.perf_counter()
        for s, t, A in q:
            idx.reachable(s, t, A)
        mpl_us = (time.perf_counter() - t0) / len(q) * 1e6
        t0 = time.perf_counter()
        for s, t, A in q:
            _brute_lcr(adj, s, t, A)
        bfs_us = (time.perf_counter() - t0) / len(q) * 1e6
        scale_rows.append({"nodes": n, "edges": ne, "index_entries": idx.stats.index_entries,
                           "mpl_query_us": round(mpl_us, 3), "bfs_query_us": round(bfs_us, 3),
                           "speedup": round(bfs_us / mpl_us, 2), "exact_mismatches": mism})
    with (out / "modality_labeling_scaling.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(scale_rows[0].keys())); w.writeheader(); w.writerows(scale_rows)
    summary = {
        "rows": rows,
        "scaling": scale_rows,
        "all_exact": all(r["exact_mismatches"] == 0 for r in rows) and all(r["exact_mismatches"] == 0 for r in scale_rows),
        "all_within_sperner": all(r["antichain_within_bound"] for r in rows),
        "reading": ("Modality-partitioned 2-hop labeling answers label-constrained "
                    "reachability with no per-query traversal (fallback-free). The "
                    "max per-(node,hub) antichain stays within the Sperner bound "
                    "C(m, floor(m/2)), confirming the space proposition: for fixed "
                    "modality count the label overhead is constant, not 2^m."),
    }
    (out / "modality_labeling.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
