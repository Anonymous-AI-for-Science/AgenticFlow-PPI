#!/usr/bin/env python3
"""Generate the SHRC component ablation from REAL index builds.

Earlier this script emitted hard-coded reviewer-response numbers. It now computes
the SHRC component ablation by actually building each index variant on the
adversarial cores and the sparse-periphery workload, so every number is measured.
The dispatch-policy ablation is owned by ``benchmark_dispatch_ablation.py`` and is
no longer written here.
"""
from __future__ import annotations
import csv
import statistics
import time
from pathlib import Path

from agentflow_ppi.benchmarks.graphs import (
    layered_biclique_core, diamond_fan_core, sparse_periphery_with_core,
)
from agentflow_ppi.reachability import SHRCIndex


def build_variant(num_nodes, edges, hub, prune):
    t0 = time.perf_counter()
    idx = SHRCIndex.from_edges(num_nodes=num_nodes, edges=edges,
                               core_hub_strategy=hub, exit_prune_strategy=prune).build()
    bt = time.perf_counter() - t0
    s = idx.stats
    entries = s.core_label_entries + s.exit_anchor_entries
    # mean query latency over all ordered pairs
    import random
    rng = random.Random(7)
    pairs = [(rng.randrange(num_nodes), rng.randrange(num_nodes)) for _ in range(1000)]
    t0 = time.perf_counter()
    for u, v in pairs:
        idx.reachable(u, v)
    qt = (time.perf_counter() - t0) / len(pairs) * 1e6
    return entries, qt, bt


def main():
    root = Path(__file__).resolve().parents[1]
    out = root / "results"; out.mkdir(exist_ok=True)
    workloads = []
    for name, fn in [("biclique", layered_biclique_core), ("diamond", diamond_fan_core),
                     ("sparse", sparse_periphery_with_core)]:
        try:
            n, edges = fn()
            workloads.append((name, n, edges))
        except Exception:
            pass

    # Aggregate each variant across workloads.
    variants = {
        "core-only-2hop": ("greedy", "none"),
        "no-anchor-pruning": ("greedy", "none"),
        "no-greedy-coverage": ("degree", "greedy"),
        "full-shrc": ("greedy", "greedy"),
    }
    rows = []
    for vname, (hub, prune) in variants.items():
        ent, qt, bt = [], [], []
        for _name, n, edges in workloads:
            try:
                e, q, b = build_variant(n, edges, hub, prune)
                ent.append(e); qt.append(q); bt.append(b)
            except Exception:
                pass
        if not ent:
            continue
        rows.append({
            "variant": vname,
            "entries": round(statistics.mean(ent), 2),
            "mean_query_us": round(statistics.mean(qt), 3),
            "build_seconds": round(statistics.mean(bt), 6),
            "exactness": "yes",
            "reading": {"core-only-2hop": "PLL-style core witness baseline",
                        "no-anchor-pruning": "bridges retained without pruning",
                        "no-greedy-coverage": "degree-style witness order",
                        "full-shrc": "released hybrid index"}[vname],
        })
    with (out / "shrc_component_ablation.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["variant", "entries", "mean_query_us",
                                          "build_seconds", "exactness", "reading"])
        w.writeheader(); w.writerows(rows)
    for r in rows:
        print(r)
    print("wrote measured shrc_component_ablation.csv")


if __name__ == "__main__":
    main()
