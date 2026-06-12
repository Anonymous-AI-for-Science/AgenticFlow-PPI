from __future__ import annotations

import csv
import statistics
import time
from pathlib import Path

from agentflow_ppi.benchmarks.baselines import BFSBaseline, GrailStyleIndex, PLLStyleIndex, PReachStyleIndex, SHRCHarness, TFLabelStyleIndex
from agentflow_ppi.benchmarks.graphs import diamond_fan_core, iter_query_pairs, layered_biclique_core, random_sparse_dag, sparse_periphery_with_core


def timed_index(factory, *args, **kwargs):
    start = time.perf_counter()
    index = factory(*args, **kwargs)
    build_s = time.perf_counter() - start
    index.stats.build_seconds = build_s
    return index


def bench_one(name, num_nodes, edges, out_rows):
    queries = list(iter_query_pairs(num_nodes, max_pairs=min(128, num_nodes * 2), seed=11))
    baselines = [
        ("shrc", lambda: timed_index(SHRCHarness, num_nodes, edges)),
        ("shrc-no-prune", lambda: timed_index(SHRCHarness, num_nodes, edges, core_hub_strategy="none", exit_prune_strategy="none")),
        ("shrc-degree", lambda: timed_index(SHRCHarness, num_nodes, edges, core_hub_strategy="degree", exit_prune_strategy="none")),
        ("online-bfs", lambda: timed_index(BFSBaseline, num_nodes, edges)),
        ("grail-style", lambda: timed_index(GrailStyleIndex, num_nodes, edges)),
        ("pll-style", lambda: timed_index(PLLStyleIndex, num_nodes, edges)),
        ("preach-style", lambda: timed_index(PReachStyleIndex, num_nodes, edges)),
        ("tf-label-style", lambda: timed_index(TFLabelStyleIndex, num_nodes, edges)),
    ]
    for label, builder in baselines:
        index = builder()
        latencies = []
        for s, t in queries:
            q0 = time.perf_counter()
            index.reachable(s, t)
            latencies.append((time.perf_counter() - q0) * 1e6)
        out_rows.append({
            "workload": name,
            "index": label,
            "num_nodes": num_nodes,
            "num_edges": len(edges),
            "build_seconds": round(index.stats.build_seconds, 6),
            "index_entries": index.stats.index_entries,
            "mean_query_us": round(statistics.mean(latencies), 3),
            "p95_query_us": round(sorted(latencies)[int(0.95 * (len(latencies) - 1))], 3),
        })


def main() -> None:
    out_rows = []
    n, e = layered_biclique_core(4, 7)
    bench_one("adversarial-layered-biclique", n, e, out_rows)
    n, e = diamond_fan_core(6, 4)
    bench_one("adversarial-diamond-fan", n, e, out_rows)
    n, e = sparse_periphery_with_core(24, 4, 2)
    bench_one("sparse-periphery-core", n, e, out_rows)
    e = random_sparse_dag(96, 0.035, seed=13)
    bench_one("random-sparse-dag", 96, e, out_rows)
    out = Path(__file__).resolve().parents[1] / "results" / "reachability_benchmarks.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()))
        writer.writeheader()
        writer.writerows(out_rows)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()


