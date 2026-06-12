"""STRING-scale SHRC measurement at 5k, 10k, and 20k nodes.

Answers practitioner concerns O2.B / O3 (scale): instead of extrapolating from a
5k-node subset, we MEASURE SHRC build time, index size, residual-core ratio,
and point-query latency on STRING-structured graphs up to the full ~20k-protein
v12 family scale. Deterministic given seeds; aggregates over a seed manifest.
"""

from __future__ import annotations

import csv
import statistics
import time
from pathlib import Path

from agentflow_ppi.data.string_scale import StringScaleGenerator, StringScaleConfig
from agentflow_ppi.reachability import SHRCIndex

SEEDS = [7, 13, 23]
SCALES = [5000, 10000, 20000]
N_QUERIES = 2000


def measure(n_nodes: int, seed: int):
    import random
    gen = StringScaleGenerator(StringScaleConfig(num_nodes=n_nodes, seed=seed))
    n, typed_edges = gen.generate()
    dag = StringScaleGenerator.to_dag_edges(typed_edges)
    t0 = time.perf_counter()
    idx = SHRCIndex.from_edges(num_nodes=n, edges=dag).build()
    build_s = time.perf_counter() - t0
    s = idx.stats
    rng = random.Random(seed)
    pairs = [(rng.randrange(n), rng.randrange(n)) for _ in range(N_QUERIES)]
    lat = []
    for u, v in pairs:
        t = time.perf_counter()
        idx.reachable(u, v)
        lat.append((time.perf_counter() - t) * 1e6)
    lat.sort()
    return {
        "nodes": n, "edges": len(dag), "core_nodes": s.core_nodes,
        "sigma": s.core_nodes / n,
        "entries": s.core_label_entries + s.exit_anchor_entries,
        "build_s": build_s,
        "mean_query_us": statistics.mean(lat),
        "p95_query_us": lat[int(0.95 * len(lat)) - 1],
    }


def main():
    root = Path(__file__).resolve().parents[1]
    out_dir = root / "results"; out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for n_nodes in SCALES:
        runs = [measure(n_nodes, seed) for seed in SEEDS]
        agg = {
            "nodes": n_nodes,
            "edges": int(statistics.mean([r["edges"] for r in runs])),
            "core_nodes": int(statistics.mean([r["core_nodes"] for r in runs])),
            "sigma": round(statistics.mean([r["sigma"] for r in runs]), 4),
            "entries": int(statistics.mean([r["entries"] for r in runs])),
            "build_s": round(statistics.mean([r["build_s"] for r in runs]), 4),
            "build_s_std": round(statistics.pstdev([r["build_s"] for r in runs]), 4),
            "mean_query_us": round(statistics.mean([r["mean_query_us"] for r in runs]), 3),
            "p95_query_us": round(statistics.mean([r["p95_query_us"] for r in runs]), 3),
            "num_seeds": len(SEEDS),
        }
        rows.append(agg)
        print(agg)
    with (out_dir / "string_scale_results.csv").open("w", newline="") as f:
        wtr = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        wtr.writeheader(); wtr.writerows(rows)
    print("wrote results/string_scale_results.csv")


if __name__ == "__main__":
    main()
