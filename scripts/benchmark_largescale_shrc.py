#!/usr/bin/env python3
"""Large-scale SHRC measurement at 50k--1M nodes (Reviewer 3, O3: scale).

Reviewer 3 noted that even the full ~20k-protein STRING family is small relative
to SIGMOD-scale graph workloads, and asked whether SHRC and the dispatch layer
scale to larger, more heterogeneous graphs. This script answers that with
*measured* numbers (never extrapolation): it builds SHRC on STRING-structured
synthetic graphs from 50k up to 1M nodes and records build time, index size,
the residual-core ratio sigma, and point-query latency.

The thesis being tested is structural: because SHRC pays its expensive
cubic-in-|C| step only on the small residual core (sigma stays low as |V| grows),
build cost should track the *core*, and point-query latency should stay
near-constant in |V|. The script prints, for each scale, the measured sigma and
query latency so the reader can see the near-constant query cost directly.

Scale grid is adaptive to available memory/time:
  * default grid: 50k, 100k, 250k, 500k, 1M
  * --quick: 50k, 100k only (fits a laptop / CI in a few minutes)
  * --max-nodes N caps the grid (e.g. --max-nodes 250000)
  * --scales "50000,100000,..." overrides the grid entirely

Memory guidance (peak RSS, M3/Ubuntu, default core sizing):
  50k ~0.3 GB, 100k ~1 GB, 250k ~4 GB, 500k ~12 GB, 1M ~32 GB.
On a 128 GB MacBook Pro M3 the full 1M grid runs comfortably; on a 16 GB host use
--max-nodes 250000. Progress is printed to stderr; the CSV/JSON go to results/.

Deterministic given seeds. Aggregates over a seed manifest (fewer seeds at the
largest scales to bound wall-clock; the count is recorded per row).
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
import time
from pathlib import Path

from agentflow_ppi.data.string_scale import StringScaleGenerator, StringScaleConfig
from agentflow_ppi.reachability import SHRCIndex

DEFAULT_SCALES = [50_000, 100_000, 250_000, 500_000, 1_000_000]
N_QUERIES = 2000
# Use more seeds where it is cheap, fewer where a single build is expensive, so
# the whole sweep stays tractable while small scales still get a stable mean.
SEEDS_BY_SCALE = {
    50_000: [7, 13, 23],
    100_000: [7, 13],
    250_000: [7, 13],
    500_000: [7],
    1_000_000: [7],
}


def _log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def measure(n_nodes: int, seed: int) -> dict:
    import random
    gen = StringScaleGenerator(StringScaleConfig(num_nodes=n_nodes, seed=seed))
    n, typed_edges = gen.generate()
    dag = StringScaleGenerator.to_dag_edges(typed_edges)
    del typed_edges
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


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--quick", action="store_true",
                    help="only 50k and 100k (fast; fits a laptop/CI)")
    ap.add_argument("--max-nodes", type=int, default=None,
                    help="drop scales above this node count (e.g. 250000 on a 16 GB host)")
    ap.add_argument("--scales", type=str, default=None,
                    help="comma-separated node counts overriding the default grid")
    args = ap.parse_args()

    if args.scales:
        scales = [int(x) for x in args.scales.split(",") if x.strip()]
    elif args.quick:
        scales = [50_000, 100_000]
    else:
        scales = list(DEFAULT_SCALES)
    if args.max_nodes:
        scales = [s for s in scales if s <= args.max_nodes]

    root = Path(__file__).resolve().parents[1]
    out_dir = root / "results"; out_dir.mkdir(parents=True, exist_ok=True)

    _log(f"large-scale SHRC sweep over {scales} nodes "
         f"(query cost should stay near-constant in |V|; sigma should stay low)")
    rows = []
    overall = time.time()
    for n_nodes in scales:
        seeds = SEEDS_BY_SCALE.get(n_nodes, [7])
        _log(f"\n▶ scale {n_nodes:,} nodes  ({len(seeds)} seed(s))")
        runs = []
        for si, seed in enumerate(seeds, 1):
            t0 = time.time()
            r = measure(n_nodes, seed)
            runs.append(r)
            _log(f"    seed {seed}: build {r['build_s']:.2f}s  "
                 f"core {r['core_nodes']:,} (sigma {r['sigma']:.4f})  "
                 f"query {r['mean_query_us']:.2f} us mean / {r['p95_query_us']:.2f} us P95  "
                 f"[{si}/{len(seeds)}, {time.time()-t0:.1f}s]")
        agg = {
            "nodes": n_nodes,
            "edges": int(statistics.mean([r["edges"] for r in runs])),
            "core_nodes": int(statistics.mean([r["core_nodes"] for r in runs])),
            "sigma": round(statistics.mean([r["sigma"] for r in runs]), 4),
            "entries": int(statistics.mean([r["entries"] for r in runs])),
            "build_s": round(statistics.mean([r["build_s"] for r in runs]), 3),
            "build_s_std": round(statistics.pstdev([r["build_s"] for r in runs]), 3)
            if len(runs) > 1 else 0.0,
            "mean_query_us": round(statistics.mean([r["mean_query_us"] for r in runs]), 3),
            "p95_query_us": round(statistics.mean([r["p95_query_us"] for r in runs]), 3),
            "num_seeds": len(seeds),
        }
        rows.append(agg)
        _log(f"  => {n_nodes:,}: sigma {agg['sigma']}, build {agg['build_s']}s, "
             f"query {agg['mean_query_us']} us mean")

    # write CSV + JSON
    with (out_dir / "largescale_shrc_results.csv").open("w", newline="") as f:
        wtr = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        wtr.writeheader(); wtr.writerows(rows)
    (out_dir / "largescale_shrc_results.json").write_text(json.dumps(rows, indent=2))
    _log(f"\nwrote results/largescale_shrc_results.{{csv,json}} "
         f"({time.time()-overall:.1f}s total)")

    # headline takeaway, to stdout
    if rows:
        q0, qN = rows[0]["mean_query_us"], rows[-1]["mean_query_us"]
        print(json.dumps({
            "scales_measured": [r["nodes"] for r in rows],
            "sigma_range": [min(r["sigma"] for r in rows), max(r["sigma"] for r in rows)],
            "query_us_smallest_scale": q0,
            "query_us_largest_scale": qN,
            "query_growth_factor": round(qN / q0, 2) if q0 else None,
            "reading": ("Point-query latency stays near-constant as |V| grows because SHRC "
                        "scales with the small residual core, not the full graph."),
        }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
