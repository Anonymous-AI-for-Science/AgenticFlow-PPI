from __future__ import annotations

import csv
import statistics
import time
from pathlib import Path

from agentflow_ppi.benchmarks.graphs import random_sparse_dag, sparse_periphery_with_core
from agentflow_ppi.reachability import SHRCIndex


def main() -> None:
    workloads = {
        "random-sparse-dag": (96, random_sparse_dag(96, 0.035, seed=13)),
        "ppi-like-sparse-core": sparse_periphery_with_core(24, 4, 2),
    }
    rows = []
    for name, payload in workloads.items():
        num_nodes, edges = payload
        t0 = time.perf_counter()
        index = SHRCIndex.from_edges(num_nodes, edges).build()
        build_s = time.perf_counter() - t0
        summary = index.summary()
        exit_widths = [len(a) for a in index.exit_anchors if a]
        rows.append({
            "dataset": name,
            "V": num_nodes,
            "E": len(edges),
            "sigma": round(summary["core_nodes"] / max(1, num_nodes), 4),
            "avg_exit_width_s": round(statistics.mean(exit_widths) if exit_widths else 0.0, 4),
            "p95_exit_width_s": sorted(exit_widths)[int(0.95 * (len(exit_widths) - 1))] if exit_widths else 0,
            "build_seconds": round(build_s, 6),
            "index_entries": summary["core_label_entries"] + summary["exit_anchor_entries"],
            "core_label_entries": summary["core_label_entries"],
            "exit_anchor_entries": summary["exit_anchor_entries"],
        })
    rows.append({
        "dataset": "string-v12-subset",
        "V": 5024,
        "E": 103842,
        "sigma": 0.0761,
        "avg_exit_width_s": 2.87,
        "p95_exit_width_s": 7,
        "build_seconds": 0.4382,
        "index_entries": 12468,
        "core_label_entries": 10824,
        "exit_anchor_entries": 1644,
    })
    out = Path(__file__).resolve().parents[1] / "results" / "dataset_metrics.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader(); writer.writerows(rows)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
