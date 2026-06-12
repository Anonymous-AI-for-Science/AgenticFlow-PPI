from __future__ import annotations

import json
from pathlib import Path

from agentflow_ppi.benchmarks.baselines import (
    BFSBaseline,
    GrailStyleIndex,
    PLLStyleIndex,
    TFLabelStyleIndex,
)
from agentflow_ppi.benchmarks.graphs import (
    diamond_fan_core,
    iter_query_pairs,
    layered_biclique_core,
    random_sparse_dag,
    sparse_periphery_with_core,
)


def main() -> None:
    workloads = {
        "layered-biclique": layered_biclique_core(4, 8),
        "diamond-fan": diamond_fan_core(5, 4),
        "sparse-periphery-core": sparse_periphery_with_core(24, 4, 2),
        "random-sparse-dag": (96, random_sparse_dag(96, 0.035, seed=7)),
    }
    constructors = {
        "grail-style": GrailStyleIndex,
        "pll-style": PLLStyleIndex,
        "tf-label-style": TFLabelStyleIndex,
    }
    out = {"workloads": {}, "all_passed": True}
    for name, (num_nodes, edges) in workloads.items():
        bfs = BFSBaseline(num_nodes, edges)
        pairs = list(iter_query_pairs(num_nodes, max_pairs=min(256, max(32, num_nodes)), seed=7))
        out["workloads"][name] = {"num_nodes": num_nodes, "num_edges": len(edges), "pairs_checked": len(pairs), "baselines": {}}
        for bname, ctor in constructors.items():
            idx = ctor(num_nodes, edges)
            mismatches = []
            for s, t in pairs:
                ref = bfs.reachable(s, t)
                got = idx.reachable(s, t)
                if ref != got:
                    mismatches.append([s, t, ref, got])
                    if len(mismatches) >= 5:
                        break
            passed = not mismatches
            out["workloads"][name]["baselines"][bname] = {
                "passed": passed,
                "mismatches": mismatches,
            }
            out["all_passed"] = out["all_passed"] and passed
    root = Path(__file__).resolve().parents[1] / "results"
    root.mkdir(parents=True, exist_ok=True)
    path = root / "baseline_invariant_checks.json"
    path.write_text(json.dumps(out, indent=2))
    print(path)


if __name__ == "__main__":
    main()
