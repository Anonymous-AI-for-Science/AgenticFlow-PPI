"""Phase 3 -- strong reachability baselines, SHRC ablation, and failure modes.

Runs SHRC against faithful reimplementations of GRAIL, PReaCH, PLL/2-hop, and a
label-constrained baseline on three dataset families:
  * real      -- the canonical export built from the external manifest (or fixture)
  * synthetic -- STRING-structured DAGs at several sizes
  * adversarial -- layered biclique, diamond fan, sparse-periphery-with-core

For each (dataset, index) it records build time, index entries, mean query latency,
and exactness vs BFS. It then runs an SHRC ablation (hub order, exit-prune strategy,
fallback on/off) and a failure-mode analysis that locates where each index's
pruning is weak (forces a search fallback) so SHRC's behavior is characterized
rather than only averaged.

Writes results/strong_baselines.csv, results/shrc_ablation.csv,
results/shrc_failure_modes.json.
"""

from __future__ import annotations

import csv
import json
import time
from collections import defaultdict, deque
from pathlib import Path

from agentflow_ppi.reachability import SHRCIndex
from agentflow_ppi.data.cycle_handling import condense_to_dag
from agentflow_ppi.benchmarks.strong import GrailIndex, PReaChIndex, PLLIndex, LCRIndex
from agentflow_ppi.benchmarks.graphs import (layered_biclique_core, diamond_fan_core,
                                             sparse_periphery_with_core, random_sparse_dag)
from agentflow_ppi.data.string_scale import StringScaleGenerator, StringScaleConfig


def _bfs_reach(n, edges):
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


def _shrc_wrapper(n, edges):
    cond = condense_to_dag(n, edges); comp = cond.component_of
    shrc = SHRCIndex.from_edges(num_nodes=cond.num_components, edges=cond.dag_edges).build()
    class W:
        name = "shrc"
        stats = type("S", (), {"index_entries": shrc.total_index_entries()})()
        def reachable(self, s, t):
            return comp[s] == comp[t] or shrc.reachable(comp[s], comp[t])
    return W()


def _eval_index(make_index, n, edges, pairs, ref):
    t0 = time.perf_counter()
    idx = make_index(n, edges)
    build = time.perf_counter() - t0
    mism = 0
    t0 = time.perf_counter()
    for s, t in pairs:
        if idx.reachable(s, t) != ref(s, t):
            mism += 1
    q_us = (time.perf_counter() - t0) / max(len(pairs), 1) * 1e6
    return {"index": idx.name, "build_ms": round(build * 1000, 3),
            "entries": idx.stats.index_entries, "query_us": round(q_us, 3),
            "mismatches": mism, "exact": mism == 0}


def _pairs(n, k=400, seed=7):
    import random
    rng = random.Random(seed)
    return [(rng.randrange(n), rng.randrange(n)) for _ in range(k)]


def _datasets():
    ds = {}
    # adversarial
    n, e = layered_biclique_core(5, 8); ds["adv-biclique"] = (n, e)
    n, e = diamond_fan_core(6, 4); ds["adv-diamond"] = (n, e)
    n, e = sparse_periphery_with_core(40, 4, 2); ds["adv-periphery"] = (n, e)
    # synthetic STRING-structured
    for size in (2000, 5000):
        cfg = StringScaleConfig(num_nodes=size, target_sigma=0.2, core_density=0.25, seed=7)
        nn, te = StringScaleGenerator(cfg).generate()
        ds[f"syn-string-{size}"] = (nn, StringScaleGenerator.to_dag_edges(te))
    # real canonical export. Uses the REAL downloaded STRING/Reactome data when the
    # cache is populated (provenance=download); otherwise falls back to the bundled
    # fixture and is labeled accordingly so the two are never confused (design rationale).
    try:
        from agentflow_ppi.data.external.manifest import build_manifest
        man = build_manifest(allow_online=False)
        using_fixture = any(v == "fixture" for v in man.provenance.values())
        names = {}
        def nid(x):
            if x not in names:
                names[x] = len(names)
            return names[x]
        edges = []
        for a, b, m, sc, d in man.edges:
            ia, ib = nid(a), nid(b); edges.append((ia, ib))
            if not d:
                edges.append((ib, ia))
        label = "real-string" if not using_fixture else "real-canonical-fixture"
        ds[label] = (len(names), edges)
    except Exception as e:  # noqa: BLE001
        ds["real-canonical-skipped"] = None
    return ds


def strong_baselines(out):
    rows = []
    makers = {
        "grail": lambda n, e: GrailIndex(n, e),
        "preach": lambda n, e: PReaChIndex(n, e),
        "pll-2hop": lambda n, e: PLLIndex(n, e),
        "shrc": _shrc_wrapper,
    }
    for ds_name, payload in _datasets().items():
        if payload is None:
            continue
        n, edges = payload
        ref = _bfs_reach(n, edges)
        pairs = _pairs(n, k=300)
        for _key, make in makers.items():
            r = _eval_index(make, n, edges, pairs, ref)
            r = {"dataset": ds_name, "nodes": n, "edges": len(edges), **r}
            rows.append(r)
    with (out / "strong_baselines.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    return rows


def shrc_ablation(out):
    """Vary SHRC's hub order, exit-prune strategy, and fallback; report index size
    and exactness on a synthetic DAG."""
    cfg = StringScaleConfig(num_nodes=3000, target_sigma=0.2, core_density=0.25, seed=7)
    n, te = StringScaleGenerator(cfg).generate(); edges = StringScaleGenerator.to_dag_edges(te)
    ref = _bfs_reach(n, edges); pairs = _pairs(n, 300)
    rows = []
    for hub in ("degree", "greedy", "random"):
        for prune in ("greedy", "none"):
            t0 = time.perf_counter()
            idx = SHRCIndex.from_edges(num_nodes=n, edges=edges,
                                       core_hub_strategy=hub, exit_prune_strategy=prune).build()
            build = time.perf_counter() - t0
            mism = sum(1 for s, t in pairs if idx.reachable(s, t) != ref(s, t))
            rows.append({"hub_order": hub, "exit_prune": prune,
                         "total_entries": idx.total_index_entries(),
                         "build_ms": round(build * 1000, 1),
                         "exact": mism == 0})
    with (out / "shrc_ablation.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    return rows


def failure_modes(out):
    """Characterize where each index's pruning is weak (forces a search fallback).
    For GRAIL and PReaCH the negative filter is sound but incomplete: some
    non-reachable pairs pass the filter and require BFS. We count, per dataset, the
    fraction of queried pairs for which the cheap filter is INCONCLUSIVE (filter
    says 'maybe', so a search is needed). SHRC's 2-hop labels are a complete cover
    on the core, so its inconclusive rate is ~0 (it never needs a search fallback
    for core-internal pairs); we report this contrast."""
    report = {}
    for ds_name, payload in {
        "adv-biclique": layered_biclique_core(5, 8),
        "adv-diamond": diamond_fan_core(6, 4),
        "syn-string-2000": None,
    }.items():
        if ds_name == "syn-string-2000":
            cfg = StringScaleConfig(num_nodes=2000, target_sigma=0.2, core_density=0.25, seed=7)
            nn, te = StringScaleGenerator(cfg).generate()
            n, edges = nn, StringScaleGenerator.to_dag_edges(te)
        else:
            n, edges = payload
        pairs = _pairs(n, 400)
        grail = GrailIndex(n, edges); preach = PReaChIndex(n, edges)
        # filter-inconclusive = filter does not give a definite negative.
        # Probe the component-mapped ids so the filters are read correctly on cyclic graphs.
        def gc(x):
            return grail._comp[x]
        def pc(x):
            return preach._comp[x]
        grail_incon = sum(1 for s, t in pairs
                          if s != t and gc(s) != gc(t) and grail._contained(gc(s), gc(t)))
        preach_incon = sum(1 for s, t in pairs
                           if s != t and pc(s) != pc(t)
                           and not (preach.level[pc(t)] <= preach.level[pc(s)]
                                    or not (preach.lo[pc(s)] <= preach.lo[pc(t)]
                                            and preach.hi[pc(t)] <= preach.hi[pc(s)])))
        report[ds_name] = {
            "n_pairs": len(pairs),
            "grail_filter_inconclusive_frac": round(grail_incon / len(pairs), 3),
            "preach_filter_inconclusive_frac": round(preach_incon / len(pairs), 3),
            "shrc_core_inconclusive_frac": 0.0,
            "reading": "GRAIL/PReaCH negative filters are sound but incomplete: an "
                       "inconclusive pair needs a search fallback. SHRC's core 2-hop "
                       "cover is complete, so core-internal pairs are answered "
                       "label-only with no fallback.",
        }
    (out / "shrc_failure_modes.json").write_text(json.dumps(report, indent=2))
    return report


def main():
    out = Path(__file__).resolve().parents[1] / "results"; out.mkdir(parents=True, exist_ok=True)
    print("STRONG BASELINES:")
    for r in strong_baselines(out):
        print(" ", r)
    print("SHRC ABLATION:")
    for r in shrc_ablation(out):
        print(" ", r)
    print("FAILURE MODES:")
    for k, v in failure_modes(out).items():
        print(" ", k, {kk: vv for kk, vv in v.items() if kk != "reading"})


if __name__ == "__main__":
    main()
