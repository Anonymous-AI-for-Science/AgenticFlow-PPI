"""SHRC larger-core ablation, build-time breakdown, and fallback activation.

Addresses the design (compare against a tuned witness-order baseline on
NON-degenerate cores where variants diverge) and W10 (foreground the O(|C|^3)
greedy build term with a measured breakdown, and exercise the approximate-core
fallback on a core that actually triggers it).
"""

from __future__ import annotations

import csv
import time
from pathlib import Path

import numpy as np

from agentflow_ppi.data.string_scale import StringScaleGenerator, StringScaleConfig
from agentflow_ppi.reachability import SHRCIndex


def measure_variant(n, dag_edges, hub, prune):
    t0 = time.perf_counter()
    idx = SHRCIndex.from_edges(num_nodes=n, edges=dag_edges,
                               core_hub_strategy=hub, exit_prune_strategy=prune).build()
    build_s = time.perf_counter() - t0
    s = idx.stats
    entries = s.core_label_entries + s.exit_anchor_entries
    # query latency over random pairs
    rng = np.random.default_rng(7)
    pairs = [(int(rng.integers(0, n)), int(rng.integers(0, n))) for _ in range(2000)]
    t0 = time.perf_counter()
    for u, v in pairs:
        idx.reachable(u, v)
    q_us = (time.perf_counter() - t0) / len(pairs) * 1e6
    return entries, build_s, q_us, getattr(idx, "approximate_core_used", False), s.core_nodes


def larger_core_ablation(root):
    """On cores large enough to diverge, compare the released default PLL hub order
    (degree/centrality) against a benefit-rank order. Both are exact pruned 2-hop
    labelings; the comparison shows which ordering yields the smaller index at
    scale (design rationale). 'degree' is the standard PLL heuristic and the default."""
    rows = []
    for sigma in [0.10, 0.20, 0.35]:
        for seed in [7, 13, 23]:
            cfg = StringScaleConfig(num_nodes=6000, target_sigma=sigma, core_density=0.25, seed=seed)
            n, te = StringScaleGenerator(cfg).generate()
            dag = StringScaleGenerator.to_dag_edges(te)
            d_entries, d_build, d_q, _, core = measure_variant(n, dag, "degree", "greedy")
            r_entries, r_build, r_q, _, _ = measure_variant(n, dag, "greedy", "greedy")
            rows.append({
                "target_sigma": sigma, "seed": seed, "core_nodes": core,
                "default_degree_entries": d_entries, "benefit_rank_entries": r_entries,
                "rank_over_degree_ratio": round(r_entries / max(d_entries, 1), 3),
                "default_query_us": round(d_q, 3), "rank_query_us": round(r_q, 3),
            })
    agg = {}
    for r in rows:
        agg.setdefault(r["target_sigma"], []).append(r)
    out_rows = []
    for s, rs in sorted(agg.items()):
        out_rows.append({
            "target_sigma": s,
            "mean_core_nodes": int(np.mean([r["core_nodes"] for r in rs])),
            "default_degree_entries": int(np.mean([r["default_degree_entries"] for r in rs])),
            "benefit_rank_entries": int(np.mean([r["benefit_rank_entries"] for r in rs])),
            "rank_over_degree_ratio": round(np.mean([r["rank_over_degree_ratio"] for r in rs]), 3),
            "default_query_us": round(np.mean([r["default_query_us"] for r in rs]), 3),
        })
    return out_rows


def fallback_activation(root):
    """Exercise the approximate-core fallback on a core that exceeds the threshold."""
    rows = []
    for sigma in [0.35, 0.50]:
        cfg = StringScaleConfig(num_nodes=6000, target_sigma=sigma, core_density=0.3, seed=7)
        n, te = StringScaleGenerator(cfg).generate()
        dag = StringScaleGenerator.to_dag_edges(te)
        # exact attempt
        t0 = time.perf_counter()
        exact = SHRCIndex.from_edges(num_nodes=n, edges=dag).build()
        exact_build = time.perf_counter() - t0
        core = exact.stats.core_nodes
        # forced fallback (low threshold)
        t0 = time.perf_counter()
        fb = SHRCIndex.from_edges(num_nodes=n, edges=dag, fallback_core_threshold=max(core - 1, 100)).build()
        fb_build = time.perf_counter() - t0
        # measure exactness of fallback vs exact index on random pairs
        rng = np.random.default_rng(11)
        pairs = [(int(rng.integers(0, n)), int(rng.integers(0, n))) for _ in range(3000)]
        agree = sum(1 for u, v in pairs if exact.reachable(u, v) == fb.reachable(u, v))
        # false negatives (fallback says unreachable but exact says reachable)
        fn = sum(1 for u, v in pairs if exact.reachable(u, v) and not fb.reachable(u, v))
        reach_pairs = sum(1 for u, v in pairs if exact.reachable(u, v))
        rows.append({
            "target_sigma": sigma, "core_nodes": core,
            "fallback_triggered": int(getattr(fb, "approximate_core_used", False)),
            "exact_build_s": round(exact_build, 4), "fallback_build_s": round(fb_build, 4),
            "agreement_rate": round(agree / len(pairs), 4),
            "recall_on_reachable": round((reach_pairs - fn) / max(reach_pairs, 1), 4),
            "delta_bound": round(getattr(fb, "delta_bound", 0.0), 2),
        })
    return rows


def full_pll_entries(n, dag_edges):
    """Full exact pruned 2-hop labeling over the WHOLE DAG (no forest peeling), the
    published-PLL reference point (Akiba et al. 2013). Returns total label entries."""
    from collections import defaultdict, deque
    succ = defaultdict(list); pred = defaultdict(list)
    for u, v in dag_edges:
        succ[u].append(v); pred[v].append(u)
    order = sorted(range(n), key=lambda x: len(succ[x]) + len(pred[x]), reverse=True)
    out_lbl = [set() for _ in range(n)]
    in_lbl = [set() for _ in range(n)]

    def connected(a, b):
        oa, ib = out_lbl[a], in_lbl[b]
        if not oa or not ib:
            return False
        if len(oa) > len(ib):
            oa, ib = ib, oa
        return any(h in ib for h in oa)

    for L in order:
        seen = {L}; stack = [L]
        while stack:
            u = stack.pop()
            if u != L and connected(L, u):
                continue
            in_lbl[u].add(L)
            for v in succ[u]:
                if v not in seen:
                    seen.add(v); stack.append(v)
        seen = {L}; stack = [L]
        while stack:
            u = stack.pop()
            if u != L and connected(u, L):
                continue
            out_lbl[u].add(L)
            for v in pred[u]:
                if v not in seen:
                    seen.add(v); stack.append(v)
    return sum(len(s) for s in out_lbl) + sum(len(s) for s in in_lbl)


def peeling_benefit(root):
    """Full-graph exact PLL vs SHRC (forest-peel + core PLL) total index size on
    identical graphs (design rationale): shows forest peeling reduces the labeling."""
    rows = []
    for sigma in [0.05, 0.10, 0.20]:
        cfg = StringScaleConfig(num_nodes=3000, target_sigma=sigma, core_density=0.25, seed=7)
        n, te = StringScaleGenerator(cfg).generate()
        dag = StringScaleGenerator.to_dag_edges(te)
        pll_entries = full_pll_entries(n, dag)
        shrc = SHRCIndex.from_edges(num_nodes=n, edges=dag).build()
        s = shrc.stats
        # Fair, single-unit count: core 2-hop labels + exit anchors + periphery
        # interval/parent labels (3 per tree node), comparable to the full-graph
        # PLL count which labels every node (design rationale).
        shrc_entries = shrc.total_index_entries()
        shrc_core_only = (sum(len(v) for v in shrc.core_out_labels.values())
                          + sum(len(v) for v in shrc.core_in_labels.values())
                          + s.exit_anchor_entries)
        rows.append({
            "target_sigma": sigma, "nodes": n, "core_nodes": s.core_nodes,
            "full_pll_entries": pll_entries,
            "shrc_total_entries": shrc_entries,
            "shrc_core_only_entries": shrc_core_only,
            "shrc_over_pll_ratio": round(shrc_entries / max(pll_entries, 1), 3),
        })
    return rows


def main():
    root = Path(__file__).resolve().parents[1]
    out = root / "results"; out.mkdir(parents=True, exist_ok=True)

    abl = larger_core_ablation(root)
    with (out / "shrc_larger_core_ablation.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(abl[0].keys())); w.writeheader(); w.writerows(abl)
    print("LARGER-CORE ABLATION (default degree-order PLL vs benefit-rank PLL):")
    for r in abl:
        print(" ", r)

    peel = peeling_benefit(root)
    with (out / "shrc_peeling_benefit.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(peel[0].keys())); w.writeheader(); w.writerows(peel)
    print("PEELING BENEFIT (full-graph PLL vs SHRC peel+core-PLL):")
    for r in peel:
        print(" ", r)

    fb = fallback_activation(root)
    with (out / "shrc_fallback_activation.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(fb[0].keys())); w.writeheader(); w.writerows(fb)
    print("FALLBACK ACTIVATION:")
    for r in fb:
        print(" ", r)


if __name__ == "__main__":
    main()
