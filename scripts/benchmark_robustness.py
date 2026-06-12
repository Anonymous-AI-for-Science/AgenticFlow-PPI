"""Robustness and failure-mode experiments.

Answers practitioner concerns:
  * the design goal / the design goal (negative-pool stress): the biological reranker is re-evaluated
    under a HARSHER all-non-reachable negative pool (negatives drawn from proteins
    NOT reachable from the source), in addition to the leakage-controlled
    reachable-unlabeled pool. We report macro-F1@2 under both regimes so the
    favorable-pool caveat is quantified rather than only acknowledged.
  * the design goal (SHRC failure mode): we sweep the residual-core ratio sigma on
    synthetic STRING-structured graphs and report where SHRC build cost and index
    size degrade toward the dense-core regime, identifying the operating point at
    which the approximate-core fallback becomes necessary.
  * the design goal (cost-model sensitivity): we sweep the dispatch gain weight (mu_gain)
    and selectivity floor and report route stability.

All measured and reproducible.
"""

from __future__ import annotations

import csv
import json
import statistics
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

from agentflow_ppi.data.cycle_handling import condense_to_dag
from agentflow_ppi.data.string_scale import StringScaleGenerator, StringScaleConfig
from agentflow_ppi.reachability import SHRCIndex

SEEDS = [7, 13, 23, 37, 101]


# ---------- shared biological helpers ----------
def load_graph(p):
    names, adj, edges = {}, defaultdict(list), []
    def nid(x):
        if x not in names: names[x] = len(names)
        return names[x]
    with p.open() as f:
        for row in csv.DictReader(f, delimiter="\t"):
            u, v = nid(row["source"]), nid(row["target"])
            adj[u].append((v, row["modality"], float(row["score"]))); edges.append((u, v))
    return names, {i: n for n, i in names.items()}, adj, edges


def expand(adj, s, hops=3):
    seen = {s}; fr = [s]; out = []
    for _ in range(hops):
        nx = []
        for u in fr:
            for v, _m, _c in adj.get(u, []):
                if v not in seen: seen.add(v); out.append(v); nx.append(v)
        fr = nx
    return out


def pscore(adj, s, v, t):
    l1 = max([c for w, _m, c in adj.get(s, []) if w == v], default=0.0)
    l2 = max([c for w, _m, c in adj.get(v, []) if w == t], default=0.0)
    if l2 == 0.0:
        for w, _m, c in adj.get(v, []):
            for x, _m2, c2 in adj.get(w, []):
                if x == t: l2 = max(l2, c * c2)
    return l1 * l2 if (l1 and l2) else max(l1, l2) * 0.5


def mscore(adj, v, mod):
    e = adj.get(v, []); return (sum(1 for _w, m, _c in e if m == mod) / len(e)) if e else 0.0


def gold(adj, s, t, mod, cands):
    pos = set()
    for v in cands:
        ie = [(m, c) for w, m, c in adj.get(s, []) if w == v]; oe = adj.get(v, [])
        if (any(m == mod for m, _c in ie) or not ie) and any(m == mod for _w, m, _c in oe): pos.add(v)
    if not pos: pos.add(max(cands, key=lambda v: mscore(adj, v, mod)))
    return pos


def f1(rank, pos, k=2):
    if not pos: return 0.0
    top = rank[:k]; tp = sum(1 for v in top if v in pos)
    p = tp / max(len(top), 1); r = tp / len(pos)
    return 0.0 if p + r == 0 else 2 * p * r / (p + r)


def feats(adj, s, v, t, mod):
    e = adj.get(v, [])
    return np.array([pscore(adj, s, v, t), np.log1p(len(e)),
                     max([c for w, _m, c in adj.get(s, []) if w == v], default=0.0),
                     mscore(adj, v, mod), (statistics.mean([c for _w, _m, c in e]) if e else 0.0)])


def negative_pool_stress(root):
    """Stress the strong symbolic ranker under a harsher negative pool, with the
    independent pathway labels (design rationale). We report how macro-F1@2 of the
    symbolic baseline degrades when source-unreachable distractors are injected."""
    from agentflow_ppi.eval.harness import build_harness, symbolic_order, f1_at_k, path_score
    h = build_harness()
    n = len(h.names)

    def nonreach_of(s):
        return [v for v in range(n) if v not in (s,) and not h.reach(s, v)]

    def run(regime):
        f1s = []
        for seed in SEEDS:
            rng = np.random.default_rng(seed)
            idx = list(range(len(h.pools))); rng.shuffle(idx)
            nt = max(1, int(round(0.2 * len(idx)))); test = set(idx[:nt])
            for i in test:
                p = h.pools[i]
                if regime == "reachable-unlabeled":
                    cands = list(p.cands)
                else:  # all-non-reachable: inject source-unreachable distractors
                    extra = nonreach_of(p.s); rng.shuffle(extra)
                    cands = list(p.cands) + extra[:max(2, len(p.cands))]
                rank = symbolic_order(h, p.s, p.t, cands)
                f1s.append(f1_at_k(rank, p.positives))
        return round(statistics.mean(f1s), 4) if f1s else 0.0

    return {
        "label_source": "independent canonical-pathway membership",
        "ranker": "symbolic path-score (the strong baseline under independent labels)",
        "reachable_unlabeled_f1_at_2": run("reachable-unlabeled"),
        "all_non_reachable_f1_at_2": run("all-non-reachable"),
        "num_query_families": len(h.pools),
        "note": "all-non-reachable injects source-unreachable distractors; lower F1 quantifies the harder regime and stresses SHRC's exact-pruning value (exact reachability removes these distractors before ranking)",
    }


def core_growth_failure(root):
    rows = []
    for sigma_target in [0.02, 0.05, 0.10, 0.20, 0.35, 0.50]:
        builds, entries, cores = [], [], []
        for seed in [7, 13, 23]:
            cfg = StringScaleConfig(num_nodes=4000, target_sigma=sigma_target, core_density=0.25, seed=seed)
            n, te = StringScaleGenerator(cfg).generate(); dag = StringScaleGenerator.to_dag_edges(te)
            t0 = time.perf_counter(); idx = SHRCIndex.from_edges(num_nodes=n, edges=dag).build(); bt = time.perf_counter() - t0
            s = idx.stats; builds.append(bt); entries.append(s.core_label_entries + s.exit_anchor_entries); cores.append(s.core_nodes / n)
        rows.append({"sigma_target": sigma_target,
                     "measured_sigma": round(statistics.mean(cores), 4),
                     "build_s": round(statistics.mean(builds), 4),
                     "index_entries": int(statistics.mean(entries)),
                     "regime": "exact" if statistics.mean(cores) < 0.15 else "dense-core (fallback relevant)"})
    return rows


def cost_model_sensitivity(root):
    """Stability of the CALIBRATED dispatch decision under the gain predictor's
    ridge regularization. We vary lambda and measure how often the admit decision
    (predicted lift > 0) flips relative to the default lambda=1.0, across seeds."""
    from agentflow_ppi.eval.harness import (
        build_harness, train_reranker, rerank, symbolic_order, f1_at_k,
        query_gain_features, SEED_MANIFEST,
    )
    h = build_harness()

    def predictor_for_lambda(train_idx, seed, lam):
        model = train_reranker(h, train_idx, seed)
        if model is None:
            return None
        X, y = [], []
        for i in train_idx:
            p = h.pools[i]
            r = f1_at_k(rerank(h, model, p.s, p.t, p.modality, p.cands), p.positives)
            sb = f1_at_k(symbolic_order(h, p.s, p.t, p.cands), p.positives)
            X.append(query_gain_features(h, p)); y.append(r - sb)
        X = np.array(X); y = np.array(y)
        mu, sd = X.mean(0), X.std(0) + 1e-9
        Xs = np.hstack([(X - mu) / sd, np.ones((len(X), 1))])
        A = Xs.T @ Xs + lam * np.eye(Xs.shape[1])
        wv = np.linalg.solve(A, Xs.T @ y)
        return (wv, mu, sd)

    def admits(pred, p):
        wv, mu, sd = pred
        x = np.append((query_gain_features(h, p) - mu) / sd, 1.0)
        return float(x @ wv) > 0.0

    rows = []
    for lam in [0.25, 0.5, 1.0, 2.0, 4.0]:
        admit_rates, flips_vs_default, total = [], 0, 0
        for seed in SEED_MANIFEST:
            rng = np.random.default_rng(seed)
            idx = list(range(len(h.pools))); rng.shuffle(idx)
            nt = max(1, int(round(0.2 * len(idx)))); test = list(idx[:nt]); train = [i for i in idx if i not in set(test)]
            pd = predictor_for_lambda(train, seed, lam)
            pdef = predictor_for_lambda(train, seed, 1.0)
            if pd is None or pdef is None:
                continue
            for i in test:
                a = admits(pd, h.pools[i]); a0 = admits(pdef, h.pools[i])
                admit_rates.append(1.0 if a else 0.0)
                total += 1
                if a != a0:
                    flips_vs_default += 1
        rows.append({"ridge_lambda": lam,
                     "admit_rate": round(statistics.mean(admit_rates), 4) if admit_rates else 0.0,
                     "route_flips_vs_default": flips_vs_default,
                     "flip_rate_vs_default": round(flips_vs_default / max(total, 1), 4)})
    return rows


def main():
    root = Path(__file__).resolve().parents[1]
    out = root / "results"; out.mkdir(parents=True, exist_ok=True)

    neg = negative_pool_stress(root)
    (out / "negative_pool_stress.json").write_text(json.dumps(neg, indent=2))
    print("NEGATIVE POOL STRESS:", json.dumps(neg))

    core_rows = core_growth_failure(root)
    with (out / "core_growth_failure.csv").open("w", newline="") as f:
        wtr = csv.DictWriter(f, fieldnames=list(core_rows[0].keys())); wtr.writeheader(); wtr.writerows(core_rows)
    print("CORE GROWTH:")
    for r in core_rows: print(" ", r)

    cm = cost_model_sensitivity(root)
    with (out / "cost_model_sensitivity_v1.csv").open("w", newline="") as f:
        wtr = csv.DictWriter(f, fieldnames=list(cm[0].keys())); wtr.writeheader(); wtr.writerows(cm)
    print("COST-MODEL SENSITIVITY:")
    for r in cm: print(" ", r)


if __name__ == "__main__":
    main()
