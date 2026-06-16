"""External reranking experiment on real (or fixture) biological data.

Builds the typed multimodal graph and SHRC index from the external manifest, then
evaluates exact symbolic ranking, an always-on learned reranker, and calibrated
cost-aware dispatch under TWO leakage-controlled regimes:
  * pathway-disjoint split (train/test pathways disjoint), and
  * protein-disjoint split (train/test proteins disjoint).

Gold mediators come from Reactome pathway position (independent of every reranker
feature), so the evaluation is not circular. Reports macro-F1@2 with bootstrap CIs
and the dispatcher's admit rate, for each split. This is the external benchmark
that replaces the toy 19-node graph as the headline biological evaluation.
"""

from __future__ import annotations

import json
import statistics
from collections import defaultdict
from pathlib import Path

import numpy as np

from agentflow_ppi.reachability import SHRCIndex
from agentflow_ppi.data.cycle_handling import condense_to_dag
from agentflow_ppi.data.external.download import DEFAULT_CACHE
from agentflow_ppi.data.external.manifest import build_manifest
from agentflow_ppi.data.external import splits as S

SEEDS = [7, 13, 23, 37, 101]
SYMBOLIC_MS = 0.05
RERANK_MS = 0.9


# --------------------------------------------------------------------------- #
# Lightweight progress reporting
# --------------------------------------------------------------------------- #
# Uses tqdm when available (nice rendering), otherwise a dependency-free fallback
# bar so the Tier-0 "numpy only" install still shows progress. All output goes to
# stderr so it never pollutes the JSON written to stdout.
import sys as _sys
import time as _time

try:
    from tqdm import tqdm as _tqdm  # type: ignore

    def progress(iterable=None, total=None, desc="", leave=True):
        return _tqdm(iterable, total=total, desc=desc, unit="it",
                     dynamic_ncols=True, file=_sys.stderr, leave=leave)

    def _phase(desc):
        print(f"\033[1m▶ {desc}\033[0m", file=_sys.stderr, flush=True)
    _HAS_TQDM = True
except Exception:  # noqa: BLE001
    _HAS_TQDM = False

    class _FallbackBar:
        """Minimal stderr progress bar: no third-party deps, carriage-return based."""
        def __init__(self, total=None, desc="", width=32):
            self.total = total if (total and total > 0) else None
            self.desc = desc
            self.width = width
            self.n = 0
            self.t0 = _time.time()
            self._render()

        def _render(self):
            elapsed = _time.time() - self.t0
            if self.total:
                frac = min(1.0, self.n / self.total)
                filled = int(self.width * frac)
                bar = "█" * filled + "░" * (self.width - filled)
                rate = self.n / elapsed if elapsed > 0 else 0.0
                eta = (self.total - self.n) / rate if rate > 0 else 0.0
                msg = (f"\r  {self.desc:<28} |{bar}| "
                       f"{self.n}/{self.total} ({frac*100:5.1f}%) "
                       f"[{elapsed:5.1f}s, ETA {eta:5.1f}s]")
            else:
                spin = "|/-\\"[self.n % 4]
                msg = (f"\r  {self.desc:<28} {spin} {self.n} it "
                       f"[{elapsed:5.1f}s]")
            print(msg, end="", file=_sys.stderr, flush=True)

        def update(self, k=1):
            self.n += k
            self._render()

        def close(self):
            self._render()
            print("", file=_sys.stderr, flush=True)

        def __iter__(self):
            return self

    def progress(iterable=None, total=None, desc="", leave=True):
        if iterable is None:
            return _FallbackBar(total=total, desc=desc)
        if total is None:
            try:
                total = len(iterable)
            except TypeError:
                total = None
        bar = _FallbackBar(total=total, desc=desc)
        for item in iterable:
            yield item
            bar.update(1)
        bar.close()

    def _phase(desc):
        print(f"\n\033[1m> {desc}\033[0m", file=_sys.stderr, flush=True)


def _build_graph(manifest):
    names, typed_adj, edges = {}, defaultdict(list), []
    def nid(x):
        if x not in names:
            names[x] = len(names)
        return names[x]
    for a, b, m, sc, d in manifest.edges:
        ia, ib = nid(a), nid(b)
        typed_adj[ia].append((ib, m, sc)); edges.append((ia, ib))
        if not d:  # undirected sources contribute both directions
            typed_adj[ib].append((ia, m, sc)); edges.append((ib, ia))
    return names, typed_adj, edges


def _path_score(typed_adj, s, v, t):
    l1 = max([c for w, _m, c in typed_adj.get(s, []) if w == v], default=0.0)
    l2 = max([c for w, _m, c in typed_adj.get(v, []) if w == t], default=0.0)
    return l1 * l2 if (l1 and l2) else max(l1, l2) * 0.5


def _mod_agree(typed_adj, v, modality):
    e = typed_adj.get(v, [])
    return (sum(1 for _w, m, _c in e if m == modality) / len(e)) if e else 0.0


def _feat(typed_adj, s, v, t, modality):
    e = typed_adj.get(v, [])
    in_conf = max([c for _w, _m, c in typed_adj.get(s, []) if _w == v], default=0.0)
    mean_conf = statistics.mean([c for _w, _m, c in e]) if e else 0.0
    return np.array([_path_score(typed_adj, s, v, t), np.log1p(len(e)), in_conf,
                     _mod_agree(typed_adj, v, modality), mean_conf], dtype=np.float64)


def _f1at2(ranked, pos):
    if not pos:
        return 0.0
    top = ranked[:2]; tp = sum(1 for v in top if v in pos)
    p = tp / max(len(top), 1); r = tp / len(pos)
    return 0.0 if p + r == 0 else 2 * p * r / (p + r)


def _train_reranker(pools, train_idx, typed_adj, seed, desc_prefix=""):
    rng = np.random.default_rng(seed); X, y = [], []
    for i in progress(train_idx, total=len(train_idx),
                      desc=f"{desc_prefix}reranker: features", leave=False):
        p = pools[i]
        for v in p["cands"]:
            X.append(_feat(typed_adj, p["s"], v, p["t"], p["modality"]))
            y.append(1.0 if v in p["pos"] else 0.0)
    if not X or len(set(y)) < 2:
        return None
    X = np.array(X); y = np.array(y)
    mu, sd = X.mean(0), X.std(0) + 1e-9; Xs = (X - mu) / sd
    w = rng.normal(0, 0.01, X.shape[1]); b = 0.0
    ITERS = 300
    for _ in progress(range(ITERS), total=ITERS, desc=f"{desc_prefix}reranker: 300-iter GD", leave=False):
        pr = 1 / (1 + np.exp(-(Xs @ w + b)))
        w -= 0.1 * (Xs.T @ (pr - y) / len(y) + 1e-4 * w); b -= 0.1 * float(np.mean(pr - y))
    return (w, b, mu, sd)


def _rerank(model, pools, i, typed_adj):
    w, b, mu, sd = model; p = pools[i]
    X = np.array([_feat(typed_adj, p["s"], v, p["t"], p["modality"]) for v in p["cands"]])
    sc = 1 / (1 + np.exp(-(((X - mu) / sd) @ w + b)))
    return [p["cands"][j] for j in np.argsort(-sc)]


def _symbolic(pools, i, typed_adj):
    p = pools[i]
    return sorted(p["cands"], key=lambda v: _path_score(typed_adj, p["s"], v, p["t"]), reverse=True)


def _query_gain_feats(pools, i, typed_adj):
    p = pools[i]; ps = [_path_score(typed_adj, p["s"], v, p["t"]) for v in p["cands"]]
    ma = [_mod_agree(typed_adj, v, p["modality"]) for v in p["cands"]]
    pss = sorted(ps, reverse=True); margin = (pss[0] - pss[1]) if len(pss) >= 2 else 0.0
    mean_ma = sum(ma) / len(ma); amb = sum(abs(x - mean_ma) for x in ma) / len(ma)
    top = int(np.argmax(ps)) if ps else 0; mismatch = mean_ma - ma[top] if ma else 0.0
    corr = float(np.corrcoef(ps, ma)[0, 1]) if (len(ps) > 1 and np.std(ps) > 1e-9 and np.std(ma) > 1e-9) else 0.0
    return np.array([len(p["cands"]), margin, mean_ma, amb,
                     statistics.pstdev(ps) if len(ps) > 1 else 0.0, mismatch, corr])


def _train_gain(pools, train_idx, typed_adj, model, desc_prefix=""):
    X, y = [], []
    for i in progress(train_idx, total=len(train_idx),
                      desc=f"{desc_prefix}gain predictor: fit", leave=False):
        r = _f1at2(_rerank(model, pools, i, typed_adj), pools[i]["pos"])
        s = _f1at2(_symbolic(pools, i, typed_adj), pools[i]["pos"])
        X.append(_query_gain_feats(pools, i, typed_adj)); y.append(r - s)
    X = np.array(X); y = np.array(y)
    mu, sd = X.mean(0), X.std(0) + 1e-9
    Xs = np.hstack([(X - mu) / sd, np.ones((len(X), 1))])
    wv = np.linalg.solve(Xs.T @ Xs + 1.0 * np.eye(Xs.shape[1]), Xs.T @ y)
    return (wv, mu, sd)


def _predict_gain(pred, pools, i, typed_adj):
    wv, mu, sd = pred
    x = np.append((_query_gain_feats(pools, i, typed_adj) - mu) / sd, 1.0)
    return float(x @ wv)


def _bootstrap_ci(xs, seed=0, n=2000):
    if not xs:
        return [0.0, 0.0]
    rng = np.random.default_rng(seed); a = np.array(xs)
    m = [float(np.mean(rng.choice(a, len(a), replace=True))) for _ in range(n)]
    return [round(float(np.percentile(m, 2.5)), 4), round(float(np.percentile(m, 97.5)), 4)]


def _evaluate(manifest, split_fn, split_name, seeds=None, max_queries=None):
    seeds = seeds if seeds is not None else SEEDS
    _phase(f"[{split_name}] building typed graph + SHRC index")
    names, typed_adj, edges = _build_graph(manifest)
    n = len(names)
    cond = condense_to_dag(n, edges); comp = cond.component_of
    shrc = SHRCIndex.from_edges(num_nodes=cond.num_components, edges=cond.dag_edges).build()
    print(f"    graph: {n} nodes, {len(edges)} edges -> "
          f"{cond.num_components} SCC-condensed nodes, "
          f"{len(cond.dag_edges)} DAG edges", file=_sys.stderr, flush=True)
    def reach(a, b):
        return comp[a] == comp[b] or shrc.reachable(comp[a], comp[b])

    # build candidate pools per query (typed expansion + exact reachability prune)
    def expand(s, hops=3):
        seen = {s}; frontier = [s]; out = []
        for _ in range(hops):
            nxt = []
            for u in frontier:
                for v, _m, _c in typed_adj.get(u, []):
                    if v not in seen:
                        seen.add(v); out.append(v); nxt.append(v)
            frontier = nxt
        return out

    pools, q_pathway = [], []
    # Optional deterministic subsample for fast runs (disclosed in output JSON).
    # Sorted by query id then truncated so the subset is reproducible, not cherry-picked.
    _queries = manifest.queries
    if max_queries is not None and max_queries < len(_queries):
        _queries = sorted(_queries, key=lambda q: q.get("qid", str(q.get("source", ""))))[:max_queries]
    for q in progress(_queries, total=len(_queries), desc=f"[{split_name}] candidate pools"):
        if q["source"] not in names or q["target"] not in names:
            continue
        s, t = names[q["source"]], names[q["target"]]
        raw = expand(s)
        cands = [v for v in raw if v not in (s, t) and reach(s, v) and reach(v, t)]
        if len(cands) < 2:
            continue
        pos = {names[g] for g in q["gold"] if g in names and names[g] in cands}
        if not pos:
            continue
        modality = "regulatory"
        pools.append({"s": s, "t": t, "modality": modality, "cands": cands, "pos": pos})
        q_pathway.append(q["pathway"])
    print(f"    kept {len(pools)} pathway-grounded query pools", file=_sys.stderr, flush=True)

    if len(pools) < 4:
        return {"split": split_name, "n_queries": len(pools),
                "note": "too few pathway-grounded queries to evaluate on this data"}

    # map split (operates on query dicts) to pool indices
    pool_queries = [{"source": manifest.queries[0]["source"], "target": "", "pathway": q_pathway[i],
                     "gold": []} for i in range(len(pools))]
    # we need source/target/gold symbols for protein-disjoint; rebuild from names
    id2n = {v: k for k, v in names.items()}
    pool_queries = []
    for idx, p in enumerate(pools):
        pool_queries.append({
            "source": id2n[p["s"]], "target": id2n[p["t"]],
            "pathway": q_pathway[idx], "gold": [id2n[v] for v in p["pos"]],
        })

    sym_f1, alw_f1, disp_f1, admit, seg_helps = [], [], [], [], []
    _phase(f"[{split_name}] training + evaluating over {len(seeds)} seeds")
    for si, seed in enumerate(progress(seeds, desc=f"[{split_name}] seeds"), 1):
        tr, te = split_fn(pool_queries, seed=seed)
        if not tr or not te:
            continue
        sp = f"    seed {seed} [{si}/{len(seeds)}] "
        model = _train_reranker(pools, tr, typed_adj, seed, desc_prefix=sp)
        if model is None:
            continue
        pred = _train_gain(pools, tr, typed_adj, model, desc_prefix=sp)
        for i in progress(te, total=len(te), desc=f"{sp}eval: test queries", leave=False):
            s = _f1at2(_symbolic(pools, i, typed_adj), pools[i]["pos"])
            r = _f1at2(_rerank(model, pools, i, typed_adj), pools[i]["pos"])
            g = _predict_gain(pred, pools, i, typed_adj)
            adm = g > 0.0 and len(pools[i]["cands"]) <= 50
            sym_f1.append(s); alw_f1.append(r); disp_f1.append(r if adm else s)
            admit.append(1 if adm else 0); seg_helps.append(1 if r > s else 0)

    if not sym_f1:
        return {"split": split_name, "n_queries": len(pools), "note": "no evaluable test folds"}
    stats = S.split_stats(pool_queries, *split_fn(pool_queries, seed=SEEDS[0]))
    return {
        "split": split_name, "n_queries": len(pools), "test_decisions": len(sym_f1),
        "protein_overlap": stats["protein_overlap"], "pathway_overlap": stats["pathway_overlap"],
        "symbolic_f1": round(statistics.mean(sym_f1), 4), "symbolic_ci": _bootstrap_ci(sym_f1),
        "always_on_f1": round(statistics.mean(alw_f1), 4), "always_on_ci": _bootstrap_ci(alw_f1),
        "dispatch_f1": round(statistics.mean(disp_f1), 4), "dispatch_ci": _bootstrap_ci(disp_f1),
        "reranker_call_rate": round(statistics.mean(admit), 4),
        "frac_queries_reranking_helps": round(statistics.mean(seg_helps), 4),
    }


def main():
    import argparse
    ap = argparse.ArgumentParser(description="External reranking benchmark "
        "(protein/pathway-disjoint). Fast-run flags subsample queries/seeds "
        "deterministically for time-boxed runs; the chosen config is recorded "
        "in the output JSON for full disclosure.")
    ap.add_argument("--quick", action="store_true",
                    help="time-boxed preset: 2 seeds, max 4000 queries per split")
    ap.add_argument("--seeds", type=str, default=None,
                    help="comma-separated seeds (default: 7,13,23,37,101)")
    ap.add_argument("--max-queries", type=int, default=None,
                    help="cap queries per split (deterministic sorted subsample)")
    args = ap.parse_args()

    seeds = SEEDS
    max_queries = None
    if args.quick:
        seeds = [7, 13]; max_queries = 4000
    if args.seeds:
        seeds = [int(x) for x in args.seeds.split(",") if x.strip()]
    if args.max_queries is not None:
        max_queries = args.max_queries

    root = Path(__file__).resolve().parents[1]
    out = root / "results"; out.mkdir(parents=True, exist_ok=True)
    print(f"progress bar backend: {'tqdm' if _HAS_TQDM else 'builtin (pip install tqdm for nicer output)'}",
          file=_sys.stderr, flush=True)
    if max_queries is not None or seeds != SEEDS:
        print(f"    [fast-run] seeds={seeds}, max_queries={max_queries} "
              f"(subsample is deterministic and recorded in the JSON; "
              f"run with no flags for the full configuration)", file=_sys.stderr, flush=True)
    _phase("loading external manifest (download cache or bundled fixture)")
    manifest = build_manifest(cache_dir=DEFAULT_CACHE, allow_online=True)
    using_fixture = any(v == "fixture" for v in manifest.provenance.values())
    print(f"    manifest: {len(manifest.queries)} queries, "
          f"{len(manifest.pathway_members)} pathways, "
          f"using_fixture={using_fixture}", file=_sys.stderr, flush=True)

    results = {
        "provenance": manifest.provenance,
        "using_fixture": using_fixture,
        "run_config": {"seeds": seeds, "max_queries": max_queries,
                       "full_config": (max_queries is None and seeds == SEEDS)},
        "graph_proteins": len({a for a, *_ in manifest.edges} | {b for _, b, *_ in manifest.edges}),
        "graph_edges": len(manifest.edges),
        "num_pathways": len(manifest.pathway_members),
        "num_queries": len(manifest.queries),
        "pathway_disjoint": _evaluate(manifest, S.pathway_disjoint_split, "pathway-disjoint",
                                      seeds=seeds, max_queries=max_queries),
        "protein_disjoint": _evaluate(manifest, S.protein_disjoint_split, "protein-disjoint",
                                      seeds=seeds, max_queries=max_queries),
    }
    (out / "external_reranking_results.json").write_text(json.dumps(results, indent=2))
    _phase("done — results written to results/external_reranking_results.json")
    print(json.dumps(results, indent=2))
    if using_fixture:
        print("\n[NOTE] Ran on bundled FIXTURES (source hosts unreachable). "
              "Run scripts/download_external_data.py on a networked host for the full datasets.")


if __name__ == "__main__":
    main()
