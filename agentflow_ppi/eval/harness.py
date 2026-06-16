"""Unified biological evaluation harness shared by every experiment script.

Reviewer E1 fix: previously the biological benchmark, the dispatch ablation, and
the end-to-end baselines each defined their own candidate pools, gold labels, and
reranker, producing inconsistent family counts (16/19/24) and incompatible label
sources. This module centralizes all of it so every downstream script uses the
SAME pathway-grounded labels, the SAME candidate generation, and the SAME trained
reranker. Differences between experiments then come only from the policy under
test, never from divergent setups.
"""

from __future__ import annotations

import csv
import math
import statistics
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple

import numpy as np

from agentflow_ppi.reachability import SHRCIndex
from agentflow_ppi.data.cycle_handling import condense_to_dag
from agentflow_ppi.data.pathway_ground_truth import (
    gold_pathway_mediators,
    query_is_pathway_grounded,
)

SEED_MANIFEST = [7, 13, 23, 37, 101]
FEATURE_NAMES = ["path_score", "log_outdeg", "in_conf", "modality_agreement", "mean_conf"]
MODALITY_FEATURE_IDX = 3
DATA_DIR = Path(__file__).resolve().parents[2] / "examples" / "biological_queries"


@dataclass
class QueryPool:
    key: str
    s: int
    t: int
    modality: str
    cands: List[int]
    positives: Set[int]


@dataclass
class Harness:
    names: Dict[str, int]
    id2n: Dict[int, str]
    typed_adj: Dict[int, List[Tuple[int, str, float]]]
    edges: List[Tuple[int, int]]
    pools: List[QueryPool]
    reach: object  # callable(a,b)->bool
    informative: Optional[Set[int]] = None  # pool indices where modality discriminates (large benchmark)


def _load_graph(tsv_path: Path):
    names: Dict[str, int] = {}
    typed_adj: Dict[int, List[Tuple[int, str, float]]] = defaultdict(list)
    edges: List[Tuple[int, int]] = []

    def nid(name: str) -> int:
        if name not in names:
            names[name] = len(names)
        return names[name]

    with tsv_path.open() as f:
        for row in csv.DictReader(f, delimiter="\t"):
            u, v = nid(row["source"]), nid(row["target"])
            typed_adj[u].append((v, row["modality"], float(row["score"])))
            edges.append((u, v))
    return names, {i: n for n, i in names.items()}, typed_adj, edges


def typed_expand(typed_adj, source, max_hops=3):
    seen = {source}; frontier = [source]; out = []
    for _ in range(max_hops):
        nxt = []
        for u in frontier:
            for v, _m, _c in typed_adj.get(u, []):
                if v not in seen:
                    seen.add(v); out.append(v); nxt.append(v)
        frontier = nxt
    return out


def path_score(typed_adj, s, v, t):
    l1 = max([c for w, _m, c in typed_adj.get(s, []) if w == v], default=0.0)
    l2 = max([c for w, _m, c in typed_adj.get(v, []) if w == t], default=0.0)
    if l2 == 0.0:
        for w, _m, c in typed_adj.get(v, []):
            for x, _m2, c2 in typed_adj.get(w, []):
                if x == t:
                    l2 = max(l2, c * c2)
    return l1 * l2 if (l1 and l2) else max(l1, l2) * 0.5


def modality_agreement(typed_adj, v, modality):
    e = typed_adj.get(v, [])
    return (sum(1 for _w, m, _c in e if m == modality) / len(e)) if e else 0.0


def features(typed_adj, s, v, t, modality):
    e = typed_adj.get(v, [])
    in_conf = max([c for _w, _m, c in typed_adj.get(s, []) if _w == v], default=0.0)
    mean_conf = statistics.mean([c for _w, _m, c in e]) if e else 0.0
    return np.array([path_score(typed_adj, s, v, t), math.log1p(len(e)),
                     in_conf, modality_agreement(typed_adj, v, modality), mean_conf],
                    dtype=np.float64)


def f1_at_k(ranked, pos, k=2):
    if not pos:
        return 0.0
    top = ranked[:k]; tp = sum(1 for v in top if v in pos)
    p = tp / max(len(top), 1); r = tp / len(pos)
    return 0.0 if p + r == 0 else 2 * p * r / (p + r)


def build_harness(max_hops: int = 3) -> Harness:
    import json
    names, id2n, typed_adj, edges = _load_graph(DATA_DIR / "named_ppi_edges.tsv")
    n = len(names)
    queries = json.loads((DATA_DIR / "real_bio_queries.json").read_text())
    cond = condense_to_dag(n, edges)
    comp = cond.component_of
    shrc = SHRCIndex.from_edges(num_nodes=cond.num_components, edges=cond.dag_edges).build()

    def reach(a, b):
        return comp[a] == comp[b] or shrc.reachable(comp[a], comp[b])

    pools: List[QueryPool] = []
    for q in queries:
        sn, tn = q["source"], q["target"]
        if sn not in names or tn not in names or not query_is_pathway_grounded(sn, tn):
            continue
        s, t = names[sn], names[tn]
        raw = typed_expand(typed_adj, s, max_hops)
        cands = [v for v in raw if v not in (s, t) and reach(s, v) and reach(v, t)]
        if len(cands) < 2:
            for v in [x for x in raw if x not in (s, t) and reach(s, x)]:
                if v not in cands:
                    cands.append(v)
                if len(cands) >= 3:
                    break
        if len(cands) < 2:
            continue
        gold_names = gold_pathway_mediators(sn, tn, [id2n[v] for v in cands])
        positives = {names[g] for g in gold_names if names.get(g) in cands}
        positives = {v for v in positives if v in cands}
        if not positives:
            continue
        pools.append(QueryPool(f"{sn}->{tn}", s, t, q["modality"], cands, positives))
    return Harness(names, id2n, typed_adj, edges, pools, reach)


def build_harness_large(num_pathways: int = 40, pathway_len: int = 8,
                        informative_fraction: float = 0.5, seed: int = 7,
                        max_hops: int = 4) -> Harness:
    """Build a large mixed-sign harness from the scalable pathway benchmark
    (reviewer W2/W3/W8/W12). Reranking helps on the modality-informative pathways
    and not on the rest, so the calibrated dispatcher must both admit and decline.

    Gold labels remain position-on-cascade (independent of edge attributes).
    """
    from agentflow_ppi.data.pathway_benchmark import (
        PathwayBenchmarkConfig, build_pathway_benchmark, gold_mediators,
        is_grounded, query_is_informative, enumerate_queries,
    )
    cfg = PathwayBenchmarkConfig(num_pathways=num_pathways, pathway_len=pathway_len,
                                 informative_fraction=informative_fraction, seed=seed)
    bench = build_pathway_benchmark(cfg)

    names: Dict[str, int] = {}
    typed_adj: Dict[int, List[Tuple[int, str, float]]] = defaultdict(list)
    edges: List[Tuple[int, int]] = []

    def nid(name: str) -> int:
        if name not in names:
            names[name] = len(names)
        return names[name]

    for u, v, m, s in bench.edges:
        iu, iv = nid(u), nid(v)
        typed_adj[iu].append((iv, m, s)); edges.append((iu, iv))
    id2n = {i: n for n, i in names.items()}
    n = len(names)

    cond = condense_to_dag(n, edges)
    comp = cond.component_of
    shrc = SHRCIndex.from_edges(num_nodes=cond.num_components, edges=cond.dag_edges).build()

    def reach(a, b):
        return comp[a] == comp[b] or shrc.reachable(comp[a], comp[b])

    pools: List[QueryPool] = []
    informative_idx: Set[int] = set()
    for (sn, tn) in enumerate_queries(bench):
        if sn not in names or tn not in names:
            continue
        s, t = names[sn], names[tn]
        raw = typed_expand(typed_adj, s, max_hops)
        cands = [v for v in raw if v not in (s, t) and reach(s, v) and reach(v, t)]
        if len(cands) < 2:
            continue
        gold_names = gold_mediators(bench, sn, tn, [id2n[v] for v in cands])
        positives = {names[g] for g in gold_names if names.get(g) in cands}
        positives = {v for v in positives if v in cands}
        if not positives:
            continue
        # native modality of the (s,t) pathway is the query intent modality
        native = None
        for nm, order in bench.pathway_order.items():
            if sn in order and tn in order and order.index(sn) < order.index(tn):
                native = bench.native_modality[nm]; break
        idx = len(pools)
        pools.append(QueryPool(f"{sn}->{tn}", s, t, native or "functional", cands, positives))
        if query_is_informative(bench, sn, tn):
            informative_idx.add(idx)
    return Harness(names, id2n, typed_adj, edges, pools, reach, informative_idx)


def train_reranker(h: Harness, train_idx: Sequence[int], seed: int, ablate_idx=None):
    rng = np.random.default_rng(seed)
    X, y = [], []
    for i in train_idx:
        p = h.pools[i]
        for v in p.cands:
            X.append(features(h.typed_adj, p.s, v, p.t, p.modality))
            y.append(1.0 if v in p.positives else 0.0)
    X = np.array(X); y = np.array(y)
    if len(set(y.tolist())) < 2:
        return None
    if ablate_idx is not None:
        X = X.copy(); X[:, ablate_idx] = 0.0
    mu, sd = X.mean(0), X.std(0) + 1e-9
    Xs = (X - mu) / sd
    w = rng.normal(0, 0.01, X.shape[1]); b = 0.0
    for _ in range(300):
        p = 1 / (1 + np.exp(-(Xs @ w + b)))
        w -= 0.1 * (Xs.T @ (p - y) / len(y) + 1e-4 * w); b -= 0.1 * float(np.mean(p - y))
    return (w, b, mu, sd, ablate_idx)


def rerank(h: Harness, model, s, t, modality, cands):
    w, b, mu, sd, ablate = model
    X = np.array([features(h.typed_adj, s, v, t, modality) for v in cands])
    if ablate is not None:
        X = X.copy(); X[:, ablate] = 0.0
    Xs = (X - mu) / sd
    sc = 1 / (1 + np.exp(-(Xs @ w + b)))
    return [cands[j] for j in np.argsort(-sc)]


def symbolic_order(h: Harness, s, t, cands):
    return sorted(cands, key=lambda v: path_score(h.typed_adj, s, v, t), reverse=True)


def expected_gain(h: Harness, s, t, modality, cands):
    """Heuristic gain proxy (modality-ambiguity). Retained for the ablation as the
    UNCALIBRATED baseline policy; superseded by the calibrated predictor below."""
    a = [modality_agreement(h.typed_adj, v, modality) for v in cands]
    ma = sum(a) / len(a)
    amb = sum(abs(x - ma) for x in a) / len(a)
    return float(np.clip(amb * 2.0, 0.0, 0.9))


def query_gain_features(h: Harness, p) -> np.ndarray:
    """Aggregate per-query features for predicting reranker F1 lift.

    These summarize the candidate frontier so the dispatcher can estimate, before
    paying for the reranker, whether reranking is likely to help. They are query-
    level (not per-candidate) and deliberately cheap to compute.
    """
    cands = p.cands
    ps = [path_score(h.typed_adj, p.s, v, p.t) for v in cands]
    ma = [modality_agreement(h.typed_adj, v, p.modality) for v in cands]
    ps_sorted = sorted(ps, reverse=True)
    # top-2 symbolic margin: large margin => symbolic order already confident
    margin = (ps_sorted[0] - ps_sorted[1]) if len(ps_sorted) >= 2 else 0.0
    mean_ma = sum(ma) / len(ma)
    amb = sum(abs(x - mean_ma) for x in ma) / len(ma)
    # Score-modality mismatch: when the highest-path-score candidate has LOW
    # modality agreement relative to the pool, the symbolic order is likely
    # misled and reranking is likely to help. This is the key discriminating
    # signal between modality-informative and uninformative queries.
    top_idx = int(np.argmax(ps)) if ps else 0
    mismatch = (mean_ma - ma[top_idx]) if ma else 0.0
    # correlation (negative => score and modality disagree)
    if len(ps) > 1 and np.std(ps) > 1e-9 and np.std(ma) > 1e-9:
        corr = float(np.corrcoef(ps, ma)[0, 1])
    else:
        corr = 0.0
    return np.array([len(cands), margin, mean_ma, amb,
                     statistics.pstdev(ps) if len(ps) > 1 else 0.0,
                     mismatch, corr],
                    dtype=np.float64)


def train_gain_predictor(h: Harness, train_idx, seed):
    """Train a calibrated predictor of reranker F1 lift on TRAIN queries only.

    Target = (rerank F1 - symbolic F1) computed on the training pools using a
    reranker trained on the same train split. The predictor is a small ridge
    regressor; the dispatcher admits the reranker iff predicted lift > 0.
    Reviewer E4: this is the calibration the dispatch decision actually needs.
    """
    model = train_reranker(h, train_idx, seed)
    if model is None:
        return None
    X, y = [], []
    for i in train_idx:
        p = h.pools[i]
        r = f1_at_k(rerank(h, model, p.s, p.t, p.modality, p.cands), p.positives)
        s = f1_at_k(symbolic_order(h, p.s, p.t, p.cands), p.positives)
        X.append(query_gain_features(h, p)); y.append(r - s)
    X = np.array(X); y = np.array(y)
    mu, sd = X.mean(0), X.std(0) + 1e-9
    Xs = np.hstack([(X - mu) / sd, np.ones((len(X), 1))])
    # ridge closed form
    lam = 1.0
    A = Xs.T @ Xs + lam * np.eye(Xs.shape[1])
    wv = np.linalg.solve(A, Xs.T @ y)
    return (wv, mu, sd)


def predict_gain(h: Harness, predictor, p) -> float:
    wv, mu, sd = predictor
    x = (query_gain_features(h, p) - mu) / sd
    x = np.append(x, 1.0)
    return float(x @ wv)
