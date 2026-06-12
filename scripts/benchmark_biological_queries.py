"""Biological reranking benchmark with INDEPENDENT pathway ground truth.

Design note: gold mediators are now defined by curated canonical-pathway
membership (agentflow_ppi/data/pathway_ground_truth), a label source that does
NOT reference edge confidence or modality counts -- the reranker's features.
This removes the previous circularity where the label-defining signal was also a
model feature.

Pipeline: (1) build the named PPI graph; (2) condense to a DAG and build SHRC;
(3) for each PATHWAY-GROUNDED query, generate mediator candidates by typed
expansion + exact SHRC reachability pruning; (4) split leak-free by family,
negatives = reachable-but-not-gold mediators; (5) train a logistic reranker and
report MEASURED macro-F1@2 over a seed manifest, for BOTH the full feature set
and a modality-ablated variant (A1 fix #2). Fully reproducible.
"""

from __future__ import annotations

import csv
import json
import math
import statistics
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from agentflow_ppi.reachability import SHRCIndex
from agentflow_ppi.data.pathway_ground_truth import (
    gold_pathway_mediators,
    query_is_pathway_grounded,
)

SEED_MANIFEST = [7, 13, 23, 37, 101]
FEATURE_NAMES = ["path_score", "log_outdeg", "in_conf", "modality_agreement", "mean_conf"]
MODALITY_FEATURE_IDX = 3  # index of modality_agreement, ablated in the control run


def load_graph(tsv_path: Path):
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
    id_to_name = {i: n for n, i in names.items()}
    return names, id_to_name, typed_adj, edges


def typed_expand(typed_adj, source, max_hops):
    seen = {source}; frontier = [source]; out = []
    for _ in range(max_hops):
        nxt = []
        for u in frontier:
            for v, _m, _c in typed_adj.get(u, []):
                if v not in seen:
                    seen.add(v); out.append(v); nxt.append(v)
        frontier = nxt
    return out


def path_score(typed_adj, source, mediator, target):
    leg1 = max([c for v, _m, c in typed_adj.get(source, []) if v == mediator], default=0.0)
    leg2 = max([c for v, _m, c in typed_adj.get(mediator, []) if v == target], default=0.0)
    if leg2 == 0.0:
        for v, _m, c in typed_adj.get(mediator, []):
            for w, _m2, c2 in typed_adj.get(v, []):
                if w == target:
                    leg2 = max(leg2, c * c2)
    return leg1 * leg2 if (leg1 and leg2) else max(leg1, leg2) * 0.5


def mediator_features(typed_adj, source, mediator, target, modality):
    ps = path_score(typed_adj, source, mediator, target)
    e = typed_adj.get(mediator, [])
    out_deg = len(e)
    in_conf = max([c for _v, _m, c in typed_adj.get(source, []) if _v == mediator], default=0.0)
    mod_match = (sum(1 for _v, m, _c in e if m == modality) / len(e)) if e else 0.0
    mean_conf = statistics.mean([c for _v, _m, c in e]) if e else 0.0
    return np.array([ps, math.log1p(out_deg), in_conf, mod_match, mean_conf], dtype=np.float64)


def f1_at_k(ranked, pos, k):
    if not pos:
        return 0.0
    topk = ranked[:k]
    tp = sum(1 for v in topk if v in pos)
    p = tp / max(len(topk), 1); r = tp / len(pos)
    return 0.0 if p + r == 0 else 2 * p * r / (p + r)


def train_logistic(X, y, seed, ablate_idx=None, epochs=300, lr=0.1):
    rng = np.random.default_rng(seed)
    Xw = X.copy()
    if ablate_idx is not None:
        Xw[:, ablate_idx] = 0.0  # remove the modality-agreement signal
    n, d = Xw.shape
    mu, sd = Xw.mean(0), Xw.std(0) + 1e-9
    Xs = (Xw - mu) / sd
    w = rng.normal(0, 0.01, size=d); b = 0.0
    for _ in range(epochs):
        p = 1.0 / (1.0 + np.exp(-(Xs @ w + b)))
        w -= lr * (Xs.T @ (p - y) / n + 1e-4 * w)
        b -= lr * float(np.mean(p - y))
    return w, b, mu, sd


def score_logistic(w, b, mu, sd, X, ablate_idx=None):
    Xw = X.copy()
    if ablate_idx is not None:
        Xw[:, ablate_idx] = 0.0
    Xs = (Xw - mu) / sd
    return 1.0 / (1.0 + np.exp(-(Xs @ w + b)))


def build_pools(queries, names, id2n, typed_adj, reach):
    """Pathway-grounded query pools with INDEPENDENT pathway labels."""
    per_query = []
    skipped = 0
    for q in queries:
        s_name, t_name = q["source"], q["target"]
        if s_name not in names or t_name not in names:
            continue
        # Only evaluate queries that are grounded on a canonical pathway, so the
        # independent labels are well-defined.
        if not query_is_pathway_grounded(s_name, t_name):
            skipped += 1
            continue
        s, t = names[s_name], names[t_name]
        raw = typed_expand(typed_adj, s, 3)
        cands = [v for v in raw if v not in (s, t) and reach(s, v) and reach(v, t)]
        if len(cands) < 2:
            for v in [x for x in raw if x not in (s, t) and reach(s, x)]:
                if v not in cands:
                    cands.append(v)
                if len(cands) >= 3:
                    break
        if len(cands) < 2:
            continue
        cand_names = [id2n[v] for v in cands]
        gold_names = gold_pathway_mediators(s_name, t_name, cand_names)
        positives = {names[g] for g in gold_names if g in names}
        positives = {v for v in positives if v in cands}
        if not positives:
            # query grounded but no gold candidate survived reachability; skip to
            # keep labels meaningful
            skipped += 1
            continue
        per_query.append({
            "key": f"{s_name}->{t_name}", "s": s, "t": t, "modality": q["modality"],
            "cands": cands, "positives": positives,
            "median_path_score": float(np.median([path_score(typed_adj, s, v, t) for v in cands])),
        })
    return per_query, skipped


def main():
    root = Path(__file__).resolve().parents[1]
    data_root = root / "examples" / "biological_queries"
    names, id2n, typed_adj, edges = load_graph(data_root / "named_ppi_edges.tsv")
    num_nodes = len(names)
    queries = json.loads((data_root / "real_bio_queries.json").read_text())

    from agentflow_ppi.data.cycle_handling import condense_to_dag
    cond = condense_to_dag(num_nodes, edges)
    comp = cond.component_of
    shrc = SHRCIndex.from_edges(num_nodes=cond.num_components, edges=cond.dag_edges).build()

    def reach(a, b):
        return comp[a] == comp[b] or shrc.reachable(comp[a], comp[b])

    per_query, skipped = build_pools(queries, names, id2n, typed_adj, reach)

    macro_base, macro_rer, macro_rer_ablate, train_seconds = [], [], [], []
    per_family = defaultdict(lambda: {"base": [], "rer": [], "abl": []})

    for seed in SEED_MANIFEST:
        rng = np.random.default_rng(seed)
        order = list(range(len(per_query))); rng.shuffle(order)
        n_test = max(1, int(round(0.2 * len(order))))
        test_idx = set(order[:n_test]); train_idx = [i for i in order if i not in test_idx]

        X, y = [], []
        for i in train_idx:
            qd = per_query[i]
            for v in qd["cands"]:
                X.append(mediator_features(typed_adj, qd["s"], v, qd["t"], qd["modality"]))
                y.append(1.0 if v in qd["positives"] else 0.0)
        X = np.array(X); y = np.array(y)
        if len(set(y.tolist())) < 2:
            continue
        t0 = time.perf_counter()
        w, b, mu, sd = train_logistic(X, y, seed)
        train_seconds.append(time.perf_counter() - t0)
        wa, ba, mua, sda = train_logistic(X, y, seed, ablate_idx=MODALITY_FEATURE_IDX)

        for i in test_idx:
            qd = per_query[i]; cands = qd["cands"]; pos = qd["positives"]
            base_rank = sorted(cands, key=lambda v: path_score(typed_adj, qd["s"], v, qd["t"]), reverse=True)
            Xte = np.array([mediator_features(typed_adj, qd["s"], v, qd["t"], qd["modality"]) for v in cands])
            sc = score_logistic(w, b, mu, sd, Xte)
            rer_rank = [cands[j] for j in np.argsort(-sc)]
            sca = score_logistic(wa, ba, mua, sda, Xte, ablate_idx=MODALITY_FEATURE_IDX)
            abl_rank = [cands[j] for j in np.argsort(-sca)]

            bf, rf, af = f1_at_k(base_rank, pos, 2), f1_at_k(rer_rank, pos, 2), f1_at_k(abl_rank, pos, 2)
            macro_base.append(bf); macro_rer.append(rf); macro_rer_ablate.append(af)
            per_family[qd["key"]]["base"].append(bf)
            per_family[qd["key"]]["rer"].append(rf)
            per_family[qd["key"]]["abl"].append(af)

    out_dir = root / "results"; out_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "label_source": "curated canonical-pathway membership (independent of reranker features)",
        "num_pathway_grounded_queries": len(per_query),
        "skipped_non_grounded": skipped,
        "macro_baseline_f1_at_2": round(statistics.mean(macro_base), 4),
        "macro_rerank_f1_at_2": round(statistics.mean(macro_rer), 4),
        "macro_rerank_modality_ablated_f1_at_2": round(statistics.mean(macro_rer_ablate), 4),
        "avg_num_candidates": round(statistics.mean([len(q["cands"]) for q in per_query]), 4),
        "median_path_score": round(statistics.median([q["median_path_score"] for q in per_query]), 4),
        "reranker_train_seconds": round(statistics.mean(train_seconds), 4) if train_seconds else 0.0,
        "num_seeds": len(SEED_MANIFEST), "eval_points": len(macro_base),
    }
    with (out_dir / "biological_query_summary.csv").open("w", newline="") as f:
        wtr = csv.DictWriter(f, fieldnames=list(summary.keys())); wtr.writeheader(); wtr.writerow(summary)

    fam_rows = []
    for key, acc in sorted(per_family.items()):
        if not acc["base"]:
            continue
        fam_rows.append({"query": key, "n_eval": len(acc["base"]),
                         "baseline_f1_at_2": round(statistics.mean(acc["base"]), 4),
                         "rerank_f1_at_2": round(statistics.mean(acc["rer"]), 4),
                         "rerank_modality_ablated_f1_at_2": round(statistics.mean(acc["abl"]), 4)})
    with (out_dir / "biological_query_results.csv").open("w", newline="") as f:
        wtr = csv.DictWriter(f, fieldnames=["query", "n_eval", "baseline_f1_at_2",
                                            "rerank_f1_at_2", "rerank_modality_ablated_f1_at_2"])
        wtr.writeheader(); wtr.writerows(fam_rows)

    protocol = {
        "label_source": "curated canonical-pathway membership; see agentflow_ppi/data/pathway_ground_truth.py",
        "independence": "labels reference only pathway position, never edge confidence or modality counts",
        "split_unit": "source-target query family",
        "num_pathway_grounded_queries": len(per_query),
        "split_ratio": {"train": 0.8, "test": 0.2},
        "negative_sampling": "reachable but non-gold mediators (exact SHRC reachability)",
        "model": "logistic reranker over path/feature evidence",
        "ablation": "modality_agreement feature zeroed to test the label-feature independence",
        "optimizer": "full-batch gradient descent", "learning_rate": 0.1, "l2": 1e-4, "epochs": 300,
        "seed_manifest": SEED_MANIFEST,
        "metric": "macro-F1@2 over test families across the seed manifest",
    }
    (out_dir / "biological_training_protocol.json").write_text(json.dumps(protocol, indent=2))
    print("Biological benchmark (independent pathway labels):")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
