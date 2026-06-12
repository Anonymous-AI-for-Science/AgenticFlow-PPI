"""Leakage-controlled splits for the external reranking benchmark.

Two split regimes, both stricter than a random query split:

  * pathway-disjoint: train and test queries come from DISJOINT Reactome pathways,
    so no pathway seen at training appears at test. Tests generalization to unseen
    pathway topology.

  * protein-disjoint: the protein set is partitioned, and a query is assigned to
    test only if BOTH its source and target (and ideally its gold mediators) live
    in the held-out protein partition, so no test protein is seen at training.
    This is the strongest control against memorization of node identity.

Both are deterministic given a seed.
"""

from __future__ import annotations

import random
from typing import Dict, List, Set, Tuple


def pathway_disjoint_split(queries: List[Dict], seed: int = 7, test_frac: float = 0.3
                           ) -> Tuple[List[int], List[int]]:
    pathways = sorted({q["pathway"] for q in queries})
    rng = random.Random(seed); rng.shuffle(pathways)
    n_test = max(1, int(round(test_frac * len(pathways))))
    test_pw = set(pathways[:n_test])
    train_idx, test_idx = [], []
    for i, q in enumerate(queries):
        (test_idx if q["pathway"] in test_pw else train_idx).append(i)
    return train_idx, test_idx


def protein_disjoint_split(queries: List[Dict], seed: int = 7, test_frac: float = 0.3
                           ) -> Tuple[List[int], List[int]]:
    proteins = sorted({q["source"] for q in queries} | {q["target"] for q in queries}
                      | {g for q in queries for g in q["gold"]})
    rng = random.Random(seed); rng.shuffle(proteins)
    n_test = max(1, int(round(test_frac * len(proteins))))
    test_prot = set(proteins[:n_test])
    train_idx, test_idx = [], []
    for i, q in enumerate(queries):
        members = {q["source"], q["target"], *q["gold"]}
        if members <= test_prot:
            test_idx.append(i)
        elif members.isdisjoint(test_prot):
            train_idx.append(i)
        # queries straddling the partition are dropped to guarantee disjointness
    return train_idx, test_idx


def split_stats(queries: List[Dict], train_idx: List[int], test_idx: List[int]) -> Dict:
    def prot_set(idxs):
        return {q["source"] for i in idxs for q in [queries[i]]} | \
               {q["target"] for i in idxs for q in [queries[i]]} | \
               {g for i in idxs for g in queries[i]["gold"]}
    tr_p, te_p = prot_set(train_idx), prot_set(test_idx)
    tr_pw = {queries[i]["pathway"] for i in train_idx}
    te_pw = {queries[i]["pathway"] for i in test_idx}
    return {
        "n_train": len(train_idx), "n_test": len(test_idx),
        "protein_overlap": len(tr_p & te_p), "pathway_overlap": len(tr_pw & te_pw),
        "train_proteins": len(tr_p), "test_proteins": len(te_p),
    }
