"""Correctness oracle and answer-equivalence test for the engine baselines.

The oracle computes, by exhaustive BFS over the canonical export, the exact set of
mediator answers for each query: the proteins m such that source -> m -> target is
reachable (respecting edge direction) and m is a curated gold mediator. Every
engine baseline must return the SAME answer set; `answer_equivalence` checks this
and reports any disagreement. This is what lets us claim the engines are compared
on identical, verified-correct results rather than on incomparable outputs.
"""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Dict, List, Set, Tuple


@dataclass
class Oracle:
    succ: Dict[int, List[int]]
    nodes: Dict[int, str]

    def reachable(self, s: int, t: int) -> bool:
        if s == t:
            return True
        seen = {s}; q = deque([s])
        while q:
            u = q.popleft()
            for v in self.succ.get(u, ()):
                if v == t:
                    return True
                if v not in seen:
                    seen.add(v); q.append(v)
        return False

    def mediators(self, s: int, t: int, gold: List[int]) -> Set[int]:
        """Exact gold mediators on a directed s->...->m->...->t path."""
        out: Set[int] = set()
        for m in gold:
            if m in (s, t):
                continue
            if self.reachable(s, m) and self.reachable(m, t):
                out.add(m)
        return out


def build_oracle(nodes: Dict[int, str], edges) -> Oracle:
    succ: Dict[int, List[int]] = defaultdict(list)
    for s, d, _m, _sc, directed in edges:
        succ[s].append(d)
        if not directed:
            succ[d].append(s)
    return Oracle(succ, nodes)


def oracle_answers(nodes, edges, queries, progress_cb=None) -> Dict[int, Set[int]]:
    """qid -> exact gold-mediator answer set.

    `progress_cb`, if given, is called as progress_cb(done, total) after each
    query's BFS so a caller can render a progress bar; building the oracle is the
    per-query BFS loop and is the part that takes time on large graphs.
    """
    oracle = build_oracle(nodes, edges)
    total = len(queries)
    out: Dict[int, Set[int]] = {}
    for i, q in enumerate(queries, 1):
        out[q["qid"]] = oracle.mediators(q["source"], q["target"], q["gold"])
        if progress_cb is not None:
            progress_cb(i, total)
    return out


@dataclass
class EquivalenceReport:
    total: int
    matching: int
    mismatches: List[Tuple[int, Set[int], Set[int]]] = field(default_factory=list)

    @property
    def all_match(self) -> bool:
        return self.matching == self.total

    def summary(self) -> Dict:
        return {"total": self.total, "matching": self.matching,
                "all_match": self.all_match,
                "num_mismatches": len(self.mismatches)}


def answer_equivalence(reference: Dict[int, Set[int]],
                       candidate: Dict[int, Set[int]]) -> EquivalenceReport:
    """Compare an engine's per-query answer sets against the oracle reference."""
    rep = EquivalenceReport(total=len(reference), matching=0)
    for qid, ref_set in reference.items():
        cand = candidate.get(qid, set())
        if set(cand) == set(ref_set):
            rep.matching += 1
        else:
            rep.mismatches.append((qid, set(ref_set), set(cand)))
    return rep
