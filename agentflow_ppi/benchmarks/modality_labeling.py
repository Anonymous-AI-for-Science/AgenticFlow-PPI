"""Modality-partitioned 2-hop labeling: fallback-free label-constrained reachability.

The prior LCR baseline answered a label-constrained query (does s reach t using only
edges whose modality lies in an allowed set L?) with a per-query label-restricted BFS.
This module removes that fallback. Each 2-hop hub entry stores, in addition to the
hub, the *minimal witness label-sets* of paths through that hub, so a query is
answered by set-containment tests over labels alone -- no graph traversal at query
time.

Construction. For each hub h (in degree order), a multi-label pruned BFS forward
(resp. backward) records, for every reachable node u, the antichain of minimal label
sets L such that h reaches u (resp. u reaches h) using exactly the modalities in L.
A label set is added only if it is not a superset of one already recorded for (u,h)
(the *minimal-witness* prune) and not already covered by a higher-ranked hub (the
2-hop prune).

Query. s reaches t under allowed set A iff there exists a hub h and an out-entry
(h, L_out) at s and an in-entry (h, L_in) at t with L_out subset A and L_in subset A.
Because a path s->h->t uses labels L_out union L_in, both must lie in A; storing
minimal sets makes the existence test exact.

Exactness is verified against a brute-force label-restricted BFS in the tests and in
the benchmark harness. The key space result (Proposition: under m modalities the
per-(node,hub) antichain has size at most C(m, floor(m/2)), and is O(m) when witness
sets are nested along shortest paths) is stated and checked empirically.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Dict, FrozenSet, Iterable, List, Optional, Set, Tuple


@dataclass
class MPLStats:
    index_entries: int          # total (hub, label-set) entries across all nodes
    max_antichain: int          # largest per-(node,hub) antichain observed
    num_modalities: int


def _antichain_add(store: List[FrozenSet[str]], new: FrozenSet[str]) -> bool:
    """Insert `new` into a minimal-set antichain. Returns False if `new` is dominated
    (a subset already present); drops any existing supersets of `new`. Keeping only
    minimal label sets is what bounds the index size."""
    for s in store:
        if s <= new:
            return False  # already have a witness with fewer/equal labels
    # remove supersets of new
    store[:] = [s for s in store if not (new < s)]
    store.append(new)
    return True


class ModalityPartitionedLabeling:
    """Fallback-free label-constrained reachability via minimal-witness 2-hop labels."""
    name = "mpl-2hop"

    def __init__(self, num_nodes: int, labeled_edges: Iterable[Tuple[int, int, str]]):
        self.n = num_nodes
        self.fwd: List[List[Tuple[int, str]]] = [[] for _ in range(num_nodes)]
        self.bwd: List[List[Tuple[int, str]]] = [[] for _ in range(num_nodes)]
        modalities: Set[str] = set()
        deg = [0] * num_nodes
        for u, v, lab in labeled_edges:
            self.fwd[u].append((v, lab)); self.bwd[v].append((u, lab))
            modalities.add(lab); deg[u] += 1; deg[v] += 1
        self.modalities = modalities
        # out_lbl[u] : hub -> antichain of minimal label-sets for u ->* hub
        # in_lbl[u]  : hub -> antichain of minimal label-sets for hub ->* u
        self.out_lbl: List[Dict[int, List[FrozenSet[str]]]] = [dict() for _ in range(num_nodes)]
        self.in_lbl: List[Dict[int, List[FrozenSet[str]]]] = [dict() for _ in range(num_nodes)]

        order = sorted(range(num_nodes), key=lambda x: deg[x], reverse=True)
        for h in order:
            self._bfs_label(h, forward=True)
            self._bfs_label(h, forward=False)

        entries = sum(len(s) for nd in self.out_lbl for s in nd.values())
        entries += sum(len(s) for nd in self.in_lbl for s in nd.values())
        max_ac = 0
        for nd in self.out_lbl:
            for s in nd.values():
                max_ac = max(max_ac, len(s))
        for nd in self.in_lbl:
            for s in nd.values():
                max_ac = max(max_ac, len(s))
        self.stats = MPLStats(index_entries=entries, max_antichain=max_ac,
                              num_modalities=len(modalities))

    def _covered(self, label_store: List, node: int, hub: int, lset: FrozenSet[str],
                 from_node_is_out: bool) -> bool:
        """2-hop prune: is (node ->* hub) with labels lset already decided by an
        existing higher-ranked hub h' such that node ->* h' ->* hub within lset?
        We approximate the standard prune by the minimal-set antichain at the node for
        already-processed hubs; exactness is preserved because the antichain keeps all
        minimal witnesses (the test verifies against BFS)."""
        # conservative prune: only skip if an equal-or-smaller label set to the SAME
        # hub already exists (handled by _antichain_add); cross-hub pruning is omitted
        # to preserve exactness, at a modest size cost.
        return False

    def _bfs_label(self, hub: int, forward: bool) -> None:
        """Multi-label BFS from `hub`. State = (node, label-set-so-far); we expand and
        keep only minimal label sets per node (antichain), recording hub-entries.
        2-hop prune: if a higher-ranked hub already provides a witness for (node,hub)
        with a subset of labels, skip -- this keeps the index near-PLL size."""
        adj = self.fwd if forward else self.bwd
        target_store = self.in_lbl if forward else self.out_lbl
        query_store = self.out_lbl if forward else self.in_lbl  # for the prune query
        start = frozenset()
        best: Dict[int, List[FrozenSet[str]]] = {hub: [start]}
        dq: deque = deque([(hub, start)])
        target_store[hub].setdefault(hub, [])
        _antichain_add(target_store[hub][hub], start)
        while dq:
            u, used = dq.popleft()
            if not any(s == used for s in best.get(u, [])):
                continue
            for v, lab in adj[u]:
                nused = used | {lab}
                fz = frozenset(nused)
                # 2-hop prune: is (hub ->* v) under labels fz already covered by an
                # already-processed (higher-rank) hub h'? i.e. hub ->* h' ->* v with
                # witness labels subset of fz. We test existing entries at v and hub.
                if self._covered_by_existing(hub, v, fz, forward):
                    continue
                store = best.setdefault(v, [])
                if _antichain_add(store, fz):
                    target_store[v].setdefault(hub, [])
                    if _antichain_add(target_store[v][hub], fz):
                        dq.append((v, fz))

    def _covered_by_existing(self, hub: int, v: int, fz: FrozenSet[str], forward: bool) -> bool:
        """True if some already-processed hub h' (h' != hub) gives hub ->* h' ->* v
        (forward) with combined witness labels subset of fz, making this entry
        redundant. Conservative: only prunes when a strictly-covering witness exists,
        preserving exactness."""
        store_hub = self.in_lbl[v] if forward else self.out_lbl[v]
        for hp, wits in store_hub.items():
            if hp == hub:
                continue
            # need hub ->* hp witness too
            other = (self.in_lbl[hp] if forward else self.out_lbl[hp]).get(hub)
            if not other:
                continue
            for w1 in other:
                if not (w1 <= fz):
                    continue
                for w2 in wits:
                    if (w1 | w2) <= fz:
                        return True
        return False

    def reachable(self, s: int, t: int, allowed: Optional[Set[str]] = None) -> bool:
        """Label-constrained reachability with NO graph traversal. If `allowed` is
        None, answers plain reachability (any modality permitted)."""
        if s == t:
            return True
        A = self.modalities if allowed is None else set(allowed)
        out_s = self.out_lbl[s]; in_t = self.in_lbl[t]
        # iterate the smaller hub set
        if len(out_s) > len(in_t):
            hubs = in_t.keys()
        else:
            hubs = out_s.keys()
        for h in hubs:
            if h not in out_s or h not in in_t:
                continue
            # exists out-witness subset A and in-witness subset A?
            out_ok = any(set(ls) <= A for ls in out_s[h])
            if not out_ok:
                continue
            in_ok = any(set(ls) <= A for ls in in_t[h])
            if in_ok:
                return True
        return False
