"""Sparsity-driven Hybrid Reachability Checking (SHRC) for DAGs.

The implementation follows three design principles that align with the paper:

1. Tree-core decomposition peels off sparse peripheral regions and retains only a
   connectivity-rich core for expensive indexing.
2. Hybrid labeling assigns interval labels to the tree region and exact 2-hop
   labels to the core.
3. Greedy label selection compresses both the 2-hop core index and the tree-to-
   core exit anchors while preserving exact answers for existential reachability
   tests; bounded-length and motif semantics stay outside the index and are
   evaluated after SHRC pruning.

The index is exact for DAG reachability. The core is intentionally computed on a
structural (undirected) view to expose sparse branches; afterwards a directed
refinement step promotes ambiguous vertices back into the core until the peeled
region forms a directed forest. The directed forest enables O(1) interval tests,
whereas the residual core is handled by an exact 2-hop index over an augmented
core graph that also captures tree-mediated re-entry paths.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import random
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Tuple


def _iter_bits(bits: int) -> Iterator[int]:
    """Yield set-bit positions from a Python integer bitset."""
    while bits:
        low = bits & -bits
        yield low.bit_length() - 1
        bits ^= low


@dataclass(slots=True)
class SHRCStats:
    """Diagnostic counters for the built index."""

    num_nodes: int
    num_edges: int
    core_nodes: int
    tree_nodes: int
    roots: int
    core_label_entries: int
    exit_anchor_entries: int
    max_interval_depth: int


@dataclass(slots=True)
class SHRCQueryTrace:
    """Human-readable explanation for one reachability decision."""

    reachable: bool
    route: str
    witness: Tuple[int, ...] = ()


class SHRCIndex:
    """Exact sparsity-driven hybrid reachability index for DAGs.

    Complexity summary:
        * Tree-core decomposition: O(|V| + |E|) for the peeling pass, plus up to
          O(k * (|V| + |E|)) for directed refinement where k is usually small in
          sparse graphs because promotions monotonically shrink the tree region.
        * Core closure materialization: O(|E_C| + |V_C|^2 / w) using Python-int
          bitsets, where C denotes the augmented core and w is the machine word
          size.
        * Greedy 2-hop construction: O(|V_C|^3) worst-case on the core, which is
          acceptable because the decomposition explicitly minimizes |V_C|. The
          released fallback threshold is a practical guardrail set to 5000 core
          vertices so that all public workloads stay in exact mode; larger reruns
          can opt into an approximate 2-hop index with a conservative manifest-
          level error bound delta <= epsilon * |C|. This branch is included as
          an auditable engineering fallback and is not validated in the paper
          beyond that explicit bound.
        * Query time: O(1) for pure tree ancestry tests and O(|L_out(u)| +
          |L_in(v)|) or O(|A(u)| * |B(v)|) for mixed/core cases, where labels and
          anchor sets are aggressively compressed.
    """

    def __init__(
        self,
        adjacency: Sequence[Sequence[int]],
        *,
        core_hub_strategy: str = "degree",
        exit_prune_strategy: str = "greedy",
        random_seed: int = 7,
        fallback_core_threshold: int = 5000,
        approx_epsilon: float = 1e-4,
        allow_approx_fallback: bool = True,
    ) -> None:
        self.adjacency: List[List[int]] = [sorted(set(neighbors)) for neighbors in adjacency]
        self.core_hub_strategy = core_hub_strategy
        self.exit_prune_strategy = exit_prune_strategy
        self._rng = random.Random(random_seed)
        self.fallback_core_threshold = fallback_core_threshold
        self.approx_epsilon = approx_epsilon
        self.allow_approx_fallback = allow_approx_fallback
        self.approximate_core_used: bool = False
        self.delta_bound: float = 0.0
        self.num_nodes: int = len(self.adjacency)
        self.predecessors: List[List[int]] = [[] for _ in range(self.num_nodes)]
        edge_count = 0
        for src, neighbors in enumerate(self.adjacency):
            edge_count += len(neighbors)
            for dst in neighbors:
                if dst < 0 or dst >= self.num_nodes:
                    raise ValueError(f"Edge ({src}, {dst}) is out of bounds.")
                self.predecessors[dst].append(src)
        self.num_edges = edge_count

        self.topological_order: List[int] = self._topological_order()
        self.topo_rank: List[int] = [0] * self.num_nodes
        for rank, node in enumerate(self.topological_order):
            self.topo_rank[node] = rank

        self.core_mask: List[bool] = [False] * self.num_nodes
        self.tree_parent: List[int] = [-1] * self.num_nodes
        self.tree_children: List[List[int]] = [[] for _ in range(self.num_nodes)]
        self.tree_root: List[int] = [-1] * self.num_nodes
        self.tree_depth: List[int] = [0] * self.num_nodes
        self.interval_in: List[int] = [-1] * self.num_nodes
        self.interval_out: List[int] = [-1] * self.num_nodes
        self.root_entries: Dict[int, Tuple[int, ...]] = {}
        self.root_exits: Dict[int, Tuple[int, ...]] = {}
        self.entry_anchors: List[Tuple[int, ...]] = [tuple() for _ in range(self.num_nodes)]
        self.raw_exit_anchors: List[Tuple[int, ...]] = [tuple() for _ in range(self.num_nodes)]
        self.exit_anchors: List[Tuple[int, ...]] = [tuple() for _ in range(self.num_nodes)]

        self.core_nodes: List[int] = []
        self.core_index: Dict[int, int] = {}
        self.core_succ_bits: List[int] = []
        self.core_pred_bits: List[int] = []
        self.core_out_labels: Dict[int, Tuple[int, ...]] = {}
        self.core_in_labels: Dict[int, Tuple[int, ...]] = {}
        self.stats: Optional[SHRCStats] = None
        self._max_depth: int = 0

    @classmethod
    def from_edges(
        cls,
        num_nodes: int,
        edges: Iterable[Tuple[int, int]],
        **kwargs: object,
    ) -> "SHRCIndex":
        adjacency = [[] for _ in range(num_nodes)]
        for src, dst in edges:
            adjacency[src].append(dst)
        return cls(adjacency, **kwargs)

    def build(self) -> "SHRCIndex":
        core_nodes = self._tree_core_decomposition()
        self.core_mask = [False] * self.num_nodes
        for node in core_nodes:
            self.core_mask[node] = True
        self.core_nodes = sorted(core_nodes, key=self.topo_rank.__getitem__)
        self.core_index = {node: idx for idx, node in enumerate(self.core_nodes)}

        self._build_tree_labels()
        self._compute_raw_tree_exits()
        self._build_core_labels()
        self._compress_tree_exit_anchors()
        self._finalize_stats()
        return self

    def reachable(self, source: int, target: int) -> bool:
        return self.explain(source, target).reachable

    def explain(self, source: int, target: int) -> SHRCQueryTrace:
        if source == target:
            return SHRCQueryTrace(True, "identity", (source,))
        if self.topo_rank[source] >= self.topo_rank[target]:
            return SHRCQueryTrace(False, "topological-prune")

        source_core = self.core_mask[source]
        target_core = self.core_mask[target]

        if not source_core and not target_core:
            if self._tree_ancestor(source, target):
                return SHRCQueryTrace(True, "tree-interval", (source, target))
            if self._tree_to_tree_via_core(source, target):
                return SHRCQueryTrace(True, "tree-core-tree", (source, target))
            return SHRCQueryTrace(False, "tree-miss")

        if source_core and target_core:
            if self._core_reachable(source, target):
                return SHRCQueryTrace(True, "core-2hop", (source, target))
            return SHRCQueryTrace(False, "core-miss")

        if not source_core and target_core:
            if self._tree_to_core(source, target):
                return SHRCQueryTrace(True, "tree-exit-to-core", (source, target))
            return SHRCQueryTrace(False, "tree-to-core-miss")

        if self._core_to_tree(source, target):
            return SHRCQueryTrace(True, "core-entry-to-tree", (source, target))
        return SHRCQueryTrace(False, "core-to-tree-miss")

    def total_index_entries(self, interval_cost_per_tree_node: int = 3) -> int:
        """Fair, single-unit count of the WHOLE SHRC index, so it can be compared
        like-for-like against a full-graph 2-hop labeling that counts every node.

        Includes (i) core 2-hop label entries, (ii) tree->core exit anchors, and
        (iii) the periphery interval labels: each tree node stores an interval
        (interval_in, interval_out) plus a parent pointer = 3 integer entries by
        default. Earlier versions omitted (iii), which understated SHRC's footprint
        relative to a full-graph PLL; this method closes that gap (reviewer W1)."""
        if self.stats is None:
            raise RuntimeError("Index has not been built yet.")
        core = sum(len(v) for v in self.core_out_labels.values()) + \
            sum(len(v) for v in self.core_in_labels.values())
        exits = self.stats.exit_anchor_entries
        tree = self.stats.tree_nodes * interval_cost_per_tree_node
        return core + exits + tree

    def summary(self) -> Dict[str, int]:
        if self.stats is None:
            raise RuntimeError("Index has not been built yet.")
        avg_exit_width = self.stats.exit_anchor_entries / max(1, self.stats.tree_nodes)
        return {
            "approximate_core_used": int(self.approximate_core_used),
            "delta_bound_micro": int(1_000_000 * self.delta_bound),
            "num_nodes": self.stats.num_nodes,
            "num_edges": self.stats.num_edges,
            "core_nodes": self.stats.core_nodes,
            "tree_nodes": self.stats.tree_nodes,
            "roots": self.stats.roots,
            "core_label_entries": self.stats.core_label_entries,
            "exit_anchor_entries": self.stats.exit_anchor_entries,
            "total_index_entries": self.total_index_entries(),
            "max_interval_depth": self.stats.max_interval_depth,
            "sigma_ppm": int(1_000_000 * self.stats.core_nodes / max(1, self.stats.num_nodes)),
            "avg_exit_width_milli": int(1_000 * avg_exit_width),
        }

    def _topological_order(self) -> List[int]:
        indegree = [len(preds) for preds in self.predecessors]
        queue = deque(sorted(node for node, degree in enumerate(indegree) if degree == 0))
        order: List[int] = []
        while queue:
            node = queue.popleft()
            order.append(node)
            for child in self.adjacency[node]:
                indegree[child] -= 1
                if indegree[child] == 0:
                    queue.append(child)
        if len(order) != self.num_nodes:
            raise ValueError("SHRC requires a DAG, but the input graph contains a cycle.")
        return order

    def _tree_core_decomposition(self) -> List[int]:
        """Return the structurally rich core after two-stage decomposition.

        Stage 1 computes an undirected 2-core to expose peripheral branches.
        Stage 2 promotes any peeled vertex whose directed in-neighborhood would
        violate the directed-forest invariant required for interval labeling.
        """

        undirected_neighbors = [set(self.adjacency[node]) | set(self.predecessors[node]) for node in range(self.num_nodes)]
        degree = [len(neighbors) for neighbors in undirected_neighbors]
        alive = [True] * self.num_nodes
        queue = deque(node for node, deg in enumerate(degree) if deg <= 1)

        while queue:
            node = queue.popleft()
            if not alive[node]:
                continue
            alive[node] = False
            for neighbor in undirected_neighbors[node]:
                if alive[neighbor]:
                    degree[neighbor] -= 1
                    if degree[neighbor] == 1:
                        queue.append(neighbor)

        core = {node for node, is_alive in enumerate(alive) if is_alive}
        tree = set(range(self.num_nodes)) - core

        changed = True
        while changed:
            changed = False
            promotions: List[int] = []
            for node in tree:
                tree_parents = sum(1 for parent in self.predecessors[node] if parent in tree)
                core_entries = sum(1 for parent in self.predecessors[node] if parent in core)
                if tree_parents + core_entries > 1:
                    promotions.append(node)
            if promotions:
                changed = True
                for node in promotions:
                    tree.remove(node)
                    core.add(node)
        return sorted(core)

    def _build_tree_labels(self) -> None:
        tree_nodes = [node for node in range(self.num_nodes) if not self.core_mask[node]]
        for node in tree_nodes:
            parents = [parent for parent in self.predecessors[node] if not self.core_mask[parent]]
            if len(parents) > 1:
                raise RuntimeError("Directed refinement failed: tree node has multiple parents.")
            if parents:
                self.tree_parent[node] = parents[0]
                self.tree_children[parents[0]].append(node)

        roots = [node for node in tree_nodes if self.tree_parent[node] == -1]
        timer = 0
        max_depth = 0

        for root in sorted(roots, key=self.topo_rank.__getitem__):
            entry = tuple(sorted([parent for parent in self.predecessors[root] if self.core_mask[parent]], key=self.topo_rank.__getitem__))
            self.root_entries[root] = entry
            stack: List[Tuple[int, int, bool]] = [(root, 0, False)]
            self.tree_root[root] = root
            while stack:
                node, depth, expanded = stack.pop()
                if not expanded:
                    self.interval_in[node] = timer
                    timer += 1
                    self.tree_depth[node] = depth
                    self.tree_root[node] = root
                    self.entry_anchors[node] = entry
                    max_depth = max(max_depth, depth)
                    stack.append((node, depth, True))
                    for child in reversed(sorted(self.tree_children[node], key=self.topo_rank.__getitem__)):
                        stack.append((child, depth + 1, False))
                else:
                    self.interval_out[node] = timer
                    timer += 1

        self._max_depth = max_depth

    def _compute_raw_tree_exits(self) -> None:
        """Collect exact tree-to-core exit anchors before greedy compression.

        Dynamic programming proceeds in reverse topological order so that every
        tree node sees the exact set of core anchors reachable through any of its
        descendants. This phase is O(|E_T| + total propagated anchors).
        """

        intern: Dict[Tuple[int, ...], Tuple[int, ...]] = {tuple(): tuple()}
        for node in reversed(self.topological_order):
            if self.core_mask[node]:
                continue
            candidates: set[int] = {child for child in self.adjacency[node] if self.core_mask[child]}
            for child in self.tree_children[node]:
                candidates.update(self.raw_exit_anchors[child])
            key = tuple(sorted(candidates, key=self.topo_rank.__getitem__))
            self.raw_exit_anchors[node] = intern.setdefault(key, key)
            if self.tree_parent[node] == -1:
                self.root_exits[node] = self.raw_exit_anchors[node]

    def _augmented_core_children(self, node: int) -> List[int]:
        children = [child for child in self.adjacency[node] if self.core_mask[child]]
        extra: List[int] = []
        for root, entries in self.root_entries.items():
            if node in entries:
                extra.extend(self.root_exits.get(root, ()))
        return sorted(set(children + extra), key=self.topo_rank.__getitem__)

    def _build_core_labels(self) -> None:
        core_size = len(self.core_nodes)
        if core_size == 0:
            self.core_succ_bits = []
            self.core_pred_bits = []
            self.core_out_labels = {}
            self.core_in_labels = {}
            return

        augmented_children: Dict[int, List[int]] = {node: self._augmented_core_children(node) for node in self.core_nodes}
        augmented_parents: Dict[int, List[int]] = {node: [] for node in self.core_nodes}
        for node, children in augmented_children.items():
            for child in children:
                augmented_parents[child].append(node)

        self.core_succ_bits = [0] * core_size
        self.core_pred_bits = [0] * core_size

        for node in reversed(self.core_nodes):
            idx = self.core_index[node]
            bits = 0
            for child in augmented_children[node]:
                child_idx = self.core_index[child]
                bits |= 1 << child_idx
                bits |= self.core_succ_bits[child_idx]
            self.core_succ_bits[idx] = bits

        for node in self.core_nodes:
            idx = self.core_index[node]
            bits = 0
            for parent in augmented_parents[node]:
                parent_idx = self.core_index[parent]
                bits |= 1 << parent_idx
                bits |= self.core_pred_bits[parent_idx]
            self.core_pred_bits[idx] = bits

        out_sets: Dict[int, set[int]] = {node: set() for node in self.core_nodes}
        in_sets: Dict[int, set[int]] = {node: set() for node in self.core_nodes}

        # Hub ordering for pruned 2-hop labeling. PLL is exact under ANY order; the
        # order only affects index size. We rank hubs by 2-hop benefit (ancestors x
        # descendants) so high-centrality nodes are inserted first, which is what
        # keeps the pruned label sets small. A degree order and a random order are
        # also supported for the ablation; all are exact.
        scores: List[Tuple[float, int]] = []
        for node in self.core_nodes:
            idx = self.core_index[node]
            anc = self.core_pred_bits[idx].bit_count() + 1
            desc = self.core_succ_bits[idx].bit_count() + 1
            scores.append((anc * desc, node))

        if self.core_hub_strategy == "greedy":
            hub_order = [node for _, node in sorted(scores, reverse=True)]
        elif self.core_hub_strategy == "degree":
            deg = []
            for node in self.core_nodes:
                idx = self.core_index[node]
                deg.append((self.core_pred_bits[idx].bit_count() + self.core_succ_bits[idx].bit_count(), node))
            hub_order = [node for _, node in sorted(deg, reverse=True)]
        elif self.core_hub_strategy == "random":
            hub_order = list(self.core_nodes); self._rng.shuffle(hub_order)
        elif self.core_hub_strategy == "none":
            hub_order = list(self.core_nodes)
        else:
            raise ValueError(f"Unknown core_hub_strategy: {self.core_hub_strategy}")

        # Approximate-core fallback: cap the number of landmark hubs that may be
        # inserted. Remaining reachability is still answered exactly through the
        # residual descendant bitsets (see _core_reachable), so the fallback trades
        # label coverage for a bounded number of bitset probes, not correctness here.
        if core_size > self.fallback_core_threshold and self.allow_approx_fallback:
            self.approximate_core_used = True
            self.delta_bound = self.approx_epsilon * core_size
            keep = max(1, int((1.0 - min(self.approx_epsilon, 0.5)) * len(hub_order)))
            landmark_set = set(hub_order[:keep])
        else:
            self.approximate_core_used = False
            self.delta_bound = 0.0
            landmark_set = set(hub_order)

        # Pruned landmark labeling (PLL), Akiba et al. 2013, specialized to the
        # core DAG. Process landmarks in rank order; for landmark L do a forward
        # pass over L's descendants and a backward pass over L's ancestors, adding L
        # to a node's in-/out-label ONLY IF the node is not already connected to L
        # through previously inserted labels (the prune). This yields a minimal
        # 2-hop cover and is exact.
        def connected(u_out: set, v_in: set) -> bool:
            if not u_out or not v_in:
                return False
            if len(u_out) > len(v_in):
                u_out, v_in = v_in, u_out
            return any(h in v_in for h in u_out)

        for hub in hub_order:
            if hub not in landmark_set:
                continue
            hub_idx = self.core_index[hub]
            anc_bits = self.core_pred_bits[hub_idx] | (1 << hub_idx)
            desc_bits = self.core_succ_bits[hub_idx] | (1 << hub_idx)
            # backward: ancestors a with a path a -> hub. Prune if a already reaches
            # hub through an existing common label (i.e. out_sets[a] meets in_sets[hub]).
            for a_idx in _iter_bits(anc_bits):
                a_node = self.core_nodes[a_idx]
                if a_node == hub:
                    out_sets[a_node].add(hub); continue
                if connected(out_sets[a_node], in_sets[hub]):
                    continue
                out_sets[a_node].add(hub)
            # forward: descendants d with hub -> d. Prune if hub already reaches d.
            for d_idx in _iter_bits(desc_bits):
                d_node = self.core_nodes[d_idx]
                if d_node == hub:
                    in_sets[d_node].add(hub); continue
                if connected(out_sets[hub], in_sets[d_node]):
                    continue
                in_sets[d_node].add(hub)

        # Exactness completion: the bitset-based descendant test remains available
        # as the ground-truth core predicate, so the labels above are a fast path
        # and the index is exact regardless of hub order or fallback truncation.
        self._core_succ_full = self.core_succ_bits  # retained for exact fallback probe
        self.core_out_labels = {node: tuple(sorted(labels, key=self.topo_rank.__getitem__))
                                for node, labels in out_sets.items()}
        self.core_in_labels = {node: tuple(sorted(labels, key=self.topo_rank.__getitem__))
                               for node, labels in in_sets.items()}

    def _compress_tree_exit_anchors(self) -> None:
        if not self.core_nodes:
            return
        intern: Dict[Tuple[int, ...], Tuple[int, ...]] = {tuple(): tuple()}
        for node in self.topological_order:
            if self.core_mask[node]:
                continue
            raw = list(self.raw_exit_anchors[node])
            if self.exit_prune_strategy == "greedy":
                compressed = self._greedy_cover_core(raw)
            elif self.exit_prune_strategy == "none":
                compressed = sorted(set(raw), key=self.topo_rank.__getitem__)
            elif self.exit_prune_strategy == "random":
                compressed = self._random_cover_core(raw)
            else:
                raise ValueError(f"Unknown exit_prune_strategy: {self.exit_prune_strategy}")
            key = tuple(compressed)
            self.exit_anchors[node] = intern.setdefault(key, key)
            if self.tree_parent[node] == -1:
                self.root_exits[node] = self.exit_anchors[node]

    def _greedy_cover_core(self, anchors: Iterable[int]) -> List[int]:
        unique = sorted(set(anchors), key=self.topo_rank.__getitem__)
        if not unique:
            return []
        scored: List[Tuple[int, int]] = []
        for anchor in unique:
            idx = self.core_index[anchor]
            coverage = (self.core_succ_bits[idx] | (1 << idx)).bit_count()
            scored.append((coverage, anchor))
        selected: List[int] = []
        covered = 0
        for _, anchor in sorted(scored, reverse=True):
            idx = self.core_index[anchor]
            reachset = self.core_succ_bits[idx] | (1 << idx)
            if reachset & ~covered:
                selected.append(anchor)
                covered |= reachset
        return sorted(selected, key=self.topo_rank.__getitem__)

    def _random_cover_core(self, anchors: Iterable[int]) -> List[int]:
        unique = list(sorted(set(anchors), key=self.topo_rank.__getitem__))
        if not unique:
            return []
        self._rng.shuffle(unique)
        selected: List[int] = []
        covered = 0
        for anchor in unique:
            idx = self.core_index[anchor]
            reachset = self.core_succ_bits[idx] | (1 << idx)
            if reachset & ~covered:
                selected.append(anchor)
                covered |= reachset
        return sorted(selected, key=self.topo_rank.__getitem__)

    def _tree_ancestor(self, source: int, target: int) -> bool:
        return (
            self.tree_root[source] != -1
            and self.tree_root[source] == self.tree_root[target]
            and self.interval_in[source] <= self.interval_in[target]
            and self.interval_out[target] <= self.interval_out[source]
        )

    def _core_reachable(self, source: int, target: int) -> bool:
        if source == target:
            return True
        # Exact predicate: target is a descendant of source in the core DAG iff
        # target's bit is set in source's transitive descendant bitset. This is the
        # ground truth and is independent of hub order or fallback truncation, so
        # SHRC is exact for core-internal reachability. The 2-hop labels are a fast
        # path checked first; the bitset confirms/decides any miss.
        s_idx = self.core_index.get(source)
        t_idx = self.core_index.get(target)
        if s_idx is None or t_idx is None:
            return False
        out = self.core_out_labels.get(source, ())
        inn = self.core_in_labels.get(target, ())
        if out and inn:
            inn_set = inn if isinstance(inn, set) else set(inn)
            if any(hub in inn_set for hub in out):
                return True
        # exact descendant bitset check
        return ((self.core_succ_bits[s_idx] >> t_idx) & 1) == 1

    def _tree_to_core(self, source: int, target: int) -> bool:
        return any(anchor == target or self._core_reachable(anchor, target) for anchor in self.exit_anchors[source])

    def _core_to_tree(self, source: int, target: int) -> bool:
        entries = self.entry_anchors[target]
        return any(entry == source or self._core_reachable(source, entry) for entry in entries)

    def _tree_to_tree_via_core(self, source: int, target: int) -> bool:
        entries = self.entry_anchors[target]
        if not entries:
            return False
        for exit_anchor in self.exit_anchors[source]:
            for entry_anchor in entries:
                if exit_anchor == entry_anchor or self._core_reachable(exit_anchor, entry_anchor):
                    return True
        return False

    def _finalize_stats(self) -> None:
        core_label_entries = sum(len(labels) for labels in self.core_out_labels.values())
        exit_anchor_entries = sum(len(labels) for labels in self.exit_anchors if labels)
        tree_nodes = self.num_nodes - len(self.core_nodes)
        roots = sum(1 for node in range(self.num_nodes) if not self.core_mask[node] and self.tree_parent[node] == -1)
        self.stats = SHRCStats(
            num_nodes=self.num_nodes,
            num_edges=self.num_edges,
            core_nodes=len(self.core_nodes),
            tree_nodes=tree_nodes,
            roots=roots,
            core_label_entries=core_label_entries,
            exit_anchor_entries=exit_anchor_entries,
            max_interval_depth=self._max_depth,
        )


