"""Scalable curated-pathway benchmark generator (reviewer W2/W3/W8/W12).

The 19-node hand-built graph was too small to support any generalizable claim, and
on it the learned reranker never helped, so the calibrated dispatcher never had a
reason to admit it. This generator builds an arbitrarily large directed signaling
network with curated linear pathways, realistic off-pathway cross-talk distractors,
and -- critically -- a CONTROLLED modality structure that makes the modality feature
genuinely discriminative on a tunable fraction of queries. That produces a
MIXED-SIGN workload: reranking helps on the modality-informative queries and does
not help (or hurts) on the rest, so a calibrated dispatcher must learn to ADMIT on
the former and DECLINE on the latter.

Construction (fully deterministic given a seed):
  * ``num_pathways`` linear cascades, each of length ``pathway_len``. Pathway p has
    nodes P{p}_0 -> P{p}_1 -> ... in curated order; consecutive nodes are connected
    by a high-confidence on-pathway edge.
  * Each on-pathway edge is assigned the pathway's "native" modality.
  * For an ``informative_fraction`` of pathways, the gold mediators additionally
    carry a distinctive outgoing-modality signature, so a modality-aware reranker
    can separate them from cross-talk distractors that share path-score but not
    modality. On the remaining pathways the modality signal is uninformative
    (gold and distractors look identical to the modality feature), so symbolic
    path-score is already sufficient and reranking cannot help.
  * Cross-talk: each node gets a few off-pathway edges to random other-pathway
    nodes with confidence comparable to on-pathway edges (these are the decoys that
    inflate the candidate frontier and that exact reachability + ranking must
    survive).

Gold labels remain position-on-cascade (pathway membership + order), independent of
edge confidence and modality counts, so the evaluation is not circular.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

MODALITIES = ["physical", "functional", "regulatory", "predicted"]


@dataclass
class PathwayBenchmarkConfig:
    num_pathways: int = 40
    pathway_len: int = 8
    informative_fraction: float = 0.5  # fraction of pathways where modality discriminates
    crosstalk_per_node: int = 3
    on_pathway_conf: Tuple[float, float] = (0.70, 0.95)
    crosstalk_conf: Tuple[float, float] = (0.65, 0.92)
    seed: int = 7


@dataclass
class PathwayBenchmark:
    edges: List[Tuple[str, str, str, float]]  # (src, dst, modality, score)
    pathway_order: Dict[str, List[str]]
    informative_pathways: set
    native_modality: Dict[str, str]


def build_pathway_benchmark(cfg: PathwayBenchmarkConfig) -> PathwayBenchmark:
    rng = random.Random(cfg.seed)
    pathway_order: Dict[str, List[str]] = {}
    native_modality: Dict[str, str] = {}
    informative: set = set()
    nodes_by_pathway: Dict[str, List[str]] = {}

    for p in range(cfg.num_pathways):
        name = f"PW{p:02d}"
        nodes = [f"P{p:02d}_{i}" for i in range(cfg.pathway_len)]
        pathway_order[name] = nodes
        nodes_by_pathway[name] = nodes
        native_modality[name] = MODALITIES[p % len(MODALITIES)]
        if rng.random() < cfg.informative_fraction:
            informative.add(name)

    edges: List[Tuple[str, str, str, float]] = []
    all_nodes = [n for ns in nodes_by_pathway.values() for n in ns]
    gold_nodes_informative: set = set()
    for name in informative:
        order = nodes_by_pathway[name]
        for i in range(1, len(order) - 1):
            gold_nodes_informative.add(order[i])

    # On-pathway edges (curated cascade).
    for name, nodes in nodes_by_pathway.items():
        mod = native_modality[name]
        inf = name in informative
        for i in range(len(nodes) - 1):
            lo, hi = cfg.on_pathway_conf
            if inf:
                # Informative: gold on-pathway edges carry native modality but
                # MODEST confidence, so score-only ranking is misled.
                conf = round(rng.uniform(0.55, 0.68), 3)
            else:
                # Non-informative: gold edges are HIGH confidence, so the symbolic
                # path-score ranker already places them correctly and reranking
                # cannot improve on it.
                conf = round(rng.uniform(0.85, 0.97), 3)
            edges.append((nodes[i], nodes[i + 1], mod, conf))
            # gold intermediates emit several extra native-modality edges so their
            # modality-agreement is dominant and distinctive.
            if inf and 0 < i < len(nodes) - 1:
                for j in range(i + 2, min(i + 4, len(nodes))):
                    edges.append((nodes[i], nodes[j], mod, round(rng.uniform(0.55, 0.68), 3)))

    # Cross-talk distractors.
    for name, nodes in nodes_by_pathway.items():
        native = native_modality[name]
        inf = name in informative
        for u in nodes[:-1]:
            # Only on INFORMATIVE pathways are gold mediators kept modality-pure
            # (no outgoing cross-talk), making the modality feature discriminative
            # there. On non-informative pathways gold nodes get the same cross-talk
            # as everyone else, so modality is uninformative and reranking cannot help.
            if inf and u in gold_nodes_informative:
                continue
            for _ in range(cfg.crosstalk_per_node):
                v = rng.choice(all_nodes)
                if v == u or (inf and v in gold_nodes_informative):
                    continue
                dm = rng.choice([m for m in MODALITIES if m != native])
                if inf:
                    s = round(rng.uniform(0.80, 0.95), 3)
                else:
                    lo, hi = cfg.crosstalk_conf
                    s = round(rng.uniform(lo, hi), 3)
                edges.append((u, v, dm, s))

    # Deduplicate keeping max score per (u,v,modality)
    best: Dict[Tuple[str, str, str], float] = {}
    for u, v, m, s in edges:
        k = (u, v, m)
        if s > best.get(k, 0.0):
            best[k] = s
    edges = [(u, v, m, s) for (u, v, m), s in best.items()]

    return PathwayBenchmark(edges, pathway_order, informative, native_modality)


def gold_mediators(bench: PathwayBenchmark, source: str, target: str,
                   candidates) -> set:
    """Position-on-cascade gold labels (independent of edge attributes)."""
    gold = set()
    for name, order in bench.pathway_order.items():
        if source in order and target in order:
            rs, rt = order.index(source), order.index(target)
            if rs < rt:
                for c in candidates:
                    if c in order and rs < order.index(c) < rt:
                        gold.add(c)
    return gold


def is_grounded(bench: PathwayBenchmark, source: str, target: str) -> bool:
    for order in bench.pathway_order.values():
        if source in order and target in order and order.index(source) < order.index(target):
            return True
    return False


def query_is_informative(bench: PathwayBenchmark, source: str, target: str) -> bool:
    """True iff the (source,target) pathway is one where modality discriminates."""
    for name, order in bench.pathway_order.items():
        if source in order and target in order and order.index(source) < order.index(target):
            return name in bench.informative_pathways
    return False


def enumerate_queries(bench: PathwayBenchmark, max_span: int = 6) -> List[Tuple[str, str]]:
    """All (source,target) pairs on a common pathway with >=1 intermediate."""
    qs = []
    for order in bench.pathway_order.values():
        for i in range(len(order)):
            for j in range(i + 2, min(i + 1 + max_span, len(order))):
                qs.append((order[i], order[j]))
    return qs
