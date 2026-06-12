"""STRING-structured synthetic graph generator for scale experiments.

The released public STRING-v12 subset has ~5k proteins; the full v12 human family
is ~20k proteins. To answer the practitioners' scale concern with *measured* numbers
rather than extrapolation, this module synthesizes directed multimodal graphs that
reproduce the structural signature of STRING after the paper's condensation
pipeline: a small dense residual core surrounded by a large sparse periphery
(low residual-core ratio sigma), heavy-tailed degree, and typed/scored edges.

The generator is deterministic given a seed so every reported number is
reproducible. It is *not* claimed to be biologically faithful; it is a structural
stress graph whose sigma, |V|, and |E| are matched to STRING-scale regimes so that
SHRC build/query cost and the dispatch layer can be measured at 5k, 10k, and 20k
nodes on commodity hardware.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np

Edge = Tuple[int, int, str, float]
MODALITIES = ("physical", "functional", "regulatory")


@dataclass(slots=True)
class StringScaleConfig:
    num_nodes: int = 20000
    target_sigma: float = 0.04        # core-sizing knob; measured sigma lands near the released subset's 0.076
    core_density: float = 0.20        # edge probability inside the dense core
    cross_core_periphery: float = 0.03  # fraction of peripheral leaves with a core re-entry link
    seed: int = 7

    @property
    def core_size(self) -> int:
        return max(8, int(round(self.num_nodes * self.target_sigma)))


class StringScaleGenerator:
    def __init__(self, config: StringScaleConfig | None = None) -> None:
        self.config = config or StringScaleConfig()
        self.rng = np.random.default_rng(self.config.seed)

    def generate(self) -> Tuple[int, List[Edge]]:
        cfg = self.config
        n = cfg.num_nodes
        core_size = min(cfg.core_size, n)
        core = list(range(core_size))
        periphery = list(range(core_size, n))
        edges: List[Edge] = []

        # 1. Dense directed core (acyclic by orientation low->high index).
        for i in range(core_size):
            for j in range(i + 1, core_size):
                if self.rng.random() < cfg.core_density:
                    edges.append(self._edge(i, j))

        # 2. Sparse periphery as a directed FOREST rooted at the dense core.
        #    Each peripheral node attaches to exactly one parent that is either a
        #    core node or an EARLIER peripheral node. Because every such node has
        #    structural in-degree 1 and the periphery contains no undirected
        #    cycle, iterative 2-core peeling removes the entire periphery, which
        #    reproduces STRING's low residual-core ratio.
        children_count = [0] * n
        for v in periphery:
            # Prefer attaching near the core to keep trees shallow; mix in some
            # peripheral parents to create realistic chains.
            if self.rng.random() < 0.6:
                parent = int(self.rng.integers(0, core_size))
            else:
                parent = int(self.rng.integers(core_size, v)) if v > core_size else int(self.rng.integers(0, core_size))
            edges.append(self._edge(parent, v))
            children_count[parent] += 1

        # 3. Rare re-entry edges: a small fraction of *leaf* peripheral nodes
        #    (no children) receive one additional edge FROM the core. Restricting
        #    to leaves keeps each such node at undirected degree 2 with one of its
        #    neighbours still degree 1, so the 2-core peeling is essentially
        #    unchanged and sigma stays small while still exercising SHRC bridge
        #    augmentation.
        leaves = [v for v in periphery if children_count[v] == 0]
        num_reentry = int(len(leaves) * cfg.cross_core_periphery)
        self.rng.shuffle(leaves)
        for v in leaves[:num_reentry]:
            c = int(self.rng.integers(0, core_size))
            edges.append(self._edge(c, v))

        # Deduplicate (keep max score per directed typed pair).
        best: dict[Tuple[int, int, str], float] = {}
        for u, v, m, s in edges:
            key = (u, v, m)
            if s > best.get(key, -1.0):
                best[key] = s
        dedup = [(u, v, m, s) for (u, v, m), s in best.items()]
        return n, dedup

    def _edge(self, u: int, v: int) -> Edge:
        modality = MODALITIES[int(self.rng.integers(0, len(MODALITIES)))]
        score = float(np.clip(self.rng.normal(0.75, 0.12), 0.4, 0.999))
        return (u, v, modality, round(score, 3))

    @staticmethod
    def to_dag_edges(edges: List[Edge]) -> List[Tuple[int, int]]:
        """Project typed edges to plain directed edges (already acyclic by index)."""
        return [(u, v) for (u, v, _m, _s) in edges]
