"""Synthetic PPI graph generator used in the paper's experiments.

The generator deliberately exposes parameters that mimic difficult database
workloads: hub skew, typed edges, and confidence-score heterogeneity.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np
import torch
from torch_geometric.data import Data


@dataclass(slots=True)
class SyntheticConfig:
    num_nodes: int = 64
    graph_dim: int = 16
    sequence_dim: int = 24
    structure_dim: int = 12
    edge_prob: float = 0.07
    hub_strength: float = 0.25
    seed: int = 7


class SyntheticPPIGenerator:
    def __init__(self, config: SyntheticConfig | None = None) -> None:
        self.config = config or SyntheticConfig()
        self.rng = np.random.default_rng(self.config.seed)

    def make_graph(self) -> Data:
        num_nodes = self.config.num_nodes
        total_dim = self.config.graph_dim + self.config.sequence_dim + self.config.structure_dim
        x = self.rng.normal(size=(num_nodes, total_dim)).astype(np.float32)

        edges: List[List[int]] = [[], []]
        hub_nodes = max(1, int(num_nodes * self.config.hub_strength))
        for src in range(num_nodes):
            for dst in range(num_nodes):
                if src == dst:
                    continue
                base = self.config.edge_prob
                if src < hub_nodes or dst < hub_nodes:
                    base *= 2.5
                if self.rng.random() < base:
                    edges[0].append(src)
                    edges[1].append(dst)

        if not edges[0]:
            edges = [[0, 1], [1, 0]]

        edge_index = torch.tensor(edges, dtype=torch.long)
        batch = torch.zeros(num_nodes, dtype=torch.long)
        y = torch.tensor([self.rng.integers(0, 2)], dtype=torch.long)
        return Data(x=torch.tensor(x), edge_index=edge_index, batch=batch, y=y)

    def make_dataset(self, num_graphs: int) -> List[Data]:
        return [self.make_graph() for _ in range(num_graphs)]


