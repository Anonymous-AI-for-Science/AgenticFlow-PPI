"""Multi-modal GIN model for protein interaction candidate scoring.

The model fuses graph-native, structure-aware, and sequence-aware feature groups
before applying GIN layers. The implementation targets PyTorch Geometric.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch
from torch import Tensor, nn
import torch.nn.functional as F
from torch_geometric.nn import GINConv, global_mean_pool


@dataclass(slots=True)
class MultiModalGINConfig:
    graph_dim: int = 16
    sequence_dim: int = 24
    structure_dim: int = 12
    hidden_dim: int = 64
    gin_layers: int = 3
    dropout: float = 0.2
    num_classes: int = 2


class MLPBlock(nn.Module):
    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.ReLU(),
            nn.Linear(out_dim, out_dim),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class MultiModalGIN(nn.Module):
    """Selective neural operator used by AgentFlow-PPI.

    Expected input feature layout:
    - graph features: node degrees, modality counts, confidence statistics
    - sequence features: k-mer or embedding-based descriptors
    - structure features: pocket, secondary-structure, or contact descriptors
    """

    def __init__(self, config: MultiModalGINConfig | None = None) -> None:
        super().__init__()
        self.config = config or MultiModalGINConfig()
        input_dim = self.config.graph_dim + self.config.sequence_dim + self.config.structure_dim
        self.fusion = nn.Sequential(
            nn.Linear(input_dim, self.config.hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.config.dropout),
        )

        self.gin_layers = nn.ModuleList()
        for _ in range(self.config.gin_layers):
            mlp = MLPBlock(self.config.hidden_dim, self.config.hidden_dim)
            self.gin_layers.append(GINConv(mlp))

        self.readout = nn.Sequential(
            nn.Linear(self.config.hidden_dim, self.config.hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.config.dropout),
            nn.Linear(self.config.hidden_dim, self.config.num_classes),
        )

    def split_modalities(self, x: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        g_end = self.config.graph_dim
        s_end = g_end + self.config.sequence_dim
        graph_x = x[:, :g_end]
        seq_x = x[:, g_end:s_end]
        struct_x = x[:, s_end:]
        return graph_x, seq_x, struct_x

    def forward(self, data) -> Tensor:
        x, edge_index, batch = data.x, data.edge_index, data.batch
        graph_x, seq_x, struct_x = self.split_modalities(x)
        fused = torch.cat([graph_x, seq_x, struct_x], dim=-1)
        h = self.fusion(fused)
        for layer in self.gin_layers:
            h = layer(h, edge_index)
            h = F.relu(h)
            h = F.dropout(h, p=self.config.dropout, training=self.training)
        pooled = global_mean_pool(h, batch)
        logits = self.readout(pooled)
        return logits

    @torch.no_grad()
    def score_candidates(self, data) -> Tensor:
        self.eval()
        logits = self.forward(data)
        return torch.softmax(logits, dim=-1)[:, 1]


