"""Train the Multi-modal GIN on synthetic graphs.

This script intentionally stays lightweight. Its main purpose is to demonstrate
that the artifact is executable end to end without depending on private data.
"""

from __future__ import annotations

import argparse

import torch
from torch import nn
from torch_geometric.loader import DataLoader

from agentflow_ppi.data.synthetic_ppi import SyntheticConfig, SyntheticPPIGenerator
from agentflow_ppi.models.multimodal_gin import MultiModalGIN, MultiModalGINConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-graphs", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=16)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    synth = SyntheticPPIGenerator(SyntheticConfig())
    dataset = synth.make_dataset(args.num_graphs)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    model = MultiModalGIN(MultiModalGINConfig())
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        for batch in loader:
            optimizer.zero_grad()
            logits = model(batch)
            loss = criterion(logits, batch.y)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item())
        avg_loss = total_loss / max(1, len(loader))
        print(f"epoch={epoch + 1} loss={avg_loss:.4f}")


if __name__ == "__main__":
    main()


