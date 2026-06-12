"""Run the Apple-Silicon-aware asynchronous research automation demo."""

from __future__ import annotations

import argparse
import asyncio
import json

from agentflow_ppi.rdflow import RDFlowConfig, RDFlowCoordinator


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--request",
        type=str,
        default="Design proofs, code, and experiments for an MPS-accelerated learned-operator graph system.",
        help="Top-level R&D request sent to the agent coordinator.",
    )
    return parser.parse_args()


async def _main() -> None:
    args = parse_args()
    coordinator = RDFlowCoordinator(RDFlowConfig())
    result = await coordinator.execute(args.request)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    asyncio.run(_main())


