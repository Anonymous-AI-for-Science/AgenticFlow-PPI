"""Run an end-to-end AgentFlow-PPI query."""

from __future__ import annotations

import argparse
import json

from agentflow_ppi.pipeline.executor import AgentFlowPipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", type=str, required=True)
    parser.add_argument("--data-root", type=str, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pipeline = AgentFlowPipeline()
    result = pipeline.execute(query=args.query, data_root=args.data_root)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()


