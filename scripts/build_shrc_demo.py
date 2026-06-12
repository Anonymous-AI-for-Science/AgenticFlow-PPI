"""Build and inspect the SHRC reachability index on a toy DAG."""

from __future__ import annotations

from agentflow_ppi.reachability import SHRCIndex


def main() -> None:
    num_nodes = 10
    edges = [
        (0, 1), (1, 2), (2, 3), (1, 4), (4, 5),
        (2, 6), (5, 7), (6, 7), (7, 8), (8, 9),
    ]
    index = SHRCIndex.from_edges(num_nodes, edges).build()
    print("SHRC summary:", index.summary())
    for source, target in [(1, 8), (4, 9), (3, 7), (0, 5)]:
        trace = index.explain(source, target)
        print(f"{source} -> {target}: {trace.reachable} via {trace.route}")


if __name__ == "__main__":
    main()


