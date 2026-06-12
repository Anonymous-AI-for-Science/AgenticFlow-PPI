"""Production graph-engine baseline benchmark.

Materializes one canonical export from the external manifest (downloaded data or
bundled fixture), then runs the in-process SHRC engine plus any available external
engines (Neo4j, PostgreSQL, TigerGraph) over the SAME snapshot, measuring load
time, memory, cold/warm latency, and timeouts, and verifying each engine's answers
against the exact BFS oracle.

Usage:
    # in-process only (no servers needed):
    python scripts/benchmark_engine_baselines.py

    # with a local Postgres and/or Neo4j (see engine docstrings for docker commands):
    export AGENTFLOW_PG_DSN=postgresql://postgres:postgres@localhost:5432/postgres
    export AGENTFLOW_NEO4J_URI=bolt://localhost:7687 AGENTFLOW_NEO4J_PASSWORD=password123
    python scripts/benchmark_engine_baselines.py
"""

from __future__ import annotations

import json
import csv
from pathlib import Path

from agentflow_ppi.data.external.download import DEFAULT_CACHE
from agentflow_ppi.data.external.manifest import build_manifest
from agentflow_ppi.engines.canonical_export import export_from_manifest
from agentflow_ppi.engines.harness import benchmark_all
from agentflow_ppi.engines.base import InProcessSHRCEngine
from agentflow_ppi.engines.sqlite_engine import SQLiteEngine
from agentflow_ppi.engines.postgres_engine import PostgresEngine
from agentflow_ppi.engines.neo4j_engine import Neo4jEngine
from agentflow_ppi.engines.tigergraph_engine import TigerGraphEngine


def main():
    root = Path(__file__).resolve().parents[1]
    out = root / "results"; out.mkdir(parents=True, exist_ok=True)
    export_dir = out / "canonical_export"

    manifest = build_manifest(cache_dir=DEFAULT_CACHE, allow_online=False)
    using_fixture = any(v == "fixture" for v in manifest.provenance.values())
    exp = export_from_manifest(manifest, export_dir)
    print(f"canonical export: {export_dir} "
          f"({json.loads(exp.meta_json.read_text())['num_nodes']} nodes, "
          f"{json.loads(exp.meta_json.read_text())['num_edges']} edges)")

    engines = [InProcessSHRCEngine(), SQLiteEngine(), PostgresEngine(), Neo4jEngine(), TigerGraphEngine()]
    results = benchmark_all(export_dir, engines)
    results["using_fixture"] = using_fixture
    results["provenance"] = manifest.provenance

    (out / "engine_baselines.json").write_text(json.dumps(results, indent=2))
    # flat CSV for the paper table
    with (out / "engine_baselines.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["engine", "available", "load_seconds", "peak_mem_mb",
                    "mean_cold_ms", "mean_warm_ms", "timed_out", "answers_match", "note"])
        for e in results["engines"]:
            eq = e.get("answer_equivalence", {})
            w.writerow([e["engine"], e["available"], e.get("load_seconds", ""),
                        e.get("peak_mem_mb", ""), e.get("mean_cold_ms", ""),
                        e.get("mean_warm_ms", ""), e.get("timed_out", ""),
                        eq.get("all_match", ""), e.get("note", "")])

    print(json.dumps(results, indent=2))
    if using_fixture:
        print("\n[NOTE] Canonical export built from FIXTURES (bio hosts unreachable). "
              "External engines are skipped unless a local server is configured; the "
              "in-process SHRC engine runs and is verified against the BFS oracle.")


if __name__ == "__main__":
    main()
