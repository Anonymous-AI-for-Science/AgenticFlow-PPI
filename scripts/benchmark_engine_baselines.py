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
import sys
import time
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


# --------------------------------------------------------------------------- #
# Lightweight progress (tqdm if available, stdlib fallback). All to stderr so
# the JSON on stdout stays clean. The harness calls a callback with per-engine
# stage strings (load -> queries -> verify -> done); we render a per-engine bar
# whose postfix names the active sub-stage.
# --------------------------------------------------------------------------- #
try:
    from tqdm import tqdm as _tqdm  # type: ignore
    _HAS_TQDM = True

    def _mkbar(total, desc):
        return _tqdm(total=total, desc=desc, unit="q", dynamic_ncols=True,
                     file=sys.stderr, leave=True)

    def _stage(bar, text):
        try:
            bar.set_postfix_str(text)
        except Exception:  # noqa: BLE001
            pass
except Exception:  # noqa: BLE001
    _HAS_TQDM = False

    class _Bar:
        def __init__(self, total, desc, width=28):
            self.total = total if (total and total > 0) else None
            self.desc = desc; self.width = width; self.n = 0; self.t0 = time.time(); self.s = ""
            self._r()

        def _r(self):
            el = time.time() - self.t0; tail = f"  <{self.s}>" if self.s else ""
            if self.total:
                fr = min(1.0, self.n / self.total); fl = int(self.width * fr)
                b = "█" * fl + "░" * (self.width - fl)
                msg = f"\r  {self.desc:<22} |{b}| {self.n}/{self.total} ({fr*100:4.0f}%) [{el:4.1f}s]{tail}"
            else:
                msg = f"\r  {self.desc:<22} {'|/-\\'[self.n % 4]} {self.n} [{el:4.1f}s]{tail}"
            print(msg, end="", file=sys.stderr, flush=True)

        def set_postfix_str(self, t): self.s = t; self._r()
        def update(self, k=1): self.n += k; self._r()
        def close(self): self.s = ""; self._r(); print("", file=sys.stderr, flush=True)

    def _mkbar(total, desc):
        return _Bar(total, desc)

    def _stage(bar, text):
        bar.set_postfix_str(text)


def _phase(text):
    print(f"\033[1m▶ {text}\033[0m", file=sys.stderr, flush=True)


def _make_progress_cb():
    """Return a callback the harness drives with per-engine stage events,
    rendering a distinct sub-stage label and a per-engine query bar."""
    state = {"bar": None, "total": 0, "engine": None}

    def cb(name, stage, **kw):
        if stage == "oracle:start":
            _phase("building BFS oracle (correctness reference)")
            state["bar"] = _mkbar(kw.get("total", 0), "oracle BFS")
            _stage(state["bar"], "per-query reachability")
        elif stage == "oracle:tick":
            if state["bar"] is not None:
                state["bar"].update(1)
        elif stage == "oracle:done":
            if state["bar"] is not None:
                state["bar"].close(); state["bar"] = None
            print(f"    oracle ready over {kw.get('num_queries', '?')} queries",
                  file=sys.stderr, flush=True)
        elif stage == "engine:start":
            print(f"\n\033[1m● engine {kw.get('index')}/{kw.get('total')}: {name}\033[0m",
                  file=sys.stderr, flush=True)
            state["engine"] = name
        elif stage == "unavailable":
            if state["bar"] is not None:
                state["bar"].close(); state["bar"] = None
            print(f"    [skip] {name} unavailable: {kw.get('note','')}", file=sys.stderr, flush=True)
        elif stage == "load-error":
            if state["bar"] is not None:
                state["bar"].close(); state["bar"] = None
            print(f"    [error] {name} load failed: {kw.get('note','')}", file=sys.stderr, flush=True)
        elif stage == "load:start":
            state["bar"] = _mkbar(None, f"{name}")
            _stage(state["bar"], "1/4 loading snapshot")
        elif stage == "load:done":
            if state["bar"] is not None:
                _stage(state["bar"], f"load {kw.get('load_seconds')}s done")
        elif stage == "queries:start":
            if state["bar"] is not None:
                state["bar"].close()
            state["total"] = kw.get("total", 0)
            state["bar"] = _mkbar(state["total"], f"{name}")
            _stage(state["bar"], "2/4 cold+warm queries")
        elif stage == "queries:tick":
            if state["bar"] is not None:
                state["bar"].update(1)
        elif stage == "queries:done":
            if state["bar"] is not None:
                _stage(state["bar"], "3/4 queries done"); state["bar"].close(); state["bar"] = None
        elif stage == "verify:start":
            print("    4/4 verifying answers vs BFS oracle...", file=sys.stderr, flush=True)
        elif stage == "engine:done":
            if kw.get("available"):
                ok = kw.get("all_match")
                mark = "\033[32m✓\033[0m" if ok else "\033[31m✗ MISMATCH\033[0m"
                print(f"    {mark} {name}: warm {kw.get('mean_warm_ms')} ms, "
                      f"oracle-equivalent={ok}", file=sys.stderr, flush=True)
            else:
                print(f"    — {name}: not available ({kw.get('note','')})", file=sys.stderr, flush=True)
    return cb


def main():
    root = Path(__file__).resolve().parents[1]
    out = root / "results"; out.mkdir(parents=True, exist_ok=True)
    export_dir = out / "canonical_export"
    print(f"progress bar backend: {'tqdm' if _HAS_TQDM else 'builtin (pip install tqdm for nicer output)'}",
          file=sys.stderr, flush=True)

    _phase("materializing canonical export from manifest")
    manifest = build_manifest(cache_dir=DEFAULT_CACHE, allow_online=False)
    using_fixture = any(v == "fixture" for v in manifest.provenance.values())
    exp = export_from_manifest(manifest, export_dir)
    meta = json.loads(exp.meta_json.read_text())
    print(f"    canonical export: {export_dir} "
          f"({meta['num_nodes']} nodes, {meta['num_edges']} edges, using_fixture={using_fixture})",
          file=sys.stderr, flush=True)

    engines = [InProcessSHRCEngine(), SQLiteEngine(), PostgresEngine(), Neo4jEngine(), TigerGraphEngine()]
    _phase(f"benchmarking {len(engines)} engine(s) over the same snapshot")
    results = benchmark_all(export_dir, engines, progress_cb=_make_progress_cb())
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

    _phase("done — results written to results/engine_baselines.{json,csv}")
    print(json.dumps(results, indent=2))
    if using_fixture:
        print("\n[NOTE] Canonical export built from FIXTURES (bio hosts unreachable). "
              "External engines are skipped unless a local server is configured; the "
              "in-process SHRC engine runs and is verified against the BFS oracle.")


if __name__ == "__main__":
    main()
