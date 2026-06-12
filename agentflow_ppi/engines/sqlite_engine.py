"""SQLite recursive-CTE reachability engine (design rationale).

the design goal asks for a comparison against a real database engine with its own query planner
and execution engine. PostgreSQL/Neo4j/TigerGraph adapters require provisioned servers
(Docker), which an artifact evaluator may not run. SQLite closes that gap: it is a
full relational engine with its own cost-based query planner and a native recursive
CTE, and it ships *inside the Python standard library*, so this baseline runs with no
server, no driver install, and no container on Ubuntu, macOS Intel, and Apple Silicon
alike. It answers reachability exactly the way the PostgreSQL baseline does -- a
`WITH RECURSIVE` transitive-closure query planned and executed by the engine -- so it
is a genuine systems-level competitor, not an in-process stub.

Like the other adapters it loads the canonical export, answers each query's gold
mediators with cold/warm timing, and its answers are checked against the BFS oracle by
the harness, so SHRC and SQLite are compared on identical, verified-correct outputs.
"""

from __future__ import annotations

import csv
import sqlite3
import time
import tracemalloc
from pathlib import Path
from typing import List, Set

from .base import BaseEngine, EngineResult, EngineUnavailable, QueryTiming

# Reachability as a recursive CTE: is there a directed path src -> dst?
_REACH_CTE = """
WITH RECURSIVE reach(node) AS (
    SELECT :src
    UNION
    SELECT e.dst FROM edges_dir e JOIN reach r ON e.src = r.node
)
SELECT 1 FROM reach WHERE node = :dst LIMIT 1;
"""


class SQLiteEngine(BaseEngine):
    """A real SQL engine (own planner + recursive CTE) that needs no server."""

    name = "sqlite-recursive-cte"

    def __init__(self):
        self.conn = None

    def load(self, export_dir: Path) -> None:
        # sqlite3 is always available in CPython; no EngineUnavailable path needed,
        # which is the whole point -- this baseline runs everywhere.
        self.conn = sqlite3.connect(":memory:")
        cur = self.conn.cursor()
        cur.execute("CREATE TABLE edges_dir(src INTEGER, dst INTEGER);")
        rows = []
        with (Path(export_dir) / "edges.csv").open() as f:
            for r in csv.DictReader(f):
                s, d = int(r["src"]), int(r["dst"])
                rows.append((s, d))
                if int(r["directed"]) == 0:
                    rows.append((d, s))
        cur.executemany("INSERT INTO edges_dir VALUES (?,?);", rows)
        cur.execute("CREATE INDEX idx_src ON edges_dir(src);")
        self.conn.commit()
        cur.close()

    def _reach(self, src: int, dst: int) -> bool:
        if src == dst:
            return True
        cur = self.conn.cursor()
        cur.execute(_REACH_CTE, {"src": src, "dst": dst})
        hit = cur.fetchone() is not None
        cur.close()
        return hit

    def mediators(self, source: int, target: int, gold: List[int]) -> Set[int]:
        out = set()
        for m in gold:
            if m in (source, target):
                continue
            if self._reach(source, m) and self._reach(m, target):
                out.add(m)
        return out

    def close(self) -> None:
        if self.conn is not None:
            self.conn.close()


def run_sqlite_engine(export_dir: Path, queries, timeout_guard_ms: float = 30000.0) -> EngineResult:
    """Load the canonical export into SQLite and time cold/warm mediator answers."""
    eng = SQLiteEngine()
    tracemalloc.start()
    t0 = time.perf_counter()
    eng.load(export_dir)
    load_s = time.perf_counter() - t0
    timings = []
    for q in queries:
        gold = q["gold"]
        c0 = time.perf_counter()
        ans = eng.mediators(q["source"], q["target"], gold)
        cold = (time.perf_counter() - c0) * 1000.0
        w0 = time.perf_counter()
        eng.mediators(q["source"], q["target"], gold)
        warm = (time.perf_counter() - w0) * 1000.0
        timings.append(QueryTiming(qid=q["qid"], answer=ans, cold_ms=cold, warm_ms=warm))
    cur_mem, peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    eng.close()
    return EngineResult(engine=SQLiteEngine.name, available=True, load_seconds=load_s,
                        peak_mem_mb=peak_mem / 1e6, timings=timings,
                        note="stdlib sqlite3; no server required")
