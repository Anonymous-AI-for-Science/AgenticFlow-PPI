"""PostgreSQL recursive-CTE baseline.

Loads the canonical export into a real PostgreSQL instance and answers the mediator
query with a recursive CTE (the standard SQL way to express reachability). The
connection string is taken from $AGENTFLOW_PG_DSN (e.g.
'postgresql://postgres:postgres@localhost:5432/agentflow'); if it is unset or the
server/driver is unreachable, EngineUnavailable is raised so the harness skips it.

To spin up a local server:
    docker run --rm -d --name agentflow-pg -e POSTGRES_PASSWORD=postgres \\
        -p 5432:5432 postgres:16
    export AGENTFLOW_PG_DSN=postgresql://postgres:postgres@localhost:5432/postgres
"""

from __future__ import annotations

import csv
import os
from pathlib import Path
from typing import List, Set

from .base import BaseEngine, EngineUnavailable


# Reachability via recursive CTE: does a directed path source -> node exist?
_REACH_CTE = """
WITH RECURSIVE reach(node) AS (
    SELECT %(src)s
    UNION
    SELECT e.dst FROM edges_dir e JOIN reach r ON e.src = r.node
)
SELECT 1 FROM reach WHERE node = %(dst)s LIMIT 1;
"""


class PostgresEngine(BaseEngine):
    name = "postgres-recursive-cte"

    def __init__(self, dsn: str = None):
        self.dsn = dsn or os.environ.get("AGENTFLOW_PG_DSN")
        self.conn = None

    def load(self, export_dir: Path) -> None:
        if not self.dsn:
            raise EngineUnavailable("AGENTFLOW_PG_DSN not set")
        try:
            import psycopg2  # noqa: F401
        except Exception as e:  # noqa: BLE001
            raise EngineUnavailable(f"psycopg2 not installed: {e}")
        try:
            import psycopg2
            self.conn = psycopg2.connect(self.dsn, connect_timeout=5)
        except Exception as e:  # noqa: BLE001
            raise EngineUnavailable(f"cannot connect to PostgreSQL: {e}")
        cur = self.conn.cursor()
        cur.execute("DROP TABLE IF EXISTS edges_dir;")
        cur.execute("CREATE TABLE edges_dir(src INT, dst INT);")
        # load directed edges (expand undirected to both directions)
        rows = []
        with (Path(export_dir) / "edges.csv").open() as f:
            for r in csv.DictReader(f):
                s, d = int(r["src"]), int(r["dst"])
                rows.append((s, d))
                if int(r["directed"]) == 0:
                    rows.append((d, s))
        cur.executemany("INSERT INTO edges_dir VALUES (%s,%s);", rows)
        cur.execute("CREATE INDEX idx_src ON edges_dir(src);")
        self.conn.commit()
        cur.close()

    def _reach(self, src: int, dst: int) -> bool:
        if src == dst:
            return True
        cur = self.conn.cursor()
        cur.execute("SET statement_timeout = 30000;")  # 30s guard
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
