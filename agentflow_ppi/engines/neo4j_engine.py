"""Neo4j Cypher baseline.

Loads the canonical export into a real Neo4j instance and answers the mediator
query with Cypher path existence (variable-length pattern). Connection params are
taken from $AGENTFLOW_NEO4J_URI / $AGENTFLOW_NEO4J_USER / $AGENTFLOW_NEO4J_PASSWORD;
if unset or the server/driver is unreachable, EngineUnavailable is raised so the
harness skips it.

To spin up a local server:
    docker run --rm -d --name agentflow-neo4j -p 7687:7687 -p 7474:7474 \\
        -e NEO4J_AUTH=neo4j/password123 neo4j:5
    export AGENTFLOW_NEO4J_URI=bolt://localhost:7687
    export AGENTFLOW_NEO4J_USER=neo4j
    export AGENTFLOW_NEO4J_PASSWORD=password123
"""

from __future__ import annotations

import csv
import os
from pathlib import Path
from typing import List, Set

from .base import BaseEngine, EngineUnavailable

# Directed reachability via a bounded variable-length path. The bound keeps the
# query tractable on a production engine; it is set generously (the diameter of the
# canonical snapshot) and recorded so the comparison is honest about semantics.
_REACH_CYPHER = (
    "MATCH (s:Protein {pid:$src}), (t:Protein {pid:$dst}) "
    "RETURN EXISTS { MATCH (s)-[:LINK*1..%d]->(t) } AS reachable"
)


class Neo4jEngine(BaseEngine):
    name = "neo4j-cypher"

    def __init__(self, uri: str = None, user: str = None, password: str = None,
                 max_hops: int = 12):
        self.uri = uri or os.environ.get("AGENTFLOW_NEO4J_URI")
        self.user = user or os.environ.get("AGENTFLOW_NEO4J_USER", "neo4j")
        self.password = password or os.environ.get("AGENTFLOW_NEO4J_PASSWORD")
        self.max_hops = max_hops
        self.driver = None

    def load(self, export_dir: Path) -> None:
        if not self.uri or not self.password:
            raise EngineUnavailable("AGENTFLOW_NEO4J_URI / _PASSWORD not set")
        try:
            from neo4j import GraphDatabase
        except Exception as e:  # noqa: BLE001
            raise EngineUnavailable(f"neo4j driver not installed: {e}")
        try:
            self.driver = GraphDatabase.driver(self.uri, auth=(self.user, self.password))
            self.driver.verify_connectivity()
        except Exception as e:  # noqa: BLE001
            raise EngineUnavailable(f"cannot connect to Neo4j: {e}")

        # directed edges (expand undirected to both directions)
        edges = []
        with (Path(export_dir) / "edges.csv").open() as f:
            for r in csv.DictReader(f):
                s, d = int(r["src"]), int(r["dst"])
                edges.append((s, d))
                if int(r["directed"]) == 0:
                    edges.append((d, s))
        nodes = []
        with (Path(export_dir) / "nodes.csv").open() as f:
            for r in csv.DictReader(f):
                nodes.append(int(r["id"]))

        with self.driver.session() as sess:
            sess.run("MATCH (n) DETACH DELETE n")
            sess.run("CREATE CONSTRAINT IF NOT EXISTS FOR (p:Protein) REQUIRE p.pid IS UNIQUE")
            sess.run("UNWIND $ids AS i CREATE (:Protein {pid:i})", ids=nodes)
            # batched relationship creation
            sess.run(
                "UNWIND $rows AS row "
                "MATCH (a:Protein {pid:row[0]}),(b:Protein {pid:row[1]}) "
                "CREATE (a)-[:LINK]->(b)",
                rows=edges)

    def _reach(self, src: int, dst: int) -> bool:
        if src == dst:
            return True
        with self.driver.session() as sess:
            rec = sess.run(_REACH_CYPHER % self.max_hops, src=src, dst=dst).single()
            return bool(rec["reachable"]) if rec else False

    def mediators(self, source: int, target: int, gold: List[int]) -> Set[int]:
        out = set()
        for m in gold:
            if m in (source, target):
                continue
            if self._reach(source, m) and self._reach(m, target):
                out.add(m)
        return out

    def close(self) -> None:
        if self.driver is not None:
            self.driver.close()
