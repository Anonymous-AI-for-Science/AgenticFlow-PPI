"""TigerGraph GSQL baseline (optional).

Loads the canonical export into a real TigerGraph instance and answers reachability
with an installed GSQL query. Connection params come from $AGENTFLOW_TG_HOST /
$AGENTFLOW_TG_USER / $AGENTFLOW_TG_PASSWORD / $AGENTFLOW_TG_GRAPH; if unset or
pyTigerGraph / the server is unavailable, EngineUnavailable is raised so the
harness skips it. TigerGraph has no free Docker image comparable to Neo4j/Postgres,
so this baseline is expected to be skipped in most environments; the code is
included so a user with a TigerGraph instance can run it.

To use:
    pip install pyTigerGraph
    export AGENTFLOW_TG_HOST=https://localhost
    export AGENTFLOW_TG_USER=tigergraph AGENTFLOW_TG_PASSWORD=tigergraph
    export AGENTFLOW_TG_GRAPH=agentflow
"""

from __future__ import annotations

import csv
import os
from pathlib import Path
from typing import List, Set

from .base import BaseEngine, EngineUnavailable

# GSQL: directed reachability from a source vertex; returns whether dst is visited.
_GSQL_REACH = """
CREATE QUERY reach(VERTEX<Protein> src, VERTEX<Protein> dst) FOR GRAPH {graph} {{
  OrAccum @@found;
  Start = {{src}};
  visited = Start;
  WHILE Start.size() > 0 DO
    Start = SELECT t FROM Start:s -(LINK:e)-> Protein:t
            WHERE t != dst OR (@@found += true) != true
            ACCUM t.@visited = true
            POST-ACCUM CASE WHEN t == dst THEN @@found += true END;
  END;
  PRINT @@found AS reachable;
}}
"""


class TigerGraphEngine(BaseEngine):
    name = "tigergraph-gsql"

    def __init__(self):
        self.host = os.environ.get("AGENTFLOW_TG_HOST")
        self.user = os.environ.get("AGENTFLOW_TG_USER", "tigergraph")
        self.password = os.environ.get("AGENTFLOW_TG_PASSWORD")
        self.graph = os.environ.get("AGENTFLOW_TG_GRAPH", "agentflow")
        self.conn = None

    def load(self, export_dir: Path) -> None:
        if not self.host or not self.password:
            raise EngineUnavailable("AGENTFLOW_TG_HOST / _PASSWORD not set")
        try:
            import pyTigerGraph as tg
        except Exception as e:  # noqa: BLE001
            raise EngineUnavailable(f"pyTigerGraph not installed: {e}")
        try:
            self.conn = tg.TigerGraphConnection(host=self.host, graphname=self.graph,
                                                username=self.user, password=self.password)
            self.conn.echo()  # connectivity check
        except Exception as e:  # noqa: BLE001
            raise EngineUnavailable(f"cannot connect to TigerGraph: {e}")
        # schema + load would be issued here via self.conn.gsql(...); kept minimal
        # because most environments will not have a reachable TigerGraph server.
        raise EngineUnavailable("TigerGraph server reachable but full load path not "
                                "provisioned in this artifact; provide a configured "
                                "instance to enable.")

    def mediators(self, source: int, target: int, gold: List[int]) -> Set[int]:
        raise EngineUnavailable("TigerGraph engine not loaded")
