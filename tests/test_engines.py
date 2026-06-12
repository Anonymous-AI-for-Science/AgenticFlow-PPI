"""Tests for the production graph-engine baselines (Phase 2).

Run fully offline: they exercise the canonical export round-trip, the BFS oracle,
the in-process SHRC engine, and the answer-equivalence comparator. External engines
(Neo4j/Postgres/TigerGraph) are not started here; their adapters are unit-checked
to raise EngineUnavailable when no server is configured, which is the behavior the
harness relies on to skip them gracefully.
"""

import sqlite3
from pathlib import Path

from agentflow_ppi.data.external.manifest import build_manifest
from agentflow_ppi.engines.canonical_export import export_from_manifest, load_export
from agentflow_ppi.engines.oracle import build_oracle, oracle_answers, answer_equivalence
from agentflow_ppi.engines.base import InProcessSHRCEngine, EngineUnavailable
from agentflow_ppi.engines.postgres_engine import PostgresEngine
from agentflow_ppi.engines.neo4j_engine import Neo4jEngine


def _export(tmp_path):
    man = build_manifest(allow_online=False)
    return export_from_manifest(man, tmp_path / "export"), tmp_path / "export"


def test_canonical_export_roundtrip(tmp_path):
    _exp, d = _export(tmp_path)
    nodes, edges, queries = load_export(d)
    assert nodes and edges and queries
    # gold ids must be valid node ids
    for q in queries:
        assert all(g in nodes for g in q["gold"])


def test_inprocess_engine_matches_oracle(tmp_path):
    _exp, d = _export(tmp_path)
    nodes, edges, queries = load_export(d)
    reference = oracle_answers(nodes, edges, queries)
    eng = InProcessSHRCEngine(); eng.load(d)
    cand = {q["qid"]: eng.mediators(q["source"], q["target"], q["gold"]) for q in queries}
    rep = answer_equivalence(reference, cand)
    assert rep.all_match, rep.mismatches


def test_recursive_cte_logic_matches_oracle(tmp_path):
    # validate the Postgres recursive-CTE SQL semantics via sqlite (same CTE)
    _exp, d = _export(tmp_path)
    nodes, edges, queries = load_export(d)
    con = sqlite3.connect(":memory:"); cur = con.cursor()
    cur.execute("CREATE TABLE edges_dir(src INT, dst INT)")
    rows = []
    for s, dd, _m, _sc, directed in edges:
        rows.append((s, dd))
        if not directed:
            rows.append((dd, s))
    cur.executemany("INSERT INTO edges_dir VALUES(?,?)", rows); con.commit()

    def reach(s, t):
        if s == t:
            return True
        cur.execute("WITH RECURSIVE reach(node) AS (SELECT ? UNION "
                    "SELECT e.dst FROM edges_dir e JOIN reach r ON e.src=r.node) "
                    "SELECT 1 FROM reach WHERE node=? LIMIT 1", (s, t))
        return cur.fetchone() is not None

    oracle = build_oracle(nodes, edges)
    for q in queries:
        sql_ans = {m for m in q["gold"] if m not in (q["source"], q["target"])
                   and reach(q["source"], m) and reach(m, q["target"])}
        assert sql_ans == oracle.mediators(q["source"], q["target"], q["gold"])


def test_external_engines_skip_without_server(monkeypatch):
    for var in ["AGENTFLOW_PG_DSN", "AGENTFLOW_NEO4J_URI", "AGENTFLOW_NEO4J_PASSWORD"]:
        monkeypatch.delenv(var, raising=False)
    try:
        PostgresEngine(dsn=None).load(Path("."))
        assert False, "expected EngineUnavailable"
    except EngineUnavailable:
        pass
    try:
        Neo4jEngine(uri=None, password=None).load(Path("."))
        assert False, "expected EngineUnavailable"
    except EngineUnavailable:
        pass
