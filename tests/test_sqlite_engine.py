"""Test the SQLite recursive-CTE engine (R2-O2): a real SQL engine, no server, that
answers reachability exactly (verified against the BFS oracle)."""

import csv
import tempfile
from pathlib import Path

from agentflow_ppi.engines.sqlite_engine import SQLiteEngine, run_sqlite_engine


def _tiny_export(d: Path):
    # chain 0->1->2->3 plus 0->4 (4 is a non-mediator distractor)
    with (d / "edges.csv").open("w", newline="") as f:
        w = csv.writer(f); w.writerow(["src", "dst", "modality", "score", "directed"])
        for s, t in [(0, 1), (1, 2), (2, 3), (0, 4)]:
            w.writerow([s, t, "functional", 0.9, 1])
    with (d / "nodes.csv").open("w", newline="") as f:
        w = csv.writer(f); w.writerow(["id", "symbol"])
        for i in range(5):
            w.writerow([i, f"P{i}"])


def test_sqlite_reachability_exact():
    with tempfile.TemporaryDirectory() as td:
        d = Path(td); _tiny_export(d)
        eng = SQLiteEngine(); eng.load(d)
        # 0 -> 3 path exists through mediators 1 and 2
        assert eng._reach(0, 3) is True
        assert eng._reach(3, 0) is False         # directed
        # mediators between 0 and 3 among {1,2,4}: 1 and 2 reach; 4 does not reach 3
        med = eng.mediators(0, 3, [1, 2, 4])
        assert med == {1, 2}
        eng.close()


def test_sqlite_engine_result_shape():
    with tempfile.TemporaryDirectory() as td:
        d = Path(td); _tiny_export(d)
        queries = [{"qid": 0, "source": 0, "target": 3, "gold": [1, 2, 4]}]
        res = run_sqlite_engine(d, queries)
        assert res.available is True
        assert res.engine == "sqlite-recursive-cte"
        assert res.timings[0].answer == {1, 2}
