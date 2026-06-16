"""Measurement harness for the production graph-engine baselines.

Builds one canonical export, runs every available engine over it, and records:
  * load_seconds      time to load the snapshot into the engine
  * peak_mem_mb       peak Python-side memory during load (tracemalloc)
  * cold_ms / warm_ms first-call vs repeat-call latency per query
  * timed_out         queries exceeding the per-query wall-clock guard
  * answer-equivalence vs the exact BFS oracle (correctness)

External engines (Neo4j/PostgreSQL/TigerGraph) are skipped gracefully with a note
when their server/driver is absent; the in-process SHRC engine always runs and is
the correctness reference. This is the harness referenced by the systems-claim
subsection of the paper.
"""

from __future__ import annotations

import time
import tracemalloc
from pathlib import Path
from typing import Dict, List

from .base import BaseEngine, EngineUnavailable, EngineResult, QueryTiming
from .oracle import oracle_answers, answer_equivalence
from .canonical_export import load_export

PER_QUERY_TIMEOUT_S = 30.0


def run_engine(engine: BaseEngine, export_dir: Path, queries: List[Dict],
               progress_cb=None) -> EngineResult:
    """Run one engine over the snapshot.

    `progress_cb`, if given, is called with short stage strings ("load",
    "queries", "oracle-check completed counts) so a caller can render progress.
    It fires only OUTSIDE the timed regions, so it never pollutes the measured
    load/cold/warm latencies.
    """
    def _emit(stage, **kw):
        if progress_cb is not None:
            progress_cb(engine.name, stage, **kw)

    _emit("load:start")
    try:
        tracemalloc.start()
        t0 = time.perf_counter()
        engine.load(export_dir)
        load_s = time.perf_counter() - t0
        _cur, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
    except EngineUnavailable as e:
        tracemalloc.stop()
        _emit("unavailable", note=str(e))
        return EngineResult(engine=engine.name, available=False, note=str(e))
    except Exception as e:  # noqa: BLE001
        tracemalloc.stop()
        _emit("load-error", note=str(e))
        return EngineResult(engine=engine.name, available=False, note=f"load error: {e}")

    _emit("load:done", load_seconds=round(load_s, 4))
    res = EngineResult(engine=engine.name, available=True,
                       load_seconds=round(load_s, 4), peak_mem_mb=round(peak / 1e6, 3))
    _emit("queries:start", total=len(queries))
    for q in queries:
        # cold call
        t0 = time.perf_counter()
        try:
            ans = engine.mediators(q["source"], q["target"], q["gold"])
        except Exception as e:  # noqa: BLE001
            res.note = f"query error: {e}"; break
        cold = (time.perf_counter() - t0) * 1000.0
        if cold > PER_QUERY_TIMEOUT_S * 1000.0:
            res.timed_out += 1
        # warm call (repeat)
        t0 = time.perf_counter()
        engine.mediators(q["source"], q["target"], q["gold"])
        warm = (time.perf_counter() - t0) * 1000.0
        res.timings.append(QueryTiming(qid=q["qid"], answer=set(ans),
                                       cold_ms=round(cold, 4), warm_ms=round(warm, 4)))
        _emit("queries:tick")
    _emit("queries:done")
    try:
        engine.close()
    except Exception:  # noqa: BLE001
        pass
    return res


def benchmark_all(export_dir: Path, engines: List[BaseEngine], progress_cb=None) -> Dict:
    def _emit(name, stage, **kw):
        if progress_cb is not None:
            progress_cb(name, stage, **kw)

    nodes, edges, queries = load_export(export_dir)
    _emit("oracle", "oracle:start", total=len(queries))
    def _oracle_cb(done, total):
        _emit("oracle", "oracle:tick", done=done, total=total)
    reference = oracle_answers(nodes, edges, queries, progress_cb=_oracle_cb)
    _emit("oracle", "oracle:done", num_queries=len(queries))

    out = {"num_nodes": len(nodes), "num_edges": len(edges), "num_queries": len(queries),
           "engines": []}
    for idx, eng in enumerate(engines, 1):
        _emit(eng.name, "engine:start", index=idx, total=len(engines))
        r = run_engine(eng, export_dir, queries, progress_cb=progress_cb)
        entry = {"engine": r.engine, "available": r.available, "note": r.note}
        if r.available:
            _emit(eng.name, "verify:start")
            equiv = answer_equivalence(reference, r.answers())
            colds = [t.cold_ms for t in r.timings]; warms = [t.warm_ms for t in r.timings]
            entry.update({
                "load_seconds": r.load_seconds, "peak_mem_mb": r.peak_mem_mb,
                "mean_cold_ms": round(sum(colds) / len(colds), 4) if colds else None,
                "mean_warm_ms": round(sum(warms) / len(warms), 4) if warms else None,
                "timed_out": r.timed_out,
                "answer_equivalence": equiv.summary(),
            })
            _emit(eng.name, "engine:done", available=True,
                  all_match=equiv.summary().get("all_match"),
                  mean_warm_ms=entry["mean_warm_ms"])
        else:
            _emit(eng.name, "engine:done", available=False, note=r.note)
        out["engines"].append(entry)
    return out
