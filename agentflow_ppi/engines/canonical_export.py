"""Canonical graph export: one matched snapshot for every engine baseline.

A fair production-engine comparison requires that Neo4j, PostgreSQL, TigerGraph,
and the in-process SHRC engine all answer the SAME query over the SAME graph. This
module materializes one canonical, frozen export from a manifest (the external
biological manifest or any typed edge graph) into three on-disk artifacts:

  * nodes.csv         id,symbol                (one row per protein)
  * edges.csv         src,dst,modality,score,directed
  * queries.csv       qid,source,target,gold   (gold = ';'-joined symbols)

Every engine loader consumes exactly these files, so input parity is guaranteed by
construction and is auditable. The export is deterministic.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple


@dataclass
class CanonicalExport:
    nodes_csv: Path
    edges_csv: Path
    queries_csv: Path
    meta_json: Path


def export_from_manifest(manifest, out_dir: Path) -> CanonicalExport:
    """Write the canonical CSV export from an ExternalManifest-like object with
    `.edges` (list of (src,tgt,modality,score,directed)) and `.queries`
    (list of {source,target,gold:[...]})."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # stable node id assignment
    symbols: Dict[str, int] = {}
    def nid(sym: str) -> int:
        if sym not in symbols:
            symbols[sym] = len(symbols)
        return symbols[sym]
    for a, b, *_ in manifest.edges:
        nid(a); nid(b)

    nodes_csv = out_dir / "nodes.csv"
    with nodes_csv.open("w", newline="") as f:
        w = csv.writer(f); w.writerow(["id", "symbol"])
        for sym, i in sorted(symbols.items(), key=lambda kv: kv[1]):
            w.writerow([i, sym])

    edges_csv = out_dir / "edges.csv"
    with edges_csv.open("w", newline="") as f:
        w = csv.writer(f); w.writerow(["src", "dst", "modality", "score", "directed"])
        for a, b, m, sc, d in manifest.edges:
            w.writerow([symbols[a], symbols[b], m, f"{sc:.4f}", int(bool(d))])

    queries_csv = out_dir / "queries.csv"
    with queries_csv.open("w", newline="") as f:
        w = csv.writer(f); w.writerow(["qid", "source", "target", "gold"])
        for qi, q in enumerate(manifest.queries):
            if q["source"] in symbols and q["target"] in symbols:
                gold = ";".join(str(symbols[g]) for g in q["gold"] if g in symbols)
                w.writerow([qi, symbols[q["source"]], symbols[q["target"]], gold])

    meta = {
        "num_nodes": len(symbols), "num_edges": len(manifest.edges),
        "num_queries": sum(1 for q in manifest.queries
                           if q["source"] in symbols and q["target"] in symbols),
        "provenance": getattr(manifest, "provenance", {}),
        "schema": {
            "nodes.csv": "id,symbol",
            "edges.csv": "src,dst,modality,score,directed (directed=1 means src->dst only)",
            "queries.csv": "qid,source,target,gold (gold = ';'-joined node ids on the canonical pathway)",
        },
    }
    meta_json = out_dir / "canonical_meta.json"
    meta_json.write_text(json.dumps(meta, indent=2))
    return CanonicalExport(nodes_csv, edges_csv, queries_csv, meta_json)


def load_export(out_dir: Path):
    """Read a canonical export back into (nodes, edges, queries) for the oracle and
    the in-process engine."""
    out_dir = Path(out_dir)
    nodes: Dict[int, str] = {}
    with (out_dir / "nodes.csv").open() as f:
        for row in csv.DictReader(f):
            nodes[int(row["id"])] = row["symbol"]
    edges: List[Tuple[int, int, str, float, bool]] = []
    with (out_dir / "edges.csv").open() as f:
        for row in csv.DictReader(f):
            edges.append((int(row["src"]), int(row["dst"]), row["modality"],
                          float(row["score"]), bool(int(row["directed"]))))
    queries: List[Dict] = []
    with (out_dir / "queries.csv").open() as f:
        for row in csv.DictReader(f):
            gold = [int(x) for x in row["gold"].split(";") if x != ""]
            queries.append({"qid": int(row["qid"]), "source": int(row["source"]),
                            "target": int(row["target"]), "gold": gold})
    return nodes, edges, queries
