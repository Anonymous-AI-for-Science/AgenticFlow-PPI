"""Build the external label manifest (typed graph + Reactome pathway labels +
pathway-grounded queries) from downloaded data or bundled fixtures.

Usage:
    python scripts/build_external_manifest.py [--cache DIR] [--out PATH] [--offline]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from agentflow_ppi.data.external.download import DEFAULT_CACHE
from agentflow_ppi.data.external.manifest import build_manifest, write_manifest


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default=str(DEFAULT_CACHE))
    ap.add_argument("--out", default=None)
    ap.add_argument("--offline", action="store_true", help="skip the online UniProt id-map")
    ap.add_argument("--string-threshold", type=int, default=700)
    args = ap.parse_args()

    root = Path(__file__).resolve().parents[1]
    out = Path(args.out) if args.out else root / "results" / "external_manifest.json"
    man = build_manifest(cache_dir=Path(args.cache), allow_online=not args.offline,
                         string_threshold=args.string_threshold)
    write_manifest(man, out)
    print(f"manifest written: {out}")
    print(f"  provenance: {man.provenance}")
    print(f"  graph: {len({a for a,*_ in man.edges} | {b for _,b,*_ in man.edges})} proteins, "
          f"{len(man.edges)} typed edges")
    print(f"  pathways: {len(man.pathway_members)}; queries: {len(man.queries)}")


if __name__ == "__main__":
    main()
