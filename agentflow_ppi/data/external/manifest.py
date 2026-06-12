"""Build an external label manifest from the downloaded biological sources.

The manifest is the bridge between raw databases and the reranking experiment. It
merges the interaction sources (STRING/BioGRID/OmniPath) into one typed,
multi-modal graph keyed on gene symbols, attaches Reactome pathway membership as
the INDEPENDENT label source (gold mediators are defined by shared-pathway
position, never by any edge attribute the reranker can see), and enumerates
pathway-grounded source->target queries with their gold intermediate mediators.

The manifest is written as JSON so the experiment and the splits are fully
reproducible and auditable.
"""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

from .download import DEFAULT_CACHE, resolve_path, is_using_fixture
from . import loaders
from .id_mapping import reactome_membership_to_symbols


@dataclass
class ExternalManifest:
    edges: List[Tuple[str, str, str, float, bool]]      # src,tgt,modality,score,directed
    pathway_members: Dict[str, List[str]]               # pathway_id -> [gene symbols]
    pathway_name: Dict[str, str]
    queries: List[Dict]                                  # {source,target,pathway,gold:[...]}
    provenance: Dict[str, str]

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)


def _merge_edges(edge_lists) -> Dict[Tuple[str, str, str], Tuple[float, bool]]:
    merged: Dict[Tuple[str, str, str], Tuple[float, bool]] = {}
    for edges in edge_lists:
        for e in edges:
            key = (e.source, e.target, e.modality)
            prev = merged.get(key)
            if prev is None or e.score > prev[0]:
                merged[key] = (e.score, e.directed)
    return merged


def _pathway_linear_order(members: List[str], directed_adj: Dict[str, Set[str]]) -> List[str]:
    """Heuristic linear order of a pathway's members by a topological-ish sort over
    the directed regulatory edges restricted to the pathway. Falls back to input
    order. The order defines 'between source and target' for gold mediators."""
    mem = set(members)
    indeg = {m: 0 for m in mem}
    adj = {m: set() for m in mem}
    for u in mem:
        for v in directed_adj.get(u, ()):
            if v in mem:
                adj[u].add(v); indeg[v] += 1
    # Kahn's algorithm; ties broken by name for determinism
    order: List[str] = []
    avail = sorted([m for m in mem if indeg[m] == 0])
    seen = set()
    while avail:
        n = avail.pop(0)
        if n in seen:
            continue
        seen.add(n); order.append(n)
        for v in sorted(adj[n]):
            indeg[v] -= 1
            if indeg[v] == 0 and v not in seen:
                avail.append(v)
        avail = sorted(set(avail))
    for m in members:  # append any cycle remnants deterministically
        if m not in seen:
            order.append(m)
    return order


def build_manifest(cache_dir: Path = DEFAULT_CACHE, allow_online: bool = True,
                   string_threshold: int = 700, min_pathway_size: int = 3,
                   max_pathway_size: int = 40) -> ExternalManifest:
    # --- interaction sources ---
    edge_lists = []
    prov = {}
    string_edges = loaders.load_string(
        resolve_path("string", "links", cache_dir), resolve_path("string", "info", cache_dir),
        score_threshold=string_threshold)
    edge_lists.append(string_edges); prov["string"] = "fixture" if is_using_fixture("string", "links", cache_dir) else "download"
    biogrid_edges = loaders.load_biogrid(resolve_path("biogrid", "tab3", cache_dir))
    edge_lists.append(biogrid_edges); prov["biogrid"] = "fixture" if is_using_fixture("biogrid", "tab3", cache_dir) else "download"
    omnipath_edges = loaders.load_omnipath(resolve_path("omnipath", "interactions", cache_dir))
    edge_lists.append(omnipath_edges); prov["omnipath"] = "fixture" if is_using_fixture("omnipath", "interactions", cache_dir) else "download"

    merged = _merge_edges(edge_lists)
    edges = [(a, b, m, sc, d) for (a, b, m), (sc, d) in merged.items()]

    # directed adjacency from regulatory (OmniPath) edges for pathway ordering
    directed_adj: Dict[str, Set[str]] = defaultdict(set)
    for a, b, m, sc, d in edges:
        if d:
            directed_adj[a].add(b)

    # --- Reactome pathway labels (independent of edge attributes) ---
    membership_uni, pathway_name, _rel = loaders.load_reactome(
        resolve_path("reactome", "uniprot2reactome", cache_dir),
        resolve_path("reactome", "relations", cache_dir))
    prov["reactome"] = "fixture" if is_using_fixture("reactome", "uniprot2reactome", cache_dir) else "download"
    membership_sym = reactome_membership_to_symbols(membership_uni, cache_dir=cache_dir, allow_online=allow_online)

    # invert to pathway -> members, restrict to graph nodes and to size band
    graph_nodes = {a for a, *_ in edges} | {b for _, b, *_ in edges}
    pw_members: Dict[str, List[str]] = defaultdict(list)
    for sym, pids in membership_sym.items():
        if sym not in graph_nodes:
            continue
        for pid in pids:
            pw_members[pid].append(sym)
    pw_members = {p: sorted(set(ms)) for p, ms in pw_members.items()
                  if min_pathway_size <= len(set(ms)) <= max_pathway_size}

    # --- enumerate pathway-grounded queries with gold mediators ---
    queries: List[Dict] = []
    for pid, members in pw_members.items():
        order = _pathway_linear_order(members, directed_adj)
        for i in range(len(order)):
            for j in range(i + 2, len(order)):
                s, t = order[i], order[j]
                gold = order[i + 1:j]
                if gold:
                    queries.append({"source": s, "target": t, "pathway": pid, "gold": gold})

    return ExternalManifest(
        edges=edges,
        pathway_members={p: ms for p, ms in pw_members.items()},
        pathway_name={p: pathway_name.get(p, p) for p in pw_members},
        queries=queries,
        provenance=prov,
    )


def write_manifest(manifest: ExternalManifest, out_path: Path) -> None:
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(manifest.to_json())
