"""Parsers for the external biological dataset file formats.

Each loader reads the exact published format and returns a normalized structure.
They operate on whatever path `download.resolve_path` returns -- a freshly
downloaded cache file or a bundled fixture -- so the same code path runs online
and offline.

Normalized types:
  * InteractionEdge(source, target, modality, score, directed)
  * PathwayMembership: dict[uniprot] -> set[pathway_id]
  * pathway hierarchy: dict[parent_id] -> set[child_id]
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Set, Tuple


@dataclass
class InteractionEdge:
    source: str          # gene symbol (normalized)
    target: str
    modality: str        # physical / functional / regulatory / predicted
    score: float         # [0,1]
    directed: bool


# ----------------------------- STRING -------------------------------------- #

def load_string(links_path: Path, info_path: Path, score_threshold: int = 700) -> List[InteractionEdge]:
    """Parse STRING protein.links (space-separated: protein1 protein2 combined_score)
    and protein.info (maps ENSP id -> preferred_name). combined_score is 0..1000.
    STRING is undirected functional association; modality='functional'."""
    ensp_to_name: Dict[str, str] = {}
    with open(info_path) as f:
        for ln in f:
            if ln.startswith("#") or not ln.strip():
                continue
            parts = ln.rstrip("\n").split("\t")
            if len(parts) >= 2:
                ensp_to_name[parts[0]] = parts[1]
    edges: List[InteractionEdge] = []
    with open(links_path) as f:
        header = f.readline()  # 'protein1 protein2 combined_score'
        for ln in f:
            p = ln.split()
            if len(p) < 3:
                continue
            a, b, sc = p[0], p[1], int(p[2])
            if sc < score_threshold:
                continue
            na, nb = ensp_to_name.get(a), ensp_to_name.get(b)
            if not na or not nb or na == nb:
                continue
            edges.append(InteractionEdge(na, nb, "functional", sc / 1000.0, directed=False))
    return edges


# ----------------------------- BioGRID ------------------------------------- #

def load_biogrid(tab3_path: Path, taxid: int = 9606,
                 systems_physical=("Affinity Capture-MS", "Two-hybrid",
                                   "Affinity Capture-Western", "Reconstituted Complex",
                                   "Co-fractionation", "Co-crystal Structure")) -> List[InteractionEdge]:
    """Parse BioGRID tab3. Uses Official Symbol Interactor A/B, restricts to the
    given organism on both sides, maps experimental system type to modality."""
    edges: List[InteractionEdge] = []
    with open(tab3_path, newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            try:
                oa = int(row.get("Organism ID Interactor A", "0") or 0)
                ob = int(row.get("Organism ID Interactor B", "0") or 0)
            except ValueError:
                continue
            if oa != taxid or ob != taxid:
                continue
            a = row.get("Official Symbol Interactor A", "").strip()
            b = row.get("Official Symbol Interactor B", "").strip()
            if not a or not b or a == b or a == "-" or b == "-":
                continue
            etype = (row.get("Experimental System Type", "") or "").strip().lower()
            modality = "physical" if etype == "physical" else "functional"
            # BioGRID is largely undirected physical evidence
            edges.append(InteractionEdge(a, b, modality, 0.8, directed=False))
    return edges


# ----------------------------- Reactome ------------------------------------ #

def load_reactome(uniprot2reactome_path: Path, relations_path: Path,
                  species: str = "Homo sapiens") -> Tuple[Dict[str, Set[str]], Dict[str, str], Dict[str, Set[str]]]:
    """Parse Reactome UniProt2Reactome_All_Levels.txt and ReactomePathwaysRelation.txt.

    Returns (uniprot -> set(pathway_id), pathway_id -> name, parent -> set(child))."""
    membership: Dict[str, Set[str]] = {}
    pathway_name: Dict[str, str] = {}
    with open(uniprot2reactome_path) as f:
        for ln in f:
            parts = ln.rstrip("\n").split("\t")
            if len(parts) < 6:
                continue
            uni, pid, _url, name, _ev, sp = parts[:6]
            if sp != species:
                continue
            membership.setdefault(uni, set()).add(pid)
            pathway_name[pid] = name
    relations: Dict[str, Set[str]] = {}
    if Path(relations_path).exists():
        with open(relations_path) as f:
            for ln in f:
                parts = ln.rstrip("\n").split("\t")
                if len(parts) >= 2:
                    relations.setdefault(parts[0], set()).add(parts[1])
    return membership, pathway_name, relations


# ----------------------------- OmniPath ------------------------------------ #

def load_omnipath(interactions_path: Path) -> List[InteractionEdge]:
    """Parse OmniPath interactions TSV. Uses gene symbols and the directionality /
    sign columns. Directed signaling edges; modality='regulatory'."""
    edges: List[InteractionEdge] = []
    with open(interactions_path, newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            a = (row.get("source_genesymbol") or "").strip()
            b = (row.get("target_genesymbol") or "").strip()
            if not a or not b or a == b:
                continue
            directed = str(row.get("is_directed", "0")).strip() in ("1", "True", "true")
            stim = str(row.get("is_stimulation", "0")).strip() in ("1", "True", "true")
            inhib = str(row.get("is_inhibition", "0")).strip() in ("1", "True", "true")
            # signed regulatory edge; score reflects evidence presence
            nrefs = len([r for r in (row.get("references", "") or "").split(";") if r.strip()])
            score = min(0.95, 0.6 + 0.1 * nrefs)
            modality = "regulatory" if (stim or inhib) else "functional"
            edges.append(InteractionEdge(a, b, modality, score, directed=directed))
    return edges
