"""Curated canonical-pathway ground truth for the named PPI workload.

Reviewer concern A1: the earlier gold-mediator definition used
modality-consistency, which is *also* a reranker feature, creating a circular
evaluation. This module replaces it with an INDEPENDENT label source: membership
on the canonical signaling cascade that mechanistically connects a source to a
target, curated from textbook pathway topology (KEGG/Reactome-style cascades).

The labels here depend ONLY on which biological pathway a protein belongs to and
on the established directionality of that cascade. They do not reference edge
confidence, modality counts, or any feature consumed by the learned reranker, so
a gain measured against these labels is not tautological.

Pathways encoded (well-established human signaling):
  * DDR_P53 : DNA-damage / p53 response
              TP53 -> {ATM, CHEK2} -> BRCA1 ; TP53/ATM -> RAD51
  * RTK_MAPK: receptor-tyrosine-kinase Ras/MAPK cascade
              EGFR/ERBB2 -> {GRB2, SHC1} -> SOS1 -> KRAS -> RAF1 -> MAPK1 -> STAT3
  * RTK_PI3K: receptor-tyrosine-kinase PI3K/AKT cascade
              EGFR/ERBB2 -> {PIK3CA, GAB1} -> AKT1 (PTEN negative regulator)

A mediator m is a GOLD intermediate for query (s, t) iff:
  (1) s, m, t all lie on a single canonical pathway P, and
  (2) m lies strictly between s and t in P's curated linear order
      (i.e., s precedes m and m precedes t in the cascade).

This is a position-on-cascade criterion, fully independent of the graph's
numeric attributes.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Set

# Canonical cascade orderings (rank = position along the pathway).
PATHWAY_ORDER: Dict[str, List[str]] = {
    # DNA-damage response: receptors/sensors -> transducers -> effectors
    "DDR_P53": ["TP53", "ATM", "CHEK2", "BRCA1", "RAD51"],
    # Ras/MAPK cascade from the receptor down to the transcriptional effector
    "RTK_MAPK": ["EGFR", "ERBB2", "SHC1", "GRB2", "SOS1", "KRAS", "RAF1", "MAPK1", "STAT3"],
    # PI3K/AKT cascade
    "RTK_PI3K": ["EGFR", "ERBB2", "SHC1", "GAB1", "PIK3CA", "AKT1"],
}

# Some proteins legitimately sit on more than one cascade (e.g. SHC1, EGFR).
PATHWAY_MEMBERS: Dict[str, Set[str]] = {p: set(order) for p, order in PATHWAY_ORDER.items()}


def _rank(pathway: str, node: str) -> Optional[int]:
    order = PATHWAY_ORDER[pathway]
    return order.index(node) if node in order else None


def gold_pathway_mediators(source: str, target: str, candidates: Sequence[str]) -> Set[str]:
    """Return curated gold mediators among ``candidates`` for query (source, target).

    A candidate is gold iff there exists a canonical pathway on which source,
    candidate, and target all lie, with source strictly before candidate and
    candidate strictly before target along that pathway's curated order.
    """
    gold: Set[str] = set()
    for pathway, order in PATHWAY_ORDER.items():
        rs = _rank(pathway, source)
        rt = _rank(pathway, target)
        if rs is None or rt is None or rs >= rt:
            continue
        for cand in candidates:
            rc = _rank(pathway, cand)
            if rc is not None and rs < rc < rt:
                gold.add(cand)
    return gold


def query_is_pathway_grounded(source: str, target: str) -> bool:
    """True iff source and target co-occur on at least one canonical pathway in order."""
    for pathway in PATHWAY_ORDER:
        rs = _rank(pathway, source)
        rt = _rank(pathway, target)
        if rs is not None and rt is not None and rs < rt:
            return True
    return False
