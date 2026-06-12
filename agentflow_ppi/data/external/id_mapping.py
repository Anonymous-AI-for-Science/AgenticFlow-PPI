"""Cross-reference ID mapping: UniProt accession <-> Ensembl protein <-> gene symbol.

The different sources key on different identifier spaces:
  * STRING        -> Ensembl protein IDs (ENSP...), resolved to gene symbols via
                     the protein.info file (handled in the STRING loader).
  * BioGRID       -> Official gene symbols (+ SWISS-PROT accessions).
  * Reactome      -> UniProt accessions.
  * OmniPath      -> UniProt accessions (+ gene symbols).

To join them we normalize everything to gene symbols. UniProt accessions from
Reactome/OmniPath are mapped to symbols via the UniProt ID-mapping REST API when
online; offline we use a bundled mapping table covering the fixture proteins. The
mapping is cached to disk so a live run pays the API cost once.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional

from .download import DEFAULT_CACHE

UNIPROT_IDMAP_RUN = "https://rest.uniprot.org/idmapping/run"
UNIPROT_IDMAP_STATUS = "https://rest.uniprot.org/idmapping/status/"
UNIPROT_IDMAP_RESULT = "https://rest.uniprot.org/idmapping/results/"

# Bundled offline mapping for the fixture proteins (UniProt accession -> gene symbol).
# These are the canonical human accessions for the fixture's signaling proteins.
_OFFLINE_UNIPROT_TO_SYMBOL: Dict[str, str] = {
    "P00533": "EGFR", "P04626": "ERBB2", "P29353": "SHC1", "P62993": "GRB2",
    "Q07889": "SOS1", "P01116": "KRAS", "P04049": "RAF1", "Q02750": "MAP2K1",
    "P28482": "MAPK1", "P40763": "STAT3", "P42336": "PIK3CA", "Q13480": "GAB1",
    "P31749": "AKT1", "P42345": "MTOR", "P04637": "TP53", "Q13315": "ATM",
    "O96017": "CHEK2", "P38398": "BRCA1", "Q06609": "RAD51", "Q00987": "MDM2",
    "P38936": "CDKN1A", "P60484": "PTEN", "P49841": "GSK3B",
}


def _cache_file(cache_dir: Path) -> Path:
    d = Path(cache_dir) / "idmapping"
    d.mkdir(parents=True, exist_ok=True)
    return d / "uniprot_to_symbol.json"


def _load_cache(cache_dir: Path) -> Dict[str, str]:
    f = _cache_file(cache_dir)
    if f.exists():
        try:
            return json.loads(f.read_text())
        except Exception:  # noqa: BLE001
            return {}
    return {}


def _save_cache(cache_dir: Path, mapping: Dict[str, str]) -> None:
    _cache_file(cache_dir).write_text(json.dumps(mapping, indent=0))


def _uniprot_idmap_online(accessions: List[str], from_db="UniProtKB_AC-ID",
                          to_db="Gene_Name", timeout: int = 60) -> Dict[str, str]:
    """Resolve UniProt accessions to gene names via the UniProt ID-mapping REST API.
    Raises on any network/HTTP failure so the caller can fall back offline."""
    import requests
    r = requests.post(UNIPROT_IDMAP_RUN, data={"from": from_db, "to": to_db,
                                               "ids": ",".join(accessions)}, timeout=timeout)
    r.raise_for_status()
    job = r.json()["jobId"]
    for _ in range(60):
        s = requests.get(UNIPROT_IDMAP_STATUS + job, timeout=timeout)
        s.raise_for_status()
        js = s.json()
        if js.get("jobStatus") in (None, "FINISHED") or "results" in js:
            break
        time.sleep(2)
    res = requests.get(UNIPROT_IDMAP_RESULT + job, params={"format": "json", "size": 500}, timeout=timeout)
    res.raise_for_status()
    out: Dict[str, str] = {}
    for rec in res.json().get("results", []):
        frm = rec.get("from"); to = rec.get("to")
        if isinstance(to, dict):
            to = to.get("geneName") or to.get("primaryAccession")
        if frm and to:
            out[frm] = to
    return out


def map_uniprot_to_symbol(accessions: Iterable[str], cache_dir: Path = DEFAULT_CACHE,
                          allow_online: bool = True) -> Dict[str, str]:
    """Return accession -> gene symbol. Uses the disk cache, then the online UniProt
    API (if allowed and reachable), then the bundled offline table. Unresolved
    accessions are simply omitted."""
    accessions = [a for a in dict.fromkeys(accessions) if a]
    mapping = _load_cache(cache_dir)
    missing = [a for a in accessions if a not in mapping]
    if missing and allow_online:
        try:
            online = _uniprot_idmap_online(missing)
            mapping.update(online)
            _save_cache(cache_dir, mapping)
            missing = [a for a in accessions if a not in mapping]
        except Exception:  # noqa: BLE001 - offline or API error; fall back
            pass
    # offline fallback for anything still missing
    for a in missing:
        if a in _OFFLINE_UNIPROT_TO_SYMBOL:
            mapping[a] = _OFFLINE_UNIPROT_TO_SYMBOL[a]
    return {a: mapping[a] for a in accessions if a in mapping}


def reactome_membership_to_symbols(membership: Dict[str, set], cache_dir: Path = DEFAULT_CACHE,
                                   allow_online: bool = True) -> Dict[str, set]:
    """Convert a UniProt-keyed pathway membership map to a gene-symbol-keyed one."""
    u2s = map_uniprot_to_symbol(list(membership.keys()), cache_dir=cache_dir, allow_online=allow_online)
    out: Dict[str, set] = {}
    for uni, pids in membership.items():
        sym = u2s.get(uni)
        if sym:
            out.setdefault(sym, set()).update(pids)
    return out
