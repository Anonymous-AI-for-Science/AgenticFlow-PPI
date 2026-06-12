"""External biological dataset source definitions.

Each source records the canonical download URL(s), the on-disk filename, the file
format, and a short provenance note. URLs are the real public endpoints; the
loaders in this package parse the exact published file formats. In an environment
with network access to these hosts, `scripts/download_external_data.py` fetches
them into a local cache; offline, the loaders fall back to the small format-faithful
fixtures bundled under `fixtures/` so the full pipeline still runs and is testable.

Hosts required for a live download (must be reachable from the run environment):
  - stringdb-downloads.org            (STRING v12.0 protein links + info)
  - downloads.thebiogrid.org          (BioGRID tab3 organism file)
  - reactome.org                      (UniProt2Reactome, pathway relations)
  - omnipathdb.org                    (OmniPath interactions REST API)
  - rest.uniprot.org                  (UniProt ID-mapping REST)
None of these are mirrored on PyPI/GitHub; a site-restricted sandbox cannot reach
them, which is why the offline fixture path exists.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

# Default organism: human (NCBI taxon 9606).
HUMAN_TAXID = 9606
STRING_VERSION = "12.0"
BIOGRID_VERSION = "4.4.246"


@dataclass(frozen=True)
class Source:
    key: str
    urls: Dict[str, str]            # logical name -> URL
    filenames: Dict[str, str]       # logical name -> local filename
    fmt: str
    note: str
    fixture: Dict[str, str] = field(default_factory=dict)  # logical name -> fixture filename


SOURCES: Dict[str, Source] = {
    "string": Source(
        key="string",
        urls={
            "links": f"https://stringdb-downloads.org/download/protein.links.v{STRING_VERSION}/{HUMAN_TAXID}.protein.links.v{STRING_VERSION}.txt.gz",
            "info": f"https://stringdb-downloads.org/download/protein.info.v{STRING_VERSION}/{HUMAN_TAXID}.protein.info.v{STRING_VERSION}.txt.gz",
        },
        filenames={
            "links": f"{HUMAN_TAXID}.protein.links.v{STRING_VERSION}.txt.gz",
            "info": f"{HUMAN_TAXID}.protein.info.v{STRING_VERSION}.txt.gz",
        },
        fmt="string_links",
        note="STRING v12.0 functional association network; combined_score in [0,1000].",
        fixture={"links": "string_links_sample.txt", "info": "string_info_sample.txt"},
    ),
    "biogrid": Source(
        key="biogrid",
        urls={
            "tab3": f"https://downloads.thebiogrid.org/Download/BioGRID/Release-Archive/BIOGRID-{BIOGRID_VERSION}/BIOGRID-ORGANISM-{BIOGRID_VERSION}.tab3.zip",
        },
        filenames={"tab3": f"BIOGRID-ORGANISM-Homo_sapiens-{BIOGRID_VERSION}.tab3.txt"},
        fmt="biogrid_tab3",
        note="BioGRID physical/genetic interactions, tab3 format (Official Symbol Interactor A/B).",
        fixture={"tab3": "biogrid_tab3_sample.txt"},
    ),
    "reactome": Source(
        key="reactome",
        urls={
            "uniprot2reactome": "https://reactome.org/download/current/UniProt2Reactome_All_Levels.txt",
            "relations": "https://reactome.org/download/current/ReactomePathwaysRelation.txt",
        },
        filenames={
            "uniprot2reactome": "UniProt2Reactome_All_Levels.txt",
            "relations": "ReactomePathwaysRelation.txt",
        },
        fmt="reactome",
        note="Reactome pathway membership (UniProt->pathway) and pathway hierarchy.",
        fixture={"uniprot2reactome": "reactome_uniprot_sample.txt",
                 "relations": "reactome_relations_sample.txt"},
    ),
    "omnipath": Source(
        key="omnipath",
        urls={
            # OmniPath REST: signed, directed interactions with stimulation/inhibition.
            "interactions": "https://omnipathdb.org/interactions?genesymbols=1&fields=sources,references,is_directed,is_stimulation,is_inhibition&license=academic",
        },
        filenames={"interactions": "omnipath_interactions.tsv"},
        fmt="omnipath",
        note="OmniPath curated signed/directed signaling interactions.",
        fixture={"interactions": "omnipath_sample.tsv"},
    ),
}


def all_keys() -> List[str]:
    return list(SOURCES.keys())
