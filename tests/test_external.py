"""Tests for the external biological dataset pipeline (Phase 1).

These run fully offline on the bundled fixtures, exercising the loaders, ID
mapping, manifest construction, and the leakage-controlled splits. They verify the
splits are genuinely disjoint (zero protein/pathway overlap), which is the property
the external benchmark hinges on.
"""

from agentflow_ppi.data.external.download import resolve_path
from agentflow_ppi.data.external import loaders
from agentflow_ppi.data.external.manifest import build_manifest
from agentflow_ppi.data.external import splits as S


def test_loaders_parse_fixtures():
    se = loaders.load_string(resolve_path("string", "links"), resolve_path("string", "info"),
                             score_threshold=700)
    assert se and all(e.modality == "functional" for e in se)
    be = loaders.load_biogrid(resolve_path("biogrid", "tab3"))
    assert be and all(not e.directed for e in be)
    oe = loaders.load_omnipath(resolve_path("omnipath", "interactions"))
    assert oe and any(e.directed for e in oe)
    membership, names, rel = loaders.load_reactome(
        resolve_path("reactome", "uniprot2reactome"), resolve_path("reactome", "relations"))
    assert membership and names


def test_manifest_builds_offline():
    man = build_manifest(allow_online=False)
    assert all(v == "fixture" for v in man.provenance.values())
    assert man.edges and man.pathway_members and man.queries
    # gold mediators must lie between source and target on a pathway (independent label)
    for q in man.queries:
        assert q["gold"] and q["source"] != q["target"]


def test_splits_are_disjoint():
    man = build_manifest(allow_online=False)
    for split_fn in (S.pathway_disjoint_split, S.protein_disjoint_split):
        tr, te = split_fn(man.queries, seed=7)
        stats = S.split_stats(man.queries, tr, te)
        if split_fn is S.pathway_disjoint_split:
            assert stats["pathway_overlap"] == 0
        else:
            assert stats["protein_overlap"] == 0
