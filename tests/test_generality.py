"""Test domain-generality (design rationale): SHRC stays exact on non-PPI graphs and composes
with the SQLite engine; the residual-core advantage is structure-dependent (honest)."""

import importlib.util
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


def _load(name):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    return mod


def test_shrc_exact_on_non_ppi_graphs():
    g = _load("benchmark_generality")
    for name, builder in g.FAMILIES.items():
        r = g.run_family(name, builder, seed=7, n_queries=150)
        # exactness must hold off-domain (the portability guarantee)
        assert r["shrc_vs_bfs_agreement"] == 1.0, name
        assert r["sqlite_vs_bfs_agreement"] == 1.0, name


def test_core_advantage_is_structure_dependent():
    """Honest finding: tree-peripheried graphs (citation/KG) keep a small core; a
    2-D mesh (road grid) does not, so SHRC's compactness advantage is structure-bound."""
    g = _load("benchmark_generality")
    cit = g.run_family("citation-dag", g.FAMILIES["citation-dag"], seed=7, n_queries=80)
    road = g.run_family("road-grid", g.FAMILIES["road-grid"], seed=7, n_queries=80)
    assert cit["residual_core_ratio"] < 0.8      # periphery peels
    assert road["residual_core_ratio"] >= 0.99   # mesh: no tree periphery
