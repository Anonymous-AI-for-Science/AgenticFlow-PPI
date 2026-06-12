"""Small regression tests for the STRING loader."""

from __future__ import annotations

import gzip
from pathlib import Path

from agentflow_ppi.data.string_loader import STRINGLoader


def test_parse_and_normalize(tmp_path: Path) -> None:
    sample = tmp_path / "9606.protein.links.detailed.v12.0.txt.gz"
    with gzip.open(sample, "wt", encoding="utf-8") as handle:
        handle.write("protein1\tprotein2\tcombined_score\n")
        handle.write("9606.ENSP1\t9606.ENSP2\t800\n")
        handle.write("9606.ENSP2\t9606.ENSP1\t800\n")

    loader = STRINGLoader(tmp_path)
    discovered = loader.discover()
    assert "protein.links.detailed" in discovered

    edge_table = loader.build_edge_table()
    assert len(edge_table) == 1
    assert abs(float(edge_table.loc[0, "combined_score"]) - 0.8) < 1e-6


