"""Version-aware STRING data loader.

The implementation is designed for modern STRING v12.x style exports, but it is
header-driven and therefore resilient to minor schema evolution. Large link files
are processed in chunks so the caller can scale to multi-gigabyte archives.

Supported families include:
- protein.links*.txt.gz
- protein.physical.links*.txt.gz
- protein.info*.txt.gz
- protein.aliases*.txt.gz
- protein.sequences*.fa.gz
- protein.network.embeddings*.h5
- protein.sequence.embeddings*.h5

The loader deliberately separates discovery, normalization, and graph
materialization so that experiments remain reproducible and easy to audit.
"""

from __future__ import annotations

import csv
import gzip
import hashlib
import io
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Mapping, MutableMapping, Optional, Sequence

import h5py
import numpy as np
import pandas as pd
import torch


_LINK_FAMILY_RE = re.compile(
    r"^(?:(?P<species>\d+)\.)?(?P<family>protein(?:\.physical)?\.links(?:\.(?:detailed|full))?)\.v(?P<version>[\d.]+)\.txt(?:\.gz)?$"
)
_INFO_FAMILY_RE = re.compile(
    r"^(?:(?P<species>\d+)\.)?(?P<family>protein\.(?:info|aliases|orthology|enrichment\.terms|homology|clusters\.proteins|clusters\.info|clusters\.tree))\.v(?P<version>[\d.]+)\.txt(?:\.gz)?$"
)
_FASTA_FAMILY_RE = re.compile(
    r"^(?:(?P<species>\d+)\.)?(?P<family>protein\.sequences)\.v(?P<version>[\d.]+)\.fa(?:\.gz)?$"
)
_H5_FAMILY_RE = re.compile(
    r"^(?:(?P<species>\d+)\.)?(?P<family>protein\.(?:network|sequence)\.embeddings)\.v(?P<version>[\d.]+)\.h5$"
)


@dataclass(slots=True)
class STRINGFileInfo:
    """Parsed metadata derived from a STRING file name."""

    path: Path
    family: str
    version: str
    species: Optional[str]


@dataclass(slots=True)
class STRINGLoadConfig:
    """Controls memory usage and normalization policy.

    Attributes
    ----------
    chunk_rows:
        Number of rows loaded per chunk for large tabular files.
    min_combined_score:
        Interactions below this normalized threshold are filtered early.
    deduplicate_undirected:
        Canonicalize symmetric pairs such that (u, v) and (v, u) collapse.
    keep_channel_columns:
        Preserve modality-specific subscore columns when present.
    compute_checksums:
        Emit SHA256 checksums in the manifest for provenance.
    """

    chunk_rows: int = 250_000
    min_combined_score: float = 0.0
    deduplicate_undirected: bool = True
    keep_channel_columns: bool = True
    compute_checksums: bool = False


@dataclass(slots=True)
class STRINGManifest:
    """Captures the exact graph snapshot used by an experiment."""

    version: Optional[str] = None
    files: Dict[str, str] = field(default_factory=dict)
    checksums: Dict[str, str] = field(default_factory=dict)
    normalization: Dict[str, object] = field(default_factory=dict)

    def to_json(self) -> str:
        return json.dumps(
            {
                "version": self.version,
                "files": self.files,
                "checksums": self.checksums,
                "normalization": self.normalization,
            },
            indent=2,
            sort_keys=True,
        )


class LazyH5Embeddings:
    """Thin lazy adapter around HDF5 embeddings.

    The adapter opens the file on demand and returns vectors only when requested,
    which avoids loading an entire embedding matrix into RAM.
    """

    def __init__(self, path: Path) -> None:
        self.path = path

    def keys(self) -> List[str]:
        with h5py.File(self.path, "r") as handle:
            return list(handle.keys())

    def get(self, protein_id: str) -> Optional[np.ndarray]:
        with h5py.File(self.path, "r") as handle:
            if protein_id in handle:
                return np.asarray(handle[protein_id])
            return None


class STRINGLoader:
    """Efficient loader for STRING exports.

    The class discovers relevant files under a root directory, parses them using
    bounded memory, and exposes both chunk-wise table iterators and convenience
    methods for building a compact PyG graph.
    """

    def __init__(self, root: str | Path, config: Optional[STRINGLoadConfig] = None) -> None:
        self.root = Path(root)
        self.config = config or STRINGLoadConfig()
        self.manifest = STRINGManifest(
            normalization={
                "score_scale": "divide_by_1000",
                "deduplicate_undirected": self.config.deduplicate_undirected,
                "min_combined_score": self.config.min_combined_score,
            }
        )

    # ------------------------------------------------------------------
    # Discovery utilities
    # ------------------------------------------------------------------
    def discover(self) -> Dict[str, STRINGFileInfo]:
        """Discover STRING files and return them keyed by family.

        The method is version-aware but not version-restrictive: it accepts any
        v12.x-style filename and records the observed version in the manifest.
        """
        discovered: Dict[str, STRINGFileInfo] = {}
        for path in sorted(self.root.iterdir()):
            if not path.is_file():
                continue
            info = self._parse_file_name(path.name)
            if info is None:
                continue
            discovered[info.family] = info
            self.manifest.files[info.family] = path.name
            self.manifest.version = info.version if self.manifest.version is None else self.manifest.version
            if self.config.compute_checksums:
                self.manifest.checksums[path.name] = self._sha256(path)
        return discovered

    def _parse_file_name(self, name: str) -> Optional[STRINGFileInfo]:
        for pattern in (_LINK_FAMILY_RE, _INFO_FAMILY_RE, _FASTA_FAMILY_RE, _H5_FAMILY_RE):
            match = pattern.match(name)
            if match:
                return STRINGFileInfo(
                    path=self.root / name,
                    family=match.group("family"),
                    version=match.group("version"),
                    species=match.groupdict().get("species"),
                )
        return None

    @staticmethod
    def _sha256(path: Path, block_size: int = 1 << 20) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            while True:
                block = handle.read(block_size)
                if not block:
                    break
                digest.update(block)
        return digest.hexdigest()

    # ------------------------------------------------------------------
    # Generic file access helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _open_text(path: Path):
        if path.suffix == ".gz":
            return gzip.open(path, "rt", encoding="utf-8", newline="")
        return path.open("rt", encoding="utf-8", newline="")

    @staticmethod
    def sniff_delimiter(path: Path) -> str:
        """Detect tab, comma, or space-delimited STRING tables.

        The official download page allows TXT, TSV, and CSV exports for links
        files, so the loader detects the delimiter rather than assuming TSV.
        """
        with STRINGLoader._open_text(path) as handle:
            sample = handle.readline()
        if "\t" in sample:
            return "\t"
        if "," in sample:
            return ","
        return " "

    def iter_table_chunks(
        self,
        path: Path,
        usecols: Optional[Sequence[str]] = None,
        dtype: Optional[Mapping[str, str]] = None,
    ) -> Iterator[pd.DataFrame]:
        """Yield pandas chunks from a large STRING table.

        This is the main memory-bounded ingestion primitive used by links,
        aliases, and protein metadata loaders.
        """
        delimiter = self.sniff_delimiter(path)
        read_kwargs = dict(
            compression="infer",
            chunksize=self.config.chunk_rows,
            usecols=list(usecols) if usecols else None,
            dtype=dtype,
        )
        if delimiter == " ":
            iterator = pd.read_csv(path, sep=r"\s+", engine="python", **read_kwargs)
        else:
            iterator = pd.read_csv(path, sep=delimiter, engine="c", **read_kwargs)
        for chunk in iterator:
            chunk.columns = [str(col).lstrip("#").strip() for col in chunk.columns]
            yield chunk

    # ------------------------------------------------------------------
    # Links and metadata loaders
    # ------------------------------------------------------------------
    def load_links(self, family: str = "protein.links.detailed") -> Iterator[pd.DataFrame]:
        """Yield normalized edge chunks from a STRING links family.

        Parameters
        ----------
        family:
            One of the discovered families such as `protein.links.detailed`,
            `protein.links.full`, or `protein.physical.links.detailed`.
        """
        discovered = self.discover()
        if family not in discovered:
            fallback = next((key for key in discovered if key.startswith("protein.links")), None)
            if fallback is None:
                raise FileNotFoundError(f"No STRING links file found under {self.root}")
            family = fallback
        info = discovered[family]

        dtype = {
            "protein1": "string",
            "protein2": "string",
            "combined_score": "float32",
        }
        for chunk in self.iter_table_chunks(info.path, dtype=dtype):
            normalized = self._normalize_links_chunk(chunk, family=family)
            if not normalized.empty:
                yield normalized

    def _normalize_links_chunk(self, chunk: pd.DataFrame, family: str) -> pd.DataFrame:
        """Normalize a link chunk in-place and return a filtered frame.

        The operation is intentionally vectorized because it runs on every edge
        chunk and therefore dominates ingestion cost for large exports.
        """
        out = chunk.copy()
        if "combined_score" in out.columns:
            out["combined_score"] = out["combined_score"].astype("float32") / 1000.0
            out = out[out["combined_score"] >= self.config.min_combined_score]

        directed = "regulatory" in family
        if self.config.deduplicate_undirected and not directed and {"protein1", "protein2"}.issubset(out.columns):
            left = out["protein1"].astype("string")
            right = out["protein2"].astype("string")
            ordered_left = left.where(left <= right, right)
            ordered_right = right.where(left <= right, left)
            out.loc[:, "protein1"] = ordered_left
            out.loc[:, "protein2"] = ordered_right
            out = out.drop_duplicates(subset=["protein1", "protein2"])

        if not self.config.keep_channel_columns:
            keep = [col for col in ["protein1", "protein2", "combined_score"] if col in out.columns]
            out = out[keep]
        out["edge_family"] = family
        return out.reset_index(drop=True)

    def load_protein_info(self) -> pd.DataFrame:
        discovered = self.discover()
        for family in discovered:
            if family == "protein.info":
                frame = pd.concat(self.iter_table_chunks(discovered[family].path), ignore_index=True)
                frame.columns = [str(col).lstrip("#").strip() for col in frame.columns]
                return frame
        raise FileNotFoundError("No protein.info file found")

    def load_aliases(self) -> Dict[str, List[str]]:
        """Load aliases into a multimap keyed by STRING protein identifier."""
        discovered = self.discover()
        if "protein.aliases" not in discovered:
            return {}
        alias_map: Dict[str, List[str]] = {}
        path = discovered["protein.aliases"].path
        for chunk in self.iter_table_chunks(path):
            protein_col = None
            for candidate in ("string_protein_id", "protein_id", "protein_external_id"):
                if candidate in chunk.columns:
                    protein_col = candidate
                    break
            if protein_col is None or "alias" not in chunk.columns:
                continue
            for protein_id, alias in zip(chunk[protein_col], chunk["alias"]):
                alias_map.setdefault(str(protein_id), []).append(str(alias))
        return alias_map

    def load_sequences(self, limit: Optional[int] = None) -> Dict[str, str]:
        """Load protein sequences from FASTA, optionally truncated for testing."""
        discovered = self.discover()
        if "protein.sequences" not in discovered:
            return {}
        sequences: Dict[str, str] = {}
        current_id: Optional[str] = None
        path = discovered["protein.sequences"].path
        with self._open_text(path) as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                if line.startswith(">"):  # FASTA header
                    current_id = line[1:].split()[0]
                    sequences[current_id] = ""
                    if limit is not None and len(sequences) > limit:
                        break
                elif current_id is not None:
                    sequences[current_id] += line
        return sequences

    def load_embeddings(self, family: str) -> Optional[LazyH5Embeddings]:
        discovered = self.discover()
        if family not in discovered:
            return None
        return LazyH5Embeddings(discovered[family].path)

    # ------------------------------------------------------------------
    # Graph materialization helpers
    # ------------------------------------------------------------------
    def build_edge_table(self, family: str = "protein.links.detailed", max_chunks: Optional[int] = None) -> pd.DataFrame:
        """Materialize normalized edge chunks into a single compact table.

        This method is convenient for small experiments and unit tests. Large
        production workflows should prefer the chunk iterator exposed by
        `load_links()`.
        """
        chunks: List[pd.DataFrame] = []
        for idx, chunk in enumerate(self.load_links(family=family)):
            chunks.append(chunk)
            if max_chunks is not None and idx + 1 >= max_chunks:
                break
        if not chunks:
            return pd.DataFrame(columns=["protein1", "protein2", "combined_score", "edge_family"])
        return pd.concat(chunks, ignore_index=True)

    def build_pyg_graph(
        self,
        family: str = "protein.links.detailed",
        max_chunks: Optional[int] = 1,
        sequence_limit: int = 5000,
    ) -> "Data":
        """Build a compact PyG graph for rapid prototyping.

        The graph uses a small engineered feature space so that the code remains
        executable without requiring full STRING embeddings. When embedding files
        are available, callers can extend this routine by fetching vectors lazily.
        """
        edges = self.build_edge_table(family=family, max_chunks=max_chunks)
        if edges.empty:
            raise RuntimeError("No edges were loaded; verify the input directory and score threshold")

        proteins = pd.Index(pd.unique(pd.concat([edges["protein1"], edges["protein2"]], ignore_index=True)))
        node_to_idx = {protein: idx for idx, protein in enumerate(proteins)}

        edge_index = torch.tensor(
            [
                edges["protein1"].map(node_to_idx).to_numpy(dtype=np.int64),
                edges["protein2"].map(node_to_idx).to_numpy(dtype=np.int64),
            ],
            dtype=torch.long,
        )

        degree = np.zeros(len(proteins), dtype=np.float32)
        np.add.at(degree, edge_index[0].numpy(), 1.0)
        np.add.at(degree, edge_index[1].numpy(), 1.0)

        sequences = self.load_sequences(limit=sequence_limit)
        seq_lengths = np.array([len(sequences.get(pid, "")) for pid in proteins], dtype=np.float32)
        confidence = edges.groupby("protein1")["combined_score"].mean().reindex(proteins, fill_value=0.0).to_numpy(dtype=np.float32)

        x = np.stack(
            [
                degree,
                seq_lengths,
                confidence,
            ],
            axis=1,
        )
        from torch_geometric.data import Data

        return Data(
            x=torch.tensor(x, dtype=torch.float32),
            edge_index=edge_index,
            edge_weight=torch.tensor(edges["combined_score"].to_numpy(dtype=np.float32)),
        )

    def write_manifest(self, path: str | Path) -> None:
        path = Path(path)
        path.write_text(self.manifest.to_json(), encoding="utf-8")


