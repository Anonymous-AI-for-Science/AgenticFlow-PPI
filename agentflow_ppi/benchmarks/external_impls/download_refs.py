"""Fetch the published reference implementations of the reachability baselines.

The in-tree baselines (agentflow_ppi/benchmarks/strong.py) are faithful, exactness-
checked reimplementations. For users who want to run the ORIGINAL authors' code,
this module records the canonical public source repositories and provides a
downloader that clones/extracts them and prints build instructions. github.com and
raw.githubusercontent.com are typically reachable even in restricted environments,
so these downloads usually succeed where the biological-database hosts do not.

We do not vendor third-party source into the artifact (licensing); we fetch it on
demand into a local directory the user controls.
"""

from __future__ import annotations

import io
import os
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

DEFAULT_DIR = Path(os.environ.get("AGENTFLOW_BASELINE_SRC",
                                  str(Path.home() / ".agentflow_ppi" / "baseline_src")))


@dataclass(frozen=True)
class RefImpl:
    key: str
    repo: str              # owner/name on GitHub
    branches: tuple        # candidate default branches to try
    build: str             # one-line build hint
    note: str

    def zip_urls(self) -> List[str]:
        return [f"https://codeload.github.com/{self.repo}/zip/refs/heads/{b}"
                for b in self.branches]


# Canonical public sources. These are the original authors' repositories (or the
# closest faithful public implementation where the authors released no code).
REF_IMPLS: Dict[str, RefImpl] = {
    "grail": RefImpl(
        key="grail",
        repo="zakimjz/grail",
        branches=("master", "main"),
        build="cd grail-* && make    # builds the GRAIL binary (C++)",
        note="GRAIL (Yildirim, Chaoji, Zaki, VLDB 2010) reference C++ source.",
    ),
    "pll": RefImpl(
        key="pll",
        repo="iwiwi/pruned-landmark-labeling",
        branches=("master", "main"),
        build="cd pruned-landmark-labeling-* && make    # PLL reference (C++)",
        note="Pruned Landmark Labeling (Akiba, Iwata, Yoshida, SIGMOD 2013) reference "
             "C++ source (2-hop cover), released by the first author.",
    ),
    "oreach": RefImpl(
        key="oreach",
        repo="KaHIP/oreach",
        branches=("main", "master"),
        build="see repo README; CMake build of the O'Reach binary (C++)",
        note="O'Reach (Hanauer, Schulz, et al.) single-source reachability; closest "
             "available successor to PReaCH. If unavailable, the in-tree PReaChIndex "
             "provides the faithful filter-based baseline.",
    ),
}


def download_ref_impl(key: str, dest_dir: Path = DEFAULT_DIR) -> Path:
    """Download and extract one reference implementation, trying each candidate
    branch. Returns the extracted dir. Raises RuntimeError if all attempts fail."""
    import requests
    impl = REF_IMPLS[key]
    dest_dir = Path(dest_dir); dest_dir.mkdir(parents=True, exist_ok=True)
    last = None
    for url in impl.zip_urls():
        try:
            r = requests.get(url, timeout=60, headers={"User-Agent": "AgentFlow-PPI-artifact/1.0"})
            r.raise_for_status()
            zf = zipfile.ZipFile(io.BytesIO(r.content))
            zf.extractall(dest_dir)
            roots = {n.split("/")[0] for n in zf.namelist()}
            return dest_dir / (sorted(roots)[0] if roots else key)
        except Exception as e:  # noqa: BLE001
            last = e
    raise RuntimeError(f"cannot download {key} from any of {impl.zip_urls()}: {last}")


def print_manifest() -> None:
    for k, impl in REF_IMPLS.items():
        print(f"[{k}] {impl.note}\n      repo:  github.com/{impl.repo}\n      build: {impl.build}")
