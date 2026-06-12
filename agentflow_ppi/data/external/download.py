"""Robust downloader for external biological datasets.

Downloads each configured source into a local cache directory, with streaming,
gzip/zip handling, resume-safe temp files, retries, and clear offline detection.
If a host is unreachable (e.g. a site-restricted sandbox), the caller can fall
back to the bundled fixtures via `resolve_path(..., allow_fixture=True)`.

Nothing here requires the bio hosts to be reachable at import time; network is
touched only when `download_source` / `download_all` is called.
"""

from __future__ import annotations

import gzip
import io
import os
import shutil
import time
import zipfile
from pathlib import Path
from typing import Dict, List, Optional

from .sources import SOURCES, Source

DEFAULT_CACHE = Path(os.environ.get("AGENTFLOW_DATA_CACHE", str(Path.home() / ".agentflow_ppi" / "external")))
FIXTURE_DIR = Path(__file__).resolve().parent / "fixtures"


class DownloadError(RuntimeError):
    pass


def _http_get(url: str, timeout: int = 60, retries: int = 3) -> bytes:
    import requests  # imported lazily so offline imports never fail
    last = None
    for attempt in range(retries):
        try:
            r = requests.get(url, timeout=timeout, stream=True,
                             headers={"User-Agent": "AgentFlow-PPI-artifact/1.0"})
            r.raise_for_status()
            return r.content
        except Exception as e:  # noqa: BLE001 - surface a uniform error
            last = e
            time.sleep(1.5 * (attempt + 1))
    raise DownloadError(f"failed to GET {url}: {last}")


def _maybe_decompress(name: str, raw: bytes) -> bytes:
    if name.endswith(".gz"):
        return gzip.decompress(raw)
    if name.endswith(".zip"):
        zf = zipfile.ZipFile(io.BytesIO(raw))
        # pick the largest .txt/.tab3 member (the organism table)
        members = [m for m in zf.namelist() if m.lower().endswith((".txt", ".tab3.txt", ".tab3"))]
        if not members:
            members = zf.namelist()
        member = max(members, key=lambda m: zf.getinfo(m).file_size)
        return zf.read(member)
    return raw


def download_source(key: str, cache_dir: Path = DEFAULT_CACHE, force: bool = False) -> Dict[str, Path]:
    """Download all files of one source into cache_dir. Returns logical name -> path.

    Raises DownloadError if the host is unreachable; callers may then fall back to
    fixtures. Decompresses .gz and extracts the main table from .zip automatically.
    """
    src: Source = SOURCES[key]
    cache_dir = Path(cache_dir) / key
    cache_dir.mkdir(parents=True, exist_ok=True)
    out: Dict[str, Path] = {}
    for logical, url in src.urls.items():
        final_name = src.filenames[logical]
        dest = cache_dir / final_name
        if dest.exists() and not force:
            out[logical] = dest
            continue
        raw = _http_get(url)
        # the URL may end in .gz/.zip while the stored filename is the decompressed table
        comp_hint = url.rsplit("/", 1)[-1]
        data = _maybe_decompress(comp_hint, raw)
        tmp = dest.with_suffix(dest.suffix + ".part")
        tmp.write_bytes(data)
        tmp.replace(dest)
        out[logical] = dest
    return out


def download_all(keys: Optional[List[str]] = None, cache_dir: Path = DEFAULT_CACHE,
                 force: bool = False) -> Dict[str, Dict[str, Path]]:
    keys = keys or list(SOURCES.keys())
    results: Dict[str, Dict[str, Path]] = {}
    for k in keys:
        results[k] = download_source(k, cache_dir=cache_dir, force=force)
    return results


def resolve_path(key: str, logical: str, cache_dir: Path = DEFAULT_CACHE,
                 allow_fixture: bool = True) -> Path:
    """Return the path to a source file: the downloaded cache copy if present,
    otherwise the bundled fixture (when allow_fixture). Raises if neither exists."""
    src = SOURCES[key]
    dest = Path(cache_dir) / key / src.filenames[logical]
    if dest.exists():
        return dest
    if allow_fixture and logical in src.fixture:
        fx = FIXTURE_DIR / src.fixture[logical]
        if fx.exists():
            return fx
    raise FileNotFoundError(
        f"{key}:{logical} not found in cache ({dest}) and no fixture available; "
        f"run scripts/download_external_data.py with network access to the source host."
    )


def is_using_fixture(key: str, logical: str, cache_dir: Path = DEFAULT_CACHE) -> bool:
    dest = Path(cache_dir) / key / SOURCES[key].filenames[logical]
    return not dest.exists()
