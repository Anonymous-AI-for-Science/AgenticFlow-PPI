"""Download the external biological datasets into a local cache.

Usage:
    python scripts/download_external_data.py [--sources string biogrid reactome omnipath]
                                             [--cache DIR] [--force]

Requires network access to the source hosts (stringdb-downloads.org,
downloads.thebiogrid.org, reactome.org, omnipathdb.org). In a site-restricted
environment these hosts are unreachable; the rest of the pipeline then runs on the
bundled fixtures automatically (see agentflow_ppi/data/external/fixtures). This
script reports clearly which sources were fetched and which fell back.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from agentflow_ppi.data.external.download import DEFAULT_CACHE, download_source, DownloadError
from agentflow_ppi.data.external.sources import all_keys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sources", nargs="*", default=all_keys())
    ap.add_argument("--cache", default=str(DEFAULT_CACHE))
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    cache = Path(args.cache)
    print(f"cache dir: {cache}")
    ok, failed = [], []
    for key in args.sources:
        try:
            paths = download_source(key, cache_dir=cache, force=args.force)
            print(f"[ok]   {key}: " + ", ".join(str(p) for p in paths.values()))
            ok.append(key)
        except DownloadError as e:
            print(f"[fail] {key}: {e}", file=sys.stderr)
            failed.append(key)
        except Exception as e:  # noqa: BLE001
            print(f"[fail] {key}: {e}", file=sys.stderr)
            failed.append(key)
    print(f"\ndownloaded: {ok}")
    if failed:
        print(f"unreachable (will use bundled fixtures): {failed}")
        print("This is expected in a site-restricted sandbox. Run on a host with "
              "access to the source databases to obtain the full datasets.")


if __name__ == "__main__":
    main()
