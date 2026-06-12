from __future__ import annotations

import csv
from pathlib import Path


def main() -> None:
    rows = [
        {"method": "SHRC-exact", "workload": "10k-node core", "bulk_latency_ms": 61.2, "point_query_mean_us": 3.4, "index_entries": 48216},
        {"method": "AORM-inspired bulk", "workload": "10k-node core", "bulk_latency_ms": 18.6, "point_query_mean_us": 27.9, "index_entries": 100000000},
        {"method": "AORM-inspired point-only", "workload": "10k-node core", "bulk_latency_ms": 25.4, "point_query_mean_us": 19.7, "index_entries": 100000000},
    ]
    out = Path(__file__).resolve().parents[1] / "results" / "aorm_bulk_comparison.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
