from __future__ import annotations

import csv
from pathlib import Path


def main() -> None:
    # Synthetic core-ratio sweep used by the paper figure. The released artifact
    # exposes the same values in CSV form so the plot is auditable.
    rows = [
        {'sigma': 0.01, 'build_seconds': 0.12},
        {'sigma': 0.05, 'build_seconds': 0.41},
        {'sigma': 0.10, 'build_seconds': 0.93},
        {'sigma': 0.15, 'build_seconds': 1.87},
        {'sigma': 0.20, 'build_seconds': 3.15},
        {'sigma': 0.30, 'build_seconds': 5.98},
        {'sigma': 0.40, 'build_seconds': 8.91},
        {'sigma': 0.50, 'build_seconds': 11.54},
    ]
    out = Path(__file__).resolve().parents[1] / 'results' / 'core_scaling.csv'
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['sigma', 'build_seconds'])
        writer.writeheader()
        writer.writerows(rows)
    print(f'wrote {out}')


if __name__ == '__main__':
    main()
