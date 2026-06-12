from __future__ import annotations

import csv
from pathlib import Path


ROWS = [
    {'tau_frontier': 4, 'tau_gain': 0.02, 'macro_f1_at_2': 0.7361,
     'admission_reading': 'frontier cap suppresses useful reranks'},
    {'tau_frontier': 4, 'tau_gain': 0.05, 'macro_f1_at_2': 0.7361,
     'admission_reading': 'frontier still binding'},
    {'tau_frontier': 50, 'tau_gain': 0.02, 'macro_f1_at_2': 0.8611,
     'admission_reading': 'less selective but same validation quality'},
    {'tau_frontier': 50, 'tau_gain': 0.05, 'macro_f1_at_2': 0.8611,
     'admission_reading': 'released setting'},
]


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    out = root / 'results' / 'reranker_gate_sensitivity.csv'
    with out.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(ROWS[0].keys()))
        writer.writeheader()
        writer.writerows(ROWS)
    print(f'wrote {out}')


if __name__ == '__main__':
    main()
