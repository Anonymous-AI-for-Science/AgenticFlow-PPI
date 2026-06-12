from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path


def mean(xs):
    return sum(xs) / len(xs) if xs else 0.0


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    results = root / 'results'
    out = results / 'reachability_summary_by_index.csv'
    by_index = defaultdict(lambda: {'entries': [], 'mean_query_us': [], 'build_seconds': []})
    with (results / 'reachability_benchmarks.csv').open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            idx = row['index']
            by_index[idx]['entries'].append(float(row['index_entries']))
            by_index[idx]['mean_query_us'].append(float(row['mean_query_us']))
            by_index[idx]['build_seconds'].append(float(row['build_seconds']))
    with out.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['index', 'avg_entries', 'avg_mean_query_us', 'avg_build_seconds'])
        writer.writeheader()
        for idx, vals in sorted(by_index.items()):
            writer.writerow({
                'index': idx,
                'avg_entries': round(mean(vals['entries']), 3),
                'avg_mean_query_us': round(mean(vals['mean_query_us']), 3),
                'avg_build_seconds': round(mean(vals['build_seconds']), 6),
            })
    meta = {
        'paper_figures': {
            'reachability_tradeoff': 'code/results/reachability_summary_by_index.csv',
            'cost_model_calibration': 'code/results/cost_model_samples.csv',
            'biological_protocol': 'code/results/biological_training_protocol.json',
            'dataset_metrics': 'code/results/dataset_metrics.csv',
            'aorm_bulk': 'code/results/aorm_bulk_comparison.csv',
            'distributed_latency': 'code/results/distributed_latency.csv',
            'coefficient_sensitivity': 'code/results/coefficient_sensitivity.csv',
            'reranker_gate_sensitivity': 'code/results/reranker_gate_sensitivity.csv',
            'core_scaling': 'code/results/core_scaling.csv',
        }
    }
    (results / 'paper_artifact_manifest.json').write_text(json.dumps(meta, indent=2))
    print('wrote derived paper artifacts')


if __name__ == '__main__':
    main()


