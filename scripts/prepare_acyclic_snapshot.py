from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from agentflow_ppi.data.cycle_handling import condense_to_dag


def main() -> None:
    parser = argparse.ArgumentParser(description='Condense a directed edge list into an acyclic snapshot.')
    parser.add_argument('--edges', type=Path, required=True, help='TSV file with source and target columns')
    parser.add_argument('--out-dir', type=Path, required=True)
    args = parser.parse_args()

    nodes = {}
    edges = []
    with args.edges.open() as f:
        reader = csv.DictReader(f, delimiter='\t')
        for row in reader:
            u = row['source']
            v = row['target']
            uid = nodes.setdefault(u, len(nodes))
            vid = nodes.setdefault(v, len(nodes))
            edges.append((uid, vid))

    result = condense_to_dag(len(nodes), edges)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        'num_original_nodes': result.num_original_nodes,
        'num_components': result.num_components,
        'num_dag_edges': len(result.dag_edges),
        'num_removed_intra_component_edges': len(result.removed_intra_component_edges),
        'component_sizes': result.component_sizes,
    }
    (args.out_dir / 'cycle_manifest.json').write_text(json.dumps(manifest, indent=2))
    with (args.out_dir / 'condensed_edges.tsv').open('w', newline='') as f:
        writer = csv.writer(f, delimiter='\t')
        writer.writerow(['source_component', 'target_component'])
        writer.writerows(result.dag_edges)
    print('wrote condensed DAG and cycle manifest')


if __name__ == '__main__':
    main()
