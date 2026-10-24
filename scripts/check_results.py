"""Validate the compact repository's recorded metrics without saved models."""
from collections import defaultdict
import csv
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RUNS = {
    'baseline_v2': 160,
    'persistence_1_routing_fix': 160,
    'threshold_0_v2': 160,
    'threshold_04_v2': 160,
    'neural_mnist_v1': 50,
    'neural_mnist_stress_v1': 25,
}


def read(path):
    with path.open(encoding='utf-8-sig', newline='') as stream:
        return list(csv.DictReader(stream))


def main():
    for name, count in RUNS.items():
        folder = ROOT / 'results' / name
        manifest = json.loads((folder / 'manifest.json').read_text())
        assert manifest['status'] == 'completed'
        assert manifest['completed_runs'] == count
        neural = name.startswith('neural_')
        filename = 'run_neural.py' if neural else 'run_experiment.py'
        source = ROOT / 'experiments' / ('neural' if neural else '') / filename
        digest = hashlib.sha256(source.read_bytes()).hexdigest()
        assert digest == manifest['source_sha256' if neural else 'code_sha256']
        assert digest == hashlib.sha256((folder / filename).read_bytes()).hexdigest()
        rows = read(folder / 'per_run.csv')
        assert len(rows) == count
        key = lambda row: (row['scenario'], row['policy'], row['seed'])
        assert len({key(row) for row in rows}) == count
        endpoints = defaultdict(list)
        for row in read(folder / 'accuracy_matrix.csv'):
            endpoints[key(row)].append(row)
        hashes = defaultdict(set)
        groups = defaultdict(list)
        for row in rows:
            points = endpoints[key(row)]
            last = max(int(point['segment']) for point in points)
            final = {point['test_domain']: float(point['accuracy'])
                     for point in points if int(point['segment']) == last}
            accuracy = np.mean(list(final.values()))
            forgetting = np.mean([
                max(float(point['accuracy']) for point in points
                    if point['test_domain'] == domain) - value
                for domain, value in final.items()
            ])
            assert abs(accuracy - float(row['final_accuracy'])) < 1e-12
            assert abs(forgetting - float(row['forgetting'])) < 1e-12
            hashes[(row['scenario'], row['seed'])].add(row['data_sha256'])
            groups[(row['scenario'], row['policy'])].append(row)
        assert all(len(values) == 1 for values in hashes.values())
        for summary in read(folder / 'summary.csv'):
            members = groups[(summary['scenario'], summary['policy'])]
            assert len(members) == int(summary['n'] if neural else summary['n_seeds'])
            for metric in ('final_accuracy', 'forgetting', 'experts'):
                values = np.array([float(row[metric]) for row in members])
                assert abs(values.mean() - float(summary[metric + '_mean'])) < 1e-12
                assert abs(values.std(ddof=1) - float(summary[metric + '_sd'])) < 1e-12
        print(f'{name}: {count} recorded runs verified')
    print('PASS: source hashes, matched data hashes, endpoint metrics and summaries')


if __name__ == '__main__':
    main()
