import argparse
import json
import random
from pathlib import Path
from statistics import fmean


def _read(directory: Path) -> tuple[str, dict[int, dict]]:
    metadata = json.loads((directory / 'config.json').read_text())
    rankings = json.loads((directory / 'development.json').read_text())
    diseases = {row['disease']: row for row in rankings['per_disease']}
    if len(diseases) != len(rankings['per_disease']):
        raise ValueError(f'{directory}: duplicate diseases')
    return metadata['benchmark_sha256'], diseases


def compare(reference: Path, candidate: Path, *, seed: int, samples: int) -> dict:
    left_sha, left = _read(reference)
    right_sha, right = _read(candidate)
    if left_sha != right_sha or left.keys() != right.keys():
        raise ValueError('compare: incompatible disease benchmark or evaluation population')
    if samples < 100:
        raise ValueError('compare: use at least 100 bootstrap samples')

    diseases = sorted(left)
    random_generator = random.Random(seed)
    comparison = {}
    for metric in ('recall', 'ndcg', 'reciprocal_rank'):
        diffs = [right[d][metric] - left[d][metric] for d in diseases]
        bootstraps = sorted(
            fmean(random_generator.choices(diffs, k=len(diffs)))
            for _ in range(samples)
        )
        comparison[metric] = {
            'reference': fmean(left[d][metric] for d in diseases),
            'candidate': fmean(right[d][metric] for d in diseases),
            'paired_difference': fmean(diffs),
            'bootstrap_95_interval': [
                bootstraps[int(0.025 * samples)],
                bootstraps[min(samples - 1, int(0.975 * samples))],
            ],
            'improved_diseases': sum(value > 0 for value in diffs),
            'declined_diseases': sum(value < 0 for value in diffs),
            'unchanged_diseases': sum(value == 0 for value in diffs),
        }
    return {
        'benchmark_sha256': left_sha,
        'development_diseases': len(diseases),
        'bootstrap_seed': seed,
        'bootstrap_replicates': samples,
        'note': 'Intervals reflect disease resampling on this selected development set; '
                'they do not adjust for checkpoint selection or optimization seeds.',
        'metrics': comparison,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('reference', type=Path)
    parser.add_argument('candidate', type=Path)
    parser.add_argument('--seed', type=int, default=163)
    parser.add_argument('--samples', type=int, default=2000)
    args = parser.parse_args()
    result = compare(args.reference, args.candidate, seed=args.seed, samples=args.samples)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
