'''Compare mean-pooled retrieval scores and failure rates across datasets.'''

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import bootstrap, spearmanr


ROOT = Path(__file__).resolve().parents[1]
LAYERS = ('3', '6', '9', '12')
DATASETS = {'word': ('cgn_lexicon', 2466),
    'cross_boundary': ('cross_boundary_lexicon100_2200', 2200)}
MODELS = {'w2v': '6', 'hubert1': '9'}


def load_scores(directory):
    '''Validate arrays and align type order across models and layers.'''
    scores = {}
    labels = {}
    for dataset, (prefix, count) in DATASETS.items():
        for model in MODELS:
            path = directory / f'{prefix}_{model}_mean_results.json'
            results = json.loads(path.read_text())
            if set(results) != set(LAYERS):
                raise ValueError(f'unexpected layers in {path}')
            for layer in LAYERS:
                keys = sorted(results[layer])
                if dataset in labels and keys != labels[dataset]:
                    raise ValueError(f'type identities differ in {path}')
                labels[dataset] = keys
                values = np.asarray([results[layer][k] for k in keys])
                if values.shape != (count, 100):
                    raise ValueError(f'unexpected score counts in {path}')
                valid = np.isfinite(values) & (values >= 0) & (values <= 1)
                if not valid.all():
                    raise ValueError(f'invalid scores in {path}')
                if not np.allclose(values * 10, np.round(values * 10)):
                    raise ValueError(f'scores are not P@10 in {path}')
                scores[dataset, model, layer] = values
    return scores, labels


def mean_difference(word, cross_boundary, axis=-1):
    '''Return word minus cross-boundary mean retrieval precision.'''
    return word.mean(axis=axis) - cross_boundary.mean(axis=axis)


def failure_profile(values, labels):
    '''Describe score severity without inferring neighbour identities.'''
    means = values.mean(axis=1)
    order = np.argsort(means, kind='stable')[:10]
    return dict(mean_pct=float(means.mean() * 100),
        type_quartiles_pct=(np.quantile(means, [.25, .5, .75]) * 100).tolist(),
        zero_match_queries_pct=float((values == 0).mean() * 100),
        perfect_queries_pct=float((values == 1).mean() * 100),
        types_below_50_pct=float((means < .5).mean() * 100),
        token_score_histogram=np.bincount(np.rint(values.ravel() * 10)
            .astype(int), minlength=11).tolist(),
        hardest_types=[dict(label=labels[i], mean_pct=float(means[i] * 100))
            for i in order])


def compare(scores, labels, n_resamples):
    '''Bootstrap independent type samples; retain descriptive diagnostics.'''
    gaps = []
    for model in MODELS:
        for layer in LAYERS:
            word = scores['word', model, layer].mean(axis=1)
            cross = scores['cross_boundary', model, layer].mean(axis=1)
            interval = bootstrap((word, cross), mean_difference,
                paired=False, vectorized=True, method='BCa', batch=128,
                n_resamples=n_resamples, rng=np.random.default_rng(20261029))
            gaps.append(dict(model=model, layer=int(layer),
                word_pct=float(word.mean() * 100),
                cross_boundary_pct=float(cross.mean() * 100),
                gap_pp=float(mean_difference(word, cross) * 100),
                ci95_pp=[float(x * 100) for x in interval.confidence_interval]))
    profiles = {}
    agreement = {}
    for dataset in DATASETS:
        profiles[dataset] = {model: failure_profile(
            scores[dataset, model, layer], labels[dataset])
            for model, layer in MODELS.items()}
        w2v = scores[dataset, 'w2v', '6'].mean(axis=1)
        hubert = scores[dataset, 'hubert1', '9'].mean(axis=1)
        agreement[dataset] = dict(type_spearman_r=float(
            spearmanr(w2v, hubert).statistic))
    return dict(n_resamples=n_resamples, seed=20261029,
        gap_definition='word minus cross-boundary, percentage points',
        bootstrap='independent types; fixed tokens and retrieval indexes',
        gaps=gaps, best_tested_layer_profiles=profiles,
        best_tested_layer_type_agreement=agreement)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, default=ROOT / 'svs_data')
    parser.add_argument('--output', type=Path,
        default=ROOT / 'data/mean_pooling_dataset_comparison.json')
    parser.add_argument('--n-resamples', type=int, default=10000)
    args = parser.parse_args()
    if args.n_resamples < 2:
        parser.error('--n-resamples must be at least 2')
    scores, labels = load_scores(args.data_dir)
    result = compare(scores, labels, args.n_resamples)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False)
        + '\n')
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
