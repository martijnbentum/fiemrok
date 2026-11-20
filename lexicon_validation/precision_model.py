'''Per-query retrieval precision of words and cross-boundary items.

Companion to word_effect: the outcome is a query's precision@10 instead of
pair similarity. The main outcome is cross-speaker precision@10: the
query's same-speaker neighbours are removed from its saved neighbour list
before the top 10 are scored, so types whose tokens come from few speakers
gain nothing from speaker similarity. Ordinary precision@10 is reported
alongside. Each dataset was searched in its own index, so is_word also
includes differences in index size and density; a control shrinks the word
index to the cross-boundary type count. Frequency is left out, as in the
word-effect model.
'''

import json

import numpy as np

from . import factor_models as fm
from . import lv_locations
from . import word_effect as we
from .neighbour_results import Result


OUTCOMES = ('cross_speaker', 'all')
TOP_K = 10
LENGTH_GROUPS = ((2, 3), (4, 5), (6, 7), (8, None))
AGE_GROUPS = ('age_under_30', 'age_50_plus', 'age_unknown')
REFERENCE_COMPONENT = 'a'


def run_precision_model(model='hubert', layer=9, pooling='mean',
    n_bootstrap=200, n_subsamples=20, seed=0,
    workspace=lv_locations.workspace):
    '''Fit the per-query precision model for one model, layer and pooling.

    model:         hubert or w2v
    layer:         3, 6, 9 or 12
    pooling:       center or mean
    n_bootstrap:   type resamples per dataset for confidence intervals
    n_subsamples:  random word-type subsets for the index-size control
    seed:          seed for the bootstrap and the subsets
    workspace:     validation workspace; output goes to factor_models/
    Returns the summary dict, also saved as JSON.
    '''
    parts = [load_queries(condition, model, layer, pooling, workspace)
        for condition in we.CONDITIONS]
    summary = dict(model=model, layer=layer, pooling=pooling, seed=seed,
        n_bootstrap=n_bootstrap)
    for outcome in OUTCOMES:
        summary[outcome] = fit_precision(parts, outcome == 'cross_speaker',
            n_bootstrap, seed)
    summary['by_length'] = word_effect_by_length(parts, n_bootstrap, seed)
    summary['index_size'] = index_size_control(parts, n_subsamples, seed)
    path = lv_locations.precision_model_path(model, layer, pooling, workspace)
    fm._save_text(path, json.dumps(summary, indent=1))
    return summary


def load_queries(condition, model, layer, pooling,
    workspace=lv_locations.workspace):
    '''Return one dataset's features and saved neighbour indices.'''
    filename = lv_locations.neighbour_results_path(model, condition, layer,
        pooling, workspace)
    with Result(filename) as result:
        features = dict(result.features)
        neighbours = result.read(0, len(result))['neighbour_indices']
    types = np.unique(features['label'], return_inverse=True)[1]
    return dict(condition=condition, features=features,
        neighbours=neighbours, types=types)


def precision(features, neighbours, cross_speaker=True, keep=None,
    top_k=TOP_K):
    '''Return per-query precision over the first top_k eligible neighbours.

    features:       token feature table
    neighbours:     saved neighbour indices, one row per query
    cross_speaker:  skip neighbours from the query's own speaker
    keep:           optional token mask; other tokens are skipped
    Returns precision and the number of neighbours scored per query; the
    precision is NaN where no neighbour was eligible.
    '''
    eligible = neighbours >= 0
    safe = np.where(eligible, neighbours, 0)
    if keep is not None: eligible &= keep[safe]
    if cross_speaker:
        speakers = features['speaker']
        eligible &= speakers[safe] != speakers[:, None]
    scored = eligible & (np.cumsum(eligible, axis=1) <= top_k)
    labels = features['label']
    correct = scored & (labels[safe] == labels[:, None])
    n_scored = scored.sum(axis=1)
    with np.errstate(invalid='ignore', divide='ignore'):
        values = correct.sum(axis=1) / n_scored
    return np.where(n_scored > 0, values, np.nan), n_scored


def query_terms(features, types, is_word, components):
    '''Return one dataset's per-query predictors before joining.'''
    n = len(types)
    terms = dict(is_word=np.full(n, float(is_word)),
        female=(features['gender'] == 'female').astype(float),
        n_phones=features['n_phones'].astype(float),
        articulation_rate=features['articulation_rate'].astype(float),
        type_n_speakers=_per_type_unique(types, features['speaker']),
        type_n_recordings=_per_type_unique(types, features['recording']))
    age = features['age']
    terms.update(age_under_30=((age > 0) & (age < 30)).astype(float),
        age_50_plus=(age >= 50).astype(float),
        age_unknown=(age == 0).astype(float))
    for component in components:
        if component == REFERENCE_COMPONENT: continue
        terms[f'component_{component}'] = (features['component']
            == component).astype(float)
    return terms


def _per_type_unique(types, values):
    '''Return, per token, the number of distinct values among its type.'''
    pairs = np.unique(np.column_stack([types, np.unique(values,
        return_inverse=True)[1]]), axis=0)
    counts = np.bincount(pairs[:, 0], minlength=types.max() + 1)
    return counts[types].astype(float)


def joint_design(parts, components=None):
    '''Return per-query design columns for both datasets.

    Continuous predictors are centred on the pooled mean, so is_word is the
    precision difference at average values; a missing articulation rate is
    set to the mean and flagged.
    '''
    if components is None:
        components = sorted(set().union(*(set(p['features']['component'])
            for p in parts)))
    columns = [query_terms(p['features'], p['types'],
        p['condition'] == 'word', components) for p in parts]
    design = {name: np.concatenate([c[name] for c in columns])
        for name in columns[0]}
    rate = design['articulation_rate']
    design['articulation_rate_unknown'] = np.isnan(rate).astype(float)
    design['articulation_rate'] = np.nan_to_num(rate, nan=np.nanmean(rate))
    for name in ('n_phones', 'articulation_rate', 'type_n_speakers',
        'type_n_recordings'):
        design[name] = design[name] - design[name].mean()
    return design


def fit_precision(parts, cross_speaker=True, n_bootstrap=200, seed=0,
    rows=None):
    '''Fit precision on the joint design with type bootstrap intervals.

    rows: optional per-dataset query masks, e.g. a phone-count group
    Queries without an eligible neighbour are left out.
    '''
    outcome = np.concatenate([precision(p['features'], p['neighbours'],
        cross_speaker)[0] for p in parts])
    design = joint_design(parts)
    keep = np.isfinite(outcome)
    if rows is not None: keep &= np.concatenate(rows)
    return _fit(design, outcome, parts, keep, n_bootstrap, seed)


def _fit(design, outcome, parts, keep, n_bootstrap, seed):
    '''Fit OLS on the kept rows; resample query types within datasets.'''
    names = [name for name, values in design.items()
        if values[keep].std() > 0]
    X = np.column_stack([np.ones(keep.sum())] +
        [design[name][keep] for name in names])
    names = ['intercept'] + names
    y = outcome[keep]
    coefficients, r2 = fm.fit_ols(X, y, np.ones(len(y)))
    generator = np.random.default_rng(seed)
    dataset = np.concatenate([np.full(len(p['types']), i)
        for i, p in enumerate(parts)])[keep]
    types = np.concatenate([p['types'] for p in parts])[keep]
    groups = [(dataset == i, types[dataset == i]) for i in range(len(parts))]
    samples = [fm.fit_ols(X, y, we._resample_weights(groups, len(y),
        generator))[0] for _ in range(n_bootstrap)]
    means = {p['condition']: float(np.mean(outcome[_dataset_rows(parts, i)
        & keep])) for i, p in enumerate(parts)}
    return dict(r2=float(r2), n=int(keep.sum()), mean_precision=means,
        coefficients=fm._with_intervals(names, coefficients, samples))


def _dataset_rows(parts, index):
    return np.concatenate([np.full(len(p['types']), i == index)
        for i, p in enumerate(parts)])


def word_effect_by_length(parts, n_bootstrap=200, seed=0):
    '''Return the cross-speaker is_word estimate per phone-count group.'''
    result = {}
    for low, high in LENGTH_GROUPS:
        if high is None: high = np.inf
        rows = [(p['features']['n_phones'] >= low)
            & (p['features']['n_phones'] <= high) for p in parts]
        fit = fit_precision(parts, True, n_bootstrap, seed, rows)
        name = f'{low}+' if high == np.inf else f'{low}-{high}'
        result[name] = dict(n=fit['n'], mean_precision=fit['mean_precision'],
            is_word=fit['coefficients']['is_word'])
    return result


def index_size_control(parts, n_subsamples=20, seed=0):
    '''Refit with the word index shrunk to the cross-boundary type count.

    Each subset drops random word types; word precision is rescored from
    the saved neighbours without the dropped tokens, and their queries are
    left out. Returns the cross-speaker is_word estimate per subset.
    '''
    word = next(p for p in parts if p['condition'] == 'word')
    other = next(p for p in parts if p['condition'] != 'word')
    n_types, n_keep = word['types'].max() + 1, other['types'].max() + 1
    design = joint_design(parts)
    other_outcome = precision(other['features'], other['neighbours'])[0]
    generator = np.random.default_rng(seed)
    estimates = []
    for _ in range(n_subsamples):
        kept_types = np.zeros(n_types, dtype=bool)
        kept_types[generator.choice(n_types, n_keep, replace=False)] = True
        keep = kept_types[word['types']]
        word_outcome = precision(word['features'], word['neighbours'],
            keep=keep)[0]
        outcome = np.concatenate([word_outcome, other_outcome])
        rows = np.concatenate([keep, np.ones(len(other_outcome), bool)])
        rows &= np.isfinite(outcome)
        fit = _fit(design, outcome, parts, rows, 0, seed)
        estimates.append(fit['coefficients']['is_word']['estimate'])
    return dict(n_word_types=int(n_keep), estimates=estimates,
        mean=float(np.mean(estimates)), min=float(np.min(estimates)),
        max=float(np.max(estimates)))


def format_summary(summary):
    '''Return a readable text summary of run_precision_model output.'''
    lines = [f"precision@10, {summary['model']} layer {summary['layer']} "
        f"{summary['pooling']}"]
    for outcome in OUTCOMES:
        fit = summary[outcome]
        means = ', '.join(f'{k} {v:.3f}'
            for k, v in fit['mean_precision'].items())
        lines.append(f"{outcome}: n={fit['n']}, R2={fit['r2']:.3f}, "
            f'mean {means}')
        for name, value in fit['coefficients'].items():
            lines.append(f'  {name:26s} {fm._interval(value)}')
    lines.append('is_word by phone count (cross-speaker):')
    for name, value in summary['by_length'].items():
        lines.append(f"  {name:6s} n={value['n']:7d} "
            f"{fm._interval(value['is_word'])}")
    size = summary['index_size']
    lines.append(f"index-size control ({size['n_word_types']} word types): "
        f"is_word mean {size['mean']:+.4f} "
        f"[{size['min']:+.4f}, {size['max']:+.4f}]")
    return '\n'.join(lines)
