'''Model pairwise embedding similarity from word identity and other factors.

Pairs are sampled once per dataset on predictors only, never on similarity,
so one sample serves every model, layer and pooling condition. The saved
sample belongs to the saved token selection; to resample, call
sample_pairs and save_pairs explicitly. The linear
model gives the word-identity effect with the other factors held constant;
matched contrasts compare same-word and different-word pairs directly.
Confidence intervals come from resampling query types.
'''

import json
from pathlib import Path
import tempfile

import numpy as np

from . import frequencies
from . import lv_locations
from . import pair_features as pf
from .neighbour_results import Result


STRATA = ('same_word', 'same_word_same_speaker', 'different_word',
    'different_word_same_speaker', 'close_phones')


def run_pair_models(condition, model='hubert', layer=9, pooling='mean',
    n_pairs=100_000, n_bootstrap=200, seed=0,
    workspace=lv_locations.workspace):
    '''Fit the pair similarity models for one retrieval condition.

    condition:    word or cross_boundary
    model:        hubert or w2v
    layer:        3, 6, 9 or 12
    pooling:      center or mean
    n_pairs:      pairs per stratum when the dataset has no saved sample
    n_bootstrap:  type resamples for confidence intervals
    seed:         seed for sampling a missing sample and for the bootstrap
    workspace:    validation workspace; output goes to factor_models/
    Returns the summary dict, also saved as JSON. The summary records the
    n_pairs and seed of the pair sample actually used.
    '''
    path = lv_locations.pair_model_path(model, condition, layer, pooling,
        workspace)
    filename = lv_locations.neighbour_results_path(model, condition, layer,
        pooling, workspace)
    with Result(filename) as result:
        features = dict(result.features)
        features.update(frequencies.frequency_columns(features,
            frequencies.load_frequencies(workspace)))
        pairs = load_pairs(condition, features, n_pairs, seed, workspace)
        vectors = np.array(result.vectors)
        similarity = cosine(vectors, pairs['queries'], pairs['others'])
    design = pair_design(features, pairs['queries'], pairs['others'])
    types = np.unique(features['label'], return_inverse=True)[1]
    summary = fit_pair_models(design, similarity, pairs['stratum'],
        types[pairs['queries']], n_bootstrap, seed)
    summary.update(model=model, condition=condition, layer=layer,
        pooling=pooling, n_pairs=int(pairs['n_pairs']),
        pair_seed=int(pairs['seed']), n_bootstrap=n_bootstrap, seed=seed,
        stratum_counts=dict(zip(STRATA, np.bincount(pairs['stratum'],
            minlength=len(STRATA)).tolist())))
    _save_text(path, json.dumps(summary, indent=1))
    return summary


def load_pairs(condition, features, n_pairs=100_000, seed=0,
    workspace=lv_locations.workspace):
    '''Load the dataset's saved pair sample, sampling it once when missing.

    n_pairs and seed only apply when sampling; a saved sample is returned
    as is, with its own settings in n_pairs and seed.
    '''
    path = lv_locations.pair_sample_path(condition, workspace)
    if path.exists():
        with np.load(path) as saved: return {k: saved[k] for k in saved.files}
    pairs = sample_pairs(features, n_pairs, seed)
    save_pairs(condition, pairs, n_pairs, seed, workspace)
    return pairs


def save_pairs(condition, pairs, n_pairs, seed,
    workspace=lv_locations.workspace):
    '''Save a pair sample with its settings, replacing any saved sample.'''
    pairs.update(n_pairs=np.int64(n_pairs), seed=np.int64(seed))
    _save_npz(lv_locations.pair_sample_path(condition, workspace), pairs)


def sample_pairs(features, n_pairs=100_000, seed=0, max_distance=2):
    '''Sample token pairs per stratum, selecting on predictors only.

    features:      token feature table
    n_pairs:       maximum pairs per stratum; rare strata may have fewer
    seed:          random seed
    max_distance:  largest phone edit distance in the close_phones stratum
    Strata are listed in STRATA; close_phones holds different-label pairs
    with equal phone count and phone edit distance 1 to max_distance.
    Returns queries, others and stratum index arrays.
    '''
    generator = np.random.default_rng(seed)
    labels = features['label']
    speaker_labels = np.char.add(np.char.add(features['speaker'].astype(str),
        '\t'), labels.astype(str))
    groups = [(labels, True), (speaker_labels, True),
        (None, False), (features['speaker'], False)]
    parts = []
    for index, (group, same_label) in enumerate(groups):
        if group is None:
            queries, others = pf.random_pairs(len(labels), n_pairs,
                generator.integers(2**32), labels)
        else:
            queries, others = _draw_within(group, labels, same_label,
                n_pairs, generator)
        parts.append((queries, others, index))
    queries, others = _close_phone_pairs(features, n_pairs, max_distance,
        generator)
    parts.append((queries, others, len(groups)))
    return dict(queries=np.concatenate([p[0] for p in parts]),
        others=np.concatenate([p[1] for p in parts]),
        stratum=np.concatenate([np.full(len(p[0]), p[2]) for p in parts]))


def _draw_within(group, labels, same_label, n_pairs, generator,
    max_tries=50):
    '''Draw pairs within groups whose labels are equal or differ.'''
    ids, counts = np.unique(group, return_inverse=True, return_counts=True)[1:]
    eligible = np.flatnonzero(counts[ids] > 1)
    order = np.argsort(ids, kind='stable')
    starts = np.concatenate([[0], np.cumsum(counts)[:-1]])
    queries = generator.choice(eligible, size=n_pairs)
    others = np.full(n_pairs, -1)
    todo = np.arange(n_pairs)
    for _ in range(max_tries):
        if not len(todo): break
        query_ids = ids[queries[todo]]
        offset = (generator.random(len(todo)) * counts[query_ids]).astype(int)
        drawn = order[starts[query_ids] + offset]
        ok = drawn != queries[todo]
        ok &= (labels[drawn] == labels[queries[todo]]) == same_label
        others[todo[ok]] = drawn[ok]
        todo = todo[~ok]
    keep = others >= 0
    return queries[keep], others[keep]


def _close_phone_pairs(features, n_pairs, max_distance, generator,
    batch_size=200_000, max_rounds=50):
    '''Draw different-label pairs of equal length at small edit distance.'''
    n_phones = features['n_phones']
    found_queries, found_others, n_found = [], [], 0
    for _ in range(max_rounds):
        if n_found >= n_pairs: break
        queries, others = _draw_within(n_phones, features['label'], False,
            batch_size, generator)
        distance = pf.phone_edit_distance(features, queries, others)
        keep = (distance >= 1) & (distance <= max_distance)
        found_queries.append(queries[keep])
        found_others.append(others[keep])
        n_found += keep.sum()
    queries = np.concatenate(found_queries)[:n_pairs]
    return queries, np.concatenate(found_others)[:n_pairs]


def cosine(vectors, queries, others, chunk_size=50_000):
    '''Return cosine similarity of L2-normalized vector pairs in chunks.'''
    similarity = np.empty(len(queries))
    for start in range(0, len(queries), chunk_size):
        stop = start + chunk_size
        a = np.asarray(vectors[queries[start:stop]], dtype=np.float64)
        b = np.asarray(vectors[others[start:stop]], dtype=np.float64)
        similarity[start:stop] = np.einsum('ij,ij->i', a, b)
    return similarity


def pair_design(features, queries, others):
    '''Return predictor columns for each pair, keyed by name.

    Expects frequency columns merged into features. Columns with missing
    values are filled with their mean and get a {name}_unknown flag; an
    absent neighbouring word counts as not shared.
    '''
    f, q, o = features, queries, others
    cross = bool(np.all(features['boundary_index'] > 0))
    frequency = 'sequence_log_freq' if cross else 'word_log_freq'
    same_word = (f['label'][q] == f['label'][o]).astype(float)
    same_speaker = pf.same_speaker(f, q, o).astype(float)
    design = dict(same_word=same_word, same_speaker=same_speaker,
        same_recording=pf.same_recording(f, q, o).astype(float),
        same_component=pf.same_component(f, q, o).astype(float),
        same_gender=pf.same_gender(f, q, o),
        age_difference=pf.age_difference(f, q, o),
        log_duration_ratio=np.log(pf.duration_ratio(f, q, o)),
        phone_edit_distance=pf.phone_edit_distance(f, q, o),
        rate_difference=pf.absolute_difference(f, 'articulation_rate', q, o),
        shared_previous_word=np.nan_to_num(pf.shared_previous_word(f, q, o)),
        shared_next_word=np.nan_to_num(pf.shared_next_word(f, q, o)),
        frequency_difference=pf.absolute_difference(f, frequency, q, o),
        mean_n_phones=(f['n_phones'][q] + f['n_phones'][o]) / 2)
    if cross:
        design['shared_left_word'] = pf.shared_left_word(f, q, o).astype(float)
        design['shared_right_word'] = pf.shared_right_word(f, q,
            o).astype(float)
    design['same_word_x_same_speaker'] = same_word * same_speaker
    return _fill_missing(design)


def _fill_missing(design):
    '''Replace NaN by the column mean and add an unknown flag column.'''
    filled = {}
    for name, values in design.items():
        missing = np.isnan(values)
        if not missing.any():
            filled[name] = values
            continue
        if missing.all(): continue
        filled[name] = np.where(missing, np.nanmean(values), values)
        filled[f'{name}_unknown'] = missing.astype(float)
    return filled


def fit_pair_models(design, similarity, stratum, query_types,
    n_bootstrap=200, seed=0):
    '''Fit the linear model and matched contrasts with type bootstrap CIs.

    design:       predictor columns from pair_design
    similarity:   cosine similarity per pair
    stratum:      stratum index per pair, see STRATA
    query_types:  type index of each pair's query, the resampling unit
    Binary predictors stay 0/1, so their coefficients are differences in
    cosine; continuous predictors are standardized.
    '''
    names, X = design_matrix(design)
    weights = np.ones(len(similarity))
    coefficients, r2 = fit_ols(X, similarity, weights)
    cells = match_cells(design)
    contrasts = matched_contrasts(design, similarity, stratum, cells,
        weights)
    generator = np.random.default_rng(seed)
    n_types = query_types.max() + 1
    samples, contrast_samples = [], []
    for _ in range(n_bootstrap):
        counts = np.bincount(generator.integers(n_types, size=n_types),
            minlength=n_types)
        weights = counts[query_types].astype(float)
        samples.append(fit_ols(X, similarity, weights)[0])
        contrast_samples.append(list(matched_contrasts(design, similarity,
            stratum, cells, weights).values()))
    linear = _with_intervals(names, coefficients, samples)
    contrasts = _with_intervals(list(contrasts), list(contrasts.values()),
        contrast_samples)
    return dict(r2=r2, n=len(similarity), coefficients=linear,
        unique_r2=unique_r2(names, X, similarity, r2),
        contrasts=contrasts, mean_similarity=_stratum_means(similarity,
        stratum))


def design_matrix(design):
    '''Stack predictors with an intercept; standardize non-binary columns.

    Constant columns are dropped, since the intercept already covers them.
    '''
    names = [name for name, values in design.items() if values.std() > 0]
    columns = [np.ones(len(design[names[0]]))]
    for name in names:
        values = design[name]
        binary = np.isin(values, (0, 1)).all()
        if not binary: values = (values - values.mean()) / values.std()
        columns.append(values)
    return ['intercept'] + names, np.column_stack(columns)


def fit_ols(X, y, weights):
    '''Return weighted least-squares coefficients and the weighted R2.'''
    Xw = X * weights[:, None]
    beta = np.linalg.pinv(Xw.T @ X) @ (Xw.T @ y)
    residual = y - X @ beta
    mean = np.average(y, weights=weights)
    total = np.sum(weights * (y - mean) ** 2)
    return beta, 1 - np.sum(weights * residual ** 2) / total


def unique_r2(names, X, y, r2):
    '''Return the R2 lost when dropping each predictor with its derived
    columns: unknown flags and interactions that start with its name.
    '''
    weights = np.ones(len(y))
    result = {}
    for name in names[1:]:
        if name.endswith('_unknown') or '_x_' in name: continue
        keep = [i for i, other in enumerate(names)
            if other != name and not other.startswith(name + '_')]
        result[name] = r2 - fit_ols(X[:, keep], y, weights)[1]
    return result


def matched_contrasts(design, similarity, stratum, cells, weights):
    '''Return same-word minus different-word mean similarity in matches.

    same_speaker:       same word vs different word, both same speaker
    other_speaker:      same word vs random different word, other speakers
    phone_distance_1/2: same word vs close_phones pair at equal distance,
                        other speakers; words only, NaN if too few pairs
    matched_cells:      differences within match_cells, weighted by
                        same-word pairs; phones are not matched
    '''
    same_word = design['same_word'] == 1
    same_speaker = design['same_speaker'] == 1
    distance = design['phone_edit_distance']
    in_stratum = {name: stratum == i for i, name in enumerate(STRATA)}

    def mean(mask): return _weighted_mean(similarity, weights, mask)

    contrasts = dict(
        same_speaker=mean(in_stratum['same_word_same_speaker'])
            - mean(in_stratum['different_word_same_speaker']),
        other_speaker=mean(same_word & ~same_speaker)
            - mean(in_stratum['different_word'] & ~same_speaker))
    for d in (1, 2):
        close = in_stratum['close_phones'] & (distance == d) & ~same_speaker
        same = same_word & (distance == d) & ~same_speaker
        enough = min(same.sum(), close.sum()) >= 30
        contrasts[f'phone_distance_{d}'] = (mean(same) - mean(close)
            if enough else np.nan)
    contrasts['matched_cells'] = _cell_contrast(design['same_word'] == 1,
        similarity, cells, weights)
    return contrasts


def _weighted_mean(values, weights, mask):
    total = weights[mask].sum()
    if total == 0: return np.nan
    return float(np.sum(values[mask] * weights[mask]) / total)


def match_cells(design):
    '''Return a cell index per pair from speaker, recording, component,
    gender, neighbouring-word and quintile duration and rate matches.
    '''
    keys = ['same_speaker', 'same_recording', 'same_component',
        'shared_previous_word', 'shared_next_word']
    columns = [design[key] for key in keys]
    # Every selected speaker has a known gender, so same_gender is 0 or 1.
    columns.append(design['same_gender'])
    for name in ('log_duration_ratio', 'rate_difference'):
        edges = np.quantile(design[name], [0.2, 0.4, 0.6, 0.8])
        columns.append(np.searchsorted(edges, design[name]))
    return np.unique(np.column_stack(columns), axis=0,
        return_inverse=True)[1].ravel()


def _cell_contrast(same_word, similarity, cells, weights):
    '''Weighted mean of same-word minus different-word within cells.'''
    n_cells = cells.max() + 1
    sums, totals = [], []
    for mask in (same_word, ~same_word):
        w = weights * mask
        sums.append(np.bincount(cells, w * similarity, n_cells))
        totals.append(np.bincount(cells, w, n_cells))
    both = (totals[0] > 0) & (totals[1] > 0)
    means = [total[both] / count[both] for total, count in zip(sums, totals)]
    difference = means[0] - means[1]
    return float(np.average(difference, weights=totals[0][both]))


def _with_intervals(names, estimates, samples):
    low, high = np.full(len(names), np.nan), np.full(len(names), np.nan)
    samples = np.asarray(samples, dtype=float).reshape(-1, len(names))
    known = np.isfinite(samples).any(axis=0)
    if known.any():
        low[known], high[known] = np.nanpercentile(samples[:, known],
            [2.5, 97.5], axis=0)
    return {name: dict(estimate=float(e), ci_low=float(lo), ci_high=float(hi))
        for name, e, lo, hi in zip(names, estimates, low, high)}


def _stratum_means(similarity, stratum):
    return {name: float(similarity[stratum == i].mean())
        for i, name in enumerate(STRATA) if np.any(stratum == i)}


def format_summary(summary):
    '''Return a readable text summary of run_pair_models output.'''
    lines = [f"{summary['condition']} {summary['model']} layer "
        f"{summary['layer']} {summary['pooling']}: n={summary['n']}, "
        f"R2={summary['r2']:.3f}", 'mean similarity per stratum:']
    for name, value in summary['mean_similarity'].items():
        count = summary['stratum_counts'][name]
        lines.append(f'  {name:30s} {value:.3f}  (n={count})')
    lines.append('coefficients [95% type-bootstrap CI], unique R2:')
    unique = summary['unique_r2']
    for name, value in summary['coefficients'].items():
        extra = f"  {unique[name]:.4f}" if name in unique else ''
        lines.append(f"  {name:30s} {_interval(value)}{extra}")
    lines.append('matched contrasts (same word minus different word):')
    for name, value in summary['contrasts'].items():
        lines.append(f'  {name:30s} {_interval(value)}')
    return '\n'.join(lines)


def _interval(value):
    return (f"{value['estimate']:+.4f} [{value['ci_low']:+.4f}, "
        f"{value['ci_high']:+.4f}]")


def _save_npz(path, arrays):
    '''Write arrays via a temporary file so failures leave no output.'''
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=f'.{path.name}.',
        suffix='.npz', delete=False) as handle:
        np.savez(handle, **arrays)
    Path(handle.name).replace(path)


def _save_text(path, text):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile('w', dir=path.parent,
        prefix=f'.{path.name}.', delete=False) as handle:
        handle.write(text)
    Path(handle.name).replace(path)
