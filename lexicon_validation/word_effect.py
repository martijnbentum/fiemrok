'''Joint pair model of the word effect over words and cross-boundary items.

The model and its terms are defined in docs/word_effect.md. Both datasets
reuse their saved pair samples from factor_models, without the close_phones
and homophone strata: the model has no homophone term, and their defining
features would otherwise shift the baseline. Homophones are analysed as a
separate contrast in factor_models.
The word effect is the same_type_x_is_word coefficient: the extra gain in
cosine similarity for the same type when that type is a word rather than a
cross-boundary phone string. Confidence intervals resample query types
within each dataset.
'''

import json

import numpy as np

from . import factor_models as fm
from . import lv_locations
from . import pair_features as pf
from .neighbour_results import Result


CONDITIONS = ('word', 'cross_boundary')
EXCLUDED_STRATA = ('close_phones', 'homophone')
TERMS = ('is_word', 'same_type', 'same_type_x_is_word',
    'same_type_x_same_speaker', 'same_speaker', 'same_recording',
    'same_component', 'same_gender', 'shared_previous_word',
    'shared_next_word', 'cb_same_left_word', 'cb_same_right_word',
    'other_type_distance')
DERIVED = {'same_type_words': ('same_type', 'same_type_x_is_word')}


def run_word_effect_model(model='hubert', layer=9, pooling='mean',
    n_bootstrap=200, seed=0, workspace=lv_locations.workspace):
    '''Fit the joint word-effect model for one model, layer and pooling.

    model:        hubert or w2v
    layer:        3, 6, 9 or 12
    pooling:      center or mean
    n_bootstrap:  type resamples per dataset for confidence intervals
    seed:         bootstrap seed; pair samples are loaded as saved
    workspace:    validation workspace; output goes to factor_models/
    Returns the summary dict, also saved as JSON.
    '''
    parts = [load_dataset_pairs(condition, model, layer, pooling, workspace)
        for condition in CONDITIONS]
    design, centre = joint_design(parts)
    similarity = np.concatenate([part['similarity'] for part in parts])
    groups = _type_groups(parts)
    summary = fit_word_effect(design, similarity, groups, n_bootstrap, seed)
    summary.update(model=model, layer=layer, pooling=pooling, seed=seed,
        n_bootstrap=n_bootstrap, distance_centre=centre,
        pairs={part['condition']: _stratum_counts(part) for part in parts})
    path = lv_locations.word_effect_path(model, layer, pooling, workspace)
    fm._save_text(path, json.dumps(summary, indent=1))
    return summary


def load_dataset_pairs(condition, model, layer, pooling,
    workspace=lv_locations.workspace):
    '''Return one dataset's features, kept pairs and their similarities.'''
    filename = lv_locations.neighbour_results_path(model, condition, layer,
        pooling, workspace)
    with Result(filename) as result:
        features = dict(result.features)
        pairs = fm.load_pairs(condition, features, workspace=workspace)
        excluded = [fm.STRATA.index(name) for name in EXCLUDED_STRATA]
        keep = ~np.isin(pairs['stratum'], excluded)
        queries, others = pairs['queries'][keep], pairs['others'][keep]
        vectors = np.array(result.vectors)
        similarity = fm.cosine(vectors, queries, others)
    types = np.unique(features['label'], return_inverse=True)[1]
    return dict(condition=condition, features=features, queries=queries,
        others=others, stratum=pairs['stratum'][keep], similarity=similarity,
        query_types=types[queries])


def joint_design(parts):
    '''Return the joint design columns keyed by TERMS and the distance centre.

    parts: load_dataset_pairs output for each dataset
    other_type_distance is the phone edit distance over the longer item's
    phone count, centred on the mean of random different-type pairs of
    both datasets, and 0 for same-type pairs.
    '''
    columns = [pair_terms(part['features'], part['queries'], part['others'],
        part['condition'] == 'word') for part in parts]
    design = {name: np.concatenate([c[name] for c in columns])
        for name in columns[0]}
    random = np.concatenate([part['stratum'] == fm.STRATA.index(
        'different_word') for part in parts])
    distance = design.pop('normalised_distance')
    centre = float(distance[random].mean())
    same_type = design['same_type']
    design['other_type_distance'] = np.where(same_type == 1, 0,
        distance - centre)
    design['same_type_x_is_word'] = same_type * design['is_word']
    design['same_type_x_same_speaker'] = same_type * design['same_speaker']
    return {name: design[name] for name in TERMS}, centre


def pair_terms(features, queries, others, is_word):
    '''Return one dataset's pair columns before joining and centring.

    Cross-boundary terms are 0 for words. Every selected speaker has a
    known gender.
    '''
    f, q, o = features, queries, others
    zeros = np.zeros(len(q))
    longer = np.maximum(f['n_phones'][q], f['n_phones'][o])
    terms = dict(is_word=zeros + is_word,
        same_type=(f['label'][q] == f['label'][o]).astype(float),
        same_speaker=pf.same_speaker(f, q, o).astype(float),
        same_recording=pf.same_recording(f, q, o).astype(float),
        same_component=pf.same_component(f, q, o).astype(float),
        same_gender=np.nan_to_num(pf.same_gender(f, q, o)),
        shared_previous_word=np.nan_to_num(pf.shared_previous_word(f, q, o)),
        shared_next_word=np.nan_to_num(pf.shared_next_word(f, q, o)),
        normalised_distance=pf.phone_edit_distance(f, q, o) / longer)
    if is_word: terms.update(cb_same_left_word=zeros, cb_same_right_word=zeros)
    else:
        terms.update(cb_same_left_word=pf.shared_left_word(f, q,
            o).astype(float), cb_same_right_word=pf.shared_right_word(f, q,
            o).astype(float))
    return terms


def _type_groups(parts):
    '''Return per dataset its row mask and query types for the bootstrap.'''
    dataset = np.concatenate([np.full(len(part['queries']), i)
        for i, part in enumerate(parts)])
    return [(dataset == i, part['query_types'])
        for i, part in enumerate(parts)]


def fit_word_effect(design, similarity, groups, n_bootstrap=200, seed=0):
    '''Fit the joint model with confidence intervals from type resamples.

    design:      columns keyed by term name, see joint_design
    similarity:  cosine similarity per pair
    groups:      per dataset a (row mask, query type per masked row) pair;
                 types are resampled within each dataset
    Coefficients are in cosine units; other_type_distance is per unit of
    normalised distance. DERIVED adds sums of coefficients, such as the
    same-type gain for words.
    '''
    names = list(design)
    X = np.column_stack([np.ones(len(similarity))] +
        [design[name] for name in names])
    names = ['intercept'] + names
    coefficients, r2 = fm.fit_ols(X, similarity, np.ones(len(similarity)))
    generator = np.random.default_rng(seed)
    samples = []
    for _ in range(n_bootstrap):
        weights = _resample_weights(groups, len(similarity), generator)
        samples.append(fm.fit_ols(X, similarity, weights)[0])
    estimates = _with_derived(names, coefficients)
    samples = [_with_derived(names, sample) for sample in samples]
    labels = list(estimates)
    return dict(r2=float(r2), n=len(similarity),
        coefficients=fm._with_intervals(labels, list(estimates.values()),
        [list(sample.values()) for sample in samples]))


def _resample_weights(groups, n_rows, generator):
    '''Return row weights from resampling query types within each dataset.'''
    weights = np.zeros(n_rows)
    for mask, types in groups:
        n_types = types.max() + 1
        counts = np.bincount(generator.integers(n_types, size=n_types),
            minlength=n_types)
        weights[mask] = counts[types]
    return weights


def _with_derived(names, coefficients):
    '''Return coefficients by name with the DERIVED sums appended.'''
    values = dict(zip(names, map(float, coefficients)))
    for name, parts in DERIVED.items():
        values[name] = sum(values[part] for part in parts)
    return values


def _stratum_counts(part):
    counts = np.bincount(part['stratum'], minlength=len(fm.STRATA))
    return {name: int(count) for name, count in zip(fm.STRATA, counts)
        if name not in EXCLUDED_STRATA}


def format_summary(summary):
    '''Return a readable text summary of run_word_effect_model output.'''
    lines = [f"word effect, {summary['model']} layer {summary['layer']} "
        f"{summary['pooling']}: n={summary['n']}, R2={summary['r2']:.3f}, "
        f"distance centre {summary['distance_centre']:.3f}",
        'coefficients [95% type-bootstrap CI]:']
    for name, value in summary['coefficients'].items():
        lines.append(f'  {name:28s} {fm._interval(value)}')
    return '\n'.join(lines)
