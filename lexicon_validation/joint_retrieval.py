'''Retrieval of words and cross-boundary items in one joint index.

Each dataset was searched in its own index, so their precision differs
partly through competitor size and density. Here every word and
cross-boundary token is searched against all tokens of both datasets
(excluding itself), so both face the same competitors. A neighbour is
correct when it has the query's type within the same dataset. The
precision model from precision_model is refitted on the joint precision,
and the share of neighbours from the other dataset is reported.
'''

import json

import numpy as np
from speech_vector_search import gpu_search

from . import factor_models as fm
from . import lv_locations
from . import precision_model as pm
from . import word_effect as we
from .neighbour_results import Result


def build_joint_neighbours(model='hubert', layer=9, pooling='mean',
    top_k=30, batch_size=512, device='cpu', overwrite=False,
    workspace=lv_locations.workspace):
    '''Search all tokens of both datasets in one index and save neighbours.

    model:       hubert or w2v
    layer:       3, 6, 9 or 12
    pooling:     center or mean
    top_k:       neighbours saved per query
    batch_size:  queries scored at once
    device:      torch device for scoring, cpu or cuda
    overwrite:   rebuild an existing file
    workspace:   validation workspace; output goes to neighbours/joint/
    Rows follow word tokens, then cross-boundary tokens, each in their
    vector-index order. Returns the saved arrays.
    '''
    path = lv_locations.joint_neighbours_path(model, layer, pooling,
        workspace)
    if path.exists() and not overwrite: return load_joint_neighbours(path)
    vectors, sizes = [], []
    for condition in we.CONDITIONS:
        filename = lv_locations.neighbour_results_path(model, condition,
            layer, pooling, workspace)
        with Result(filename) as result:
            vectors.append(np.array(result.vectors))
        sizes.append(len(vectors[-1]))
    vectors = np.concatenate(vectors)
    scorer = gpu_search.GpuScorer(vectors, device=device)
    neighbours = np.empty((len(vectors), top_k), dtype=np.int32)
    similarities = np.empty((len(vectors), top_k), dtype=np.float32)
    for start, scores in scorer.cosine_batches(vectors, batch_size):
        stop = start + len(scores)
        rows, values = gpu_search.k_nearest_neighbours_rows(scores, top_k,
            np.arange(start, stop))
        neighbours[start:stop], similarities[start:stop] = rows, values
    arrays = dict(neighbours=neighbours, similarities=similarities,
        sizes=np.array(sizes), conditions=np.array(we.CONDITIONS))
    fm._save_npz(path, arrays)
    return arrays


def load_joint_neighbours(path):
    with np.load(path) as saved: return {k: saved[k] for k in saved.files}


def joint_features(parts):
    '''Return label and speaker columns over both datasets' tokens.

    Labels are prefixed with the dataset, so a type is only correct within
    its own dataset.
    '''
    labels = [np.char.add(f"{p['condition']}:", p['features']['label']
        .astype(str)) for p in parts]
    speakers = [p['features']['speaker'] for p in parts]
    return dict(label=np.concatenate(labels),
        speaker=np.concatenate(speakers))


def run_joint_precision_model(model='hubert', layer=9, pooling='mean',
    n_bootstrap=200, seed=0, device='cpu',
    workspace=lv_locations.workspace):
    '''Fit the precision model on precision@10 from the joint index.

    Builds the joint neighbours first when missing; see
    build_joint_neighbours for device. Returns the summary dict, also
    saved as JSON.
    '''
    joint = build_joint_neighbours(model, layer, pooling, device=device,
        workspace=workspace)
    parts = [pm.load_queries(condition, model, layer, pooling, workspace)
        for condition in we.CONDITIONS]
    sizes = [len(p['types']) for p in parts]
    if list(joint['sizes']) != sizes:
        raise ValueError('joint neighbours do not match the datasets')
    features = joint_features(parts)
    neighbours = joint['neighbours'].astype(np.int64)
    summary = dict(model=model, layer=layer, pooling=pooling, seed=seed,
        n_bootstrap=n_bootstrap,
        other_dataset_share=other_dataset_share(neighbours, sizes))
    for outcome in pm.OUTCOMES:
        precision = pm.precision(features, neighbours,
            outcome == 'cross_speaker')[0]
        summary[outcome] = pm.fit_outcome(parts, precision, n_bootstrap,
            seed)
    path = lv_locations.joint_precision_path(model, layer, pooling,
        workspace)
    fm._save_text(path, json.dumps(summary, indent=1))
    return summary


def other_dataset_share(neighbours, sizes, top_k=pm.TOP_K):
    '''Return, per query dataset, the share of top_k neighbours from the
    other dataset.
    '''
    dataset = np.repeat(np.arange(len(sizes)), sizes)
    other = dataset[neighbours[:, :top_k]] != dataset[:, None]
    return {condition: float(other[dataset == i].mean())
        for i, condition in enumerate(we.CONDITIONS)}


def format_summary(summary):
    '''Return a readable text summary of run_joint_precision_model output.'''
    shares = ', '.join(f'{k} {v:.3f}'
        for k, v in summary['other_dataset_share'].items())
    lines = [f"joint index precision@10, {summary['model']} layer "
        f"{summary['layer']} {summary['pooling']}",
        f'top-10 neighbours from the other dataset: {shares}']
    for outcome in pm.OUTCOMES:
        fit = summary[outcome]
        means = ', '.join(f'{k} {v:.3f}'
            for k, v in fit['mean_precision'].items())
        effect = fm._interval(fit['coefficients']['is_word'])
        lines.append(f"{outcome}: n={fit['n']}, mean {means}, "
            f'is_word {effect}')
    return '\n'.join(lines)
