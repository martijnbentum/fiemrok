import json
from pathlib import Path

import faiss
import numpy
from progressbar import progressbar
import speech_vector_search as svs


def export_neighbours(model_name, layer, pooling, directory='svs_data',
    top_k=30, batch_size=512):
    '''Export ranked non-self neighbours for every token in both datasets.
    model_name: hubert or wav2vec2
    layer:      3, 6, 9, or 12
    pooling:    center or mean
    directory:  saved indexes and output JSONL files
    top_k:      neighbours to retain per query, default 30
    batch_size: queries searched together, default 512
    Returns a dictionary mapping dataset names to output paths.
    Source keys are stored as hexadecimal strings for JSON serialization.
    '''
    models = {'hubert': 'hubert1', 'wav2vec2': 'w2v'}
    if model_name not in models:
        raise ValueError('model_name must be hubert or wav2vec2')
    if not isinstance(layer, int) or layer not in (3, 6, 9, 12):
        raise ValueError('layer must be 3, 6, 9, or 12')
    if pooling not in ('center', 'mean'):
        raise ValueError('pooling must be center or mean')
    if not isinstance(top_k, int) or top_k < 1:
        raise ValueError('top_k must be a positive integer')
    if not isinstance(batch_size, int) or batch_size < 1:
        raise ValueError('batch_size must be a positive integer')
    model = models[model_name]
    word_suffix = '_mean' if pooling == 'mean' else ''
    names = {'word': f'{model}_{layer}{word_suffix}',
        'cross_boundary':
        f'cross_boundary_lexicon100_2200_{model}_{layer}_{pooling}'}
    directory = Path(directory)
    paths = {}
    old_threads = faiss.omp_get_max_threads()
    try:
        faiss.omp_set_num_threads(1)
        for dataset, name in names.items():
            index = svs.VectorIndex.load(str(directory), name)
            filename = directory / f'{name}_neighbours{top_k}.jsonl'
            _write_neighbours(index, filename, top_k, batch_size)
            paths[dataset] = filename
            print(f'Saved {filename}')
            del index
    finally:
        faiss.omp_set_num_threads(old_threads)
    return paths


def _write_neighbours(index, filename, top_k, batch_size):
    '''Stream all queries, publishing the output only after completion.'''
    n_tokens = len(index.metadatas)
    if n_tokens <= top_k:
        raise ValueError('index needs more entries than top_k')
    temporary = filename.with_suffix('.jsonl.tmp')
    with temporary.open('w', encoding='utf-8') as output:
        for start in progressbar(range(0, n_tokens, batch_size)):
            results = index.queries(index.vectors[start:start + batch_size],
                top_k=top_k + 1)
            for offset, result in enumerate(results):
                query = index.metadatas[start + offset]
                query_keys = set(query.source_echoframe_keys)
                neighbours = [dict(label=md.label,
                    source_keys_hex=md.source_echoframe_keys_hex,
                    similarity=float(score))
                    for md, score in zip(result['metadatas'], result['scores'])
                    if query_keys.isdisjoint(md.source_echoframe_keys)][:top_k]
                if len(neighbours) != top_k:
                    raise ValueError(f'expected {top_k} non-self neighbours')
                row = dict(label=query.label,
                    source_keys_hex=query.source_echoframe_keys_hex,
                    neighbours=neighbours)
                output.write(json.dumps(row, ensure_ascii=False) + '\n')
    temporary.replace(filename)

def precision_at_k(result, query_metadata, top_k=10):
    '''Score neighbours after excluding the query token by source key.'''
    if top_k < 1:
        raise ValueError('top_k must be positive')
    query_keys = set(query_metadata.source_echoframe_keys)
    neighbours = [md for md in result['metadatas']
        if query_keys.isdisjoint(md.source_echoframe_keys)][:top_k]
    if len(neighbours) < top_k:
        raise ValueError(f'expected {top_k} non-self neighbours')
    matches = sum(md.label == query_metadata.label for md in neighbours)
    return matches / top_k


def evaluate_index(vector_index, top_k=10, batch_size=256):
    '''Return per-word token scores, searching bounded batches of vectors.
    batch_size: number of queries per search; use 1 for individual queries
    '''
    if len(vector_index.metadatas) <= top_k:
        raise ValueError('index needs more entries than top_k')
    n_tokens = len(vector_index.metadatas)
    print(f'evaluating {n_tokens} queries in batches of {batch_size}')
    output = {}
    for start in progressbar(range(0, n_tokens, batch_size)):
        stop = min(start + batch_size, n_tokens)
        if batch_size == 1:
            results = [vector_index.query_by_index(start, top_k=top_k + 1)]
        else:
            results = vector_index.queries(vector_index.vectors[start:stop],
                top_k=top_k + 1)
        for offset, result in enumerate(results):
            metadata = vector_index.metadatas[start + offset]
            score = precision_at_k(result, metadata, top_k=top_k)
            output.setdefault(metadata.label, []).append(score)
    return output


def analyse_results(results, o, layer):
    '''Score ten non-self neighbours per saved query, in index order.'''
    r = results[layer]
    vs, md = o[layer]
    output = {}
    for index, d in enumerate(r):
        gt_vmd = vs.metadatas[index]
        gt_label = gt_vmd.label
        acc = precision_at_k(d, gt_vmd, top_k=10)
        if gt_label not in output:
            output[gt_label] = []
        output[gt_label].append(acc)
    return output

def per_word(output):
    o = {}
    for label, accs in output.items():
        o[label] = numpy.mean(accs)
    return o 

def overall_result(output):
    accs = []
    for label, acc in output.items():
        accs += acc
    return numpy.mean(accs), numpy.std(accs)


def plot_precision(results_per_layer, save_path='_plot.png'):
    '''bar plot of average precision at 10 with 95% CI per layer.
    results_per_layer   {layer: output_dict} where output_dict is from
                        analyse_results(); each value is a list of per-item
                        precision scores for one word type
    save_path           path to save the plot
    '''
    import matplotlib.pyplot as plt
    from imgcat import imgcat

    layers = sorted(results_per_layer.keys())
    means, cis = [], []
    for layer in layers:
        output = results_per_layer[layer]
        mean, std = overall_result(output)
        n = sum(len(v) for v in output.values())
        ci = 1.96 * std / numpy.sqrt(n)
        means.append(mean)
        cis.append(ci)

    x = range(len(layers))
    fig, ax = plt.subplots(figsize=(max(4, len(layers) * 0.6), 4))
    ax.bar(x, means, yerr=cis, capsize=5, color='steelblue', alpha=0.85)
    ax.set_xticks(list(x))
    ax.set_xticklabels([str(l) for l in layers])
    ax.set_xlabel('layer')
    ax.set_ylabel('average precision at 10')
    ax.set_title('CGN lexicon: average precision at 10 with 95% CI')
    ax.set_ylim(0, 1)
    fig.tight_layout()
    fig.savefig(save_path)
    imgcat(open(save_path, 'rb'))
    return fig, ax


    
