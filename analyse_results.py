import numpy

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


def evaluate_index(vector_index, top_k=10):
    '''Return word-to-token precision lists without retaining all searches.'''
    if len(vector_index.metadatas) <= top_k:
        raise ValueError('index needs more entries than top_k')
    from progressbar import progressbar
    output = {}
    for index in progressbar(range(len(vector_index.metadatas))):
        metadata = vector_index.metadatas[index]
        result = vector_index.query_by_index(index, top_k=top_k + 1)
        score = precision_at_k(result, metadata, top_k=top_k)
        output.setdefault(metadata.label, []).append(score)
    return output


def analyse_results(results, o, layer):
    r = results[layer]
    vs, md = o[layer]
    output = {}
    for index, d in enumerate(r):
        vmds = d['metadatas']
        n = len(vmds)
        gt_vmd = vs.metadatas[index]
        gt_label = gt_vmd.label
        acc = precision_at_k(d, gt_vmd, top_k=n - 1)
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


    
