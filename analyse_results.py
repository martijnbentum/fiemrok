import numpy
import matplotlib.pyplot as plt
from imgcat import imgcat

def analyse_results(results, o, layer):
    r = results[layer]
    vs, md = o[layer]
    output = {}
    for index, d in enumerate(r):
        vmds = d['metadatas']
        n = len(vmds)
        gt_vmd = vs.metadatas[index]
        gt_label = gt_vmd.label
        gt_efk = gt_vmd.source_echoframe_keys[0]
        first_efk = vmds[0].source_echoframe_keys[0]
        second_efk = vmds[1].source_echoframe_keys[0]
        if gt_efk != first_efk and gt_efk != second_efk:
            m = f'{index} {gt_label} {layer}'
            m += f' {gt_efk} != {first_efk}'
            m += f' and {gt_efk} != {second_efk}'
            raise ValueError(m)
        labels = [x.label for x in vmds[1:]]
        acc = labels.count(gt_label) / (n-1)
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


    
