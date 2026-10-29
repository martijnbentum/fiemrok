'''Compare lexicon and cross-boundary retrieval with type bootstrap CIs.'''

import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np
from scipy.stats import bootstrap

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import PercentFormatter


ROOT = Path(__file__).resolve().parents[1]
LAYERS = [3, 6, 9, 12]
SEED = 20261029
N_RESAMPLES = 10000
DATASETS = [
    ('Word-aligned', 'cgn_lexicon', 2466, 'centre'),
    ('Across-word', 'cross_boundary_lexicon100_2200', 2200, 'center'),
]


def summarize(scores, n_types):
    '''Return mean and 95% BCa interval by resampling whole type means.'''
    values = np.asarray([scores[key] for key in sorted(scores)], dtype=float)
    if values.shape != (n_types, 100):
        raise ValueError(f'expected {n_types} types with 100 scores each')
    if not np.isfinite(values).all() or ((values < 0) | (values > 1)).any():
        raise ValueError('scores must be finite and between 0 and 1')
    type_means = values.mean(axis=1)
    # Resetting the seed pairs the draws across conditions in each dataset.
    result = bootstrap((type_means,), np.mean, n_resamples=N_RESAMPLES,
        batch=128, confidence_level=0.95, method='BCa',
        rng=np.random.default_rng(SEED))
    low, high = result.confidence_interval
    return type_means.mean(), low, high


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, default=ROOT / 'svs_data')
    parser.add_argument('--output', type=Path,
        default=ROOT / 'notes/assets/lexicon_cross_boundary_by_dataset')
    args = parser.parse_args()
    plt.rcParams.update({'font.size': 11, 'svg.fonttype': 'none'})
    fig, axes = plt.subplots(1, 2, figsize=(11, 6.4), sharey=True)
    models = [('w2v', 'Wav2Vec 2.0', '#0072B2'),
        ('hubert1', 'HuBERT', '#D55E00')]
    poolings = [('mean', '-', 'o'), ('centre', '--', 's')]
    for ax, (dataset, prefix, n_types, centre_suffix) in zip(axes, DATASETS):
        type_set = None
        for model, title, color in models:
            for pooling, style, marker in poolings:
                suffix = 'mean' if pooling == 'mean' else centre_suffix
                filename = f'{prefix}_{model}_{suffix}_results.json'
                results = json.loads((args.data_dir / filename).read_text())
                if set(results) != {str(layer) for layer in LAYERS}:
                    raise ValueError(f'unexpected layers in {filename}')
                summaries = []
                for layer in LAYERS:
                    scores = results[str(layer)]
                    keys = set(scores)
                    if type_set is not None and keys != type_set:
                        raise ValueError(f'type sets differ for {dataset}')
                    type_set = keys
                    summary = summarize(scores, n_types)
                    summaries.append(summary)
                    mean, low, high = summary
                    print(f'{dataset} | {title} | {pooling} | {layer} | '
                        f'{mean:.4f} | [{low:.4f}, {high:.4f}]', flush=True)
                means, lows, highs = np.asarray(summaries).T
                intervals = np.asarray([means - lows, highs - means])
                ax.plot(LAYERS, means, color=color, linestyle=style,
                    linewidth=1.95, zorder=2)
                ax.plot(LAYERS, means, color=color, linestyle='none',
                    marker=marker, markersize=2.5, alpha=0.65, zorder=4)
                ax.errorbar(LAYERS, means, yerr=intervals, fmt='none',
                    ecolor=color, elinewidth=1.8, capsize=5, capthick=1.8,
                    zorder=3)
        ax.set_title(dataset, fontsize=15, pad=12)
        ax.set_xlabel('Layer', labelpad=8)
        ax.set_xticks(LAYERS)
        ax.set_xlim(2.5, 12.5)
        ax.set_ylim(0, 1)
        ax.set_yticks(np.arange(0, 1.01, 0.2))
        ax.yaxis.set_major_formatter(PercentFormatter(xmax=1, decimals=0))
        ax.set_axisbelow(True)
        ax.yaxis.grid(True, color='#E3E7EB', linewidth=0.8)
        ax.spines[['top', 'right']].set_visible(False)
        ax.spines[['left', 'bottom']].set_color('#BCC3CA')
        ax.tick_params(axis='both', length=3, color='#BCC3CA')
    axes[0].set_ylabel('Mean precision at 10', labelpad=10)
    handles = [Line2D([], [], color=color, linewidth=4, label=title)
        for _, title, color in models]
    handles += [Line2D([], [], color='#444444', linestyle=style,
        linewidth=1.95, label=label)
        for label, style in [('Mean pooling', '-'),
        ('Centre pooling', '--')]]
    fig.suptitle('Word-aligned and across-word retrieval', fontsize=18,
        y=0.97)
    fig.legend(handles=handles, loc='upper center', ncol=4,
        bbox_to_anchor=(0.5, 0.91), frameon=False, handlelength=2.5,
        columnspacing=1.7)
    fig.subplots_adjust(left=0.085, right=0.98, bottom=0.12, top=0.79,
        wspace=0.12)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for suffix in ['.png', '.svg', '.pdf']:
        fig.savefig(args.output.with_suffix(suffix), dpi=200)
    svg_path = args.output.with_suffix('.svg')
    lines = svg_path.read_text().splitlines()
    svg_path.write_text('\n'.join(line.rstrip() for line in lines) + '\n')
    plt.close(fig)


if __name__ == '__main__':
    main()
